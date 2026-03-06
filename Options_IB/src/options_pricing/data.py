"""Stage 1 — Data Collection via Interactive Brokers.

Connects to TWS/IB Gateway, pulls the options chain for a given ticker,
filters to the expiry closest to a target DTE window, removes illiquid
strikes, and returns a clean DataFrame ready for calibration.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, Option, util


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7496          # TWS live
DEFAULT_CLIENT_ID = 1
DEFAULT_TICKER = "SPY"
DEFAULT_EXCHANGE = "SMART"
DEFAULT_CURRENCY = "USD"
TARGET_DTE_LOW = 30
TARGET_DTE_HIGH = 45
MIN_BID = 0.01               # filter strikes with bid below this (low for delayed data)
MIN_VOLUME = 0               # disabled — volume is often 0 in delayed/closed market data


@dataclass
class OptionsSnapshot:
    """Container for a single-expiry options snapshot."""

    ticker: str
    spot: float
    expiry: str                # ISO format YYYYMMDD
    dte: int
    risk_free_rate: float
    chains: pd.DataFrame       # columns: strike, mid, bid, ask, iv, volume, oi, right


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def connect(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    client_id: int = DEFAULT_CLIENT_ID,
) -> IB:
    """Connect to TWS / IB Gateway and return the IB instance."""
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib


def disconnect(ib: IB) -> None:
    ib.disconnect()


def _safe_float(val, default: float = np.nan) -> float:
    """Safely convert an IB field to float, returning default for None/-1/NaN."""
    if val is None or val == -1:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Core data pipeline
# ---------------------------------------------------------------------------

def _pick_expiry(
    expiries: list[str],
    dte_low: int = TARGET_DTE_LOW,
    dte_high: int = TARGET_DTE_HIGH,
) -> str:
    """Select the expiry closest to the midpoint of the target DTE window."""
    today = dt.date.today()
    target_mid = (dte_low + dte_high) / 2
    best = None
    best_diff = float("inf")
    for exp_str in expiries:
        exp_date = dt.datetime.strptime(exp_str, "%Y%m%d").date()
        dte = (exp_date - today).days
        if dte < 1:
            continue
        diff = abs(dte - target_mid)
        if diff < best_diff:
            best_diff = diff
            best = exp_str
    if best is None:
        raise ValueError("No valid expiries found in the chain.")
    return best


def _batch_qualify(ib: IB, contracts: list, batch_size: int = 50) -> list:
    """Qualify contracts in batches to avoid IB rate limits."""
    for i in range(0, len(contracts), batch_size):
        batch = contracts[i : i + batch_size]
        ib.qualifyContracts(*batch)
    return [c for c in contracts if c.conId > 0]


def _request_market_data(
    ib: IB,
    contracts: list[Option],
    batch_size: int = 50,
) -> list:
    """Request snapshot market data for a list of option contracts."""
    all_tickers = []
    for i in range(0, len(contracts), batch_size):
        batch = contracts[i : i + batch_size]
        tickers = [ib.reqMktData(c, "", False, False) for c in batch]
        ib.sleep(5)
        all_tickers.extend(tickers)
    return all_tickers


def fetch_options_chain(
    ib: IB,
    ticker: str = DEFAULT_TICKER,
    exchange: str = DEFAULT_EXCHANGE,
    currency: str = DEFAULT_CURRENCY,
    dte_low: int = TARGET_DTE_LOW,
    dte_high: int = TARGET_DTE_HIGH,
    risk_free_rate: float = 0.05,
) -> OptionsSnapshot:
    """Fetch and filter an options chain from IB for a single expiry."""

    # 1. Qualify the underlying and get spot price
    #    Type 4 = delayed-frozen: returns live when available, falls back to
    #    last available delayed/closing data when live subscription is missing
    #    or market is closed. Best of both worlds.
    # Type 3 = delayed data. Works without funding the account.
    # Delayed data is 15-min behind but sufficient for options analysis.
    ib.reqMarketDataType(3)
    stock = Stock(ticker, exchange, currency)
    ib.qualifyContracts(stock)
    ib.reqMktData(stock, "", False, False)
    ib.sleep(3)

    tickers_list = ib.reqTickers(stock)
    if not tickers_list:
        raise RuntimeError(f"No ticker data returned for {ticker}")
    spot_ticker = tickers_list[0]

    spot = _safe_float(spot_ticker.marketPrice())
    if np.isnan(spot):
        spot = _safe_float(spot_ticker.last)
    if np.isnan(spot):
        spot = _safe_float(spot_ticker.close)
    if np.isnan(spot):
        # Try delayed fields
        if hasattr(spot_ticker, "delayedLast"):
            spot = _safe_float(spot_ticker.delayedLast)
        if np.isnan(spot) and hasattr(spot_ticker, "delayedClose"):
            spot = _safe_float(spot_ticker.delayedClose)
    if np.isnan(spot):
        raise RuntimeError(
            f"Cannot determine spot price for {ticker}. "
            "Check your IB market data subscriptions."
        )

    # 2. Get all available option chains
    chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    if not chains:
        raise RuntimeError(f"No option chains returned for {ticker}")

    # Use SMART exchange chain (largest set of strikes/expiries)
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
    expiries = sorted(chain.expirations)
    strikes = sorted(chain.strikes)

    # 3. Pick target expiry
    expiry = _pick_expiry(expiries, dte_low, dte_high)
    exp_date = dt.datetime.strptime(expiry, "%Y%m%d").date()
    dte = (exp_date - dt.date.today()).days

    # 4. Filter strikes to a reasonable range around spot (±30%)
    strike_lo = spot * 0.70
    strike_hi = spot * 1.30
    filtered_strikes = [k for k in strikes if strike_lo <= k <= strike_hi]

    # Remove half-dollar strikes ($2.50 increments) — these are weekly-only
    # and don't exist for standard monthly/monthly-like expiries. IB returns
    # them in the combined strike list across all expiries.
    pre_count = len(filtered_strikes)
    filtered_strikes = [k for k in filtered_strikes if k == int(k)]
    n_removed = pre_count - len(filtered_strikes)
    if n_removed > 0:
        print(f"  Removed {n_removed} half-dollar strikes (weekly-only, "
              f"don't exist for {expiry})")

    # 5. Build option contracts for both calls and puts
    contracts = []
    for right in ("C", "P"):
        for k in filtered_strikes:
            opt = Option(ticker, expiry, k, right, exchange)
            contracts.append(opt)

    contracts = _batch_qualify(ib, contracts)

    # 6. Request market data snapshots
    tickers = _request_market_data(ib, contracts)

    # 7. Parse into rows
    try:
        rows = []
        for t in tickers:
            c = t.contract
            bid = _safe_float(t.bid, 0.0)
            ask = _safe_float(t.ask, 0.0)
            # Fall back to delayed bid/ask if live fields are empty
            if bid <= 0:
                bid = _safe_float(getattr(t, "delayedBid", None), 0.0)
            if ask <= 0:
                ask = _safe_float(getattr(t, "delayedAsk", None), 0.0)
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0

            iv = np.nan
            if t.modelGreeks is not None:
                iv = _safe_float(getattr(t.modelGreeks, "impliedVol", None))
            # Fallback: use lastGreeks
            if np.isnan(iv) and hasattr(t, "lastGreeks") and t.lastGreeks is not None:
                iv = _safe_float(getattr(t.lastGreeks, "impliedVol", None))

            vol = _safe_float(getattr(t, "volume", None), 0.0)
            if vol <= 0:
                vol = _safe_float(getattr(t, "delayedVolume", None), 0.0)
            oi = _safe_float(getattr(t, "openInterest", None), 0.0)

            rows.append({
                "strike": c.strike,
                "right": c.right,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "iv": iv,
                "volume": vol,
                "oi": oi,
            })
    finally:
        # Always cancel market data subscriptions to avoid leaking
        for t in tickers:
            try:
                ib.cancelMktData(t.contract)
            except Exception:
                pass

    df = pd.DataFrame(rows)

    # 8. Filter illiquid strikes
    n_before = len(df)
    df = df[df["bid"] >= MIN_BID]
    df = df[df["mid"] > 0]
    df = df.dropna(subset=["iv"])
    # Volume filter disabled for delayed data compatibility
    if MIN_VOLUME > 0 and (df["volume"] > 0).any():
        df = df[df["volume"] >= MIN_VOLUME]
    df = df.sort_values(["right", "strike"]).reset_index(drop=True)

    n_filtered = n_before - len(df)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} illiquid strikes, {len(df)} remaining")

    if df.empty:
        raise RuntimeError(
            f"No liquid options found for {ticker} expiry {expiry}. "
            "Check that the market is open or loosen filter thresholds."
        )

    return OptionsSnapshot(
        ticker=ticker,
        spot=spot,
        expiry=expiry,
        dte=dte,
        risk_free_rate=risk_free_rate,
        chains=df,
    )


# ---------------------------------------------------------------------------
# Convenience: run as standalone
# ---------------------------------------------------------------------------

def pull(
    ticker: str = DEFAULT_TICKER,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    client_id: int = DEFAULT_CLIENT_ID,
    dte_low: int = TARGET_DTE_LOW,
    dte_high: int = TARGET_DTE_HIGH,
    risk_free_rate: float = 0.05,
) -> OptionsSnapshot:
    """High-level helper: connect, fetch, disconnect, return snapshot."""
    ib = connect(host, port, client_id)
    try:
        snapshot = fetch_options_chain(
            ib, ticker,
            dte_low=dte_low,
            dte_high=dte_high,
            risk_free_rate=risk_free_rate,
        )
    finally:
        disconnect(ib)
    return snapshot


# ---------------------------------------------------------------------------
# CSV export / import — use when IB data subscription is limited
# ---------------------------------------------------------------------------

def save_snapshot(snapshot: OptionsSnapshot, path: str) -> None:
    """Save an OptionsSnapshot to CSV with metadata in the header."""
    import json
    meta = {
        "ticker": snapshot.ticker,
        "spot": snapshot.spot,
        "expiry": snapshot.expiry,
        "dte": snapshot.dte,
        "risk_free_rate": snapshot.risk_free_rate,
    }
    with open(path, "w") as f:
        f.write(f"# {json.dumps(meta)}\n")
        snapshot.chains.to_csv(f, index=False)
    print(f"Snapshot saved to {path}")


def load_snapshot(path: str) -> OptionsSnapshot:
    """Load an OptionsSnapshot from a CSV previously saved with save_snapshot,
    or from a manually created CSV.

    For manual CSVs, the file must have columns:
        strike, right, bid, ask, mid, iv, volume, oi

    And a JSON comment on the first line:
        # {"ticker": "SPY", "spot": 580.0, "expiry": "20260417", "dte": 45, "risk_free_rate": 0.05}
    """
    import json
    with open(path) as f:
        header_line = f.readline().strip()
        if not header_line.startswith("#"):
            raise ValueError(
                "First line must be a JSON comment: # {\"ticker\": ..., \"spot\": ..., ...}"
            )
        meta = json.loads(header_line[1:].strip())
        df = pd.read_csv(f)

    required = {"ticker", "spot", "expiry", "dte", "risk_free_rate"}
    missing = required - set(meta.keys())
    if missing:
        raise ValueError(f"Missing metadata fields: {missing}")

    required_cols = {"strike", "right", "mid", "iv"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing CSV columns: {missing_cols}")

    return OptionsSnapshot(
        ticker=meta["ticker"],
        spot=float(meta["spot"]),
        expiry=str(meta["expiry"]),
        dte=int(meta["dte"]),
        risk_free_rate=float(meta["risk_free_rate"]),
        chains=df,
    )


def fetch_multiple_chains(
    ib: IB,
    ticker: str = DEFAULT_TICKER,
    dte_targets: list[int] | None = None,
    exchange: str = DEFAULT_EXCHANGE,
    currency: str = DEFAULT_CURRENCY,
    risk_free_rate: float = 0.05,
) -> list[OptionsSnapshot]:
    """Fetch options chains for multiple DTE targets in a single IB session.

    Deduplicates expiries that resolve to the same date.
    """
    if dte_targets is None:
        dte_targets = [30, 60, 90, 180]

    snapshots = []
    seen_expiries: set[str] = set()

    for dte_target in sorted(dte_targets):
        margin = max(7, dte_target // 10)
        dte_lo = max(1, dte_target - margin)
        dte_hi = dte_target + margin
        try:
            snap = fetch_options_chain(
                ib, ticker, exchange, currency,
                dte_low=dte_lo,
                dte_high=dte_hi,
                risk_free_rate=risk_free_rate,
            )
            if snap.expiry in seen_expiries:
                print(f"  Skipping duplicate expiry {snap.expiry} "
                      f"(already fetched for a nearby DTE target)")
                continue
            seen_expiries.add(snap.expiry)
            snapshots.append(snap)
        except Exception as e:
            print(f"  Warning: Could not fetch for ~{dte_target} DTE: {e}")

    if not snapshots:
        raise RuntimeError("No expiries could be fetched for any DTE target.")

    return snapshots


def pull_multiple(
    ticker: str = DEFAULT_TICKER,
    dte_targets: list[int] | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    client_id: int = DEFAULT_CLIENT_ID,
    risk_free_rate: float = 0.05,
) -> list[OptionsSnapshot]:
    """Fetch multiple expiries in a single IB session."""
    ib = connect(host, port, client_id)
    try:
        snapshots = fetch_multiple_chains(
            ib, ticker, dte_targets,
            risk_free_rate=risk_free_rate,
        )
    finally:
        disconnect(ib)
    return snapshots


if __name__ == "__main__":
    snap = pull()
    print(f"Ticker: {snap.ticker}  Spot: {snap.spot:.2f}  "
          f"Expiry: {snap.expiry}  DTE: {snap.dte}")
    print(f"\nCalls: {len(snap.chains[snap.chains['right'] == 'C'])}  "
          f"Puts: {len(snap.chains[snap.chains['right'] == 'P'])}")
    print(snap.chains.to_string(index=False))
