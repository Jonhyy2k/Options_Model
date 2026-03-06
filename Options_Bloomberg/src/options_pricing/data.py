"""Stage 1 — Data Collection via Bloomberg Terminal.

Connects to the Bloomberg API (blpapi), pulls the options chain for a given
ticker, filters to the expiry closest to a target DTE window, removes illiquid
strikes, and returns a clean DataFrame ready for calibration.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import blpapi


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "SPY"
TARGET_DTE_LOW = 30
TARGET_DTE_HIGH = 45
MIN_BID = 0.01
BBG_HOST = "localhost"
BBG_PORT = 8194


@dataclass
class OptionsSnapshot:
    """Container for a single-expiry options snapshot."""

    ticker: str
    spot: float
    expiry: str                # YYYYMMDD
    dte: int
    risk_free_rate: float
    chains: pd.DataFrame       # columns: strike, mid, bid, ask, iv, volume, oi, right


# ---------------------------------------------------------------------------
# Bloomberg helpers
# ---------------------------------------------------------------------------

def _start_session(host: str = BBG_HOST, port: int = BBG_PORT) -> blpapi.Session:
    """Start a Bloomberg API session."""
    options = blpapi.SessionOptions()
    options.setServerHost(host)
    options.setServerPort(port)
    session = blpapi.Session(options)
    if not session.start():
        raise RuntimeError(
            "Failed to start Bloomberg session. "
            "Make sure the Bloomberg Terminal is running."
        )
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata service.")
    return session


def _send_and_collect(session: blpapi.Session, request) -> list:
    """Send a request and collect all response messages."""
    session.sendRequest(request)
    messages = []
    while True:
        event = session.nextEvent(10000)
        for msg in event:
            messages.append(msg)
        if event.eventType() == blpapi.Event.RESPONSE:
            break
    return messages


def _bdp(session: blpapi.Session, security: str, fields: list[str]) -> dict:
    """Bloomberg Data Point — single security, multiple fields."""
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", security)
    for f in fields:
        request.append("fields", f)

    result = {}
    for msg in _send_and_collect(session, request):
        if not msg.hasElement("securityData"):
            continue
        sec_data = msg.getElement("securityData")
        for i in range(sec_data.numValues()):
            item = sec_data.getValueAsElement(i)
            if not item.hasElement("fieldData"):
                continue
            fd = item.getElement("fieldData")
            for f in fields:
                if fd.hasElement(f):
                    el = fd.getElement(f)
                    try:
                        result[f] = el.getValueAsFloat()
                    except Exception:
                        try:
                            result[f] = str(el.getValueAsString())
                        except Exception:
                            pass
    return result


def _bds(session: blpapi.Session, security: str, field: str,
         overrides: dict | None = None) -> list[dict]:
    """Bloomberg Data Set — returns bulk data as a list of dicts."""
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    request.append("securities", security)
    request.append("fields", field)

    if overrides:
        ovrd_elem = request.getElement("overrides")
        for k, v in overrides.items():
            o = ovrd_elem.appendElement()
            o.setElement("fieldId", k)
            o.setElement("value", str(v))

    rows = []
    for msg in _send_and_collect(session, request):
        if not msg.hasElement("securityData"):
            continue
        sec_data = msg.getElement("securityData")
        for i in range(sec_data.numValues()):
            item = sec_data.getValueAsElement(i)
            if not item.hasElement("fieldData"):
                continue
            fd = item.getElement("fieldData")
            if not fd.hasElement(field):
                continue
            bulk = fd.getElement(field)
            for j in range(bulk.numValues()):
                row_elem = bulk.getValueAsElement(j)
                row = {}
                for k in range(row_elem.numElements()):
                    el = row_elem.getElement(k)
                    name = str(el.name())
                    try:
                        row[name] = el.getValueAsFloat()
                    except Exception:
                        try:
                            row[name] = str(el.getValueAsString())
                        except Exception:
                            row[name] = None
                rows.append(row)
    return rows


def _bulk_ref(session: blpapi.Session, securities: list[str],
              fields: list[str]) -> dict[str, dict]:
    """Bulk reference data for multiple securities."""
    service = session.getService("//blp/refdata")
    request = service.createRequest("ReferenceDataRequest")
    for sec in securities:
        request.append("securities", sec)
    for f in fields:
        request.append("fields", f)

    results = {}
    for msg in _send_and_collect(session, request):
        if not msg.hasElement("securityData"):
            continue
        sec_data = msg.getElement("securityData")
        for i in range(sec_data.numValues()):
            item = sec_data.getValueAsElement(i)
            sec_name = item.getElementAsString("security")
            row = {}
            if item.hasElement("fieldData"):
                fd = item.getElement("fieldData")
                for f in fields:
                    if fd.hasElement(f):
                        el = fd.getElement(f)
                        try:
                            row[f] = el.getValueAsFloat()
                        except Exception:
                            try:
                                row[f] = str(el.getValueAsString())
                            except Exception:
                                row[f] = None
            results[sec_name] = row
    return results


# ---------------------------------------------------------------------------
# Core data pipeline
# ---------------------------------------------------------------------------

def _to_bbg(ticker: str) -> str:
    """Ensure Bloomberg format: 'AAPL' -> 'AAPL US Equity'."""
    if "Equity" not in ticker and " " not in ticker:
        return f"{ticker} US Equity"
    return ticker


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
        try:
            exp_date = dt.datetime.strptime(exp_str, "%Y%m%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if dte < 1:
            continue
        diff = abs(dte - target_mid)
        if diff < best_diff:
            best_diff = diff
            best = exp_str
    if best is None:
        raise ValueError("No valid expiries found.")
    return best


def fetch_options_chain(
    session: blpapi.Session,
    ticker: str = DEFAULT_TICKER,
    dte_low: int = TARGET_DTE_LOW,
    dte_high: int = TARGET_DTE_HIGH,
    risk_free_rate: float = 0.05,
) -> OptionsSnapshot:
    """Fetch and filter an options chain from Bloomberg for a single expiry."""

    bbg_ticker = _to_bbg(ticker)
    display_ticker = ticker.split()[0] if " " in ticker else ticker

    # 1. Spot price
    spot_data = _bdp(session, bbg_ticker, ["PX_LAST"])
    spot = spot_data.get("PX_LAST")
    if spot is None or (isinstance(spot, float) and np.isnan(spot)):
        raise RuntimeError(f"Cannot determine spot price for {bbg_ticker}")
    spot = float(spot)
    print(f"  Spot: ${spot:.2f}")

    # 2. Get option chain tickers from Bloomberg
    #    OPT_CHAIN returns all option tickers for the underlying
    chain_rows = _bds(session, bbg_ticker, "OPT_CHAIN")

    option_tickers = []
    for row in chain_rows:
        # Bloomberg returns the ticker in various field names depending on version
        for key in ("Security Description", "security_description",
                     "Ticker", "ticker"):
            if key in row and row[key]:
                option_tickers.append(str(row[key]).strip())
                break

    if not option_tickers:
        raise RuntimeError(f"No option chain returned for {bbg_ticker}")

    print(f"  Found {len(option_tickers)} option tickers in chain")

    # 3. Get expiry + strike + type for all options to filter to target expiry
    #    Request in batches to avoid Bloomberg message size limits
    batch_size = 200
    fields = ["OPT_EXPIRE_DT", "OPT_STRIKE_PX", "OPT_PUT_CALL"]
    meta_map = {}

    for i in range(0, len(option_tickers), batch_size):
        batch = option_tickers[i : i + batch_size]
        batch_data = _bulk_ref(session, batch, fields)
        meta_map.update(batch_data)

    # Parse expiries
    expiry_set = set()
    for sec, data in meta_map.items():
        exp_raw = data.get("OPT_EXPIRE_DT", "")
        exp_str = str(exp_raw).replace("-", "").replace("/", "")[:8]
        if len(exp_str) == 8 and exp_str.isdigit():
            expiry_set.add(exp_str)

    if not expiry_set:
        raise RuntimeError("Could not parse any expiry dates from chain")

    # Pick target expiry
    expiry = _pick_expiry(sorted(expiry_set), dte_low, dte_high)
    exp_date = dt.datetime.strptime(expiry, "%Y%m%d").date()
    dte = (exp_date - dt.date.today()).days
    print(f"  Selected expiry: {expiry} ({dte} DTE)")

    # 4. Filter to target expiry options
    target_options = []
    for sec, data in meta_map.items():
        exp_raw = str(data.get("OPT_EXPIRE_DT", "")).replace("-", "").replace("/", "")[:8]
        if exp_raw != expiry:
            continue

        strike = data.get("OPT_STRIKE_PX")
        if strike is None:
            continue
        strike = float(strike)

        # Filter to ±30% of spot
        if strike < spot * 0.70 or strike > spot * 1.30:
            continue

        pc = str(data.get("OPT_PUT_CALL", "")).upper()
        if "CALL" in pc or pc == "C":
            right = "C"
        elif "PUT" in pc or pc == "P":
            right = "P"
        else:
            continue

        target_options.append((sec, strike, right))

    if not target_options:
        raise RuntimeError(f"No options found for expiry {expiry}")

    print(f"  {len(target_options)} options for target expiry (±30% of spot)")

    # 5. Fetch market data for target options
    price_fields = ["PX_BID", "PX_ASK", "PX_MID", "IVOL_MID",
                     "VOLUME", "OPEN_INT"]
    target_secs = [t[0] for t in target_options]
    price_data = {}
    for i in range(0, len(target_secs), batch_size):
        batch = target_secs[i : i + batch_size]
        batch_data = _bulk_ref(session, batch, price_fields)
        price_data.update(batch_data)

    # 6. Build DataFrame
    rows = []
    for sec, strike, right in target_options:
        data = price_data.get(sec, {})

        bid = float(data.get("PX_BID", 0) or 0)
        ask = float(data.get("PX_ASK", 0) or 0)
        mid = float(data.get("PX_MID", 0) or 0)
        if mid <= 0 and bid > 0 and ask > 0:
            mid = (bid + ask) / 2

        iv = data.get("IVOL_MID")
        if iv is not None and iv != "":
            iv = float(iv) / 100.0  # Bloomberg returns IV as percentage
        else:
            iv = np.nan

        vol = float(data.get("VOLUME", 0) or 0)
        oi = float(data.get("OPEN_INT", 0) or 0)

        rows.append({
            "strike": strike,
            "right": right,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "iv": iv,
            "volume": vol,
            "oi": oi,
        })

    df = pd.DataFrame(rows)

    # 7. Filter
    n_before = len(df)
    df = df[df["bid"] >= MIN_BID]
    df = df[df["mid"] > 0]
    df = df.dropna(subset=["iv"])
    df = df.sort_values(["right", "strike"]).reset_index(drop=True)

    n_filtered = n_before - len(df)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} illiquid strikes, {len(df)} remaining")

    if df.empty:
        raise RuntimeError(
            f"No liquid options found for {display_ticker} expiry {expiry}."
        )

    print(f"  {len(df)} liquid options loaded from Bloomberg")

    return OptionsSnapshot(
        ticker=display_ticker,
        spot=spot,
        expiry=expiry,
        dte=dte,
        risk_free_rate=risk_free_rate,
        chains=df,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def pull(
    ticker: str = DEFAULT_TICKER,
    host: str = BBG_HOST,
    port: int = BBG_PORT,
    dte_low: int = TARGET_DTE_LOW,
    dte_high: int = TARGET_DTE_HIGH,
    risk_free_rate: float = 0.05,
) -> OptionsSnapshot:
    """High-level helper: connect, fetch, return snapshot."""
    session = _start_session(host, port)
    try:
        snapshot = fetch_options_chain(
            session, ticker,
            dte_low=dte_low,
            dte_high=dte_high,
            risk_free_rate=risk_free_rate,
        )
    finally:
        session.stop()
    return snapshot


def fetch_multiple_chains(
    session: "blpapi.Session",
    ticker: str = DEFAULT_TICKER,
    dte_targets: list[int] | None = None,
    risk_free_rate: float = 0.05,
) -> list[OptionsSnapshot]:
    """Fetch options chains for multiple DTE targets in a single Bloomberg session.

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
                session, ticker,
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
    host: str = BBG_HOST,
    port: int = BBG_PORT,
    risk_free_rate: float = 0.05,
) -> list[OptionsSnapshot]:
    """Fetch multiple expiries in a single Bloomberg session."""
    session = _start_session(host, port)
    try:
        snapshots = fetch_multiple_chains(
            session, ticker, dte_targets,
            risk_free_rate=risk_free_rate,
        )
    finally:
        session.stop()
    return snapshots


# ---------------------------------------------------------------------------
# CSV export / import
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
    """Load an OptionsSnapshot from CSV."""
    import json
    with open(path) as f:
        header_line = f.readline().strip()
        if not header_line.startswith("#"):
            raise ValueError(
                "First line must be a JSON comment: "
                '# {"ticker": ..., "spot": ..., ...}'
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
