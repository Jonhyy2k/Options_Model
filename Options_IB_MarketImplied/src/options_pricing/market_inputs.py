"""Infer market-implied carry inputs from put-call parity.

This module estimates the discount factor and forward directly from the same
options chain used for calibration.  The main regression is:

    C(K) - P(K) = B(0, T) * (F(0, T) - K) = alpha + beta * K

with:
    beta = -B(0, T)
    alpha = B(0, T) * F(0, T)

For the pricing models we convert the inferred quantities to an implied
financing rate r and implied continuous dividend / carry yield q:

    r = -ln(B) / T
    q = r - ln(F / S) / T

The implementation uses a weighted least-squares fit on matched call/put
pairs, emphasizing tighter and more liquid strikes while downweighting deep
ITM/OTM contracts where single-stock put-call parity is noisier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketImpliedInputs:
    """Summary of the inferred market inputs and fit diagnostics."""

    source: str
    method: str
    discount_factor: float
    forward_price: float
    implied_rate: float
    dividend_yield: float
    carry_rate: float
    regression_intercept: float
    regression_slope: float
    intercept_se: float | None
    slope_se: float | None
    discount_factor_se: float | None
    forward_se: float | None
    implied_rate_se: float | None
    dividend_yield_se: float | None
    r_squared: float | None
    rmse: float | None
    weighted_rmse: float | None
    max_abs_residual: float | None
    n_pairs_total: int
    n_pairs_used: int
    strike_min: float | None
    strike_max: float | None
    forward_bid: float | None
    forward_ask: float | None
    average_pair_width: float | None
    fallback_rate: float
    notes: tuple[str, ...] = ()


def _fallback_inputs(
    spot: float,
    T: float,
    fallback_rate: float,
    notes: list[str] | None = None,
) -> MarketImpliedInputs:
    """Return a flat-rate fallback when parity inference is unavailable."""
    discount_factor = float(np.exp(-fallback_rate * T))
    forward_price = float(spot * np.exp(fallback_rate * T))
    return MarketImpliedInputs(
        source="flat_rate",
        method="user_supplied_rate",
        discount_factor=discount_factor,
        forward_price=forward_price,
        implied_rate=float(fallback_rate),
        dividend_yield=0.0,
        carry_rate=float(fallback_rate),
        regression_intercept=np.nan,
        regression_slope=np.nan,
        intercept_se=None,
        slope_se=None,
        discount_factor_se=None,
        forward_se=None,
        implied_rate_se=None,
        dividend_yield_se=None,
        r_squared=None,
        rmse=None,
        weighted_rmse=None,
        max_abs_residual=None,
        n_pairs_total=0,
        n_pairs_used=0,
        strike_min=None,
        strike_max=None,
        forward_bid=None,
        forward_ask=None,
        average_pair_width=None,
        fallback_rate=float(fallback_rate),
        notes=tuple(notes or ()),
    )


def _build_matched_pairs(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Create a matched call/put table by strike."""
    calls = (
        df[df["right"] == "C"]
        .copy()
        .rename(
            columns={
                "bid": "call_bid",
                "ask": "call_ask",
                "mid": "call_mid",
                "iv": "call_iv",
                "volume": "call_volume",
                "oi": "call_oi",
            }
        )
    )
    puts = (
        df[df["right"] == "P"]
        .copy()
        .rename(
            columns={
                "bid": "put_bid",
                "ask": "put_ask",
                "mid": "put_mid",
                "iv": "put_iv",
                "volume": "put_volume",
                "oi": "put_oi",
            }
        )
    )
    keep_call = ["strike", "call_bid", "call_ask", "call_mid", "call_iv", "call_volume", "call_oi"]
    keep_put = ["strike", "put_bid", "put_ask", "put_mid", "put_iv", "put_volume", "put_oi"]
    pairs = calls[keep_call].merge(puts[keep_put], on="strike", how="inner")
    if pairs.empty:
        return pairs

    pairs["synthetic_mid"] = pairs["call_mid"] - pairs["put_mid"]
    pairs["synthetic_bid"] = pairs["call_bid"] - pairs["put_ask"]
    pairs["synthetic_ask"] = pairs["call_ask"] - pairs["put_bid"]
    pairs["pair_width"] = np.maximum(pairs["synthetic_ask"] - pairs["synthetic_bid"], 0.0)
    pairs["log_moneyness"] = np.log(np.maximum(pairs["strike"], 1e-9) / max(float(spot), 1e-9))
    pairs["abs_log_moneyness"] = np.abs(pairs["log_moneyness"])

    call_spread = np.maximum(pairs["call_ask"] - pairs["call_bid"], 0.0)
    put_spread = np.maximum(pairs["put_ask"] - pairs["put_bid"], 0.0)
    pairs["call_spread_ratio"] = call_spread / np.maximum(pairs["call_ask"], 1e-6)
    pairs["put_spread_ratio"] = put_spread / np.maximum(pairs["put_ask"], 1e-6)

    pair_volume = np.minimum(
        np.maximum(pairs["call_volume"].fillna(0.0), 0.0),
        np.maximum(pairs["put_volume"].fillna(0.0), 0.0),
    )
    pair_oi = np.minimum(
        np.maximum(pairs["call_oi"].fillna(0.0), 0.0),
        np.maximum(pairs["put_oi"].fillna(0.0), 0.0),
    )
    pairs["liquidity_score"] = np.sqrt(pair_volume + 1.0) * np.sqrt(np.log1p(pair_oi) + 1.0)
    return pairs.sort_values("strike").reset_index(drop=True)


def infer_market_inputs(
    df: pd.DataFrame,
    spot: float,
    T: float,
    fallback_rate: float,
    *,
    use_market_implied: bool = True,
    max_spread_ratio: float = 0.60,
    atm_moneyness_band: float = 0.18,
    min_pairs: int = 4,
) -> MarketImpliedInputs:
    """Estimate B(0,T), F(0,T), implied r, and implied q from put-call parity."""
    if not use_market_implied:
        return _fallback_inputs(
            spot,
            T,
            fallback_rate,
            notes=["Using flat carry inputs because --carry-mode=flat_rate."],
        )

    notes: list[str] = []
    if T <= 0 or spot <= 0:
        return _fallback_inputs(
            spot,
            max(T, 1e-9),
            fallback_rate,
            notes=["Invalid spot or time-to-expiry for parity inference."],
        )

    needed_cols = {"strike", "right", "mid", "bid", "ask", "volume", "oi"}
    missing_cols = needed_cols - set(df.columns)
    if missing_cols:
        return _fallback_inputs(
            spot,
            T,
            fallback_rate,
            notes=[
                "Option chain is missing bid/ask or liquidity columns required for parity inference: "
                + ", ".join(sorted(missing_cols))
            ],
        )

    pairs = _build_matched_pairs(df, spot)
    n_pairs_total = len(pairs)
    if n_pairs_total < min_pairs:
        return _fallback_inputs(
            spot,
            T,
            fallback_rate,
            notes=[f"Only {n_pairs_total} matched call/put pairs available; need at least {min_pairs}."],
        )

    valid = pairs[
        np.isfinite(pairs["synthetic_mid"])
        & np.isfinite(pairs["pair_width"])
        & np.isfinite(pairs["call_spread_ratio"])
        & np.isfinite(pairs["put_spread_ratio"])
        & (pairs["pair_width"] >= 0.0)
        & (pairs["call_spread_ratio"] <= max_spread_ratio)
        & (pairs["put_spread_ratio"] <= max_spread_ratio)
    ].copy()
    if len(valid) < min_pairs:
        return _fallback_inputs(
            spot,
            T,
            fallback_rate,
            notes=[f"Matched pairs exist, but only {len(valid)} survive spread filters."],
        )

    atm_mask = np.abs(valid["strike"] / spot - 1.0) <= atm_moneyness_band
    selected = valid[atm_mask].copy()
    if len(selected) < min_pairs:
        selected = valid.copy()
        notes.append(
            "ATM-focused parity filter left too few strikes; regression widened to all valid matched pairs."
        )

    spread_floor = np.maximum(selected["pair_width"].to_numpy(dtype=float), 0.05)
    liquidity = np.maximum(selected["liquidity_score"].to_numpy(dtype=float), 1.0)
    moneyness_weight = np.exp(-((selected["abs_log_moneyness"].to_numpy(dtype=float) / 0.22) ** 2))
    weights_raw = liquidity * moneyness_weight / (spread_floor ** 2)
    weights_raw = np.maximum(weights_raw, 1e-8)
    weights = weights_raw * (len(weights_raw) / weights_raw.sum())

    X = np.column_stack([np.ones(len(selected)), selected["strike"].to_numpy(dtype=float)])
    y = selected["synthetic_mid"].to_numpy(dtype=float)
    XtWX = X.T @ (weights[:, np.newaxis] * X)
    XtWy = X.T @ (weights * y)
    try:
        coeffs = np.linalg.solve(XtWX, XtWy)
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return _fallback_inputs(
            spot,
            T,
            fallback_rate,
            notes=["Weighted parity regression matrix was singular; reverted to flat carry inputs."],
        )

    alpha, beta = coeffs
    residuals = y - X @ coeffs
    rss_w = float(np.sum(weights * residuals**2))
    dof = max(len(selected) - X.shape[1], 1)
    sigma2 = rss_w / dof
    cov = sigma2 * XtWX_inv
    intercept_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    slope_se = float(np.sqrt(max(cov[1, 1], 0.0)))

    discount_factor = float(-beta)
    if not np.isfinite(discount_factor) or discount_factor <= 0:
        return _fallback_inputs(
            spot,
            T,
            fallback_rate,
            notes=[f"Parity regression produced an invalid discount factor ({discount_factor})."],
        )

    forward_price = float(alpha / discount_factor)
    if not np.isfinite(forward_price) or forward_price <= 0:
        return _fallback_inputs(
            spot,
            T,
            fallback_rate,
            notes=[f"Parity regression produced an invalid forward price ({forward_price})."],
        )

    r_squared = None
    y_bar = float(np.average(y, weights=weights))
    ss_tot = float(np.sum(weights * (y - y_bar) ** 2))
    if ss_tot > 1e-12:
        r_squared = 1.0 - rss_w / ss_tot

    rmse = float(np.sqrt(np.mean(residuals**2)))
    weighted_rmse = float(np.sqrt(rss_w / np.sum(weights)))
    max_abs_residual = float(np.max(np.abs(residuals)))
    average_pair_width = float(np.mean(selected["pair_width"]))

    forward_se = None
    if discount_factor > 0:
        grad = np.array([1.0 / discount_factor, alpha / (discount_factor**2)])
        forward_var = float(grad @ cov @ grad)
        forward_se = float(np.sqrt(max(forward_var, 0.0)))

    implied_rate = float(-np.log(discount_factor) / T)
    carry_rate = float(np.log(forward_price / spot) / T)
    dividend_yield = float(implied_rate - carry_rate)

    discount_factor_se = slope_se
    implied_rate_se = None
    if discount_factor_se is not None:
        implied_rate_se = float(abs(discount_factor_se / (discount_factor * T)))

    dividend_yield_se = None
    if implied_rate_se is not None and forward_se is not None:
        carry_rate_se = abs(forward_se / (forward_price * T))
        dividend_yield_se = float(np.sqrt(implied_rate_se**2 + carry_rate_se**2))

    finite_bounds = np.isfinite(selected["synthetic_bid"]) & np.isfinite(selected["synthetic_ask"])
    forward_bid = None
    forward_ask = None
    if finite_bounds.any():
        strikes = selected.loc[finite_bounds, "strike"].to_numpy(dtype=float)
        syn_bid = selected.loc[finite_bounds, "synthetic_bid"].to_numpy(dtype=float)
        syn_ask = selected.loc[finite_bounds, "synthetic_ask"].to_numpy(dtype=float)
        if len(strikes) > 0:
            bid_candidates = syn_bid / discount_factor + strikes
            ask_candidates = syn_ask / discount_factor + strikes
            forward_bid = float(np.nanmax(bid_candidates))
            forward_ask = float(np.nanmin(ask_candidates))
            if forward_bid > forward_ask:
                notes.append(
                    "Synthetic forward bid/ask bounds overlap poorly; single-stock early exercise or dividend effects may be material."
                )

    if dividend_yield < -0.10:
        notes.append(
            "Implied dividend/carry yield is materially negative. This can happen with borrow/funding effects or noisy American-option parity."
        )
    if r_squared is not None and r_squared < 0.97:
        notes.append(
            "Parity fit R^2 is below 0.97; interpret the inferred carry inputs as an estimate rather than an exact no-arbitrage quantity."
        )

    return MarketImpliedInputs(
        source="market_implied",
        method="weighted_put_call_parity_regression",
        discount_factor=discount_factor,
        forward_price=forward_price,
        implied_rate=implied_rate,
        dividend_yield=dividend_yield,
        carry_rate=carry_rate,
        regression_intercept=float(alpha),
        regression_slope=float(beta),
        intercept_se=intercept_se,
        slope_se=slope_se,
        discount_factor_se=discount_factor_se,
        forward_se=forward_se,
        implied_rate_se=implied_rate_se,
        dividend_yield_se=dividend_yield_se,
        r_squared=r_squared,
        rmse=rmse,
        weighted_rmse=weighted_rmse,
        max_abs_residual=max_abs_residual,
        n_pairs_total=n_pairs_total,
        n_pairs_used=len(selected),
        strike_min=float(selected["strike"].min()),
        strike_max=float(selected["strike"].max()),
        forward_bid=forward_bid,
        forward_ask=forward_ask,
        average_pair_width=average_pair_width,
        fallback_rate=float(fallback_rate),
        notes=tuple(notes),
    )
