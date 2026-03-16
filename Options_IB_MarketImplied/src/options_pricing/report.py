"""Metrics report — writes all calculated metrics to a structured text file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from options_pricing.data import OptionsSnapshot
    from options_pricing.calibration import CalibrationResult
    from options_pricing.averaging import AveragedPrices
    from options_pricing.density import RiskNeutralDensity


def write_report(
    path: str,
    snapshot: "OptionsSnapshot",
    cal_results: dict[str, "CalibrationResult"],
    avg: "AveragedPrices",
    rnd: "RiskNeutralDensity",
    target: float | None = None,
) -> None:
    """Write a comprehensive metrics report to a text file."""

    df = snapshot.chains
    call_mask = df["right"].values == "C"
    put_mask = df["right"].values == "P"

    lines = []
    w = lines.append
    sep = "=" * 72

    w(sep)
    w(f"  OPTIONS PRICING REPORT — {snapshot.ticker}")
    w(sep)
    w("")

    # --- Market Data Summary ---
    w("1. MARKET DATA SUMMARY")
    w("-" * 40)
    w(f"  Underlying:        {snapshot.ticker}")
    w(f"  Spot Price:        ${snapshot.spot:.2f}")
    w(f"  Expiry:            {snapshot.expiry}")
    w(f"  DTE:               {snapshot.dte} days")
    w(f"  T (years):         {snapshot.dte / 365:.4f}")
    w(f"  Pricing Rate (r):  {snapshot.risk_free_rate:.2%}")
    w(f"  Dividend Yield(q): {snapshot.dividend_yield:.2%}")
    w(f"  Total Options:     {len(df)}")
    w(f"    Calls:           {call_mask.sum()}")
    w(f"    Puts:            {put_mask.sum()}")
    w(f"  Strike Range:      ${df['strike'].min():.0f} – ${df['strike'].max():.0f}")
    iv = df["iv"].values
    w(f"  IV Range:          {iv.min():.4f} – {iv.max():.4f}")
    w(f"  IV Mean:           {iv.mean():.4f}")
    w("")

    # --- Market-Implied Inputs ---
    w("2. MARKET-IMPLIED FORWARD / DISCOUNT INPUTS")
    w("-" * 40)
    mi = snapshot.market_inputs
    if mi is None:
        w("  No market-input diagnostics available.")
        w("")
    else:
        w(f"  Carry Source:      {snapshot.carry_source}")
        w(f"  Method:            {mi.method}")
        w(f"  Fallback Rate:     {mi.fallback_rate:.2%}")
        w(f"  Discount Factor:   {mi.discount_factor:.6f}")
        w(f"  Forward Price:     ${mi.forward_price:.2f}")
        w(f"  Implied Rate:      {mi.implied_rate:.2%}")
        w(f"  Implied Carry:     {mi.carry_rate:.2%}")
        w(f"  Implied Div Yield: {mi.dividend_yield:.2%}")
        if mi.discount_factor_se is not None:
            w(f"  DF Std Error:      {mi.discount_factor_se:.6f}")
        if mi.forward_se is not None:
            w(f"  Forward Std Err:   ${mi.forward_se:.2f}")
        if mi.implied_rate_se is not None:
            w(f"  Rate Std Error:    {mi.implied_rate_se:.2%}")
        if mi.dividend_yield_se is not None:
            w(f"  Div Yield Std Err: {mi.dividend_yield_se:.2%}")
        if mi.forward_bid is not None and mi.forward_ask is not None:
            w(f"  Forward Bounds:    ${mi.forward_bid:.2f} – ${mi.forward_ask:.2f}")
        w(f"  Pairs Used:        {mi.n_pairs_used} / {mi.n_pairs_total}")
        if mi.strike_min is not None and mi.strike_max is not None:
            w(f"  Pair Strike Band:  ${mi.strike_min:.0f} – ${mi.strike_max:.0f}")
        if mi.r_squared is not None:
            w(f"  Parity Fit R^2:    {mi.r_squared:.4f}")
        if mi.rmse is not None:
            w(f"  Parity RMSE:       ${mi.rmse:.4f}")
        if mi.weighted_rmse is not None:
            w(f"  Wtd Parity RMSE:   ${mi.weighted_rmse:.4f}")
        if mi.max_abs_residual is not None:
            w(f"  Max |Residual|:    ${mi.max_abs_residual:.4f}")
        if mi.notes:
            w("  Notes:")
            for note in mi.notes:
                w(f"    - {note}")
        w("")

    # --- Calibration Results ---
    w("3. MODEL CALIBRATION RESULTS")
    w("-" * 40)
    for name, res in sorted(cal_results.items()):
        w(f"  [{name.upper()}]")
        w(f"    SSE:             {res.residual_sse:.6f}")
        w(f"    RMSE:            {np.sqrt(res.residual_sse / len(df)):.6f}")
        for pname, pval in res.params.items():
            w(f"    {pname:16s} {pval:.6f}")
        # Per-model pricing error stats
        errors = res.model_prices - df["mid"].values
        w(f"    Mean Error:      {errors.mean():.6f}")
        w(f"    Max |Error|:     {np.abs(errors).max():.6f}")
        w("")

    # --- Model Weights ---
    if avg.weighting_method == "bic":
        section_title = "4. MODEL WEIGHTS (BIC POSTERIOR APPROXIMATION)"
    elif avg.weighting_method == "aic":
        section_title = "4. MODEL WEIGHTS (AIC WEIGHTS)"
    else:
        section_title = "4. MODEL WEIGHTS (INVERSE SSE)"
    w(section_title)
    w("-" * 40)
    w(f"  Score basis:      {avg.score_label} (lower is better)")
    for name, weight in sorted(avg.weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(weight * 40)
        score = avg.scores[name]
        w(f"  {name:8s}  {weight:6.1%}  {avg.score_label}={score:10.4f}  {bar}")
    w("")

    # --- Model Price Correlation ---
    w("5. MODEL PRICE CORRELATION MATRIX")
    w("-" * 40)
    names = sorted(avg.per_model_prices.keys())
    price_matrix = np.column_stack([avg.per_model_prices[n] for n in names])
    corr = np.corrcoef(price_matrix.T)
    header = "          " + "".join(f"{n:>10s}" for n in names)
    w(header)
    for i, n in enumerate(names):
        row = f"  {n:8s}" + "".join(f"{corr[i, j]:10.4f}" for j in range(len(names)))
        w(row)
    w("")

    # --- Risk-Neutral Distribution ---
    w("6. RISK-NEUTRAL DISTRIBUTION")
    w("-" * 40)
    w(f"  RN Mean (E[S_T]):  ${rnd.rn_mean:.2f}")
    w(f"  RN Std Dev:        ${rnd.rn_std:.2f}")
    if snapshot.forward_price is not None:
        w(f"  Market Forward:    ${snapshot.forward_price:.2f}")
        w(f"  RN Mean vs Fwd:    {(rnd.rn_mean / snapshot.forward_price - 1):+.2%}")
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    if rnd.rn_std > 1e-10:
        skew_integrand = ((rnd.strikes - rnd.rn_mean) / rnd.rn_std) ** 3 * rnd.pdf
        rn_skew = _trapz(skew_integrand, rnd.strikes)
        kurt_integrand = ((rnd.strikes - rnd.rn_mean) / rnd.rn_std) ** 4 * rnd.pdf
        rn_kurt = _trapz(kurt_integrand, rnd.strikes)
    else:
        rn_skew = 0.0
        rn_kurt = 3.0
    w(f"  RN Skewness:       {rn_skew:.4f}")
    w(f"  RN Excess Kurtosis:{rn_kurt - 3:.4f}")
    w("")

    # Percentiles (interpolate for accuracy)
    w("  Percentiles:")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        p = pct / 100
        strike_val = np.interp(p, rnd.cdf, rnd.strikes)
        w(f"    {pct:3d}th:           ${strike_val:.2f}")
    w("")

    # Probability intervals
    w("  Probability Intervals:")
    for lo_pct, hi_pct in [(10, 90), (25, 75), (5, 95)]:
        lo_strike = np.interp(lo_pct / 100, rnd.cdf, rnd.strikes)
        hi_strike = np.interp(hi_pct / 100, rnd.cdf, rnd.strikes)
        w(f"    {100 - 2 * lo_pct}% CI:           ${lo_strike:.2f} – ${hi_strike:.2f}")
    w("")

    # --- Target Analysis ---
    if target is not None:
        w("7. PRICE TARGET ANALYSIS")
        w("-" * 40)
        w(f"  Your Target:       ${target:.2f}")
        dist_from_spot = (target - snapshot.spot) / snapshot.spot
        w(f"  vs Spot:           {dist_from_spot:+.2%}")
        dist_from_mean = (target - rnd.rn_mean) / rnd.rn_mean
        w(f"  vs RN Mean:        {dist_from_mean:+.2%}")
        if snapshot.forward_price is not None:
            dist_from_fwd = (target - snapshot.forward_price) / snapshot.forward_price
            w(f"  vs Forward:        {dist_from_fwd:+.2%}")

        if target >= rnd.strikes.min() and target <= rnd.strikes.max():
            prob_below = float(np.interp(target, rnd.strikes, rnd.cdf))
            prob_above = 1 - prob_below
            w(f"  P(S > target):     {prob_above:.2%}")
            w(f"  P(S < target):     {prob_below:.2%}")
        else:
            w(f"  Target outside strike range ({rnd.strikes.min():.0f}–{rnd.strikes.max():.0f})")
        w("")

    w(sep)
    if avg.weighting_method == "bic":
        footer = "  Generated by options-pricing (BIC posterior model averaging)"
    elif avg.weighting_method == "aic":
        footer = "  Generated by options-pricing (AIC-weighted model averaging)"
    else:
        footer = "  Generated by options-pricing (inverse-SSE model averaging)"
    w(footer)
    w(sep)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report saved to {path}")


def write_multi_dte_summary(
    path: str,
    results: list[tuple],
    target: float | None = None,
) -> None:
    """Write a multi-horizon summary comparing results across DTEs.

    Parameters
    ----------
    results : list of (OptionsSnapshot, cal_results, AveragedPrices, RiskNeutralDensity)
    """
    lines: list[str] = []
    w = lines.append
    sep = "=" * 72

    ticker = results[0][0].ticker
    spot = results[0][0].spot
    weighting_method = results[0][2].weighting_method
    if weighting_method == "bic":
        weighting_desc = "BIC posterior approximation"
    elif weighting_method == "aic":
        weighting_desc = "AIC weights"
    else:
        weighting_desc = "inverse SSE"

    w(sep)
    w(f"  MULTI-HORIZON OPTIONS ANALYSIS — {ticker}")
    w(sep)
    w("")
    w(f"  Underlying:     {ticker}")
    w(f"  Spot Price:     ${spot:.2f}")
    w(f"  Weighting:      {weighting_desc}")
    if target is not None:
        w(f"  Price Target:   ${target:.2f} ({(target / spot - 1):+.1%} from spot)")
    w(f"  Horizons:       {len(results)}")
    w("")

    # --- Summary table ---
    w("HORIZON COMPARISON")
    w("-" * 72)
    hdr = f"  {'DTE':>5s}  {'Expiry':>10s}  {'Forward':>10s}  {'r':>8s}  {'q':>8s}  {'RN Mean':>10s}  {'50% CI':>20s}"
    if target is not None:
        hdr += f"  {'P(>target)':>10s}"
    w(hdr)
    w("-" * 72)

    for snap, cal_results, avg, rnd in results:
        p25 = float(np.interp(0.25, rnd.cdf, rnd.strikes))
        p75 = float(np.interp(0.75, rnd.cdf, rnd.strikes))
        row = (f"  {snap.dte:5d}  {snap.expiry:>10s}  "
               f"${snap.forward_price:8.2f}  {snap.risk_free_rate:7.2%}  {snap.dividend_yield:7.2%}  "
               f"${rnd.rn_mean:8.2f}  "
               f"${p25:6.0f} – ${p75:.0f}")
        if target is not None:
            if rnd.strikes.min() <= target <= rnd.strikes.max():
                prob = 1 - float(np.interp(target, rnd.strikes, rnd.cdf))
                row += f"  {prob:9.2%}"
            else:
                row += f"  {'N/A':>9s}"
        w(row)
    w("")

    # --- Per-horizon detail ---
    for snap, cal_results, avg, rnd in results:
        w(f"--- {snap.dte} DTE (Expiry: {snap.expiry}) ---")
        n_calls = int((snap.chains["right"] == "C").sum())
        n_puts = int((snap.chains["right"] == "P").sum())
        w(f"  Options: {len(snap.chains)} | Calls: {n_calls} | Puts: {n_puts}")
        w(f"  Carry: {snap.carry_source} | Forward ${snap.forward_price:.2f} | "
          f"r {snap.risk_free_rate:.2%} | q {snap.dividend_yield:.2%}")
        w(f"  RN Mean: ${rnd.rn_mean:.2f}  RN Std: ${rnd.rn_std:.2f}")
        if snap.market_inputs is not None and snap.market_inputs.r_squared is not None:
            w(f"  Parity fit: R^2={snap.market_inputs.r_squared:.4f} | "
              f"Pairs={snap.market_inputs.n_pairs_used}/{snap.market_inputs.n_pairs_total}")

        # BMA weights
        w("  BMA: " + ", ".join(
            f"{name} {wt:.1%}" for name, wt in
            sorted(avg.weights.items(), key=lambda x: -x[1])
        ))

        # Best model
        best_name, best_res = min(cal_results.items(),
                                  key=lambda x: x[1].residual_sse)
        w(f"  Best model: {best_name} (SSE={best_res.residual_sse:.4f})")

        # Percentiles
        p10 = float(np.interp(0.10, rnd.cdf, rnd.strikes))
        p90 = float(np.interp(0.90, rnd.cdf, rnd.strikes))
        w(f"  80% CI: ${p10:.0f} – ${p90:.0f}")

        if target is not None:
            if rnd.strikes.min() <= target <= rnd.strikes.max():
                prob = 1 - float(np.interp(target, rnd.strikes, rnd.cdf))
                w(f"  P(S > ${target:.2f}): {prob:.2%}")
        w("")

    w(sep)
    method = weighting_method
    if method == "bic":
        footer = "  Generated by options-pricing (Multi-Horizon BIC posterior averaging)"
    elif method == "aic":
        footer = "  Generated by options-pricing (Multi-Horizon AIC-weighted averaging)"
    else:
        footer = "  Generated by options-pricing (Multi-Horizon inverse-SSE averaging)"
    w(footer)
    w(sep)

    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Multi-DTE summary saved to {path}")
