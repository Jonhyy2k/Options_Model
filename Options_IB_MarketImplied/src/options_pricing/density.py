"""Stage 5 — Breeden-Litzenberger risk-neutral density extraction.

Take the model-averaged call price curve C(K), fit a smooth spline,
then compute the second derivative analytically:

    q(K) = e^{rT} * d²C/dK²

where q(K) is the risk-neutral probability density function.

Supports tail extrapolation via calibrated models to extend the PDF
beyond the observed strike range.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import UnivariateSpline

if TYPE_CHECKING:
    from options_pricing.calibration import CalibrationResult


@dataclass
class RiskNeutralDensity:
    """Risk-neutral PDF extracted via Breeden-Litzenberger."""

    strikes: np.ndarray       # fine grid of strikes
    pdf: np.ndarray           # probability density at each strike
    cdf: np.ndarray           # cumulative distribution
    call_spline: UnivariateSpline
    rn_mean: float            # risk-neutral expected value of S_T
    rn_std: float             # risk-neutral std dev
    observed_range: tuple[float, float]   # (K_min, K_max) of actual market data


def extend_call_curve(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    S: float,
    T: float,
    r: float,
    q: float,
    cal_results: dict[str, "CalibrationResult"],
    weights: dict[str, float],
    extension_pct: float = 0.50,
    n_extend: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the observed call price curve using calibrated BMA models.

    Generates synthetic BMA-weighted call prices at strikes beyond the
    observed range so the PDF tails are properly captured.

    Parameters
    ----------
    strikes : observed call strikes (sorted)
    call_prices : observed/BMA call prices at those strikes
    S : spot price
    T : time to expiry (years)
    r : risk-free rate
    q : continuous dividend / carry yield
    cal_results : calibrated model results
    weights : BMA model weights
    extension_pct : how far to extend beyond spot (0.50 = ±50%)
    n_extend : number of synthetic strikes per side

    Returns
    -------
    (extended_strikes, extended_prices) — combined observed + synthetic
    """
    from options_pricing.calibration import MODELS

    K_min_obs = strikes.min()
    K_max_obs = strikes.max()

    # Extend range to ±extension_pct of spot
    K_lo = S * (1 - extension_pct)
    K_hi = S * (1 + extension_pct)

    # Only extend where we don't already have data
    ext_left = np.linspace(K_lo, K_min_obs, n_extend, endpoint=False) if K_lo < K_min_obs else np.array([])
    ext_right = np.linspace(K_max_obs, K_hi, n_extend + 1)[1:] if K_hi > K_max_obs else np.array([])

    def _bma_call_price(K_arr: np.ndarray) -> np.ndarray:
        """Compute BMA-weighted call prices at arbitrary strikes."""
        prices = np.zeros(len(K_arr))
        for name, res in cal_results.items():
            model = MODELS[name]
            w = weights[name]
            params = tuple(res.params.values())
            try:
                rights = np.array(["C"] * len(K_arr))
                p = model.price_for_calibration(params, S, K_arr, T, r, rights, q=q)
                valid = np.isfinite(p) & (p >= 0)
                prices[valid] += w * p[valid]
            except Exception:
                pass
        return prices

    # Generate synthetic prices at extended strikes
    all_strikes = [strikes]
    all_prices = [call_prices]

    if len(ext_left) > 0:
        left_prices = _bma_call_price(ext_left)
        all_strikes.insert(0, ext_left)
        all_prices.insert(0, left_prices)

    if len(ext_right) > 0:
        right_prices = _bma_call_price(ext_right)
        all_strikes.append(ext_right)
        all_prices.append(right_prices)

    combined_K = np.concatenate(all_strikes)
    combined_C = np.concatenate(all_prices)

    # Sort and deduplicate
    order = np.argsort(combined_K)
    combined_K = combined_K[order]
    combined_C = combined_C[order]

    unique_mask = np.concatenate([[True], np.diff(combined_K) > 0.01])
    combined_K = combined_K[unique_mask]
    combined_C = combined_C[unique_mask]

    n_ext = len(combined_K) - len(strikes)
    if n_ext > 0:
        print(f"  Extended call curve by {n_ext} synthetic strikes "
              f"(range: ${combined_K.min():.0f}–${combined_K.max():.0f}, "
              f"observed: ${K_min_obs:.0f}–${K_max_obs:.0f})")

    return combined_K, combined_C


def extract_density(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    T: float,
    discount_factor: float,
    observed_range: tuple[float, float] | None = None,
    n_grid: int = 500,
) -> RiskNeutralDensity:
    """Extract the risk-neutral PDF from (optionally extended) call prices.

    Parameters
    ----------
    strikes : sorted array of strikes (calls only), possibly extended
    call_prices : BMA call prices at those strikes
    T : time to expiry (years)
    discount_factor : market discount factor B(0,T)
    observed_range : (K_min, K_max) of actual market data before extension
    n_grid : number of points in the output strike grid

    Returns
    -------
    RiskNeutralDensity
    """
    # Sort by strike
    order = np.argsort(strikes)
    K = strikes[order].astype(float)
    C = call_prices[order].astype(float)

    if observed_range is None:
        observed_range = (float(K.min()), float(K.max()))

    # Validate: call prices should be monotonically decreasing
    diffs = np.diff(C)
    n_violations = np.sum(diffs > 0)
    if n_violations > 0:
        warnings.warn(
            f"Call prices not monotonically decreasing ({n_violations} violations). "
            "Smoothing spline will handle this, but data may be noisy."
        )

    # Remove duplicate strikes
    unique_mask = np.concatenate([[True], np.diff(K) > 0])
    K = K[unique_mask]
    C = C[unique_mask]

    if len(K) < 4:
        raise ValueError(
            f"Need at least 4 unique call strikes for spline fit, got {len(K)}"
        )

    # Fit smoothing spline
    weights = np.ones_like(K)
    K_mid = (K.min() + K.max()) / 2
    K_range = K.max() - K.min()
    weights *= 1.0 + 0.5 * np.exp(-((K - K_mid) / (K_range * 0.3)) ** 2)

    # Give slightly higher weight to observed data vs extrapolated
    obs_lo, obs_hi = observed_range
    for i, k in enumerate(K):
        if obs_lo <= k <= obs_hi:
            weights[i] *= 1.5

    s = len(K) * np.mean((np.diff(C)) ** 2) * 0.5
    s = max(s, 1e-6)

    spline = UnivariateSpline(K, C, w=weights, s=s, k=4)

    # Fine grid over the full range (including extensions)
    K_fine = np.linspace(K.min(), K.max(), n_grid)

    # Second derivative of the spline
    d2C = spline.derivative(n=2)(K_fine)

    # Breeden-Litzenberger: q(K) = e^{rT} * C''(K)
    discount = 1.0 / max(float(discount_factor), 1e-12)
    pdf = discount * d2C

    # Clamp negative values (spline artifacts)
    pdf = np.maximum(pdf, 0.0)

    # Normalise
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = _trapz(pdf, K_fine)
    if area <= 1e-12:
        warnings.warn(
            "PDF area is near zero after clamping. "
            "Risk-neutral density extraction may be unreliable."
        )
        pdf = np.ones_like(pdf) / (K_fine[-1] - K_fine[0])
    else:
        pdf = pdf / area

    # CDF via cumulative trapezoidal integration
    dk = np.diff(K_fine)
    avg_pdf = 0.5 * (pdf[:-1] + pdf[1:])
    cdf = np.zeros_like(pdf)
    cdf[1:] = np.cumsum(avg_pdf * dk)

    # Ensure CDF is monotonic and bounded [0, 1]
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    # Risk-neutral moments
    rn_mean = _trapz(K_fine * pdf, K_fine)
    rn_var = _trapz((K_fine - rn_mean) ** 2 * pdf, K_fine)
    rn_std = np.sqrt(max(rn_var, 0.0))

    return RiskNeutralDensity(
        strikes=K_fine,
        pdf=pdf,
        cdf=cdf,
        call_spline=spline,
        rn_mean=rn_mean,
        rn_std=rn_std,
        observed_range=observed_range,
    )
