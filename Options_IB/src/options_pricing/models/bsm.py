"""Black-Scholes-Merton analytical pricing model."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float,
    r: float,
    sigma: float,
    right: str = "C",
) -> np.ndarray:
    """Compute BSM option price.

    Parameters
    ----------
    S : spot price
    K : strike(s)
    T : time to expiry in years
    r : risk-free rate (annualised, continuous)
    sigma : volatility (annualised)
    right : "C" for call, "P" for put

    Returns
    -------
    np.ndarray of option prices, same shape as K.
    """
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if right == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# -- Calibration interface --------------------------------------------------
# Each model exposes:
#   PARAM_NAMES  – tuple of parameter names
#   BOUNDS       – list of (lo, hi) for each parameter
#   price_for_calibration(params, S, K, T, r, rights) -> np.ndarray

PARAM_NAMES = ("sigma",)
BOUNDS = [(0.01, 1.5)]


def price_for_calibration(
    params: tuple,
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    rights: np.ndarray,
) -> np.ndarray:
    """Vectorised pricing across strikes and rights for calibration."""
    (sigma,) = params
    prices = np.empty_like(K, dtype=float)
    for right_val in ("C", "P"):
        mask = rights == right_val
        if mask.any():
            prices[mask] = price(S, K[mask], T, r, sigma, right=right_val)
    return prices
