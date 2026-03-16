"""Merton jump-diffusion pricing model.

Extends BSM by adding a compound Poisson jump process.  The price is an
infinite series of BSM prices weighted by Poisson probabilities — in practice
we truncate at ~40 terms which converges well.

Parameters beyond BSM:
    lam   – jump intensity (expected number of jumps per year)
    mu_j  – mean of log-jump size
    sig_j – std dev of log-jump size
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.special import gammaln


def price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sig_j: float,
    q: float = 0.0,
    right: str = "C",
    n_terms: int = 40,
) -> np.ndarray:
    """Merton jump-diffusion option price.

    Parameters
    ----------
    S, K, T, r, sigma : standard BSM inputs
    lam : jump intensity (jumps / year)
    mu_j : mean of log-jump size
    sig_j : std dev of log-jump size
    q : continuous dividend / carry yield
    right : "C" or "P"
    n_terms : series truncation

    Returns
    -------
    np.ndarray of option prices.
    """
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)

    # Compensator: E[e^J - 1] where J ~ N(mu_j, sig_j^2)
    k_comp = np.exp(mu_j + 0.5 * sig_j**2) - 1.0
    lam_prime = lam * (1 + k_comp)

    total = np.zeros_like(K, dtype=float)
    for n in range(n_terms):
        # Adjusted vol and drift for n jumps
        sigma_n = np.sqrt(sigma**2 + n * sig_j**2 / T)
        b_n = (r - q) - lam * k_comp + n * (mu_j + 0.5 * sig_j**2) / T

        d1 = (np.log(S / K) + (b_n + 0.5 * sigma_n**2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)
        spot_pv = S * np.exp((b_n - r) * T)
        strike_pv = K * np.exp(-r * T)

        if right == "C":
            bsm_n = spot_pv * norm.cdf(d1) - strike_pv * norm.cdf(d2)
        else:
            bsm_n = strike_pv * norm.cdf(-d2) - spot_pv * norm.cdf(-d1)

        # Poisson weight (log-space to avoid factorial overflow)
        log_weight = -lam_prime * T + n * np.log(lam_prime * T + 1e-300) - gammaln(n + 1)
        weight = np.exp(log_weight)
        total += weight * bsm_n

    return total


# -- Calibration interface ---------------------------------------------------
PARAM_NAMES = ("sigma", "lam", "mu_j", "sig_j")
BOUNDS = [
    (0.01, 1.5),    # sigma
    (0.0, 5.0),     # lam  (jump intensity)
    (-0.3, 0.3),    # mu_j (mean log-jump)
    (0.01, 0.4),    # sig_j (jump vol)
]


def price_for_calibration(
    params: tuple,
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    rights: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    sigma, lam, mu_j, sig_j = params
    prices = np.empty_like(K, dtype=float)
    for right_val in ("C", "P"):
        mask = rights == right_val
        if mask.any():
            prices[mask] = price(
                S, K[mask], T, r, sigma, lam, mu_j, sig_j, q=q, right=right_val
            )
    return prices
