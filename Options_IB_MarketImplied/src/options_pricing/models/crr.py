"""Cox-Ross-Rubinstein binomial tree pricing model."""

from __future__ import annotations

import numpy as np


def price(
    S: float,
    K: float | np.ndarray,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    right: str = "C",
    n_steps: int = 200,
) -> np.ndarray:
    """CRR binomial tree option price (European).

    Parameters
    ----------
    S : spot price
    K : strike(s)
    T : time to expiry in years
    r : risk-free rate
    sigma : volatility
    q : continuous dividend / carry yield
    right : "C" or "P"
    n_steps : number of tree steps

    Returns
    -------
    np.ndarray of option prices.
    """
    K = np.asarray(K, dtype=float)
    scalar = K.ndim == 0
    K = np.atleast_1d(K)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal asset prices at step n_steps
    # S * u^j * d^(n_steps - j) for j = 0..n_steps
    j = np.arange(n_steps + 1)
    S_T = S * u ** (2 * j - n_steps)  # equivalent to u^j * d^(n-j)

    results = np.empty(K.shape, dtype=float)
    for idx, k in enumerate(K):
        if right == "C":
            payoff = np.maximum(S_T - k, 0.0)
        else:
            payoff = np.maximum(k - S_T, 0.0)

        # Backward induction
        V = payoff.copy()
        for _ in range(n_steps):
            V = disc * (p * V[1:] + (1 - p) * V[:-1])
        results[idx] = V[0]

    return results[0] if scalar else results


# -- Calibration interface ---------------------------------------------------
PARAM_NAMES = ("sigma",)
BOUNDS = [(0.01, 1.5)]


def price_for_calibration(
    params: tuple,
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    rights: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    (sigma,) = params
    prices = np.empty_like(K, dtype=float)
    for right_val in ("C", "P"):
        mask = rights == right_val
        if mask.any():
            prices[mask] = price(S, K[mask], T, r, sigma, q=q, right=right_val)
    return prices
