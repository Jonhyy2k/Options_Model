"""Heston stochastic volatility pricing model.

Uses Gauss-Laguerre quadrature for fast numerical integration of the
characteristic function, fully vectorised across strikes.

Parameters:
    v0    – initial variance
    kappa – mean-reversion speed
    theta – long-run variance
    xi    – vol-of-vol
    rho   – correlation between spot and variance Brownian motions
"""

from __future__ import annotations

import numpy as np

# Pre-compute Gauss-Laguerre nodes and weights once at import time.
# 64 points gives excellent accuracy for Heston integrals.
_GL_N = 64
_GL_NODES, _GL_WEIGHTS = np.polynomial.laguerre.laggauss(_GL_N)


def _characteristic_function(
    phi: np.ndarray,
    S: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    j: int,
) -> np.ndarray:
    """Vectorised Heston characteristic function over an array of phi values.

    Uses the Albrecher et al. (2007) formulation to avoid branch cuts.
    """
    if j == 1:
        u = 0.5
        b = kappa - rho * xi
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(S)

    d = np.sqrt(
        (rho * xi * phi * 1j - b) ** 2
        - xi**2 * (2 * u * phi * 1j - phi**2)
    )
    g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)

    exp_dT = np.exp(d * T)

    C = (
        r * phi * 1j * T
        + (a / xi**2)
        * (
            (b - rho * xi * phi * 1j + d) * T
            - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
        )
    )
    D = (
        (b - rho * xi * phi * 1j + d)
        / xi**2
        * ((1.0 - exp_dT) / (1.0 - g * exp_dT))
    )

    return np.exp(C + D * v0 + 1j * phi * x)


def price(
    S: float,
    K: float | np.ndarray,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    right: str = "C",
) -> np.ndarray:
    """Heston model European option price, vectorised across strikes.

    Uses Gauss-Laguerre quadrature (64 nodes) — no per-strike loops.
    """
    K = np.asarray(K, dtype=float)
    scalar = K.ndim == 0
    K = np.atleast_1d(K)

    phi = _GL_NODES  # shape (N,)
    w = _GL_WEIGHTS  # shape (N,)

    # Characteristic functions evaluated at all phi — shape (N,)
    cf1 = _characteristic_function(phi, S, T, r, v0, kappa, theta, xi, rho, j=1)
    cf2 = _characteristic_function(phi, S, T, r, v0, kappa, theta, xi, rho, j=2)

    # log(K) — shape (M,)
    logK = np.log(K)

    # Integrand matrix: shape (M, N)
    # exp(-i*phi*logK) has shape (M, N) via broadcasting
    exp_factor = np.exp(-1j * np.outer(logK, phi))

    # Re[ exp(-i*phi*logK) * cf / (i*phi) ] — shape (M, N)
    # Multiply by Gauss-Laguerre weights and the exp(phi) correction
    # (Gauss-Laguerre integrates f(x)*exp(-x), so we multiply back by exp(x))
    denom = 1j * phi  # shape (N,)
    correction = np.exp(phi) * w  # shape (N,)

    integrand1 = np.real(exp_factor * cf1[np.newaxis, :] / denom[np.newaxis, :])
    integrand2 = np.real(exp_factor * cf2[np.newaxis, :] / denom[np.newaxis, :])

    P1 = 0.5 + np.sum(integrand1 * correction[np.newaxis, :], axis=1) / np.pi
    P2 = 0.5 + np.sum(integrand2 * correction[np.newaxis, :], axis=1) / np.pi

    call = S * P1 - K * np.exp(-r * T) * P2

    if right == "C":
        prices = call
    else:
        prices = call - S + K * np.exp(-r * T)

    return prices[0] if scalar else prices


# -- Calibration interface ---------------------------------------------------
PARAM_NAMES = ("v0", "kappa", "theta", "xi", "rho")
BOUNDS = [
    (0.001, 1.0),     # v0    (initial variance)
    (0.1, 20.0),      # kappa (mean-reversion speed)
    (0.001, 1.0),     # theta (long-run variance)
    (0.01, 2.0),      # xi    (vol-of-vol)
    (-0.99, 0.99),    # rho   (correlation)
]


def price_for_calibration(
    params: tuple,
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    rights: np.ndarray,
) -> np.ndarray:
    v0, kappa, theta, xi, rho = params
    prices = np.empty_like(K, dtype=float)
    for right_val in ("C", "P"):
        mask = rights == right_val
        if mask.any():
            prices[mask] = price(
                S, K[mask], T, r, v0, kappa, theta, xi, rho, right=right_val
            )
    return prices
