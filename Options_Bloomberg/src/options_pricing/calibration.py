"""Stage 3 — Model calibration via differential evolution.

For each model, find parameters that minimise sum of squared errors between
model prices and observed market mid prices across all strikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import differential_evolution

from options_pricing.models import bsm, crr, merton, heston

if TYPE_CHECKING:
    from options_pricing.data import OptionsSnapshot

# Registry of all models
MODELS = {
    "bsm": bsm,
    "crr": crr,
    "merton": merton,
    "heston": heston,
}


@dataclass
class CalibrationResult:
    """Result of calibrating a single model."""

    name: str
    params: dict[str, float]
    residual_sse: float          # sum of squared errors
    model_prices: np.ndarray     # fitted prices per strike


def _objective(
    params: tuple,
    model_module,
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    rights: np.ndarray,
    market_prices: np.ndarray,
) -> float:
    """Sum of squared errors between model and market prices."""
    try:
        model_prices = model_module.price_for_calibration(params, S, K, T, r, rights)
        if not np.all(np.isfinite(model_prices)):
            return 1e12
        return float(np.sum((model_prices - market_prices) ** 2))
    except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError):
        return 1e12


def calibrate_model(
    name: str,
    snapshot: "OptionsSnapshot",
    maxiter: int = 300,
    seed: int = 42,
    tol: float = 1e-8,
) -> CalibrationResult:
    """Calibrate a single model to observed market data."""
    model = MODELS[name]
    df = snapshot.chains
    K = df["strike"].values.astype(float)
    rights = df["right"].values
    market_prices = df["mid"].values.astype(float)
    S = snapshot.spot
    T = snapshot.dte / 365.0
    r = snapshot.risk_free_rate

    if len(K) == 0 or np.std(market_prices) < 1e-10:
        raise ValueError(f"Insufficient data to calibrate {name}")

    result = differential_evolution(
        _objective,
        bounds=model.BOUNDS,
        args=(model, S, K, T, r, rights, market_prices),
        maxiter=maxiter,
        seed=seed,
        tol=tol,
        polish=True,
        updating="deferred",
    )

    best_params = dict(zip(model.PARAM_NAMES, result.x))
    model_prices = model.price_for_calibration(result.x, S, K, T, r, rights)

    return CalibrationResult(
        name=name,
        params=best_params,
        residual_sse=result.fun,
        model_prices=model_prices,
    )


def calibrate_all(
    snapshot: "OptionsSnapshot",
    maxiter: int = 300,
    seed: int = 42,
) -> dict[str, CalibrationResult]:
    """Calibrate all models and return results keyed by model name."""
    results = {}
    for name in MODELS:
        print(f"  Calibrating {name}...", end=" ", flush=True)
        try:
            res = calibrate_model(name, snapshot, maxiter=maxiter, seed=seed)
            print(f"SSE = {res.residual_sse:.6f}  params = {res.params}")
            results[name] = res
        except Exception as e:
            print(f"FAILED: {e}")
    if not results:
        raise RuntimeError("All model calibrations failed")
    return results
