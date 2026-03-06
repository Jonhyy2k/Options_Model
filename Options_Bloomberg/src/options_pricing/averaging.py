"""Stage 4 — Bayesian Model Averaging.

Weight each model inversely by its calibration SSE. For any strike, the
model-averaged price is the weighted sum of all four model prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from options_pricing.calibration import CalibrationResult


@dataclass
class AveragedPrices:
    """Container for BMA-weighted prices."""

    strikes: np.ndarray
    rights: np.ndarray
    prices: np.ndarray               # model-averaged price per strike
    weights: dict[str, float]         # normalised model weights
    per_model_prices: dict[str, np.ndarray]


def compute_weights(
    results: dict[str, CalibrationResult],
) -> dict[str, float]:
    """Inverse-SSE weights, normalised to sum to 1.

    Models with lower SSE get higher weight.  A small epsilon prevents
    division by zero if a model achieves near-perfect fit.
    """
    eps = 1e-12
    raw = {name: 1.0 / (res.residual_sse + eps) for name, res in results.items()}
    total = sum(raw.values())
    return {name: w / total for name, w in raw.items()}


def average(
    results: dict[str, CalibrationResult],
    strikes: np.ndarray,
    rights: np.ndarray,
) -> AveragedPrices:
    """Compute model-averaged prices across all calibrated models.

    Parameters
    ----------
    results : dict of CalibrationResult from calibrate_all()
    strikes : array of strikes (from the snapshot DataFrame)
    rights : array of "C"/"P" (matching strikes)

    Returns
    -------
    AveragedPrices
    """
    weights = compute_weights(results)
    blended = np.zeros(len(strikes), dtype=float)
    per_model = {}

    for name, res in results.items():
        w = weights[name]
        per_model[name] = res.model_prices
        blended += w * res.model_prices

    return AveragedPrices(
        strikes=strikes,
        rights=rights,
        prices=blended,
        weights=weights,
        per_model_prices=per_model,
    )
