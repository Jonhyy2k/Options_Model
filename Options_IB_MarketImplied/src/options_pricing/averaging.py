"""Stage 4 — model averaging.

Default weights use a BIC/Laplace approximation to posterior model
probabilities. That is a practical form of Bayesian model averaging:

    p(M_j | D) ~ exp(-0.5 * BIC_j)

For diagnostics, simpler weighting rules are also available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from options_pricing.calibration import CalibrationResult

DEFAULT_WEIGHTING_METHOD = "bic"
_VALID_WEIGHTING_METHODS = {"bic", "aic", "inverse_sse"}


@dataclass
class AveragedPrices:
    """Container for BMA-weighted prices."""

    strikes: np.ndarray
    rights: np.ndarray
    prices: np.ndarray               # model-averaged price per strike
    weights: dict[str, float]         # normalised model weights
    per_model_prices: dict[str, np.ndarray]
    weighting_method: str
    score_label: str
    scores: dict[str, float]


def compute_weights(
    results: dict[str, CalibrationResult],
    n_obs: int,
    method: str = DEFAULT_WEIGHTING_METHOD,
) -> tuple[dict[str, float], str, dict[str, float]]:
    """Compute model weights.

    Parameters
    ----------
    results : calibration results keyed by model name
    n_obs : number of observed option prices used in calibration
    method : one of ``"bic"``, ``"aic"``, ``"inverse_sse"``

    Returns
    -------
    (weights, score_label, scores)
    """
    if method not in _VALID_WEIGHTING_METHODS:
        choices = ", ".join(sorted(_VALID_WEIGHTING_METHODS))
        raise ValueError(f"Unknown weighting method '{method}'. Choose from: {choices}")
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")

    eps = 1e-12
    scores: dict[str, float] = {}

    if method == "inverse_sse":
        raw = {name: 1.0 / (max(float(res.residual_sse), eps)) for name, res in results.items()}
        total = sum(raw.values())
        weights = {name: value / total for name, value in raw.items()}
        scores = {name: float(res.residual_sse) for name, res in results.items()}
        return weights, "SSE", scores

    log_weights = {}
    for name, res in results.items():
        k = len(res.params)
        sse = max(float(res.residual_sse), eps)
        sigma2_mle = sse / n_obs
        loglike = -0.5 * n_obs * (np.log(2.0 * np.pi * sigma2_mle) + 1.0)

        if method == "aic":
            score = 2 * k - 2 * loglike
            score_label = "AIC"
        else:
            score = k * np.log(n_obs) - 2 * loglike
            score_label = "BIC"

        scores[name] = float(score)
        log_weights[name] = -0.5 * float(score)

    log_norm = max(log_weights.values())
    raw = {name: np.exp(log_w - log_norm) for name, log_w in log_weights.items()}
    total = sum(raw.values())
    weights = {name: value / total for name, value in raw.items()}
    return weights, score_label, scores


def average(
    results: dict[str, CalibrationResult],
    strikes: np.ndarray,
    rights: np.ndarray,
    weighting_method: str = DEFAULT_WEIGHTING_METHOD,
) -> AveragedPrices:
    """Compute model-averaged prices across all calibrated models.

    Parameters
    ----------
    results : dict of CalibrationResult from calibrate_all()
    strikes : array of strikes (from the snapshot DataFrame)
    rights : array of "C"/"P" (matching strikes)
    weighting_method : model weighting rule

    Returns
    -------
    AveragedPrices
    """
    weights, score_label, scores = compute_weights(
        results,
        n_obs=len(strikes),
        method=weighting_method,
    )
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
        weighting_method=weighting_method,
        score_label=score_label,
        scores=scores,
    )
