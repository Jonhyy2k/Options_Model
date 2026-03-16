import unittest

import numpy as np

from options_pricing.averaging import average, compute_weights
from options_pricing.calibration import CalibrationResult


def _result(name: str, sse: float, n_params: int, prices: list[float]) -> CalibrationResult:
    params = {f"p{i}": float(i) for i in range(n_params)}
    return CalibrationResult(
        name=name,
        params=params,
        residual_sse=sse,
        model_prices=np.array(prices, dtype=float),
    )


class AveragingTests(unittest.TestCase):
    def test_bic_weights_penalize_extra_parameters_on_near_tie(self) -> None:
        results = {
            "simple": _result("simple", sse=100.0, n_params=1, prices=[1.0, 2.0]),
            "complex": _result("complex", sse=99.0, n_params=5, prices=[2.0, 3.0]),
        }

        weights, score_label, _ = compute_weights(results, n_obs=40, method="bic")

        self.assertEqual(score_label, "BIC")
        self.assertGreater(weights["simple"], weights["complex"])

    def test_bic_weights_still_reward_clear_fit_improvement(self) -> None:
        results = {
            "simple": _result("simple", sse=100.0, n_params=1, prices=[1.0, 2.0]),
            "complex": _result("complex", sse=10.0, n_params=5, prices=[2.0, 3.0]),
        }

        weights, _, _ = compute_weights(results, n_obs=40, method="bic")

        self.assertGreater(weights["complex"], 0.99)

    def test_average_uses_requested_weighting_method(self) -> None:
        results = {
            "simple": _result("simple", sse=100.0, n_params=1, prices=[1.0, 2.0]),
            "complex": _result("complex", sse=99.0, n_params=5, prices=[3.0, 4.0]),
        }
        strikes = np.array([100.0, 110.0])
        rights = np.array(["C", "C"])

        avg = average(results, strikes, rights, weighting_method="bic")

        expected = (
            avg.weights["simple"] * results["simple"].model_prices
            + avg.weights["complex"] * results["complex"].model_prices
        )
        np.testing.assert_allclose(avg.prices, expected)
        self.assertEqual(avg.weighting_method, "bic")
        self.assertEqual(avg.score_label, "BIC")


if __name__ == "__main__":
    unittest.main()
