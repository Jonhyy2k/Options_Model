import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from options_pricing.data import load_snapshot
from options_pricing.market_inputs import infer_market_inputs
from options_pricing.models import bsm


def _synthetic_chain(
    *,
    S: float = 100.0,
    T: float = 0.25,
    r: float = 0.045,
    q: float = 0.015,
    sigma: float = 0.25,
) -> pd.DataFrame:
    rows = []
    strikes = np.arange(80.0, 121.0, 5.0)
    for strike in strikes:
        for right in ("C", "P"):
            mid = float(bsm.price(S, strike, T, r, sigma, q=q, right=right))
            spread = 0.15 + 0.002 * abs(strike - S)
            bid = max(mid - spread / 2, 0.01)
            ask = mid + spread / 2
            rows.append(
                {
                    "strike": float(strike),
                    "right": right,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "iv": sigma,
                    "volume": 1_000,
                    "oi": 10_000,
                }
            )
    return pd.DataFrame(rows)


class MarketInputTests(unittest.TestCase):
    def test_infer_market_inputs_recovers_forward_and_carry(self) -> None:
        S = 100.0
        T = 0.25
        r = 0.045
        q = 0.015
        chain = _synthetic_chain(S=S, T=T, r=r, q=q)

        inputs = infer_market_inputs(chain, S, T, fallback_rate=0.05)

        self.assertEqual(inputs.source, "market_implied")
        self.assertGreaterEqual(inputs.n_pairs_used, 4)
        self.assertIsNotNone(inputs.r_squared)
        self.assertGreater(inputs.r_squared, 0.999)
        self.assertAlmostEqual(inputs.discount_factor, np.exp(-r * T), places=3)
        self.assertAlmostEqual(inputs.forward_price, S * np.exp((r - q) * T), places=2)
        self.assertAlmostEqual(inputs.implied_rate, r, places=3)
        self.assertAlmostEqual(inputs.dividend_yield, q, places=3)

    def test_infer_market_inputs_falls_back_with_insufficient_pairs(self) -> None:
        chain = _synthetic_chain().query("right == 'C'").copy()
        inputs = infer_market_inputs(chain, 100.0, 0.25, fallback_rate=0.05)

        self.assertEqual(inputs.source, "flat_rate")
        self.assertEqual(inputs.dividend_yield, 0.0)
        self.assertAlmostEqual(inputs.implied_rate, 0.05, places=8)
        self.assertTrue(inputs.notes)

    def test_load_snapshot_rebuilds_market_inputs(self) -> None:
        chain = _synthetic_chain()
        meta = {
            "ticker": "TEST",
            "spot": 100.0,
            "expiry": "20261218",
            "dte": 91,
            "risk_free_rate": 0.05,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.csv"
            with open(path, "w") as f:
                f.write(f"# {json.dumps(meta)}\n")
                chain.to_csv(f, index=False)

            snapshot = load_snapshot(str(path), carry_mode="market_implied")

        self.assertEqual(snapshot.carry_source, "market_implied")
        self.assertIsNotNone(snapshot.market_inputs)
        self.assertGreater(snapshot.forward_price, 0.0)
        self.assertGreater(snapshot.discount_factor, 0.0)
        self.assertGreaterEqual(snapshot.market_inputs.n_pairs_used, 4)


if __name__ == "__main__":
    unittest.main()
