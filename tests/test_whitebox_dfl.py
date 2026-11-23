"""Offline unit tests for Whitebox DFL, 500-stock Dykstra projection, PIT alignment, backtest, and stats."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from csi500_alpha.backtest import BarQuote, run_backtest
from csi500_alpha.factors import align_industry_pit, align_quarterly_pit
from csi500_alpha.models import WhiteboxDFL
from csi500_alpha.projection import (
    DifferentiableIEProjection,
    build_constraint_matrix_numpy,
    compute_active_bounds_numpy,
    precompute_eq_operator_numpy,
    project_portfolio_numpy,
)
from csi500_alpha.stats import cscv_pbo, deflated_sharpe_ratio, newey_west_t_stat


class TestWhiteboxDFL(unittest.TestCase):
    def _synthetic_500_cross_section(self, n: int = 500, n_ind: int = 30, seed: int = 19):
        rng = np.random.default_rng(seed)
        raw_a = rng.dirichlet(np.full(n, 2.0))
        anchor = np.minimum(raw_a, 0.035)
        anchor = (anchor / anchor.sum()).astype(np.float32)
        ind_codes = np.array([f"SW_{i % n_ind:02d}" for i in range(n)])
        rng.shuffle(ind_codes)
        beta = np.clip(rng.normal(1.0, 0.25, size=n), 0.3, 2.0).astype(np.float32)
        size_z = rng.normal(0.0, 1.0, size=n).astype(np.float32)
        size_z = ((size_z - size_z.mean()) / size_z.std()).astype(np.float32)
        score = rng.normal(0.0, 1.0, size=n).astype(np.float32)
        return anchor, ind_codes, beta, size_z, score

    def test_projection_constraints_500_stocks(self) -> None:
        anchor, ind_codes, beta, size_z, score = self._synthetic_500_cross_section()
        A = build_constraint_matrix_numpy(ind_codes, beta, size_z)
        B_eq = precompute_eq_operator_numpy(A, eps=1e-7)
        lo, hi = compute_active_bounds_numpy(anchor, active_limit=0.015, single_cap=0.04)

        proj = DifferentiableIEProjection(num_steps=30, eps=1e-7)
        dw0 = torch.from_numpy((0.005 * score).astype(np.float32))
        dw_t = proj(dw0, torch.from_numpy(A), torch.from_numpy(lo), torch.from_numpy(hi), B_eq=torch.from_numpy(B_eq)).detach().numpy()

        self.assertLess(abs(float(dw_t.sum())), 1e-5)
        max_ind = max(abs(float(dw_t[ind_codes == g].sum())) for g in np.unique(ind_codes))
        self.assertLess(max_ind, 1e-3)
        self.assertLess(abs(float((beta - beta.mean()) @ dw_t)), 1e-3)
        self.assertLess(abs(float((size_z - size_z.mean()) @ dw_t)), 1e-3)
        self.assertTrue(np.all(dw_t >= lo - 1e-6))
        self.assertTrue(np.all(dw_t <= hi + 1e-6))

        dw_np, w_np = project_portfolio_numpy(
            score, anchor, ind_codes, beta, size_z, gamma=0.005, active_limit=0.015, single_cap=0.04, num_steps=30, eps=1e-7
        )
        self.assertAlmostEqual(float(w_np.sum()), 1.0, places=6)
        self.assertTrue(np.all(w_np >= -1e-7) and np.all(w_np <= 0.04 + 1e-7))
        self.assertLess(max(abs(float(dw_np[ind_codes == g].sum())) for g in np.unique(ind_codes)), 1e-4)

    def test_whitebox_dfl_gradients_and_prior_shrinkage(self) -> None:
        torch.manual_seed(7)
        n = 150
        anchor, ind_codes, beta, size_z, _ = self._synthetic_500_cross_section(n=n, n_ind=10, seed=7)
        A = build_constraint_matrix_numpy(ind_codes, beta, size_z)
        B_eq = precompute_eq_operator_numpy(A, eps=1e-7)
        lo, hi = compute_active_bounds_numpy(anchor, active_limit=0.015, single_cap=0.04)

        net = WhiteboxDFL(num_factors=10, gamma=0.005, train_steps=12, eps=1e-7)
        # Verify exact 10 learnable parameters
        self.assertEqual(sum(p.numel() for p in net.parameters()), 10)

        x = torch.randn(n, 10)
        ret = torch.randn(n) * 0.03
        dw, score_z, theta = net.forward_portfolio(
            x, torch.from_numpy(A), torch.from_numpy(lo), torch.from_numpy(hi), B_eq=torch.from_numpy(B_eq)
        )
        self.assertAlmostEqual(float(theta.detach().sum()), 1.0, places=5)
        self.assertTrue(bool((theta.detach() > 0).all()))

        prior = torch.full((10,), 0.1)
        loss = -(dw * ret).sum() + 2.0 * ((theta - prior) ** 2).sum()
        loss.backward()
        self.assertIsNotNone(net.logits.grad)
        self.assertTrue(torch.isfinite(net.logits.grad).all().item())
        self.assertGreater(float(net.logits.grad.abs().sum()), 0.0)

        # Verify L2 prior shrinkage pulls biased weights back toward 0.1
        with torch.no_grad():
            net.logits.copy_(torch.tensor([2.0] + [0.0] * 9))
            init_dist = float(((net.factor_weights() - prior) ** 2).sum())
        opt = torch.optim.SGD([net.logits], lr=0.5)
        for _ in range(5):
            opt.zero_grad()
            th = net.factor_weights()
            pen = 2.0 * ((th - prior) ** 2).sum()
            pen.backward()
            opt.step()
        self.assertLess(float(((net.factor_weights() - prior) ** 2).sum()), init_dist)

    def test_pit_quarterly_and_industry_alignment(self) -> None:
        signals = pd.DataFrame({
            "signal_date": pd.to_datetime(["2023-10-20", "2023-11-03", "2024-04-01", "2024-04-30"]).astype("datetime64[us]"),
            "code": ["sz.000001"] * 4,
        })
        quarterly = pd.DataFrame({
            "code": ["sz.000001"] * 3,
            "report_date": pd.to_datetime(["2023-09-30", "2023-12-31", "2024-03-31"]).astype("datetime64[ms]"),
            "pub_date": pd.to_datetime(["2024-10-30", "2024-04-03", "2024-04-27"]).astype("datetime64[ms]"),
            "roe": [10.0, 14.0, 4.0],
            "np_yoy": [5.0, 8.0, 12.0],
            "cfo_per_share": [1.0, 2.0, 0.5],
        })
        aligned = align_quarterly_pit(signals, quarterly)
        self.assertTrue(pd.isna(aligned.loc[0, "roe"]))
        self.assertEqual(aligned.loc[1, "roe"], 10.0)
        self.assertEqual(aligned.loc[2, "roe"], 10.0)
        self.assertEqual(aligned.loc[3, "roe"], 4.0)

        ind_df = pd.DataFrame({
            "code": ["sz.000001", "sz.000001"],
            "industry": ["SW_801780_银行", "SW_801790_非银金融"],
            "start_date": pd.to_datetime(["2020-01-01", "2024-04-15"]).astype("datetime64[ms]"),
        })
        ind_out = align_industry_pit(signals, ind_df)
        self.assertEqual(ind_out.loc[2, "industry"], "SW_801780_银行")
        self.assertEqual(ind_out.loc[3, "industry"], "SW_801790_非银金融")

    def test_backtest_hfq_continuity_across_split(self) -> None:
        days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        # 2-for-1 split on day 2: raw halves from 20 -> 10, hfq stays 40 -> 40 (zero return)
        bars = {
            days[0]: {"sh.600000": BarQuote(20.0, 20.0, 20.0, 40.0, 40.0, True)},
            days[1]: {"sh.600000": BarQuote(10.0, 10.0, 10.0, 40.0, 40.0, True)},
            days[2]: {"sh.600000": BarQuote(10.5, 11.0, 10.0, 42.0, 44.0, True)},
        }
        preds = pd.DataFrame({
            "signal_date": [pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-02")],
            "entry_date": [days[0], days[1]],
            "code": ["sh.600000", "sh.600000"],
            "target_weight": [0.5, 0.5],
        })
        nav = run_backtest("test", preds, bars, pd.DatetimeIndex(days), fee_bps=0, initial_cash=100_000.0)
        # 100,000 * (1 - 0.003) * 0.5 = 49,850 -> 2,400 raw shares = 48,000 RMB invested at raw=20 / hfq=40
        # Day 1: 2-for-1 split -> 4,800 raw shares @ raw=10 = 48,000 RMB (0 spurious trades!)
        # Day 2: +10% return (hfq 40 -> 44) -> 48,000 * 1.10 + 52,000 cash = 104,800.0 exact!
        self.assertAlmostEqual(float(nav.loc[0, "equity"]), 100_000.0, places=4)
        self.assertAlmostEqual(float(nav.loc[1, "equity"]), 100_000.0, places=4)
        self.assertAlmostEqual(float(nav.loc[1, "gross_traded"]), 0.0, places=4)
        self.assertAlmostEqual(float(nav.loc[2, "equity"]), 104_800.0, places=4)

    def test_stats_n3_valid(self) -> None:
        rng = np.random.default_rng(9)
        mat = rng.normal(0.0003, 0.003, size=(1200, 3))
        m, t_nw, p_nw = newey_west_t_stat(mat[:, 0], lag=4)
        self.assertTrue(all(np.isfinite(v) for v in (m, t_nw, p_nw)))
        sr0, dsr = deflated_sharpe_ratio(mat, 0, periods_per_year=252)
        self.assertTrue(np.isfinite(sr0) and 0.0 <= dsr <= 1.0)
        pbo, med_rank, splits = cscv_pbo(mat, n_blocks=16, max_splits=100)
        self.assertEqual(splits, 100)
        self.assertTrue(0.0 <= pbo <= 1.0 and 0.0 <= med_rank <= 1.0)


if __name__ == "__main__":
    unittest.main()
