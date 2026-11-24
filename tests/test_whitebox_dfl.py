"""Offline unit tests for Whitebox DFL, 500-stock Dykstra projection, PIT alignment, backtest, and stats."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from csi500_alpha.backtest import BarQuote, order_fee_rate, run_backtest
from csi500_alpha.factors import align_industry_pit, align_quarterly_pit
from csi500_alpha.fetch_csi500 import (
    BaoSession,
    _group_missing_year_chunks,
    _is_cached_code_valid,
    _load_v3_cached_years,
    _standardize_bs_hfq,
    _standardize_bs_raw,
    _to_iso_date,
    _validate_raw_hfq_consistency,
    collect_member_date_spans,
    collect_unique_member_codes,
)
from csi500_alpha.models import WhiteboxDFL
from csi500_alpha.projection import (
    DifferentiableIEProjection,
    build_constraint_matrix_numpy,
    compute_active_bounds_numpy,
    precompute_eq_operator_numpy,
    project_portfolio_numpy,
)
from csi500_alpha.stats import cscv_pbo, deflated_sharpe_ratio, newey_west_t_stat, paired_return_test


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
        self.assertLess(float(((net.factor_weights().detach() - prior) ** 2).sum()), init_dist)

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
            "start_date": pd.to_datetime(["2023-12-01", "2024-04-15"]).astype("datetime64[ms]"),
        })
        ind_out = align_industry_pit(signals, ind_df)
        self.assertEqual(ind_out.loc[0, "industry"], "Other")
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
        # Day 1: 2-for-1 split -> 4,800 raw shares @ raw=10 (48,000 RMB), desired=4,985 -> buys 100 more shares (1,000 RMB) -> 49,000 RMB invested
        # Day 2: +10% return (hfq 40 -> 44) -> 49,000 * 1.10 + 51,000 cash = 104,900.0 exact!
        self.assertAlmostEqual(float(nav.loc[0, "equity"]), 100_000.0, places=4)
        self.assertAlmostEqual(float(nav.loc[1, "equity"]), 100_000.0, places=4)
        self.assertAlmostEqual(float(nav.loc[1, "gross_traded"]), 1000.0, places=4)
        self.assertAlmostEqual(float(nav.loc[2, "equity"]), 104_900.0, places=4)

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

    def test_baostock_bar_and_valuation_standardization(self) -> None:
        self.assertEqual(_to_iso_date("20140601"), "2014-06-01")
        self.assertEqual(_to_iso_date("2025-12-31"), "2025-12-31")

        raw_input = pd.DataFrame([
            {
                "date": "2024-01-02",
                "code": "sz.000001",
                "open": "10.0",
                "high": "10.6",
                "low": "9.9",
                "close": "10.5",
                "preclose": "10.0",
                "volume": "1000000",
                "amount": "10500000.0",
                "adjustflag": "3",
                "turn": "1.5",
                "tradestatus": "1",
                "pctChg": "5.0",
                "peTTM": "6.5",
                "pbMRQ": "0.75",
                "psTTM": "1.2",
                "isST": "0",
            },
            {
                "date": "2024-01-03",
                "code": "sz.000001",
                "open": "10.5",
                "high": "10.5",
                "low": "10.5",
                "close": "10.5",
                "preclose": "10.5",
                "volume": "",
                "amount": "",
                "adjustflag": "3",
                "turn": "",
                "tradestatus": "0",
                "pctChg": "0.0",
                "peTTM": "6.5",
                "pbMRQ": "0.75",
                "psTTM": "1.2",
                "isST": "0",
            },
            {
                "date": "2024-01-04",
                "code": "sz.000001",
                "open": "21.0",
                "high": "21.0",
                "low": "21.0",
                "close": "21.0",
                "preclose": "20.0",
                "volume": "100",
                "amount": "2100",
                "adjustflag": "1",
                "turn": "0.1",
                "tradestatus": "1",
                "pctChg": "5.0",
                "peTTM": "6.6",
                "pbMRQ": "0.76",
                "psTTM": "1.2",
                "isST": "0",
            },
        ])
        raw_out, val_out = _standardize_bs_raw(raw_input, "sz.000001")
        self.assertEqual(len(raw_out), 2)
        self.assertAlmostEqual(float(raw_out.loc[0, "change_amt"]), 0.5, places=6)
        self.assertEqual(float(raw_out.loc[0, "tradestatus"]), 1.0)
        self.assertEqual(float(raw_out.loc[1, "tradestatus"]), 0.0)
        self.assertTrue(pd.isna(raw_out.loc[1, "turn"]) and pd.isna(raw_out.loc[1, "volume"]))
        self.assertAlmostEqual(float(val_out.loc[0, "float_mv"]), 10500000.0 / 0.015, places=2)
        self.assertAlmostEqual(float(val_out.loc[0, "pe_ttm"]), 6.5, places=6)
        self.assertAlmostEqual(float(val_out.loc[0, "pb_mrq"]), 0.75, places=6)
        self.assertTrue(pd.isna(val_out.loc[1, "float_mv"]))

        hfq_input = pd.DataFrame([
            {
                "date": "2022-01-25",
                "code": "sh.688223",
                "open": "10.0",
                "high": "10.5",
                "low": "9.8",
                "close": "10.2",
                "volume": "100",
                "amount": "1000",
                "adjustflag": "1",
                "turn": "0.5",
            },
            {
                "date": "2022-01-26",
                "code": "sh.688223",
                "open": "5.0",
                "high": "5.2",
                "low": "4.9",
                "close": "5.1",
                "volume": "200",
                "amount": "1000",
                "adjustflag": "3",
                "turn": "0.5",
            },
        ])
        hfq_out = _standardize_bs_hfq(hfq_input, "sh.688223")
        self.assertEqual(len(hfq_out), 1)
        self.assertEqual(hfq_out.loc[0, "date"], pd.Timestamp("2022-01-25"))

        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            mem_dir = raw_dir / "members"
            mem_dir.mkdir(parents=True)
            (mem_dir / "metadata.json").write_text(json.dumps({"pit_members": True}), encoding="utf-8")
            pd.DataFrame({"code": ["sh.600000", "sz.000001"]}).to_parquet(mem_dir / "2024-01-05.parquet", index=False)
            pd.DataFrame({"code": ["sz.300001"]}).to_parquet(mem_dir / "csindex_latest.parquet", index=False)
            self.assertEqual(collect_unique_member_codes(raw_dir), ["sh.600000", "sz.000001"])

    def test_baostock_chunking_consistency_and_v3_cache_reuse(self) -> None:
        chunks_full = _group_missing_year_chunks(
            list(range(2014, 2026)),
            pd.Timestamp("2014-06-01"),
            pd.Timestamp("2025-12-31"),
            max_years=5,
        )
        self.assertEqual(
            chunks_full,
            [
                ("2014-06-01", "2018-12-31"),
                ("2019-01-01", "2023-12-31"),
                ("2024-01-01", "2025-12-31"),
            ],
        )
        chunks_gapped = _group_missing_year_chunks(
            [2014, 2015, 2016, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            pd.Timestamp("2014-06-01"),
            pd.Timestamp("2025-12-31"),
            max_years=5,
        )
        self.assertEqual(
            chunks_gapped,
            [
                ("2014-06-01", "2016-12-31"),
                ("2019-01-01", "2023-12-31"),
                ("2024-01-01", "2025-12-31"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            mem_dir = raw_dir / "members"
            mem_dir.mkdir(parents=True)
            (mem_dir / "metadata.json").write_text(json.dumps({"pit_members": True}), encoding="utf-8")
            pd.DataFrame({"snapshot_date": [pd.Timestamp("2020-01-10")] * 2, "code": ["sh.600000", "sh.600004"]}).to_parquet(
                mem_dir / "2020-01-10.parquet", index=False
            )
            pd.DataFrame({"snapshot_date": [pd.Timestamp("2020-06-12")] * 2, "code": ["sh.600000", "sz.000001"]}).to_parquet(
                mem_dir / "2020-06-12.parquet", index=False
            )

            spans = collect_member_date_spans(raw_dir, "2014-06-01", "2025-12-31", warmup_days=420, tail_days=60)
            self.assertEqual(
                spans["sh.600000"],
                (pd.Timestamp("2020-01-10") - pd.Timedelta(days=420), pd.Timestamp("2020-06-12") + pd.Timedelta(days=60)),
            )
            self.assertEqual(
                spans["sz.000001"],
                (pd.Timestamp("2020-06-12") - pd.Timedelta(days=420), pd.Timestamp("2020-06-12") + pd.Timedelta(days=60)),
            )

            # Truncated raw (2820) vs hfq (2000) validation & cache check
            d_2820 = pd.bdate_range("2014-06-03", periods=2820)
            d_2000 = pd.bdate_range("2014-06-03", periods=2000)
            r_2820 = pd.DataFrame({"date": d_2820})
            h_2000 = pd.DataFrame({"date": d_2000})
            with self.assertRaises(ValueError):
                _validate_raw_hfq_consistency(r_2820, h_2000, "sh.600062")

            rf = raw_dir / "sh.600062_r.parquet"
            hf = raw_dir / "sh.600062_h.parquet"
            vf = raw_dir / "sh.600062_v.parquet"
            r_2820.to_parquet(rf, index=False)
            h_2000.to_parquet(hf, index=False)
            r_2820.to_parquet(vf, index=False)
            self.assertFalse(_is_cached_code_valid(rf, hf, vf))
            r_2820.to_parquet(hf, index=False)
            self.assertTrue(_is_cached_code_valid(rf, hf, vf))

            # v3 cache loader requires both raw & adjusted for a year
            v3_dir = root / "v3"
            (v3_dir / "raw" / "2019").mkdir(parents=True)
            (v3_dir / "adjusted" / "2019").mkdir(parents=True)
            (v3_dir / "raw" / "2020").mkdir(parents=True)
            pd.DataFrame({"date": [pd.Timestamp("2019-01-02")], "close": ["10.0"]}).to_parquet(
                v3_dir / "raw" / "2019" / "sh.600000.parquet", index=False
            )
            pd.DataFrame({"date": [pd.Timestamp("2019-01-02")], "close": ["20.0"]}).to_parquet(
                v3_dir / "adjusted" / "2019" / "sh.600000.parquet", index=False
            )
            pd.DataFrame({"date": [pd.Timestamp("2020-01-02")], "close": ["11.0"]}).to_parquet(
                v3_dir / "raw" / "2020" / "sh.600000.parquet", index=False
            )
            raw_parts, hfq_parts, cached_years = _load_v3_cached_years(v3_dir, "sh.600000", [2019, 2020])
            self.assertEqual(cached_years, {2019})
            self.assertEqual(len(raw_parts), 1)
            self.assertEqual(len(hfq_parts), 1)

        # Verify BaoSession guards send_msg & restores globals cleanly on exception without network calls
        import baostock.util.socketutil as sock_util

        orig_socket = socket.socket
        orig_timeout = socket.getdefaulttimeout()
        with mock.patch.object(sock_util, "send_msg", return_value=None) as patched_orig:
            with self.assertRaises(RuntimeError):
                with BaoSession(timeout=45.0) as sess:
                    self.assertIsNot(socket.socket, orig_socket)
                    self.assertIsNot(sock_util.send_msg, patched_orig)
                    try:
                        sock_util.send_msg("noop")
                    except ConnectionError as exc:
                        raise RuntimeError("triggered_guard") from exc
            self.assertIs(sock_util.send_msg, patched_orig)
        self.assertIs(socket.socket, orig_socket)
        self.assertEqual(socket.getdefaulttimeout(), orig_timeout)

    def test_v2_strict_open_fees_participation_and_paired_stats(self) -> None:
        self.assertAlmostEqual(order_fee_rate(pd.Timestamp("2023-08-25"), is_sell=False, fee_mode="ashare_realistic"), 0.0010)
        self.assertAlmostEqual(order_fee_rate(pd.Timestamp("2023-08-25"), is_sell=True, fee_mode="ashare_realistic"), 0.0020)
        self.assertAlmostEqual(order_fee_rate(pd.Timestamp("2023-08-28"), is_sell=True, fee_mode="ashare_realistic"), 0.0015)

        days = pd.to_datetime(["2024-01-02", "2024-01-03"])
        # Open tradable even when close_ok=False; prev_amount=100,000 with 5% participation caps buy at 5,000 RMB
        bars = {
            days[0]: {"sh.600000": BarQuote(10.0, np.nan, 10.0, 20.0, np.nan, open_tradable=True, close_ok=False, prev_amount=100_000.0)},
            days[1]: {"sh.600000": BarQuote(10.0, 10.0, 10.0, 20.0, 20.0, open_tradable=True, close_ok=True, prev_amount=100_000.0)},
        }
        preds = pd.DataFrame({
            "signal_date": [pd.Timestamp("2023-12-29")],
            "entry_date": [days[0]],
            "code": ["sh.600000"],
            "target_weight": [0.5],
        })
        nav = run_backtest("cap_test", preds, bars, pd.DatetimeIndex(days), fee_mode="fixed_0bp", initial_cash=100_000.0, max_participation=0.05)
        self.assertAlmostEqual(float(nav.loc[0, "gross_traded"]), 5_000.0, places=4)
        self.assertEqual(int(nav.loc[0, "cap_clipped_orders"]), 1)

        rng = np.random.default_rng(123)
        bench = rng.normal(0.0002, 0.01, size=600)
        ret_b = bench + rng.normal(0.0001, 0.002, size=600)
        ret_a = ret_b + rng.normal(0.0004, 0.001, size=600)
        res = paired_return_test(ret_a, ret_b, bench_ret=bench, lag=4, n_boot=300, block_size=10, seed=42)
        self.assertGreater(res["nw_tstat"], 2.0)
        self.assertLess(res["ann_diff_ci_lo"], res["ann_diff_ci_hi"])
        mat = np.column_stack([ret_a - bench, ret_b - bench, rng.normal(0.0, 0.002, size=600)])
        sr3, _ = deflated_sharpe_ratio(mat, 0, periods_per_year=252, n_trials_override=3)
        sr10, _ = deflated_sharpe_ratio(mat, 0, periods_per_year=252, n_trials_override=10)
        self.assertGreater(sr10, sr3)


if __name__ == "__main__":
    unittest.main()
