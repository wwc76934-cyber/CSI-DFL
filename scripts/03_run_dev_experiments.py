"""Step 3: Run v2 rolling models (B0, S0, S0_LowTurn_Band, S1, S2), multi-fee/capacity backtests, 5-seed check, and 7-factor pure-PIT ablation."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from csi500_alpha.backtest import build_quote_lookup, run_backtest, summarize_performance
from csi500_alpha.config import load_config
from csi500_alpha.factors import FACTORS, FACTORS_PURE_PIT_7, _to_ns
from csi500_alpha.models import prepare_week_records, run_all_models


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config(ROOT / "configs" / "experiment.toml")
    raw_dir = ROOT / "data" / "raw"
    proc_dir = ROOT / "data" / "processed"
    rep_dir = ROOT / "reports"
    pred_dir = proc_dir / "predictions"
    bt_dir = proc_dir / "backtests"
    pred_dir.mkdir(parents=True, exist_ok=True)
    bt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(proc_dir / "panel_csi500_dev.parquet")
    daily_bars = pd.read_parquet(proc_dir / "daily_bars_csi500_dev.parquet")
    benchmarks = pd.read_parquet(proc_dir / "benchmarks_csi500_dev.parquet")
    cal_df = pd.read_parquet(raw_dir / "calendar.parquet")
    trading_days = pd.DatetimeIndex(_to_ns(cal_df["date"]).sort_values().unique())

    eval_start = pd.Timestamp(cfg["dates"]["eval_start"])
    dev_end = pd.Timestamp(cfg["dates"]["dev_end"])
    eval_days = trading_days[(trading_days >= eval_start) & (trading_days <= dev_end)]
    bars = build_quote_lookup(daily_bars, eval_days)

    # --- 1. Primary 10-Factor Ladder ---
    print("[exp] Preparing 10-factor weekly tensors & projection operators...", flush=True)
    weeks = prepare_week_records(panel, cfg, factor_cols=FACTORS)
    print(f"[exp] Prepared {len(weeks)} weekly cross-sections in {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    preds, window_audit, ridge_w, dfl_w, dfl_turn_rec = run_all_models(weeks, cfg, factor_cols=FACTORS)
    print(f"[exp] Completed primary B0/S0/S0_LowTurn/S1/S2 runs in {time.perf_counter() - t1:.1f}s", flush=True)

    window_audit.to_csv(rep_dir / "training_window_audit.csv", index=False, encoding="utf-8-sig")
    ridge_w.to_csv(rep_dir / "ridge_factor_weights_history.csv", index=False, encoding="utf-8-sig")
    dfl_w.to_csv(rep_dir / "dfl_factor_weights_history.csv", index=False, encoding="utf-8-sig")

    for name, df in preds.items():
        df.to_parquet(pred_dir / f"{name}.parquet", index=False)

    strategies = list(cfg["models"]["evaluated_strategies"])
    fee_modes = list(cfg["portfolio"]["fee_modes"])
    init_cash = float(cfg["portfolio"]["initial_cash"])
    single_cap = float(cfg["portfolio"]["single_stock_cap"])
    cash_res = float(cfg["portfolio"]["cash_reserve"])

    overall_rows, yearly_rows, daily_frames = [], [], []
    nav_store: dict[tuple[str, str], pd.DataFrame] = {}

    for fee_mode in fee_modes:
        # Run B0_FloatMV_Anchor first so all other strategies compare against same-engine, same-cost B0!
        anchor_nav = run_backtest(
            strategy_name="B0_FloatMV_Anchor",
            predictions=preds["B0_FloatMV_Anchor"],
            bars=bars,
            eval_days=eval_days,
            fee_mode=fee_mode,
            initial_cash=init_cash,
            single_cap=single_cap,
            cash_reserve=cash_res,
        )
        nav_store[("B0_FloatMV_Anchor", fee_mode)] = anchor_nav

        for name in strategies:
            nav_df = anchor_nav if name == "B0_FloatMV_Anchor" else run_backtest(
                strategy_name=name,
                predictions=preds[name],
                bars=bars,
                eval_days=eval_days,
                fee_mode=fee_mode,
                initial_cash=init_cash,
                single_cap=single_cap,
                cash_reserve=cash_res,
            )
            nav_store[(name, fee_mode)] = nav_df
            overall, yearly_df, daily_active = summarize_performance(
                name, fee_mode, nav_df, benchmarks, preds[name],
                initial_cash=init_cash, anchor_nav_df=anchor_nav,
            )
            overall_rows.append(overall)
            yearly_rows.append(yearly_df)
            daily_frames.append(daily_active.assign(strategy=name, fee_mode=fee_mode))
            nav_df.to_parquet(bt_dir / f"{name}_{fee_mode}_nav.parquet", index=False)
            print(
                f"[bt {fee_mode:16s}] {name:18s} | Net {overall['net_return']:+.2%} | "
                f"vsH {overall['excess_h00905_pp']:+.2%} | vsB0 {overall['excess_floatmv_pp']:+.2%} | "
                f"IR_B0 {overall['ir_floatmv']:+.3f} | Turn {overall['weekly_two_way_turnover']:.2%} | "
                f"RealInd {overall['realized_max_ind_active_exp']:.2%}",
                flush=True,
            )

    pd.DataFrame(overall_rows).to_csv(rep_dir / "dev_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(yearly_rows, ignore_index=True).to_csv(rep_dir / "dev_yearly.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_parquet(bt_dir / "daily_active_returns.parquet", index=False)

    # Reconcile DFL training drifted turnover vs target undrifted turnover vs actual executed account turnover
    s2_nav_real = nav_store[("S2_WhiteboxDFL", "ashare_realistic")].copy()
    s2_nav_real["executed_turn"] = s2_nav_real["gross_traded"] / s2_nav_real["opening_equity"]
    rec_df = dfl_turn_rec.merge(
        s2_nav_real[["date", "executed_turn"]].rename(columns={"date": "entry_date"}),
        on="entry_date",
        how="left",
    )
    rec_df.to_csv(rep_dir / "dfl_turnover_reconcile.csv", index=False, encoding="utf-8-sig")

    # --- 2. Capacity & Liquidity Stress Test (1M / 10M / 100M RMB under ashare_realistic + 5% prev_amount cap) ---
    print("\n[exp] Running capacity stress test (1M / 10M / 100M RMB, 5% prev_amount participation cap)...", flush=True)
    cap_rows = []
    max_part = float(cfg["portfolio"]["max_prev_amount_participation"])
    cap_strats = ["B0_FloatMV_Anchor", "S0_EqualWeight", "S0_LowTurn_Band", "S2_WhiteboxDFL"]
    for cap_cash in cfg["portfolio"]["capacity_cash_list"]:
        cap_f = float(cap_cash)
        b0_cap = run_backtest(
            strategy_name="B0_FloatMV_Anchor",
            predictions=preds["B0_FloatMV_Anchor"],
            bars=bars,
            eval_days=eval_days,
            fee_mode="ashare_realistic",
            initial_cash=cap_f,
            single_cap=single_cap,
            cash_reserve=cash_res,
            max_participation=max_part,
        )
        for name in cap_strats:
            nav_c = b0_cap if name == "B0_FloatMV_Anchor" else run_backtest(
                strategy_name=name,
                predictions=preds[name],
                bars=bars,
                eval_days=eval_days,
                fee_mode="ashare_realistic",
                initial_cash=cap_f,
                single_cap=single_cap,
                cash_reserve=cash_res,
                max_participation=max_part,
            )
            ov, _, _ = summarize_performance(
                name, "ashare_realistic", nav_c, benchmarks, preds[name],
                initial_cash=cap_f, anchor_nav_df=b0_cap,
            )
            ov["capital_rmb"] = int(cap_cash)
            ov["max_participation"] = max_part
            cap_rows.append(ov)
            print(
                f"[cap {int(cap_cash/1e6):3d}M] {name:18s} | Net {ov['net_return']:+.2%} | "
                f"vsB0 {ov['excess_floatmv_pp']:+.2%} | IR_B0 {ov['ir_floatmv']:+.3f} | "
                f"Cash {ov['avg_cash_weight']:.2%} | Fill {ov['order_fill_rate']:.2%} | Clipped {ov['cap_clipped_orders']}",
                flush=True,
            )
    pd.DataFrame(cap_rows).to_csv(rep_dir / "capacity_stress_test.csv", index=False, encoding="utf-8-sig")

    # --- 3. 5-Seed DFL Stability Check ---
    print("\n[exp] Running 5-seed stability check for S2_WhiteboxDFL...", flush=True)
    seed_rows = []
    for seed in cfg["models"]["dfl_stability_seeds"]:
        s_preds, _, _, _, _ = run_all_models(
            weeks, cfg, factor_cols=FACTORS, seed_override=int(seed),
            strategies_override=["S2_WhiteboxDFL"], verbose=False,
        )
        for fm in ["fixed_20bp", "ashare_realistic"]:
            nav_s = run_backtest(
                strategy_name="S2_WhiteboxDFL",
                predictions=s_preds["S2_WhiteboxDFL"],
                bars=bars,
                eval_days=eval_days,
                fee_mode=fm,
                initial_cash=init_cash,
                single_cap=single_cap,
                cash_reserve=cash_res,
            )
            ov, _, _ = summarize_performance(
                "S2_WhiteboxDFL", fm, nav_s, benchmarks, s_preds["S2_WhiteboxDFL"],
                initial_cash=init_cash, anchor_nav_df=nav_store[("B0_FloatMV_Anchor", fm)],
            )
            ov["seed"] = int(seed)
            seed_rows.append(ov)
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(rep_dir / "dfl_multiseed_stability.csv", index=False, encoding="utf-8-sig")
    print(seed_df[["seed", "fee_mode", "net_return", "excess_h00905_pp", "excess_floatmv_pp", "ir_floatmv", "weekly_two_way_turnover"]].to_string(index=False), flush=True)

    # --- 4. 7-Factor Pure-PIT Ablation (removing 3 restated Eastmoney quarterly factors) ---
    print("\n[exp] Running 7-factor pure-PIT ablation (excluding ROE, NP_YOY, CFO_YIELD)...", flush=True)
    weeks_7f = prepare_week_records(panel, cfg, factor_cols=FACTORS_PURE_PIT_7)
    preds_7f, _, _, dfl_w_7f, _ = run_all_models(
        weeks_7f, cfg, factor_cols=FACTORS_PURE_PIT_7, verbose=False,
    )
    dfl_w_7f.to_csv(rep_dir / "dfl_7f_pure_pit_weights.csv", index=False, encoding="utf-8-sig")

    abl_rows, abl_daily = [], []
    for fm in ["fixed_0bp", "fixed_10bp", "fixed_20bp", "ashare_realistic"]:
        b0_7 = nav_store[("B0_FloatMV_Anchor", fm)]
        for name in strategies:
            nav_7 = b0_7 if name == "B0_FloatMV_Anchor" else run_backtest(
                strategy_name=name,
                predictions=preds_7f[name],
                bars=bars,
                eval_days=eval_days,
                fee_mode=fm,
                initial_cash=init_cash,
                single_cap=single_cap,
                cash_reserve=cash_res,
            )
            ov, _, d_act = summarize_performance(
                name, fm, nav_7, benchmarks, preds_7f[name],
                initial_cash=init_cash, anchor_nav_df=b0_7,
            )
            abl_rows.append(ov)
            abl_daily.append(d_act.assign(strategy=name, fee_mode=fm))
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(rep_dir / "pure_pit_7f_ablation.csv", index=False, encoding="utf-8-sig")
    pd.concat(abl_daily, ignore_index=True).to_parquet(bt_dir / "pure_pit_7f_daily_active.parquet", index=False)
    print(
        abl_df.loc[abl_df["fee_mode"].isin(["fixed_20bp", "ashare_realistic"]), [
            "fee_mode", "strategy", "net_return", "excess_h00905_pp", "excess_floatmv_pp", "ir_floatmv", "weekly_two_way_turnover"
        ]].to_string(index=False),
        flush=True,
    )
    print(f"\n[exp] Full v2 suite finished in {time.perf_counter() - t0:.1f}s total.", flush=True)


if __name__ == "__main__":
    main()
