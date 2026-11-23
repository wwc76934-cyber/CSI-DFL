"""Step 3: Run 2017-2025 rolling models (S0, S1, S2_WhiteboxDFL) and open-to-open backtests."""

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
from csi500_alpha.factors import _to_ns
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

    print("[exp] Preparing weekly tensors & projection operators...", flush=True)
    weeks = prepare_week_records(panel, cfg)
    print(f"[exp] Prepared {len(weeks)} weekly cross-sections in {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    preds, window_audit, ridge_w, dfl_w = run_all_models(weeks, cfg)
    print(f"[exp] Completed S0/S1/S2 runs in {time.perf_counter() - t1:.1f}s", flush=True)

    window_audit.to_csv(rep_dir / "training_window_audit.csv", index=False, encoding="utf-8-sig")
    ridge_w.to_csv(rep_dir / "ridge_factor_weights_history.csv", index=False, encoding="utf-8-sig")
    dfl_w.to_csv(rep_dir / "dfl_factor_weights_history.csv", index=False, encoding="utf-8-sig")

    for name, df in preds.items():
        df.to_parquet(pred_dir / f"{name}.parquet", index=False)

    eval_start = pd.Timestamp(cfg["dates"]["eval_start"])
    dev_end = pd.Timestamp(cfg["dates"]["dev_end"])
    eval_days = trading_days[(trading_days >= eval_start) & (trading_days <= dev_end)]
    bars = build_quote_lookup(daily_bars, eval_days)

    overall_rows, yearly_rows, daily_frames = [], [], []
    for fee_bps in cfg["portfolio"]["fee_bps_list"]:
        for name in cfg["models"]["evaluated_strategies"]:
            df_pred = preds[name]
            nav_df = run_backtest(
                strategy_name=name,
                predictions=df_pred,
                bars=bars,
                eval_days=eval_days,
                fee_bps=int(fee_bps),
                initial_cash=float(cfg["portfolio"]["initial_cash"]),
                single_cap=float(cfg["portfolio"]["single_stock_cap"]),
                cash_reserve=float(cfg["portfolio"]["cash_reserve"]),
            )
            overall, yearly_df, daily_active = summarize_performance(
                name, int(fee_bps), nav_df, benchmarks, df_pred,
                initial_cash=float(cfg["portfolio"]["initial_cash"]),
            )
            overall_rows.append(overall)
            yearly_rows.append(yearly_df)
            daily_frames.append(daily_active.assign(strategy=name, fee_bps=int(fee_bps)))
            nav_df.to_parquet(bt_dir / f"{name}_{fee_bps}bp_nav.parquet", index=False)
            print(
                f"[bt {fee_bps:2d}bp] {name:18s} | Net {overall['net_return']:+.2%} | "
                f"ExH {overall['excess_h00905_pp']:+.2%} | TE {overall['te_h00905']:.2%} | "
                f"IR {overall['ir_h00905']:+.3f} | Turn {overall['weekly_two_way_turnover']:.2%}",
                flush=True,
            )

    pd.DataFrame(overall_rows).to_csv(rep_dir / "dev_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(yearly_rows, ignore_index=True).to_csv(rep_dir / "dev_yearly.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_parquet(bt_dir / "daily_active_returns.parquet", index=False)
    print(f"[exp] Done in {time.perf_counter() - t0:.1f}s total.", flush=True)


if __name__ == "__main__":
    main()
