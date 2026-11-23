"""Step 4: Compute Newey-West, PSR/DSR (N=3), CSCV PBO, and write reports/DEV_REPORT.md."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from csi500_alpha.config import load_config
from csi500_alpha.factors import FACTORS
from csi500_alpha.stats import (
    annualized_ir_from_returns,
    cscv_pbo,
    deflated_sharpe_ratio,
    newey_west_t_stat,
    probabilistic_sharpe_ratio,
)


def compute_signal_metrics(pred_dir: Path, strategies: list[str], nw_lag: int) -> pd.DataFrame:
    rows = []
    for name in strategies:
        df = pd.read_parquet(pred_dir / f"{name}.parquet")
        weekly = []
        for day, g in df.groupby("signal_date", sort=True):
            valid = g.dropna(subset=["risk_adjusted_target", "score"])
            if len(valid) < 100:
                continue
            ic = valid["score"].rank().corr(valid["risk_adjusted_target"].rank())
            top50 = set(valid.sort_values(["score", "code"], ascending=[False, True]).head(50)["code"])
            best50 = set(valid.sort_values(["risk_adjusted_target", "code"], ascending=[False, True]).head(50)["code"])
            weekly.append((pd.Timestamp(day), ic, len(top50 & best50) / 50.0))
        wdf = pd.DataFrame(weekly, columns=["signal_date", "rank_ic", "top50_hit"])
        for period, mask in [
            ("all", pd.Series(True, index=wdf.index)),
            ("2017-2022", wdf["signal_date"].dt.year <= 2022),
            ("2023-2025", wdf["signal_date"].dt.year >= 2023),
        ]:
            sub = wdf.loc[mask]
            mean_ic, t_nw, p_nw = newey_west_t_stat(sub["rank_ic"].to_numpy(), lag=nw_lag)
            sd = float(sub["rank_ic"].std(ddof=1))
            rows.append({
                "strategy": name,
                "period": period,
                "weeks": len(sub),
                "mean_rank_ic": mean_ic,
                "icir_weekly": mean_ic / sd if sd > 0 else np.nan,
                "ic_nw_tstat": t_nw,
                "ic_nw_pval": p_nw,
                "top50_hit": float(sub["top50_hit"].mean()),
            })
    return pd.DataFrame(rows)


def compute_portfolio_stats(
    daily_df: pd.DataFrame,
    strategies: list[str],
    nw_lag: int,
    n_blocks: int,
    max_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stat_rows, pbo_rows = [], []
    for fee_bps in sorted(daily_df["fee_bps"].unique()):
        sub = daily_df.loc[daily_df["fee_bps"] == fee_bps]
        for bench_col, bench_label in [("active_h00905", "H00905"), ("active_floatmv", "FloatMV")]:
            wide = sub.pivot(index="date", columns="strategy", values=bench_col)[strategies].dropna()
            mat = wide.to_numpy(dtype=np.float64)
            pbo, med_rank, n_splits = cscv_pbo(mat, n_blocks=n_blocks, max_splits=max_splits)
            pbo_rows.append({
                "fee_bps": fee_bps,
                "benchmark": bench_label,
                "num_strategies_N": len(strategies),
                "cscv_blocks": n_blocks,
                "splits_evaluated": n_splits,
                "pbo": pbo,
                "median_oos_rank_pct": med_rank,
            })
            for j, name in enumerate(strategies):
                arr = mat[:, j]
                mean_d, t_nw, p_nw = newey_west_t_stat(arr, lag=nw_lag)
                ir_ann = annualized_ir_from_returns(arr, 252)
                _, psr, sk, ex_kurt = probabilistic_sharpe_ratio(arr, 0.0, 252)
                sr0_ann, dsr = deflated_sharpe_ratio(mat, j, 252)
                stat_rows.append({
                    "fee_bps": fee_bps,
                    "benchmark": bench_label,
                    "strategy": name,
                    "ann_ir": ir_ann,
                    "daily_active_bps": mean_d * 10000.0,
                    "nw_tstat": t_nw,
                    "nw_pval": p_nw,
                    "psr": psr,
                    "dsr_N3": dsr,
                    "dsr_hurdle_ir": sr0_ann,
                    "skewness": sk,
                    "excess_kurtosis": ex_kurt,
                })
    return pd.DataFrame(stat_rows), pd.DataFrame(pbo_rows)


def write_markdown_report(
    rep_dir: Path,
    comp_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    pbo_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    ridge_w: pd.DataFrame,
    dfl_w: pd.DataFrame,
) -> None:
    pit_members = bool(audit_df["pit_members"].iloc[0]) if "pit_members" in audit_df.columns else False
    lines = [
        "# 中证 500 指数增强开发期实验与统计审计报告（2017–2025）",
        "",
        f"> 严格依据 [docs/PRE_ANALYSIS_PLAN.md](../docs/PRE_ANALYSIS_PLAN.md) 执行；点时历史成分：`pit_members = {pit_members}`；未读取 2026 年数据。",
        "",
        "## 1. 三模型阶梯对比（2017–2025 全期，官方基准 H00905）",
        "",
    ]
    for fee in sorted(comp_df["fee_bps"].unique()):
        sub = comp_df.loc[comp_df["fee_bps"] == fee]
        lines.append(f"### 单边费率 {fee} bp")
        lines.append("")
        lines.append("| 策略 | 累计净收益 | vs H00905 超额 | vs FloatMV 超额 | 跟踪误差 TE | IR (H00905) | IR (FloatMV) | 最大相对回撤 | 周双边换手 | 最大行业主动暴露 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sub.itertuples(index=False):
            lines.append(
                f"| {r.strategy} | {r.net_return:+.2%} | {r.excess_h00905_pp:+.2%} | {r.excess_floatmv_pp:+.2%} | "
                f"{r.te_h00905:.2%} | {r.ir_h00905:+.3f} | {r.ir_floatmv:+.3f} | {r.max_rel_dd_h00905:.2%} | "
                f"{r.weekly_two_way_turnover:.2%} | {r.max_ind_active_exp:.2%} |"
            )
        lines.append("")

    lines.extend(["## 2. 分段与抗单年稳健性（10 bp）", ""])
    y10 = yearly_df.loc[(yearly_df["fee_bps"] == 10) & yearly_df["period"].isin(["2017-2022", "2023-2025", "ex_2020", "all"])]
    lines.append("| 策略 | 区间 | 组合收益 | H00905 | 超额 (H00905) | IR (H00905) | 超额 (FloatMV) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in y10.itertuples(index=False):
        lines.append(
            f"| {r.strategy} | {r.period} | {r.net_return:+.2%} | {r.h00905_return:+.2%} | "
            f"{r.excess_h00905_pp:+.2%} | {r.ir_h00905:+.3f} | {r.excess_floatmv_pp:+.2%} |"
        )
    lines.append("")

    lines.extend(["## 3. 白盒 10 因子平均权重演化对比", ""])
    lines.append("| 因子 | 等权基准 | 滚动 Ridge 均值 | 白盒 DFL 均值 |")
    lines.append("|---|---:|---:|---:|")
    for f in FACTORS:
        lines.append(f"| {f} | 10.00% | {ridge_w[f].mean():.2%} | {dfl_w[f].mean():.2%} |")
    lines.append("")

    lines.extend(["## 4. 信号层截面排序能力与组合统计显著性（N = 3）", ""])
    lines.append("| 策略 | 区间 | 周数 | 平均 Rank IC | 周度 ICIR | NW t-stat | Top50 命中率 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in sig_df.itertuples(index=False):
        lines.append(f"| {r.strategy} | {r.period} | {r.weeks} | {r.mean_rank_ic:+.4f} | {r.icir_weekly:+.3f} | {r.ic_nw_tstat:+.2f} | {r.top50_hit:.2%} |")
    lines.append("")

    s_h = stat_df.loc[stat_df["benchmark"] == "H00905"]
    lines.append("| 费率 | 策略 | 年化 IR | NW t-stat | NW p-value | PSR | DSR (N=3) | DSR 门槛 IR |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in s_h.itertuples(index=False):
        lines.append(f"| {r.fee_bps} bp | {r.strategy} | {r.ann_ir:+.3f} | {r.nw_tstat:+.2f} | {r.nw_pval:.4f} | {r.psr:.2%} | {r.dsr_N3:.2%} | {r.dsr_hurdle_ir:.3f} |")
    lines.append("")

    lines.append("### CSCV 回测过拟合概率（PBO）")
    lines.append("")
    lines.append("| 费率 | 基准 | 策略数 N | PBO | 样本外中位排名分位 |")
    lines.append("|---:|---|---:|---:|---:|")
    for r in pbo_df.itertuples(index=False):
        lines.append(f"| {r.fee_bps} bp | {r.benchmark} | {r.num_strategies_N} | {r.pbo:.2%} | {r.median_oos_rank_pct:.2%} |")
    lines.append("")

    (rep_dir / "DEV_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "experiment.toml")
    proc_dir = ROOT / "data" / "processed"
    rep_dir = ROOT / "reports"
    strategies = list(cfg["models"]["evaluated_strategies"])
    nw_lag = int(cfg["stats"]["newey_west_lag"])
    n_blocks = int(cfg["stats"]["cscv_blocks"])
    max_splits = int(cfg["stats"]["cscv_max_splits"])

    comp_df = pd.read_csv(rep_dir / "dev_comparison.csv")
    yearly_df = pd.read_csv(rep_dir / "dev_yearly.csv")
    audit_df = pd.read_csv(rep_dir / "factor_data_audit.csv")
    ridge_w = pd.read_csv(rep_dir / "ridge_factor_weights_history.csv")
    dfl_w = pd.read_csv(rep_dir / "dfl_factor_weights_history.csv")
    daily_df = pd.read_parquet(proc_dir / "backtests" / "daily_active_returns.parquet")

    sig_df = compute_signal_metrics(proc_dir / "predictions", strategies, nw_lag)
    stat_df, pbo_df = compute_portfolio_stats(daily_df, strategies, nw_lag, n_blocks, max_splits)

    sig_df.to_csv(rep_dir / "dev_signal_metrics.csv", index=False, encoding="utf-8-sig")
    stat_df.to_csv(rep_dir / "dev_stats_audit.csv", index=False, encoding="utf-8-sig")
    pbo_df.to_csv(rep_dir / "dev_pbo_audit.csv", index=False, encoding="utf-8-sig")
    write_markdown_report(rep_dir, comp_df, yearly_df, sig_df, stat_df, pbo_df, audit_df, ridge_w, dfl_w)

    print(f"[stats] Wrote DEV_REPORT.md and CSV audits to {rep_dir}")
    print("\n=== Portfolio Statistical Audit (vs H00905) ===")
    print(stat_df.loc[stat_df["benchmark"] == "H00905"].round(4).to_string(index=False))
    print("\n=== CSCV PBO ===")
    print(pbo_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
