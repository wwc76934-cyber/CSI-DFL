"""Step 4: Run v2 paired statistical tests, block bootstrap CIs, DSR (N=3 & N=10), turnover reconciliation, and write honest DEV_REPORT.md."""

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
    paired_return_test,
    probabilistic_sharpe_ratio,
)


def compute_signal_metrics(pred_dir: Path, strategies: list[str], nw_lag: int) -> pd.DataFrame:
    rows = []
    for name in strategies:
        if name == "B0_FloatMV_Anchor":
            continue
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
    dsr_cons_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_strats = [s for s in strategies if s != "B0_FloatMV_Anchor"]
    stat_rows, pbo_rows = [], []
    for fee_mode in daily_df["fee_mode"].unique():
        sub = daily_df.loc[daily_df["fee_mode"] == fee_mode]
        for bench_col, bench_label in [("active_h00905", "H00905"), ("active_floatmv", "B0_FloatMV")]:
            wide = sub.pivot(index="date", columns="strategy", values=bench_col)[active_strats].dropna()
            mat = wide.to_numpy(dtype=np.float64)
            pbo, med_rank, n_splits = cscv_pbo(mat, n_blocks=n_blocks, max_splits=max_splits)
            pbo_rows.append({
                "fee_mode": fee_mode,
                "benchmark": bench_label,
                "num_strategies": len(active_strats),
                "cscv_blocks": n_blocks,
                "splits_evaluated": n_splits,
                "pbo": pbo,
                "median_oos_rank_pct": med_rank,
            })
            for j, name in enumerate(active_strats):
                arr = mat[:, j]
                mean_d, t_nw, p_nw = newey_west_t_stat(arr, lag=nw_lag)
                ir_ann = annualized_ir_from_returns(arr, 252)
                _, psr, sk, ex_kurt = probabilistic_sharpe_ratio(arr, 0.0, 252)
                sr0_3, dsr_3 = deflated_sharpe_ratio(mat, j, 252, n_trials_override=3)
                sr0_10, dsr_10 = deflated_sharpe_ratio(mat, j, 252, n_trials_override=dsr_cons_n)
                stat_rows.append({
                    "fee_mode": fee_mode,
                    "benchmark": bench_label,
                    "strategy": name,
                    "ann_ir": ir_ann,
                    "daily_active_bps": mean_d * 10000.0,
                    "nw_tstat": t_nw,
                    "nw_pval": p_nw,
                    "psr": psr,
                    "dsr_N3": dsr_3,
                    "dsr_N10": dsr_10,
                    "dsr_hurdle_N3": sr0_3,
                    "dsr_hurdle_N10": sr0_10,
                    "skewness": sk,
                    "excess_kurtosis": ex_kurt,
                })
    return pd.DataFrame(stat_rows), pd.DataFrame(pbo_rows)


def compute_paired_tests(
    daily_df: pd.DataFrame,
    nw_lag: int,
    n_boot: int,
    block_days: int,
) -> pd.DataFrame:
    pairs = [
        ("S2_WhiteboxDFL", "S1_RollingRidge"),
        ("S2_WhiteboxDFL", "S0_EqualWeight"),
        ("S2_WhiteboxDFL", "S0_LowTurn_Band"),
        ("S0_LowTurn_Band", "S0_EqualWeight"),
        ("S2_WhiteboxDFL", "B0_FloatMV_Anchor"),
        ("S0_EqualWeight", "B0_FloatMV_Anchor"),
        ("S0_LowTurn_Band", "B0_FloatMV_Anchor"),
    ]
    rows = []
    for fee_mode in daily_df["fee_mode"].unique():
        sub = daily_df.loc[daily_df["fee_mode"] == fee_mode]
        ret_wide = sub.pivot(index="date", columns="strategy", values="port_ret").dropna()
        bm_series = sub.loc[sub["strategy"] == "B0_FloatMV_Anchor"].set_index("date")["port_ret"].reindex(ret_wide.index)
        for a_name, b_name in pairs:
            if a_name not in ret_wide.columns or b_name not in ret_wide.columns:
                continue
            res = paired_return_test(
                ret_wide[a_name].to_numpy(),
                ret_wide[b_name].to_numpy(),
                bench_ret=bm_series.to_numpy(),
                lag=nw_lag,
                n_boot=n_boot,
                block_size=block_days,
                seed=42,
            )
            rows.append({"fee_mode": fee_mode, "pair": f"{a_name} - {b_name}", "strat_a": a_name, "strat_b": b_name, **res})
    return pd.DataFrame(rows)


def write_markdown_report(
    rep_dir: Path,
    comp_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    pbo_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    ridge_w: pd.DataFrame,
    dfl_w: pd.DataFrame,
) -> None:
    q_fb = float(audit_df["quarterly_fallback_rate_2017_2025"].iloc[0])
    open_diff = int(audit_df["open_vs_close_diff_rows"].iloc[0])

    lines = [
        "# 中证 500 指数增强 v2 严谨审计与实证报告（2017–2025）",
        "",
        "> **定位声明**：本项目是方法严谨、结论诚实的中证 500 指数增强研究原型，严格遵守 [docs/PRE_ANALYSIS_PLAN.md](../docs/PRE_ANALYSIS_PLAN.md)（v2），未读取 `2026+` 保留期数据。",
        f"> **数据透明度声明**：周度 PIT 成分 `562` 期（`1,323/1,326` 覆盖，3 只并购更名股剔除）；申万一级行业**严格 PIT 零未来回填**；开盘与收盘有效性标志差异 **`{open_diff}` 行**；东财季报 `ak.stock_yjbb_em` 因刷新“最新公告日期”导致法定截止日回退比例达 **`{q_fb:.2%}`**（属法定截止日对齐重述报表，非首次披露 PIT），本文通过第 6 节 **7 因子纯 PIT 消融实验**量化其影响。",
        "",
        "## 1. 同引擎全归因对比（含毛收益 0 bp、真实 A 股成本与 20 bp 保守基准）",
        "",
    ]

    mode_titles = [
        ("fixed_0bp", "毛收益归因（0 bp，无交易成本）"),
        ("ashare_realistic", "A 股真实分段成本（买 10 bp；卖 2023-08-28 前 20 bp / 后 15 bp）"),
        ("fixed_20bp", "保守基准情形（双边各 20 bp）"),
        ("fixed_10bp", "乐观机构参考（双边各 10 bp）"),
    ]
    for fm, title in mode_titles:
        sub = comp_df.loc[comp_df["fee_mode"] == fm]
        if sub.empty:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| 策略 | 累计净收益 | 年化 CAGR | vs H00905 | vs 同引擎 B0 锚 | IR (vs B0) | 周双边换手 | 实际最大行业暴露 | 实盘成交率 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sub.itertuples(index=False):
            lines.append(
                f"| {r.strategy} | {r.net_return:+.2%} | {r.cagr:+.2%} | {r.excess_h00905_pp:+.2%} | "
                f"{r.excess_floatmv_pp:+.2%} | {r.ir_floatmv:+.3f} | {r.weekly_two_way_turnover:.2%} | "
                f"{r.realized_max_ind_active_exp:.2%} | {r.order_fill_rate:.2%} |"
            )
        lines.append("")

    lines.extend(["## 2. 关键模型两两配对检验（Newey-West t 检验 + 区块自助法 95% CI）", ""])
    lines.append("> 直接检验同日净收益差 `A - B`，判断模型增量是否真正统计显著。")
    lines.append("")
    lines.append("| 成本口径 | 配对对比 (A - B) | 年化收益差 | 95% 区块自助法 CI | IR 差 (vs B0) | NW t-stat | NW p-value |")
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for r in paired_df.loc[paired_df["fee_mode"].isin(["ashare_realistic", "fixed_20bp", "fixed_10bp"])].itertuples(index=False):
        lines.append(
            f"| {r.fee_mode} | `{r.pair}` | {r.ann_diff:+.2%} | `[{r.ann_diff_ci_lo:+.2%}, {r.ann_diff_ci_hi:+.2%}]` | "
            f"{r.ir_gap:+.3f} | {r.nw_tstat:+.2f} | {r.nw_pval:.4f} |"
        )
    lines.append("")

    lines.extend(["## 3. 单策略相对同引擎 B0 锚与 H00905 显著性（含 DSR N=3 与 N=10 敏感性）", ""])
    lines.append("| 成本口径 | 基准 | 策略 | 年化 IR | NW t-stat | NW p-value | PSR | DSR (N=3) | DSR (N=10) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in stat_df.loc[stat_df["fee_mode"].isin(["ashare_realistic", "fixed_20bp"])].itertuples(index=False):
        lines.append(
            f"| {r.fee_mode} | {r.benchmark} | {r.strategy} | {r.ann_ir:+.3f} | {r.nw_tstat:+.2f} | {r.nw_pval:.4f} | "
            f"{r.psr:.2%} | {r.dsr_N3:.2%} | {r.dsr_N10:.2%} |"
        )
    lines.append("")

    lines.extend(["## 4. 分段稳健性与衰减检验（A 股真实成本 `ashare_realistic`）", ""])
    yr = yearly_df.loc[(yearly_df["fee_mode"] == "ashare_realistic") & yearly_df["period"].isin(["2017-2022", "2023-2025", "ex_2020", "all"])]
    lines.append("| 策略 | 区间 | 组合收益 | vs H00905 | vs 同引擎 B0 锚 | IR (vs B0) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in yr.itertuples(index=False):
        lines.append(
            f"| {r.strategy} | {r.period} | {r.net_return:+.2%} | {r.excess_h00905_pp:+.2%} | {r.excess_floatmv_pp:+.2%} | {r.ir_floatmv:+.3f} |"
        )
    lines.append("")

    lines.extend(["## 5. 训练换手对账、5 种子稳定性与资金容量压力测试", ""])
    if not rec_df.empty:
        corr_dr = float(rec_df["full_drifted_turn"].corr(rec_df["executed_turn"]))
        corr_px = float(rec_df["proxy_active_delta_turn"].corr(rec_df["executed_turn"]))
        lines.append(
            f"- **DFL 换手对账（周度均值）**：旧代理主动偏离变化 `{rec_df['proxy_active_delta_turn'].mean():.2%}`（与实盘相关 `{corr_px:.3f}`） "
            f"→ 新全组合漂移换手 `{rec_df['full_drifted_turn'].mean():.2%}`（与实盘相关 **`{corr_dr:.3f}`**） vs 账户实际成交换手 `{rec_df['executed_turn'].mean():.2%}`。"
        )
    if not seed_df.empty:
        for fm in ["ashare_realistic", "fixed_20bp"]:
            s_sub = seed_df.loc[seed_df["fee_mode"] == fm]
            lines.append(
                f"- **5 随机种子稳定性（`{fm}`）**：`S2_WhiteboxDFL` 净收益范围 `{s_sub['net_return'].min():+.2%} ~ {s_sub['net_return'].max():+.2%}`，"
                f"vs B0 IR 范围 `{s_sub['ir_floatmv'].min():+.3f} ~ {s_sub['ir_floatmv'].max():+.3f}`（标准差 `{s_sub['ir_floatmv'].std():.4f}`）。"
            )
    lines.append("")
    lines.append("### 资金规模容量压力测试（`ashare_realistic` + 前一日成交额 5% 参与率上限）")
    lines.append("")
    lines.append("| 初始资金 | 策略 | 累计净收益 | vs 同引擎 B0 锚 | IR (vs B0) | 均值现金占比 | 订单成交率 | 流动性截断单数 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in cap_df.itertuples(index=False):
        lines.append(
            f"| {int(r.capital_rmb/1e6)}M | {r.strategy} | {r.net_return:+.2%} | {r.excess_floatmv_pp:+.2%} | "
            f"{r.ir_floatmv:+.3f} | {r.avg_cash_weight:.2%} | {r.order_fill_rate:.2%} | {int(r.cap_clipped_orders)} |"
        )
    lines.append("")

    lines.extend(["## 6. 7 因子纯 PIT 消融对照（剔除 3 个东财重述季报因子）", ""])
    lines.append("> 仅保留 7 个严格时点可核验的 BaoStock 日度估值（`EP_TTM`, `BP_MRQ`）与量价趋势因子，验证策略在零季报重述依赖下的表现。")
    lines.append("")
    lines.append("| 成本口径 | 策略（7 因子纯 PIT） | 累计净收益 | vs H00905 | vs 同引擎 B0 锚 | IR (vs B0) | 周双边换手 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in abl_df.loc[abl_df["fee_mode"].isin(["ashare_realistic", "fixed_20bp"])].itertuples(index=False):
        lines.append(
            f"| {r.fee_mode} | {r.strategy} | {r.net_return:+.2%} | {r.excess_h00905_pp:+.2%} | "
            f"{r.excess_floatmv_pp:+.2%} | {r.ir_floatmv:+.3f} | {r.weekly_two_way_turnover:.2%} |"
        )
    lines.append("")

    lines.extend(["## 7. 白盒 10 因子平均权重与信号层 Rank IC", ""])
    lines.append("| 因子 | 等权先验 | S1 滚动 Ridge 均值 | S2 白盒 DFL 均值 |")
    lines.append("|---|---:|---:|---:|")
    for f in FACTORS:
        lines.append(f"| {f} | 10.00% | {ridge_w[f].mean():.2%} | {dfl_w[f].mean():.2%} |")
    lines.append("")

    lines.append("| 策略 | 区间 | 周数 | 平均 Rank IC | 周度 ICIR | NW t-stat | Top50 命中率 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in sig_df.itertuples(index=False):
        lines.append(f"| {r.strategy} | {r.period} | {r.weeks} | {r.mean_rank_ic:+.4f} | {r.icir_weekly:+.3f} | {r.ic_nw_tstat:+.2f} | {r.top50_hit:.2%} |")
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
    dsr_cons_n = int(cfg["stats"]["dsr_conservative_n"])
    n_boot = int(cfg["stats"]["bootstrap_reps"])
    block_days = int(cfg["stats"]["bootstrap_block_days"])

    comp_df = pd.read_csv(rep_dir / "dev_comparison.csv")
    yearly_df = pd.read_csv(rep_dir / "dev_yearly.csv")
    audit_df = pd.read_csv(rep_dir / "factor_data_audit.csv")
    ridge_w = pd.read_csv(rep_dir / "ridge_factor_weights_history.csv")
    dfl_w = pd.read_csv(rep_dir / "dfl_factor_weights_history.csv")
    cap_df = pd.read_csv(rep_dir / "capacity_stress_test.csv")
    seed_df = pd.read_csv(rep_dir / "dfl_multiseed_stability.csv")
    abl_df = pd.read_csv(rep_dir / "pure_pit_7f_ablation.csv")
    rec_df = pd.read_csv(rep_dir / "dfl_turnover_reconcile.csv")
    daily_df = pd.read_parquet(proc_dir / "backtests" / "daily_active_returns.parquet")

    sig_df = compute_signal_metrics(proc_dir / "predictions", strategies, nw_lag)
    stat_df, pbo_df = compute_portfolio_stats(daily_df, strategies, nw_lag, n_blocks, max_splits, dsr_cons_n)
    paired_df = compute_paired_tests(daily_df, nw_lag, n_boot, block_days)

    sig_df.to_csv(rep_dir / "dev_signal_metrics.csv", index=False, encoding="utf-8-sig")
    stat_df.to_csv(rep_dir / "dev_stats_audit.csv", index=False, encoding="utf-8-sig")
    pbo_df.to_csv(rep_dir / "dev_pbo_audit.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(rep_dir / "dev_paired_tests.csv", index=False, encoding="utf-8-sig")

    write_markdown_report(
        rep_dir, comp_df, yearly_df, sig_df, stat_df, paired_df, pbo_df,
        audit_df, cap_df, seed_df, abl_df, rec_df, ridge_w, dfl_w,
    )

    print(f"[stats] Wrote v2 DEV_REPORT.md and CSV audits to {rep_dir}")
    print("\n=== Paired Net Return Tests (ashare_realistic & fixed_20bp) ===")
    print(
        paired_df.loc[paired_df["fee_mode"].isin(["ashare_realistic", "fixed_20bp"]), [
            "fee_mode", "pair", "ann_diff", "ann_diff_ci_lo", "ann_diff_ci_hi", "ir_gap", "nw_tstat", "nw_pval"
        ]].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
