"""Step 5: Generate high-resolution defense & README figures in reports/figures/."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from csi500_alpha.factors import FACTORS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160


COLORS = {
    "S0_EqualWeight": "#4C72B0",
    "S1_RollingRidge": "#DD8452",
    "S2_WhiteboxDFL": "#C44E52",
}
LABELS = {
    "S0_EqualWeight": "S0 等权基准 (1/10)",
    "S1_RollingRidge": "S1 两阶段滚动 Ridge",
    "S2_WhiteboxDFL": "S2 10参数白盒 DFL (本文)",
}


def plot_excess_curves(bt_dir: Path, out_dir: Path) -> Path:
    daily = pd.read_parquet(bt_dir / "daily_active_returns.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for ax, fee in zip(axes, [10, 20]):
        sub = daily.loc[daily["fee_bps"] == fee]
        for strat in ["S0_EqualWeight", "S1_RollingRidge", "S2_WhiteboxDFL"]:
            g = sub.loc[sub["strategy"] == strat].sort_values("date")
            rel_nav = ((1.0 + g["port_ret"]) / (1.0 + g["h00905_ret"])).cumprod()
            lw = 2.4 if strat == "S2_WhiteboxDFL" else 1.6
            ls = "-" if strat != "S1_RollingRidge" else "--"
            ax.plot(g["date"], (rel_nav - 1.0) * 100.0, label=LABELS[strat], color=COLORS[strat], linewidth=lw, linestyle=ls)

        ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
        ax.axvspan(pd.Timestamp("2023-01-01"), g["date"].max(), color="#F0F4F8", alpha=0.8, label="2023–2025 量化收紧期")
        ax.set_title(f"单边费率 {fee} bp：相对中证500全收益 (H00905) 累计超额净值", fontsize=11, fontweight="bold")
        ax.set_ylabel("几何累计超额收益 (%)" if fee == 10 else "")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.8, frameon=True)

    fig.tight_layout()
    target = out_dir / "nav_and_excess_10bp_20bp.png"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_yearly_excess(rep_dir: Path, out_dir: Path) -> Path:
    yearly = pd.read_csv(rep_dir / "dev_yearly.csv")
    years = [str(y) for y in range(2017, 2026)]
    sub = yearly.loc[(yearly["fee_bps"] == 10) & (yearly["period"].isin(years))].copy()

    x = np.arange(len(years))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12.0, 4.6))

    for idx, strat in enumerate(["S0_EqualWeight", "S1_RollingRidge", "S2_WhiteboxDFL"]):
        s_df = sub.loc[sub["strategy"] == strat].set_index("period").reindex(years)
        vals = s_df["excess_h00905_pp"].to_numpy() * 100.0
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, label=LABELS[strat], color=COLORS[strat], alpha=0.9)
        if strat == "S2_WhiteboxDFL":
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + (0.25 if v >= 0 else -0.8),
                    f"{v:+.1f}%",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.8,
                    color="#8B1A1A",
                    fontweight="bold",
                )

    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("年度超额收益 vs H00905 (百分点 %)")
    ax.set_title("2017–2025 分年度超额收益对比（单边 10 bp）：白盒 DFL 连续 9 年正超额", fontsize=11.5, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9.2)
    fig.tight_layout()

    target = out_dir / "yearly_excess_comparison.png"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_factor_weights(rep_dir: Path, out_dir: Path) -> Path:
    rw = pd.read_csv(rep_dir / "ridge_factor_weights_history.csv")
    dw = pd.read_csv(rep_dir / "dfl_factor_weights_history.csv")
    dw["signal_date"] = pd.to_datetime(dw["signal_date"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0))

    y = np.arange(len(FACTORS))
    h = 0.34
    r_mean = rw[FACTORS].mean().to_numpy() * 100.0
    d_mean = dw[FACTORS].mean().to_numpy() * 100.0

    ax1.barh(y - h / 2, r_mean, height=h, color=COLORS["S1_RollingRidge"], label="S1 滚动 Ridge 平均权重", alpha=0.88)
    ax1.barh(y + h / 2, d_mean, height=h, color=COLORS["S2_WhiteboxDFL"], label="S2 白盒 DFL 平均权重", alpha=0.92)
    ax1.axvline(10.0, color="#4C72B0", linestyle="--", linewidth=1.3, label="等权先验 (10.0%)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(FACTORS, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 31.5)
    ax1.set_xlabel("平均因子权重 (%)")
    ax1.set_title("10 因子平均权重对比：DFL 主动抑制高换手短量价", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.6, frameon=True)

    highlight = [
        ("BP_MRQ", "#2E7D32", "BP_MRQ (估值安全边际)"),
        ("CFO_YIELD", "#1565C0", "CFO_YIELD (现金流收益率)"),
        ("INTRADAY_REV_20", "#EF6C00", "INTRADAY_REV_20 (日内反转)"),
        ("CORR_PV_10", "#6A1B9A", "CORR_PV_10 (10日量价相关-高换手)"),
    ]
    for col, color, label in highlight:
        ax2.plot(dw["signal_date"], dw[col] * 100.0, label=label, color=color, linewidth=1.8)
    ax2.axhline(10.0, color="#666666", linestyle=":", linewidth=1.0, label="等权先验 (10%)")
    ax2.set_ylim(1.5, 25.0)
    ax2.set_ylabel("白盒 DFL 因子权重 (%)")
    ax2.set_title("S2 白盒 DFL 关键因子权重时序演化 (2017–2025)", fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", ncol=2, fontsize=7.9, frameon=True)

    fig.tight_layout()
    target = out_dir / "factor_weights_ridge_vs_dfl.png"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> None:
    bt_dir = ROOT / "data" / "processed" / "backtests"
    rep_dir = ROOT / "reports"
    out_dir = rep_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_excess_curves(bt_dir, out_dir)
    p2 = plot_yearly_excess(rep_dir, out_dir)
    p3 = plot_factor_weights(rep_dir, out_dir)
    print(f"[plot] Saved 3 figures:\n  - {p1}\n  - {p2}\n  - {p3}")


if __name__ == "__main__":
    main()
