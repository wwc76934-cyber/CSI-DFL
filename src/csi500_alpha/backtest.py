"""CSI 500 open-to-open backtest engine with lot sizing, limit-up/down, and hfq total-return continuity."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd


CENT = D("0.01")
GEM_CHANGE = pd.Timestamp("2020-08-24")


@dataclass(frozen=True)
class BarQuote:
    raw_open: float
    raw_close: float
    raw_preclose: float
    hfq_open: float
    hfq_close: float
    bar_ok: bool


def money(value: D) -> D:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def stock_rules(code: str, day: pd.Timestamp) -> tuple[int, int, D]:
    if code.startswith(("sh.688", "sh.689")):
        return 200, 1, D("0.20")
    if code.startswith(("sz.300", "sz.301", "sz.302")) and day >= GEM_CHANGE:
        return 100, 100, D("0.20")
    return 100, 100, D("0.10")


def build_quote_lookup(bars_df: pd.DataFrame, eval_days: pd.DatetimeIndex) -> dict[pd.Timestamp, dict[str, BarQuote]]:
    sub = bars_df.loc[
        bars_df["date"].isin(eval_days),
        ["date", "code", "raw_open", "raw_close", "raw_preclose", "hfq_open", "hfq_close", "bar_ok"],
    ]
    lookup: dict[pd.Timestamp, dict[str, BarQuote]] = {}
    for day, grp in sub.groupby("date", sort=True):
        day_ts = pd.Timestamp(day)
        lookup[day_ts] = {
            row.code: BarQuote(
                float(row.raw_open) if pd.notna(row.raw_open) else np.nan,
                float(row.raw_close) if pd.notna(row.raw_close) else np.nan,
                float(row.raw_preclose) if pd.notna(row.raw_preclose) else np.nan,
                float(row.hfq_open) if pd.notna(row.hfq_open) else np.nan,
                float(row.hfq_close) if pd.notna(row.hfq_close) else np.nan,
                bool(row.bar_ok) if pd.notna(row.bar_ok) else False,
            )
            for row in grp.itertuples(index=False)
        }
    return lookup


def run_backtest(
    strategy_name: str,
    predictions: pd.DataFrame,
    bars: dict[pd.Timestamp, dict[str, BarQuote]],
    eval_days: pd.DatetimeIndex,
    fee_bps: int,
    initial_cash: float = 1_000_000.0,
    single_cap: float = 0.04,
    cash_reserve: float = 0.003,
) -> pd.DataFrame:
    """Simulate weekly open-to-open portfolio execution using raw lot sizing + hfq total-return continuity."""
    fee_rate = fee_bps / 10000.0
    cash = float(initial_cash)
    hfq_shares: dict[str, float] = {}
    last_hfq_mark: dict[str, float] = {}

    by_entry = {pd.Timestamp(d): g for d, g in predictions.groupby("entry_date", sort=True)}
    first_day = min(by_entry.keys())
    days = [pd.Timestamp(d) for d in eval_days if pd.Timestamp(d) >= first_day]

    nav_rows = []
    for day in days:
        quotes = bars.get(day, {})
        day_fee = 0.0
        gross_traded = 0.0
        opening_equity = np.nan

        def hfq_mark(code: str, field: str) -> float:
            q = quotes.get(code)
            if q is not None:
                v = getattr(q, field)
                if np.isfinite(v) and v > 0:
                    return float(v)
            return last_hfq_mark[code]

        if day in by_entry:
            grp = by_entry[day]
            open_stock_val = sum(qty * hfq_mark(c, "hfq_open") for c, qty in hfq_shares.items())
            opening_equity = cash + open_stock_val

            target_w = dict(zip(grp["code"], grp["target_weight"].astype(float)))
            codes_all = set(hfq_shares) | {c for c, w in target_w.items() if w > 1e-10}

            plans = []
            for code in codes_all:
                q = quotes.get(code)
                old_hfq = hfq_shares.get(code, 0.0)
                w_tgt = target_w.get(code, 0.0)

                if q is None or not q.bar_ok or not np.isfinite(q.raw_open) or q.raw_open <= 0 or not np.isfinite(q.hfq_open) or q.hfq_open <= 0:
                    continue

                old_val = old_hfq * float(q.hfq_open)
                old_raw = int(round(old_val / float(q.raw_open))) if old_hfq > 0 else 0
                minimum, step, limit_rate = stock_rules(code, day)

                if w_tgt <= 1e-10:
                    desired_raw = 0
                else:
                    budget = opening_equity * (1.0 - cash_reserve) * w_tgt
                    sizing_cost = float(q.raw_open) * (1.0 + fee_rate if day == days[0] else 1.0)
                    desired_raw = max(0, int(budget // sizing_cost))

                change_raw = desired_raw - old_raw
                if change_raw > 0:
                    change_raw = (change_raw // step) * step
                    if change_raw < minimum:
                        change_raw = 0
                elif change_raw < 0 and desired_raw > 0:
                    change_raw = -((-change_raw // step) * step)
                    if -change_raw < minimum:
                        change_raw = 0

                if change_raw != 0:
                    plans.append((code, q, old_hfq, old_raw, desired_raw, change_raw, minimum, step, limit_rate))

            # Execute sells before buys to free cash
            plans.sort(key=lambda p: (0 if p[5] < 0 else 1, -target_w.get(p[0], 0.0), p[0]))
            for code, q, old_hfq, old_raw, desired_raw, req_raw, minimum, step, limit_rate in plans:
                fill_raw = 0
                pre = q.raw_preclose
                if np.isfinite(pre) and pre > 0:
                    up_lim = float(money(D(str(pre)) * (D("1") + limit_rate)))
                    dn_lim = float(money(D(str(pre)) * (D("1") - limit_rate)))
                    if req_raw > 0 and q.raw_open >= up_lim:
                        fill_raw = 0
                    elif req_raw < 0 and q.raw_open <= dn_lim:
                        fill_raw = 0
                    else:
                        fill_raw = req_raw
                        if req_raw > 0:
                            affordable = int(cash // (q.raw_open * step * (1.0 + fee_rate))) * step
                            fill_raw = min(affordable, req_raw)
                            if fill_raw < minimum:
                                fill_raw = 0

                if fill_raw == 0:
                    continue

                if desired_raw == 0 and fill_raw == -old_raw:
                    # Full exit: sell exact hfq value held to avoid rounding dust
                    gross = old_hfq * float(q.hfq_open)
                    fee = abs(gross) * fee_rate
                    cash += gross - fee
                    gross_traded += abs(gross)
                    day_fee += fee
                    hfq_shares.pop(code, None)
                else:
                    gross = fill_raw * float(q.raw_open)
                    fee = abs(gross) * fee_rate
                    if gross > 0 and cash < gross + fee - 1e-6:
                        continue
                    cash -= gross + fee
                    gross_traded += abs(gross)
                    day_fee += fee
                    new_hfq = max(0.0, old_hfq + gross / float(q.hfq_open))
                    if new_hfq > 1e-9:
                        hfq_shares[code] = new_hfq
                    else:
                        hfq_shares.pop(code, None)

        stock_val = 0.0
        for code, qty in hfq_shares.items():
            px = hfq_mark(code, "hfq_close")
            last_hfq_mark[code] = px
            stock_val += qty * px

        equity = cash + stock_val
        nav_rows.append((strategy_name, day, cash, stock_val, equity, day_fee, gross_traded, opening_equity))

    return pd.DataFrame(
        nav_rows,
        columns=["strategy", "date", "cash", "stock_value", "equity", "fee", "gross_traded", "opening_equity"],
    )


def summarize_performance(
    strategy_name: str,
    fee_bps: int,
    nav_df: pd.DataFrame,
    benchmarks_df: pd.DataFrame,
    predictions: pd.DataFrame,
    initial_cash: float = 1_000_000.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    d = nav_df.set_index("date").sort_index()
    b = benchmarks_df.set_index("date").sort_index().reindex(d.index)
    if b[["h00905", "float_mv_nav"]].isna().any().any():
        raise ValueError("Missing benchmark dates for NAV comparison")

    port_r = d["equity"].div(d["equity"].shift(fill_value=initial_cash)) - 1.0
    h_r = b["h00905_nav"].div(b["h00905_nav"].shift(fill_value=1.0)) - 1.0
    f_r = b["float_mv_nav"].div(b["float_mv_nav"].shift(fill_value=1.0)) - 1.0

    daily_active = pd.DataFrame({
        "date": d.index,
        "port_ret": port_r.to_numpy(),
        "h00905_ret": h_r.to_numpy(),
        "floatmv_ret": f_r.to_numpy(),
        "active_h00905": (port_r - h_r).to_numpy(),
        "active_floatmv": (port_r - f_r).to_numpy(),
    })

    def slice_metrics(mask: pd.Series, label: str) -> dict[str, Any]:
        pr, hr, fr = port_r.loc[mask], h_r.loc[mask], f_r.loc[mask]
        ah, af = pr - hr, pr - fr
        net = float((1.0 + pr).prod() - 1.0)
        h_cum = float((1.0 + hr).prod() - 1.0)
        f_cum = float((1.0 + fr).prod() - 1.0)
        nav_curve = np.r_[1.0, (1.0 + pr).cumprod().to_numpy()]
        rel_h = np.r_[1.0, ((1.0 + pr) / (1.0 + hr)).cumprod().to_numpy()]
        rel_f = np.r_[1.0, ((1.0 + pr) / (1.0 + fr)).cumprod().to_numpy()]
        te_h = float(ah.std(ddof=1) * np.sqrt(252)) if ah.std(ddof=1) > 0 else 0.0
        te_f = float(af.std(ddof=1) * np.sqrt(252)) if af.std(ddof=1) > 0 else 0.0
        ir_h = float(np.sqrt(252) * ah.mean() / ah.std(ddof=1)) if ah.std(ddof=1) > 0 else np.nan
        ir_f = float(np.sqrt(252) * af.mean() / af.std(ddof=1)) if af.std(ddof=1) > 0 else np.nan
        beta_h = float(pr.cov(hr) / hr.var()) if hr.var() > 0 else np.nan
        return {
            "strategy": strategy_name,
            "fee_bps": fee_bps,
            "period": label,
            "days": len(pr),
            "net_return": net,
            "h00905_return": h_cum,
            "excess_h00905_pp": net - h_cum,
            "te_h00905": te_h,
            "ir_h00905": ir_h,
            "floatmv_return": f_cum,
            "excess_floatmv_pp": net - f_cum,
            "te_floatmv": te_f,
            "ir_floatmv": ir_f,
            "max_dd": float((1.0 - nav_curve / np.maximum.accumulate(nav_curve)).max()),
            "max_rel_dd_h00905": float((1.0 - rel_h / np.maximum.accumulate(rel_h)).max()),
            "max_rel_dd_floatmv": float((1.0 - rel_f / np.maximum.accumulate(rel_f)).max()),
            "realized_beta_h00905": beta_h,
        }

    years = pd.Series(d.index.year, index=d.index)
    yearly_rows = [slice_metrics(years.eq(y), str(y)) for y in sorted(years.unique())]
    sub_rows = [
        slice_metrics(pd.Series(True, index=d.index), "all"),
        slice_metrics(years.le(2022), "2017-2022"),
        slice_metrics(years.ge(2023), "2023-2025"),
        slice_metrics(years.ne(2020), "ex_2020"),
    ]
    yearly_df = pd.DataFrame(yearly_rows + sub_rows)

    weekly_turn = float(d["gross_traded"].div(d["opening_equity"]).sum() * 5.0 / len(d))
    ind_exp = float(predictions.groupby(["signal_date", "industry"])["active_weight"].sum().abs().max())
    beta_exp = float((predictions["beta"] * predictions["active_weight"]).groupby(predictions["signal_date"]).sum().abs().max())
    size_exp = float((predictions["size_z"] * predictions["active_weight"]).groupby(predictions["signal_date"]).sum().abs().max())

    overall = dict(sub_rows[0])
    overall["weekly_two_way_turnover"] = weekly_turn
    overall["max_ind_active_exp"] = ind_exp
    overall["max_beta_active_exp"] = beta_exp
    overall["max_size_active_exp"] = size_exp
    overall["total_fees"] = float(d["fee"].sum())
    return overall, yearly_df, daily_active
