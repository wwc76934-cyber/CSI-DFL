"""CSI 500 open-to-open backtest engine with strict open tradability, realistic A-share costs, capacity limits, and post-trade exposure audit."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd


CENT = D("0.01")
GEM_CHANGE = pd.Timestamp("2020-08-24")
STAMP_DUTY_CUT = pd.Timestamp("2023-08-28")


@dataclass(frozen=True)
class BarQuote:
    raw_open: float
    raw_close: float
    raw_preclose: float
    hfq_open: float
    hfq_close: float
    open_tradable: bool
    close_ok: bool = True
    prev_amount: float = float("inf")


def money(value: D) -> D:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def stock_rules(code: str, day: pd.Timestamp) -> tuple[int, int, D]:
    if code.startswith(("sh.688", "sh.689")):
        return 200, 1, D("0.20")
    if code.startswith(("sz.300", "sz.301", "sz.302")) and day >= GEM_CHANGE:
        return 100, 100, D("0.20")
    return 100, 100, D("0.10")


def order_fee_rate(day: pd.Timestamp, is_sell: bool, fee_mode: str = "fixed_20bp", fee_bps: int | None = None) -> float:
    """Return one-way transaction cost rate (commission + slippage + A-share stamp duty)."""
    if fee_mode == "ashare_realistic":
        # Buy: 2.5bp commission + 7.5bp open slippage = 10bp
        # Sell: stamp duty (10bp pre-2023-08-28, 5bp post) + 2.5bp comm + 7.5bp open slippage
        if not is_sell:
            return 10.0 / 10000.0
        stamp = 10.0 if day < STAMP_DUTY_CUT else 5.0
        return (stamp + 2.5 + 7.5) / 10000.0
    if fee_mode.startswith("fixed_") and fee_mode.endswith("bp"):
        bps = int(fee_mode.replace("fixed_", "").replace("bp", ""))
        return bps / 10000.0
    if fee_bps is not None:
        return float(fee_bps) / 10000.0
    raise ValueError(f"Unsupported fee_mode={fee_mode!r}")


def build_quote_lookup(bars_df: pd.DataFrame, eval_days: pd.DatetimeIndex) -> dict[pd.Timestamp, dict[str, BarQuote]]:
    has_open_flag = "open_tradable" in bars_df.columns
    has_close_flag = "close_ok" in bars_df.columns
    has_prev_amt = "prev_amount" in bars_df.columns

    cols = ["date", "code", "raw_open", "raw_close", "raw_preclose", "hfq_open", "hfq_close"]
    if has_open_flag:
        cols.append("open_tradable")
    if has_close_flag:
        cols.append("close_ok")
    if not has_open_flag or not has_close_flag:
        cols.append("bar_ok")
    if has_prev_amt:
        cols.append("prev_amount")

    sub = bars_df.loc[bars_df["date"].isin(eval_days), cols]
    lookup: dict[pd.Timestamp, dict[str, BarQuote]] = {}
    for day, grp in sub.groupby("date", sort=True):
        day_ts = pd.Timestamp(day)
        day_map: dict[str, BarQuote] = {}
        for row in grp.itertuples(index=False):
            open_ok = bool(getattr(row, "open_tradable")) if has_open_flag else bool(getattr(row, "bar_ok"))
            close_ok = bool(getattr(row, "close_ok")) if has_close_flag else bool(getattr(row, "bar_ok"))
            p_amt_raw = getattr(row, "prev_amount") if has_prev_amt else np.inf
            p_amt = float(p_amt_raw) if (p_amt_raw is not None and pd.notna(p_amt_raw)) else float("nan")
            day_map[row.code] = BarQuote(
                float(row.raw_open) if pd.notna(row.raw_open) else np.nan,
                float(row.raw_close) if pd.notna(row.raw_close) else np.nan,
                float(row.raw_preclose) if pd.notna(row.raw_preclose) else np.nan,
                float(row.hfq_open) if pd.notna(row.hfq_open) else np.nan,
                float(row.hfq_close) if pd.notna(row.hfq_close) else np.nan,
                open_ok,
                close_ok,
                p_amt,
            )
        lookup[day_ts] = day_map
    return lookup


def run_backtest(
    strategy_name: str,
    predictions: pd.DataFrame,
    bars: dict[pd.Timestamp, dict[str, BarQuote]],
    eval_days: pd.DatetimeIndex,
    fee_bps: int | None = None,
    fee_mode: str | None = None,
    initial_cash: float = 1_000_000.0,
    single_cap: float = 0.04,
    cash_reserve: float = 0.003,
    max_participation: float | None = None,
) -> pd.DataFrame:
    """Simulate weekly open-to-open portfolio execution using strictly pre-open tradability and hfq continuity."""
    resolved_mode = fee_mode if fee_mode is not None else f"fixed_{int(fee_bps or 0)}bp"
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
        realized_max_ind_exp = np.nan
        realized_beta_exp = np.nan
        realized_size_exp = np.nan
        planned_orders = 0
        filled_orders = 0
        cap_clipped_orders = 0

        def hfq_mark(code: str, field: str) -> float:
            q = quotes.get(code)
            if q is not None:
                ok = q.open_tradable if field == "hfq_open" else q.close_ok
                v = getattr(q, field)
                if ok and np.isfinite(v) and v > 0:
                    return float(v)
                if field == "hfq_close" and q.open_tradable and np.isfinite(q.hfq_open) and q.hfq_open > 0:
                    return float(q.hfq_open)
            return last_hfq_mark[code]

        if day in by_entry:
            grp = by_entry[day]
            buy_fee_rate = order_fee_rate(day, is_sell=False, fee_mode=resolved_mode, fee_bps=fee_bps)
            sell_fee_rate = order_fee_rate(day, is_sell=True, fee_mode=resolved_mode, fee_bps=fee_bps)

            open_stock_val = sum(qty * hfq_mark(c, "hfq_open") for c, qty in hfq_shares.items())
            opening_equity = cash + open_stock_val

            target_w = dict(zip(grp["code"], grp["target_weight"].astype(float)))
            codes_all = set(hfq_shares) | {c for c, w in target_w.items() if w > 1e-10}

            plans = []
            for code in codes_all:
                q = quotes.get(code)
                old_hfq = hfq_shares.get(code, 0.0)
                w_tgt = target_w.get(code, 0.0)

                # Strictly open-only tradability check (no close-of-day look-ahead!)
                if (
                    q is None
                    or not q.open_tradable
                    or not np.isfinite(q.raw_open)
                    or q.raw_open <= 0
                    or not np.isfinite(q.hfq_open)
                    or q.hfq_open <= 0
                ):
                    continue

                old_val = old_hfq * float(q.hfq_open)
                old_raw = int(round(old_val / float(q.raw_open))) if old_hfq > 0 else 0
                minimum, step, limit_rate = stock_rules(code, day)

                if w_tgt <= 1e-10:
                    desired_raw = 0
                else:
                    budget = opening_equity * (1.0 - cash_reserve) * w_tgt
                    sizing_cost = float(q.raw_open) * (1.0 + buy_fee_rate if day == days[0] else 1.0)
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
                    planned_orders += 1
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
                        # Strictly pre-open liquidity participation cap using prior-day amount
                        if max_participation is not None and np.isfinite(q.prev_amount) and q.prev_amount > 0:
                            max_notional = float(max_participation) * float(q.prev_amount)
                            max_sh = int(max_notional // float(q.raw_open))
                            if fill_raw > 0:
                                capped = (max_sh // step) * step
                                if capped < fill_raw:
                                    cap_clipped_orders += 1
                                    fill_raw = capped if capped >= minimum else 0
                            elif fill_raw < 0:
                                capped_sell = (max_sh // step) * step if desired_raw > 0 else max_sh
                                if capped_sell < -fill_raw:
                                    cap_clipped_orders += 1
                                    fill_raw = -capped_sell if capped_sell >= minimum else 0

                        if fill_raw > 0:
                            affordable = int(cash // (q.raw_open * step * (1.0 + buy_fee_rate))) * step
                            fill_raw = min(affordable, fill_raw)
                            if fill_raw < minimum:
                                fill_raw = 0

                if fill_raw == 0:
                    continue

                filled_orders += 1
                if desired_raw == 0 and fill_raw == -old_raw:
                    gross = old_hfq * float(q.hfq_open)
                    fee = abs(gross) * sell_fee_rate
                    cash += gross - fee
                    gross_traded += abs(gross)
                    day_fee += fee
                    hfq_shares.pop(code, None)
                else:
                    gross = fill_raw * float(q.raw_open)
                    f_rate = buy_fee_rate if gross > 0 else sell_fee_rate
                    fee = abs(gross) * f_rate
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

            # Audit realized post-trade holdings exposures vs anchor on rebalance open
            post_vals = {c: qty * hfq_mark(c, "hfq_open") for c, qty in hfq_shares.items()}
            post_eq = cash + sum(post_vals.values())
            if post_eq > 0 and {"industry", "anchor_weight", "beta", "size_z"}.issubset(grp.columns):
                w_real_map = {c: v / post_eq for c, v in post_vals.items()}
                g_codes = grp["code"].astype(str).tolist()
                w_real_vec = np.array([w_real_map.get(c, 0.0) for c in g_codes], dtype=np.float64)
                dw_real = w_real_vec - grp["anchor_weight"].to_numpy(dtype=np.float64)
                inds = grp["industry"].astype(str).to_numpy()
                ind_df = pd.Series(dw_real).groupby(inds).sum().abs()
                realized_max_ind_exp = float(ind_df.max()) if len(ind_df) else 0.0
                realized_beta_exp = float(abs(np.dot(grp["beta"].to_numpy(dtype=np.float64), dw_real)))
                realized_size_exp = float(abs(np.dot(grp["size_z"].to_numpy(dtype=np.float64), dw_real)))

        stock_val = 0.0
        for code, qty in hfq_shares.items():
            px = hfq_mark(code, "hfq_close")
            last_hfq_mark[code] = px
            stock_val += qty * px

        equity = cash + stock_val
        nav_rows.append((
            strategy_name, day, cash, stock_val, equity, day_fee, gross_traded, opening_equity,
            realized_max_ind_exp, realized_beta_exp, realized_size_exp,
            planned_orders, filled_orders, cap_clipped_orders,
        ))

    return pd.DataFrame(
        nav_rows,
        columns=[
            "strategy", "date", "cash", "stock_value", "equity", "fee", "gross_traded", "opening_equity",
            "realized_max_ind_exp", "realized_beta_exp", "realized_size_exp",
            "planned_orders", "filled_orders", "cap_clipped_orders",
        ],
    )


def summarize_performance(
    strategy_name: str,
    fee_mode: str,
    nav_df: pd.DataFrame,
    benchmarks_df: pd.DataFrame,
    predictions: pd.DataFrame,
    initial_cash: float = 1_000_000.0,
    anchor_nav_df: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    d = nav_df.set_index("date").sort_index()
    b = benchmarks_df.set_index("date").sort_index().reindex(d.index)
    if b[["h00905", "float_mv_nav"]].isna().any().any():
        raise ValueError("Missing benchmark dates for NAV comparison")

    port_r = d["equity"].div(d["equity"].shift(fill_value=initial_cash)) - 1.0
    h_r = b["h00905_nav"].div(b["h00905_nav"].shift(fill_value=1.0)) - 1.0
    if anchor_nav_df is not None:
        a = anchor_nav_df.set_index("date").sort_index().reindex(d.index)
        f_r = a["equity"].div(a["equity"].shift(fill_value=initial_cash)) - 1.0
    else:
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
        n_yrs = len(pr) / 252.0
        cagr_p = float((1.0 + net) ** (1.0 / n_yrs) - 1.0) if n_yrs > 0 else np.nan
        cagr_h = float((1.0 + h_cum) ** (1.0 / n_yrs) - 1.0) if n_yrs > 0 else np.nan
        cagr_f = float((1.0 + f_cum) ** (1.0 / n_yrs) - 1.0) if n_yrs > 0 else np.nan
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
            "fee_mode": fee_mode,
            "period": label,
            "days": len(pr),
            "net_return": net,
            "cagr": cagr_p,
            "h00905_return": h_cum,
            "excess_h00905_pp": net - h_cum,
            "ann_excess_h00905": cagr_p - cagr_h,
            "te_h00905": te_h,
            "ir_h00905": ir_h,
            "floatmv_return": f_cum,
            "excess_floatmv_pp": net - f_cum,
            "ann_excess_floatmv": cagr_p - cagr_f,
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
    overall["target_max_ind_active_exp"] = ind_exp
    overall["target_max_beta_active_exp"] = beta_exp
    overall["target_max_size_active_exp"] = size_exp
    overall["realized_max_ind_active_exp"] = float(d["realized_max_ind_exp"].max()) if d["realized_max_ind_exp"].notna().any() else np.nan
    overall["realized_max_beta_active_exp"] = float(d["realized_beta_exp"].max()) if d["realized_beta_exp"].notna().any() else np.nan
    overall["realized_max_size_active_exp"] = float(d["realized_size_exp"].max()) if d["realized_size_exp"].notna().any() else np.nan
    overall["avg_cash_weight"] = float((d["cash"] / d["equity"]).mean())
    tot_planned = int(d["planned_orders"].sum())
    tot_filled = int(d["filled_orders"].sum())
    overall["order_fill_rate"] = float(tot_filled / tot_planned) if tot_planned > 0 else 1.0
    overall["cap_clipped_orders"] = int(d["cap_clipped_orders"].sum())
    overall["total_fees"] = float(d["fee"].sum())
    return overall, yearly_df, daily_active
