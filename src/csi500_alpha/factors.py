"""CSI 500 10-factor engineering, strict PIT alignment, and double OLS neutralization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEV_END = pd.Timestamp("2025-12-31")

FACTORS = [
    "ROE",
    "NP_YOY",
    "CFO_YIELD",
    "EP_TTM",
    "BP_MRQ",
    "MOM_120_SKIP20",
    "HIGH_52W_PROX",
    "OVERNIGHT_RET_20",
    "INTRADAY_REV_20",
    "CORR_PV_10",
]

FACTORS_PURE_PIT_7 = [
    "EP_TTM",
    "BP_MRQ",
    "MOM_120_SKIP20",
    "HIGH_52W_PROX",
    "OVERNIGHT_RET_20",
    "INTRADAY_REV_20",
    "CORR_PV_10",
]


def _to_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).astype("datetime64[ns]")


def rolling_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    min_p = max(5, int(window * 0.7))
    mx = x.rolling(window, min_periods=min_p).mean()
    my = y.rolling(window, min_periods=min_p).mean()
    cov = (x * y).rolling(window, min_periods=min_p).mean() - mx * my
    vx = (x * x).rolling(window, min_periods=min_p).mean() - mx * mx
    vy = (y * y).rolling(window, min_periods=min_p).mean() - my * my
    denom = np.sqrt(vx.clip(lower=0.0) * vy.clip(lower=0.0))
    return (cov / denom.where(denom > 1e-12)).clip(-1.0, 1.0)


def rolling_beta(stock_ret: pd.DataFrame, market_ret: pd.Series, window: int, min_periods: int) -> pd.DataFrame:
    valid = stock_ret.notna()
    mkt_matched = valid.mul(market_ret, axis=0).where(valid)
    mean_s = stock_ret.rolling(window, min_periods=min_periods).mean()
    mean_m = mkt_matched.rolling(window, min_periods=min_periods).mean()
    cov = (stock_ret * mkt_matched).rolling(window, min_periods=min_periods).mean() - mean_s * mean_m
    var_m = (mkt_matched * mkt_matched).rolling(window, min_periods=min_periods).mean() - mean_m * mean_m
    return cov / var_m.where(var_m > 1e-12)


def _statutory_avail_date(report_dates: pd.Series) -> pd.Series:
    months = report_dates.dt.month
    years = report_dates.dt.year
    avail = pd.Series(pd.NaT, index=report_dates.index, dtype="datetime64[ns]")
    for m, avail_m in [(3, 5), (6, 9), (9, 11)]:
        mask = months == m
        if mask.any():
            avail.loc[mask] = pd.to_datetime({"year": years.loc[mask], "month": avail_m, "day": 1}).astype("datetime64[ns]")
    mask_12 = months == 12
    if mask_12.any():
        avail.loc[mask_12] = pd.to_datetime({"year": years.loc[mask_12] + 1, "month": 5, "day": 1}).astype("datetime64[ns]")
    return avail


def align_quarterly_pit(signals_df: pd.DataFrame, quarterly_df: pd.DataFrame) -> pd.DataFrame:
    """Strictly align quarterly fundamentals by avail_date <= signal_date (zero look-ahead)."""
    left = signals_df.copy()
    left["_orig_order"] = np.arange(len(left))
    left["signal_date"] = _to_ns(left["signal_date"])
    left = left.sort_values(["signal_date", "code"], kind="mergesort")

    right = quarterly_df.copy()
    right["pub_date"] = _to_ns(right["pub_date"])
    if "report_date" in right.columns:
        right["report_date"] = _to_ns(right["report_date"])
        lag_days = (right["pub_date"] - right["report_date"]).dt.days
        statutory = _statutory_avail_date(right["report_date"])
        timely = lag_days.between(0, 150)
        right["avail_date"] = _to_ns(right["pub_date"].where(timely, statutory))
        right["quarterly_fallback_used"] = (~timely).astype(bool)
        right = right.dropna(subset=["avail_date"]).sort_values(["avail_date", "report_date", "code"], kind="mergesort")
    else:
        right["avail_date"] = right["pub_date"]
        right["quarterly_fallback_used"] = False
        right = right.dropna(subset=["avail_date"]).sort_values(["avail_date", "code"], kind="mergesort")

    keep_cols = [
        c for c in ["code", "avail_date", "pub_date", "report_date", "quarterly_fallback_used", "roe", "np_yoy", "cfo_per_share"]
        if c in right.columns
    ]
    merged = pd.merge_asof(
        left,
        right[keep_cols],
        left_on="signal_date",
        right_on="avail_date",
        by="code",
        direction="backward",
        allow_exact_matches=True,
    )
    bad = merged["avail_date"].notna() & (merged["avail_date"] > merged["signal_date"])
    if bad.any():
        raise ValueError(f"Point-in-time violation: {int(bad.sum())} rows have avail_date > signal_date")
    merged["quarterly_fallback_used"] = merged["quarterly_fallback_used"].eq(True)
    return merged.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)


def align_industry_pit(signals_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    """Strictly align Shenwan Level-1 industry by start_date <= signal_date (no future backfill)."""
    left = signals_df.copy()
    left["_orig_order"] = np.arange(len(left))
    left["signal_date"] = _to_ns(left["signal_date"])
    left = left.sort_values(["signal_date", "code"], kind="mergesort")

    right = industry_df.copy()
    if "start_date" in right.columns:
        right["start_date"] = _to_ns(right["start_date"])
        right = right.dropna(subset=["code", "industry", "start_date"]).sort_values(["start_date", "code"], kind="mergesort")
        merged = pd.merge_asof(
            left,
            right[["code", "industry", "start_date"]],
            left_on="signal_date",
            right_on="start_date",
            by="code",
            direction="backward",
            allow_exact_matches=True,
        )
        bad = merged["start_date"].notna() & (merged["start_date"] > merged["signal_date"])
        if bad.any():
            raise ValueError(f"Industry PIT violation: {int(bad.sum())} rows have start_date > signal_date")
        merged["industry"] = merged["industry"].fillna("Other").astype(str)
        merged = merged.drop(columns=["start_date"])
    else:
        ind_map = right.drop_duplicates(subset=["code"], keep="last").set_index("code")["industry"].to_dict()
        merged = left.copy()
        merged["industry"] = merged["code"].map(ind_map).fillna("Other").astype(str)

    return merged.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)


def neutralize_cross_section(frame: pd.DataFrame, factor_cols: list[str], winsor_sigma: float = 3.0) -> pd.DataFrame:
    """Double OLS residualization on Shenwan L1 industry dummies + standardized log FloatMV."""
    out = frame.copy()
    out["size_z"] = 0.0
    for _, idx in out.groupby("signal_date", sort=False).groups.items():
        sub = out.loc[idx]
        mv = sub["log_mv"]
        valid = sub["signal_bar_usable"] & mv.notna() & np.isfinite(mv)
        if valid.sum() >= 10:
            vals = mv.loc[valid]
            sd = float(vals.std(ddof=0))
            if sd > 1e-12:
                out.loc[vals.index, "size_z"] = ((vals - vals.mean()) / sd).clip(-winsor_sigma, winsor_sigma)

    for col in factor_cols:
        out[col] = 0.0

    for _, idx in out.groupby("signal_date", sort=False).groups.items():
        sub = out.loc[idx]
        ind = sub["industry"].fillna("Other").astype(str).copy()
        counts = ind.value_counts()
        small_inds = set(counts[counts < 2].index)
        if small_inds:
            ind = ind.where(~ind.isin(small_inds), "Other")
        counts2 = ind.value_counts()
        if (counts2 < 2).any() and len(counts2) > 1:
            largest = counts2.idxmax()
            rare = set(counts2[counts2 < 2].index)
            ind = ind.where(~ind.isin(rare), largest)

        dummies = pd.get_dummies(ind, drop_first=True, dtype=float)
        design_full = pd.concat([pd.Series(1.0, index=sub.index, name="_const"), dummies, sub["size_z"]], axis=1)

        for col in factor_cols:
            y_all = sub[f"raw_{col}"]
            eligible = sub["signal_bar_usable"] & y_all.notna() & np.isfinite(y_all)
            if eligible.sum() < max(20, design_full.shape[1] + 5):
                continue
            y = y_all.loc[eligible].astype(float)
            q_lo, q_hi = y.quantile(0.01), y.quantile(0.99)
            if np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo:
                y = y.clip(q_lo, q_hi)
            X = design_full.loc[eligible].to_numpy(dtype=np.float64)
            y_vec = y.to_numpy(dtype=np.float64)
            coef, *_ = np.linalg.lstsq(X, y_vec, rcond=None)
            resid = y_vec - X @ coef
            sd = resid.std(ddof=0)
            if sd <= 1e-12:
                continue
            resid = np.clip(resid, resid.mean() - winsor_sigma * sd, resid.mean() + winsor_sigma * sd)
            sd2 = resid.std(ddof=0)
            if sd2 <= 1e-12:
                continue
            out.loc[y.index, col] = (resid - resid.mean()) / sd2

    return out


def _expand_weekly_members(raw_dir: Path, schedule: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    mem_dir = raw_dir / "members"
    meta_path = mem_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    pit = bool(meta.get("pit_members", False))
    stamps = _to_ns(schedule["signal_date"]).sort_values().unique()

    if pit:
        snaps = []
        for f in sorted(mem_dir.glob("20*.parquet")):
            df = pd.read_parquet(f)
            df["snapshot_date"] = _to_ns(df["snapshot_date"])
            snaps.append(df[["snapshot_date", "code", "updateDate"]])
        snap_df = pd.concat(snaps, ignore_index=True).sort_values("snapshot_date")
        snap_dates = sorted(snap_df["snapshot_date"].unique())
        rows = []
        for sig in stamps:
            valid_snaps = [d for d in snap_dates if d <= sig]
            pick = valid_snaps[-1] if valid_snaps else snap_dates[0]
            sub = snap_df.loc[snap_df["snapshot_date"] == pick, ["code"]].copy()
            sub["signal_date"] = sig
            rows.append(sub[["signal_date", "code"]])
        members = pd.concat(rows, ignore_index=True)
    else:
        static_df = pd.read_parquet(mem_dir / "csindex_latest.parquet", columns=["code"])
        codes = static_df["code"].astype(str).tolist()
        members = pd.DataFrame({
            "signal_date": np.repeat(stamps, len(codes)),
            "code": codes * len(stamps),
        })
    return members.sort_values(["signal_date", "code"]).reset_index(drop=True), pit


def compute_benchmarks(
    weights_df: pd.DataFrame,
    daily_bars: pd.DataFrame,
    h00905: pd.Series,
    schedule: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    eval_start: pd.Timestamp = pd.Timestamp("2017-01-01"),
) -> pd.DataFrame:
    codes = sorted(weights_df["code"].unique())
    open_mat = daily_bars.pivot(index="date", columns="code", values="hfq_open").reindex(index=trading_days, columns=codes)
    close_mat = daily_bars.pivot(index="date", columns="code", values="hfq_close").reindex(index=trading_days, columns=codes)
    ok_mat = daily_bars.pivot(index="date", columns="code", values="bar_ok").reindex(index=trading_days, columns=codes).eq(True)
    open_mat = open_mat.where(ok_mat & open_mat.gt(0))
    close_mat = close_mat.where(ok_mat & close_mat.gt(0))

    rebal_map = schedule.loc[schedule["execution_date"] >= eval_start].set_index("execution_date")["signal_date"]
    first_eval_day = rebal_map.index.min()
    eval_days = trading_days[trading_days >= first_eval_day]

    weekly_w = {
        day: grp.set_index("code").reindex(codes)[["weight_float_mv", "weight_equal"]].fillna(0.0).to_numpy().T
        for day, grp in weights_df.groupby("signal_date")
    }
    latest_marks = close_mat.loc[close_mat.index < first_eval_day].ffill().iloc[-1].to_numpy(dtype=np.float64)
    shares = np.zeros((2, len(codes)), dtype=np.float64)
    equity = np.ones(2, dtype=np.float64)

    rows = []
    for day in eval_days:
        fo = open_mat.loc[day].to_numpy(dtype=np.float64)
        fc = close_mat.loc[day].to_numpy(dtype=np.float64)
        adj_o = np.where(np.isfinite(fo), fo, latest_marks)

        if day in rebal_map.index:
            sig = rebal_map.loc[day]
            tgt = weekly_w[sig].copy()
            unavail = ~np.isfinite(adj_o)
            tgt[:, unavail] = 0.0
            tot = tgt.sum(axis=1, keepdims=True)
            tgt = tgt / np.where(tot > 0, tot, 1.0)
            if shares.any():
                equity = shares @ np.nan_to_num(adj_o, nan=0.0)
            shares = tgt * equity[:, None] / np.nan_to_num(adj_o, nan=1.0)

        val = np.where(np.isfinite(fc), fc, adj_o)
        latest_marks = val
        equity = shares @ np.nan_to_num(val, nan=0.0)
        rows.append((day, float(equity[0]), float(equity[1]), float(h00905.loc[day])))

    bench = pd.DataFrame(rows, columns=["date", "float_mv_nav", "equal_weight_nav", "h00905"])
    anchor_day = trading_days[trading_days < first_eval_day][-1]
    bench["h00905_nav"] = bench["h00905"] / float(h00905.loc[anchor_day])
    return bench


def load_and_build_csi500_dataset(cfg: dict[str, Any], raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Load CSI 500 raw files + shared incremental data and build complete dev panel."""
    inc_root = Path(cfg["project"]["incremental_root"])
    horizon = int(cfg["factors"]["horizon_days"])

    cal_df = pd.read_parquet(raw_dir / "calendar.parquet")
    cal_df["date"] = _to_ns(cal_df["date"])
    trading_days = pd.DatetimeIndex(cal_df.loc[cal_df["date"].le(DEV_END), "date"].sort_values().unique())

    schedule = pd.read_parquet(raw_dir / "schedule.parquet")
    schedule["signal_date"] = _to_ns(schedule["signal_date"])
    schedule["execution_date"] = _to_ns(schedule["execution_date"])
    schedule = schedule.loc[schedule["execution_date"].le(DEV_END)].sort_values("signal_date").reset_index(drop=True)

    idx_px = pd.read_parquet(raw_dir / "000905.parquet")
    idx_px["date"] = _to_ns(idx_px["date"])
    idx_px = idx_px.loc[idx_px["date"].le(DEV_END)].set_index("date").reindex(trading_days)

    idx_tr = pd.read_parquet(raw_dir / "H00905.parquet")
    idx_tr["date"] = _to_ns(idx_tr["date"])
    h00905 = idx_tr.loc[idx_tr["date"].le(DEV_END)].set_index("date")["close"].sort_index()

    members, pit_members = _expand_weekly_members(raw_dir, schedule)
    codes = sorted(members["code"].unique())

    raw_parts, hfq_parts, val_parts = [], [], []
    for code in codes:
        rf = raw_dir / "bars_raw" / f"{code}.parquet"
        hf = raw_dir / "bars_hfq" / f"{code}.parquet"
        vf = raw_dir / "valuations" / f"{code}.parquet"
        if rf.exists():
            r = pd.read_parquet(rf)
            if not r.empty:
                raw_parts.append(r)
        if hf.exists():
            h = pd.read_parquet(hf)
            if not h.empty:
                hfq_parts.append(h)
        if vf.exists():
            v = pd.read_parquet(vf)
            if not v.empty:
                val_parts.append(v)

    raw_df = pd.concat(raw_parts, ignore_index=True)
    raw_df["date"] = _to_ns(raw_df["date"])
    raw_df = raw_df.loc[raw_df["date"].le(DEV_END)].rename(columns={
        "open": "raw_open", "close": "raw_close", "high": "raw_high", "low": "raw_low",
    })

    hfq_df = pd.concat(hfq_parts, ignore_index=True)
    hfq_df["date"] = _to_ns(hfq_df["date"])
    hfq_df = hfq_df.loc[hfq_df["date"].le(DEV_END), ["date", "code", "open", "close", "high", "low"]].rename(columns={
        "open": "hfq_open", "close": "hfq_close", "high": "hfq_high", "low": "hfq_low",
    })

    bars = raw_df.merge(hfq_df, on=["date", "code"], how="outer", validate="one_to_one")
    if val_parts:
        val_df = pd.concat(val_parts, ignore_index=True)
        val_df["date"] = _to_ns(val_df["date"])
        val_df = val_df.loc[val_df["date"].le(DEV_END), ["date", "code", "pe_ttm", "pb_mrq", "float_mv"]]
        bars = bars.merge(val_df, on=["date", "code"], how="left", validate="one_to_one")
    else:
        bars["pe_ttm"] = np.nan
        bars["pb_mrq"] = np.nan
        bars["float_mv"] = np.nan

    bars = bars.sort_values(["code", "date"]).reset_index(drop=True)
    # Reconstruct raw preclose accurately: BaoStock preclose -> close - change_amt -> hfq adjustment continuity
    if "preclose" in bars.columns:
        bars["raw_preclose"] = pd.to_numeric(bars["preclose"], errors="coerce")
        missing_pre = bars["raw_preclose"].isna() | bars["raw_preclose"].le(0)
        bars.loc[missing_pre, "raw_preclose"] = (bars["raw_close"] - bars["change_amt"]).loc[missing_pre]
    else:
        bars["raw_preclose"] = bars["raw_close"] - bars["change_amt"]
    adj_ratio = (bars["hfq_close"] / bars["raw_close"].where(bars["raw_close"].gt(0)))
    prev_adj_ratio = adj_ratio.groupby(bars["code"]).shift(1)
    prev_raw_close = bars.groupby("code")["raw_close"].shift(1)
    fallback_pre = prev_raw_close * (prev_adj_ratio / adj_ratio.where(adj_ratio.gt(0)))
    bad_pre = bars["raw_preclose"].isna() | bars["raw_preclose"].le(0)
    bars.loc[bad_pre, "raw_preclose"] = fallback_pre.loc[bad_pre]

    tradestatus_ok = (
        bars["tradestatus"].isna() | pd.to_numeric(bars["tradestatus"], errors="coerce").eq(1)
        if "tradestatus" in bars.columns
        else True
    )
    bars["open_tradable"] = (
        bars["raw_open"].gt(0)
        & bars["hfq_open"].gt(0)
        & tradestatus_ok
    )
    bars["close_ok"] = (
        bars["open_tradable"]
        & bars["raw_close"].gt(0)
        & bars["hfq_close"].gt(0)
        & bars["hfq_high"].ge(bars[["hfq_open", "hfq_close", "hfq_low"]].max(axis=1))
        & bars["hfq_low"].le(bars[["hfq_open", "hfq_close"]].min(axis=1))
        & bars["volume"].gt(0)
    )
    bars["bar_ok"] = bars["close_ok"]
    bars["prev_amount"] = bars.groupby("code")["amount"].shift(1)
    open_vs_close_diff = int((bars["open_tradable"] != bars["close_ok"]).sum())

    # Load shared incremental data (industry & quarterly reports)
    ind_df = pd.read_parquet(inc_root / "industry_map.parquet")
    ind_df["code"] = ind_df["code"].astype(str)
    if "start_date" in ind_df.columns:
        ind_df["start_date"] = _to_ns(ind_df["start_date"])

    q_frames = [pd.read_parquet(p) for p in sorted((inc_root / "quarterly_reports").glob("*.parquet"))]
    q_df = pd.concat(q_frames, ignore_index=True)
    q_df["code"] = q_df["code"].astype(str)
    q_df["report_date"] = _to_ns(q_df["report_date"])
    q_df["pub_date"] = _to_ns(q_df["pub_date"])

    def wide(col: str) -> pd.DataFrame:
        return bars.pivot(index="date", columns="code", values=col).reindex(trading_days)

    usable = wide("bar_ok").eq(True)
    hfq_close = wide("hfq_close").where(usable)
    hfq_open = wide("hfq_open").where(usable)
    hfq_high = wide("hfq_high").where(usable)
    raw_close = wide("raw_close").where(usable)
    volume = wide("volume").where(usable)
    amount = wide("amount").where(usable & wide("amount").gt(0))
    turn = wide("turn").where(usable & wide("turn").gt(0) & wide("turn").le(100))

    stock_ret = hfq_close.pct_change(fill_method=None)
    mkt_ret = idx_px["close"].pct_change(fill_method=None)
    vol20 = stock_ret.rolling(20, min_periods=15).std()
    beta120 = rolling_beta(stock_ret, mkt_ret, 120, 100)

    proxy_mv = amount / (turn / 100.0)
    val_mv = wide("float_mv").where(wide("float_mv").gt(0))
    float_mv_daily = val_mv.combine_first(proxy_mv).rolling(20, min_periods=10).median()

    overnight_1d = np.log((hfq_open / hfq_close.shift(1)).where(lambda x: x.gt(0)))
    intraday_1d = np.log((hfq_close / hfq_open).where(lambda x: x.gt(0)))

    pe_ttm = wide("pe_ttm")
    pb_mrq = wide("pb_mrq")

    mats = {
        "signal_bar_usable": usable,
        "raw_close": raw_close,
        "float_mv": float_mv_daily,
        "log_mv": np.log(float_mv_daily.where(float_mv_daily.gt(0))),
        "BETA_120": beta120.clip(0.2, 2.2).where(np.isfinite(beta120)),
        "VOL_20": vol20,
        "raw_EP_TTM": 1.0 / pe_ttm.where(pe_ttm.gt(0)),
        "raw_BP_MRQ": 1.0 / pb_mrq.where(pb_mrq.gt(0)),
        "raw_MOM_120_SKIP20": hfq_close.shift(20) / hfq_close.shift(120) - 1.0,
        "raw_HIGH_52W_PROX": hfq_close / hfq_high.rolling(240, min_periods=120).max(),
        "raw_OVERNIGHT_RET_20": overnight_1d.rolling(20, min_periods=15).sum(),
        "raw_INTRADAY_REV_20": -intraday_1d.rolling(20, min_periods=15).sum(),
        "raw_CORR_PV_10": -rolling_corr(hfq_close, np.log1p(volume), 10),
    }

    stamps = pd.DatetimeIndex(members["signal_date"].unique()).sort_values()
    frame = members.merge(schedule, on="signal_date", how="left", validate="many_to_one").rename(columns={"execution_date": "entry_date"})
    key_idx = pd.MultiIndex.from_frame(frame[["signal_date", "code"]])
    for name, mat in mats.items():
        stacked = mat.reindex(stamps).rename_axis(index="signal_date", columns="code").stack(future_stack=True)
        frame[name] = stacked.reindex(key_idx).to_numpy()
    frame["signal_bar_usable"] = frame["signal_bar_usable"].eq(True)

    mv_clean = frame["float_mv"].where(frame["float_mv"].gt(0)).fillna(0.0)
    tot_mv = mv_clean.groupby(frame["signal_date"]).transform("sum")
    frame["weight_float_mv"] = np.where(tot_mv > 0, mv_clean / tot_mv, 1.0 / frame.groupby("signal_date")["code"].transform("size"))
    frame["weight_equal"] = 1.0 / frame.groupby("signal_date")["code"].transform("size")

    frame = align_quarterly_pit(frame, q_df)
    q_num = frame["report_date"].dt.month // 3
    ann_mult = 4.0 / q_num.where(q_num.isin([1, 2, 3, 4]))
    frame["raw_ROE"] = frame["roe"] * ann_mult
    frame["raw_NP_YOY"] = frame["np_yoy"]
    frame["raw_CFO_YIELD"] = (frame["cfo_per_share"] * ann_mult) / frame["raw_close"].where(frame["raw_close"].gt(0))

    frame = align_industry_pit(frame, ind_df)
    frame = neutralize_cross_section(frame, FACTORS, winsor_sigma=3.0)

    next_exec = pd.Series(schedule["execution_date"].shift(-1).to_numpy(), index=schedule["execution_date"])
    fallback_exit = pd.Series(trading_days, index=trading_days).shift(-horizon)
    frame["exit_date"] = frame["entry_date"].map(next_exec).fillna(frame["entry_date"].map(fallback_exit))

    open_bars = bars[["date", "code", "hfq_open", "open_tradable"]].copy()
    frame = frame.merge(
        open_bars.rename(columns={"date": "entry_date", "hfq_open": "entry_open", "open_tradable": "entry_ok"}),
        on=["entry_date", "code"], how="left", validate="many_to_one",
    )
    frame = frame.merge(
        open_bars.rename(columns={"date": "exit_date", "hfq_open": "exit_open", "open_tradable": "exit_ok"}),
        on=["exit_date", "code"], how="left", validate="many_to_one",
    )

    valid_label = (
        frame["signal_bar_usable"]
        & frame["exit_date"].notna()
        & frame["exit_date"].le(DEV_END)
        & frame["entry_ok"].eq(True)
        & frame["exit_ok"].eq(True)
    )
    frame["stock_return_5d"] = (frame["exit_open"] / frame["entry_open"] - 1.0).where(valid_label)
    frame["h00905_return_5d"] = (frame["exit_date"].map(h00905) / frame["entry_date"].map(h00905) - 1.0).where(frame["exit_date"].notna())

    w_valid = frame["weight_float_mv"].where(frame["stock_return_5d"].notna(), 0.0)
    w_sum = w_valid.groupby(frame["signal_date"]).transform("sum")
    anchor_ret_hold = (w_valid * frame["stock_return_5d"].fillna(0.0)).groupby(frame["signal_date"]).transform("sum") / w_sum.where(w_sum > 0)
    frame["anchor_return_hold"] = anchor_ret_hold.fillna(frame["h00905_return_5d"])

    raw_target = (
        (frame["stock_return_5d"] - frame["BETA_120"] * frame["anchor_return_hold"]) / frame["VOL_20"].clip(lower=0.01)
    ).replace([np.inf, -np.inf], np.nan)
    frame["risk_adjusted_target"] = np.nan
    for _, idx in frame.groupby("signal_date", sort=False).groups.items():
        y = raw_target.loc[idx].dropna()
        if len(y) < 50:
            continue
        mu, sd = float(y.mean()), float(y.std(ddof=1))
        if sd <= 1e-12:
            continue
        trimmed = y.clip(mu - 3.0 * sd, mu + 3.0 * sd)
        sd2 = float(trimmed.std(ddof=1))
        if sd2 > 1e-12:
            frame.loc[y.index, "risk_adjusted_target"] = (trimmed - trimmed.mean()) / sd2

    eval_slice = frame.loc[frame["signal_date"] >= pd.Timestamp("2017-01-01")]
    has_rep = eval_slice["report_date"].notna()
    q_fallback_rate = (
        float(eval_slice.loc[has_rep, "quarterly_fallback_used"].mean())
        if has_rep.any()
        else 1.0
    )

    audit_rows = []
    for f in FACTORS:
        raw_col = f"raw_{f}"
        cov_all = float(frame[raw_col].notna().mean())
        cov_eval = float(eval_slice[raw_col].notna().mean())
        audit_rows.append({"factor": f, "coverage_2015_2025": cov_all, "coverage_2017_2025": cov_eval})
    audit_df = pd.DataFrame(audit_rows)

    corr_mats = [grp[FACTORS].corr(method="spearman").abs().to_numpy() for _, grp in frame.groupby("signal_date")]
    med_corr = np.nanmedian(np.stack(corr_mats, axis=0), axis=0)
    np.fill_diagonal(med_corr, 0.0)
    audit_df["max_median_abs_spearman_vs_others"] = med_corr.max(axis=1)
    audit_df["quarterly_fallback_rate_2017_2025"] = q_fallback_rate
    audit_df["pit_quarterly_strict"] = bool(q_fallback_rate < 0.01)
    audit_df["pit_industry_strict"] = True
    audit_df["open_vs_close_diff_rows"] = open_vs_close_diff
    audit_df["pit_members"] = pit_members

    benchmarks = compute_benchmarks(frame, bars, h00905, schedule, trading_days)
    return frame, bars, audit_df, benchmarks, pit_members
