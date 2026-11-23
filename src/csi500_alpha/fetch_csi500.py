"""Resumable CSI 500 data fetcher (index, constituents, daily bars, valuations)."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import pandas as pd

from .config import load_config, project_root


def normalize_a_share_code(raw_code: str) -> str:
    code = str(raw_code).strip()
    if code.startswith(("sh.", "sz.", "bj.")):
        return code.lower()
    digits = "".join(ch for ch in code if ch.isdigit()).zfill(6)
    if digits.startswith(("6", "9")):
        return f"sh.{digits}"
    if digits.startswith(("0", "2", "3")):
        return f"sz.{digits}"
    if digits.startswith(("4", "8")):
        return f"bj.{digits}"
    return digits


def code_to_digits(code: str) -> str:
    return "".join(ch for ch in str(code) if ch.isdigit()).zfill(6)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def fetch_index_and_calendar(
    raw_dir: Path,
    start_date: str = "20140601",
    end_date: str = "20251231",
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch CSI 500 price (000905) and total return (H00905) indices and derive calendar/schedule."""
    import akshare as ak

    idx_path = raw_dir / "000905.parquet"
    tr_path = raw_dir / "H00905.parquet"
    cal_path = raw_dir / "calendar.parquet"
    sched_path = raw_dir / "schedule.parquet"

    if not (idx_path.exists() and tr_path.exists() and cal_path.exists() and sched_path.exists()) or force:
        for sym, target in [("000905", idx_path), ("H00905", tr_path)]:
            last_err: Exception | None = None
            for attempt in range(1, 4):
                try:
                    df = ak.index_hist_csindex(symbol=sym, start_date=start_date, end_date=end_date)
                    if df is None or df.empty:
                        raise RuntimeError(f"index_hist_csindex({sym}) returned empty")
                    out = pd.DataFrame({
                        "date": pd.to_datetime(df["日期"]),
                        "open": pd.to_numeric(df["开盘"], errors="coerce"),
                        "high": pd.to_numeric(df["最高"], errors="coerce"),
                        "low": pd.to_numeric(df["最低"], errors="coerce"),
                        "close": pd.to_numeric(df["收盘"], errors="coerce"),
                    }).dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
                    _atomic_parquet(out, target)
                    print(f"[index] {sym}: saved {len(out):,} days ({out['date'].min().date()} .. {out['date'].max().date()})", flush=True)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    print(f"[index] {sym} attempt {attempt}/3 failed: {exc}", flush=True)
                    time.sleep(2 * attempt)
            if last_err is not None:
                raise RuntimeError(f"Failed to fetch index {sym}: {last_err}")

        idx_df = pd.read_parquet(idx_path)
        days = pd.Series(pd.to_datetime(idx_df["date"]).sort_values().unique(), name="date")
        cal_df = pd.DataFrame({"date": days})
        _atomic_parquet(cal_df, cal_path)

        sched = pd.DataFrame({"signal_date": days, "execution_date": days.shift(-1)})
        sched = sched.loc[
            sched["signal_date"].ge(pd.Timestamp("2015-01-01"))
            & sched["execution_date"].notna()
            & sched["signal_date"].dt.to_period("W-SUN").ne(sched["execution_date"].dt.to_period("W-SUN"))
        ].reset_index(drop=True)
        _atomic_parquet(sched, sched_path)
        print(f"[calendar] Saved {len(cal_df):,} trading days and {len(sched):,} weekly rebalance pairs", flush=True)
    else:
        print("[index] Using cached 000905 / H00905 / calendar / schedule", flush=True)

    return pd.read_parquet(idx_path), pd.read_parquet(tr_path)


def fetch_members(
    raw_dir: Path,
    source: str = "baostock",
    force: bool = False,
) -> Path:
    """Fetch CSI 500 constituents either via BaoStock PIT snapshots or CSIndex latest snapshot."""
    mem_dir = raw_dir / "members"
    mem_dir.mkdir(parents=True, exist_ok=True)
    meta_path = mem_dir / "metadata.json"

    if source == "csindex":
        import akshare as ak

        target = mem_dir / "csindex_latest.parquet"
        if not target.exists() or force:
            df = ak.index_stock_cons_csindex(symbol="000905")
            if df is None or df.empty:
                raise RuntimeError("ak.index_stock_cons_csindex('000905') returned empty")
            out = pd.DataFrame({
                "updateDate": pd.to_datetime(df["日期"], errors="coerce"),
                "code": df["成分券代码"].astype(str).map(normalize_a_share_code),
                "code_name": df["成分券名称"].astype(str),
            }).dropna(subset=["code"]).drop_duplicates("code").sort_values("code").reset_index(drop=True)
            _atomic_parquet(out, target)
            meta_path.write_text(
                json.dumps({"source": "csindex_latest", "pit_members": False, "count": len(out)}, indent=2),
                encoding="utf-8",
            )
            print(f"[members] Saved static CSIndex CSI 500 snapshot ({len(out)} codes, pit_members=False)", flush=True)
        else:
            print(f"[members] Using cached {target}", flush=True)
        return mem_dir

    # Default: BaoStock true Point-in-Time month-end snapshots (2014-12 .. 2025-12)
    import baostock as bs

    cal_path = raw_dir / "calendar.parquet"
    if not cal_path.exists():
        raise FileNotFoundError("Missing calendar.parquet; run --step index first.")
    cal = pd.read_parquet(cal_path)
    days = pd.to_datetime(cal["date"]).sort_values()
    month_ends = days.groupby(days.dt.to_period("M")).max()
    snap_dates = [d for d in month_ends if pd.Timestamp("2014-12-01") <= d <= pd.Timestamp("2025-12-31")]

    socket.setdefaulttimeout(20)
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(
            f"BaoStock login failed [{lg.error_code}] {lg.error_msg}. "
            "If BaoStock stays unreachable, you may rerun with --members-source csindex (non-PIT fallback)."
        )
    try:
        written = 0
        for idx, d in enumerate(snap_dates, 1):
            tag = d.strftime("%Y-%m-%d")
            target = mem_dir / f"{tag}.parquet"
            if target.exists() and not force:
                continue
            rs = bs.query_zz500_stocks(date=tag)
            if rs.error_code != "0":
                raise RuntimeError(f"query_zz500_stocks({tag}) failed [{rs.error_code}] {rs.error_msg}")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                raise RuntimeError(f"query_zz500_stocks({tag}) returned 0 rows")
            df = pd.DataFrame(rows, columns=rs.fields)
            df["snapshot_date"] = d
            df["updateDate"] = pd.to_datetime(df["updateDate"], errors="coerce")
            df["code"] = df["code"].astype(str).map(normalize_a_share_code)
            _atomic_parquet(df, target)
            written += 1
            if written % 20 == 0 or idx == len(snap_dates):
                print(f"[members] BaoStock PIT snapshot {idx}/{len(snap_dates)} ({tag})", flush=True)
            time.sleep(0.05)
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    meta_path.write_text(
        json.dumps({"source": "baostock", "pit_members": True, "snapshots": len(snap_dates)}, indent=2),
        encoding="utf-8",
    )
    print(f"[members] Ready {len(snap_dates)} BaoStock PIT CSI 500 snapshots (pit_members=True)", flush=True)
    return mem_dir


def collect_unique_member_codes(raw_dir: Path) -> list[str]:
    mem_dir = raw_dir / "members"
    files = sorted(mem_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No member parquet files in {mem_dir}; run --step members first.")
    codes: set[str] = set()
    for f in files:
        df = pd.read_parquet(f, columns=["code"])
        codes.update(df["code"].astype(str).tolist())
    return sorted(codes)


def _standardize_em_bars(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "code", "open", "close", "high", "low", "volume", "amount", "turn", "change_amt", "pct_chg"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["日期"]),
        "code": code,
        "open": pd.to_numeric(df["开盘"], errors="coerce"),
        "close": pd.to_numeric(df["收盘"], errors="coerce"),
        "high": pd.to_numeric(df["最高"], errors="coerce"),
        "low": pd.to_numeric(df["最低"], errors="coerce"),
        "volume": pd.to_numeric(df["成交量"], errors="coerce"),
        "amount": pd.to_numeric(df["成交额"], errors="coerce"),
        "turn": pd.to_numeric(df["换手率"], errors="coerce"),
        "change_amt": pd.to_numeric(df["涨跌额"], errors="coerce"),
        "pct_chg": pd.to_numeric(df["涨跌幅"], errors="coerce"),
    })
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_bars(
    raw_dir: Path,
    start_date: str = "20140601",
    end_date: str = "20251231",
    force: bool = False,
    sleep_sec: float = 0.08,
) -> None:
    """Fetch raw and hfq daily bars for all CSI 500 historical/current constituent codes."""
    import akshare as ak

    codes = collect_unique_member_codes(raw_dir)
    raw_out = raw_dir / "bars_raw"
    hfq_out = raw_dir / "bars_hfq"
    raw_out.mkdir(parents=True, exist_ok=True)
    hfq_out.mkdir(parents=True, exist_ok=True)

    failed = []
    for idx, code in enumerate(codes, 1):
        rf = raw_out / f"{code}.parquet"
        hf = hfq_out / f"{code}.parquet"
        if rf.exists() and hf.exists() and not force:
            continue

        digits = code_to_digits(code)
        ok = False
        for attempt in range(1, 4):
            try:
                if not rf.exists() or force:
                    df_raw = ak.stock_zh_a_hist_em(symbol=digits, period="daily", start_date=start_date, end_date=end_date, adjust="")
                    _atomic_parquet(_standardize_em_bars(df_raw, code), rf)
                if not hf.exists() or force:
                    df_hfq = ak.stock_zh_a_hist_em(symbol=digits, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
                    _atomic_parquet(_standardize_em_bars(df_hfq, code), hf)
                ok = True
                break
            except Exception as exc:
                if attempt == 3:
                    failed.append({"code": code, "error": str(exc)})
                else:
                    time.sleep(1.0 * attempt)

        if idx % 50 == 0 or idx == len(codes):
            print(f"[bars] Progress {idx}/{len(codes)} (failed={len(failed)})", flush=True)
        if ok:
            time.sleep(sleep_sec)

    if failed:
        pd.DataFrame(failed).to_csv(raw_dir / "bars_errors.csv", index=False, encoding="utf-8-sig")
        print(f"[bars] Finished with {len(failed)} failed codes logged to {raw_dir / 'bars_errors.csv'}", flush=True)
    else:
        print(f"[bars] All {len(codes)} codes ready!", flush=True)


def fetch_valuations(
    raw_dir: Path,
    end_date: str = "2025-12-31",
    force: bool = False,
    sleep_sec: float = 0.08,
) -> None:
    """Fetch daily valuation metrics (PE_TTM, PB_MRQ, FloatMV) via ak.stock_value_em."""
    import akshare as ak

    codes = collect_unique_member_codes(raw_dir)
    val_dir = raw_dir / "valuations"
    val_dir.mkdir(parents=True, exist_ok=True)
    end_ts = pd.Timestamp(end_date)

    failed = []
    for idx, code in enumerate(codes, 1):
        vf = val_dir / f"{code}.parquet"
        if vf.exists() and not force:
            continue
        digits = code_to_digits(code)
        ok = False
        for attempt in range(1, 4):
            try:
                df = ak.stock_value_em(symbol=digits)
                if df is None or df.empty:
                    out = pd.DataFrame(columns=["date", "code", "pe_ttm", "pb_mrq", "ps_ttm", "float_mv"])
                else:
                    out = pd.DataFrame({
                        "date": pd.to_datetime(df["数据日期"], errors="coerce"),
                        "code": code,
                        "pe_ttm": pd.to_numeric(df["PE(TTM)"], errors="coerce"),
                        "pb_mrq": pd.to_numeric(df["市净率"], errors="coerce"),
                        "ps_ttm": pd.to_numeric(df["市销率"], errors="coerce") if "市销率" in df.columns else float("nan"),
                        "float_mv": pd.to_numeric(df["流通市值"], errors="coerce"),
                    })
                    out = out.loc[out["date"].notna() & out["date"].le(end_ts)].sort_values("date").drop_duplicates("date").reset_index(drop=True)
                _atomic_parquet(out, vf)
                ok = True
                break
            except Exception as exc:
                if attempt == 3:
                    failed.append({"code": code, "error": str(exc)})
                else:
                    time.sleep(1.0 * attempt)

        if idx % 50 == 0 or idx == len(codes):
            print(f"[valuations] Progress {idx}/{len(codes)} (failed={len(failed)})", flush=True)
        if ok:
            time.sleep(sleep_sec)

    if failed:
        pd.DataFrame(failed).to_csv(raw_dir / "valuations_errors.csv", index=False, encoding="utf-8-sig")
        print(f"[valuations] Finished with {len(failed)} failed codes logged to {raw_dir / 'valuations_errors.csv'}", flush=True)
    else:
        print(f"[valuations] All {len(codes)} codes ready!", flush=True)


def _probe_baostock_zz500() -> None:
    import baostock as bs

    socket.setdefaulttimeout(20)
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed [{lg.error_code}] {lg.error_msg}")
    try:
        rs = bs.query_zz500_stocks(date="2024-01-05")
        if rs.error_code != "0":
            raise RuntimeError(f"query_zz500_stocks failed [{rs.error_code}] {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        print(f"   BaoStock query_zz500_stocks OK: {len(rows)} rows")
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def probe_apis() -> None:
    import akshare as ak

    print("[1/4] Probing AKShare CSI 500 index (H00905 & 000905)...", flush=True)
    try:
        df_tr = ak.index_hist_csindex(symbol="H00905", start_date="20240101", end_date="20240131")
        df_px = ak.index_hist_csindex(symbol="000905", start_date="20240101", end_date="20240131")
        print(f"   H00905 rows={len(df_tr)}, 000905 rows={len(df_px)}", flush=True)
    except Exception as exc:
        print(f"   [FAIL] {exc}", flush=True)

    print("[2/4] Probing AKShare daily bars (stock_zh_a_hist_em)...", flush=True)
    try:
        df_b = ak.stock_zh_a_hist_em(symbol="000001", period="daily", start_date="20240101", end_date="20240131", adjust="")
        print(f"   rows={len(df_b)}, cols={list(df_b.columns)}", flush=True)
    except Exception as exc:
        print(f"   [FAIL] {exc}", flush=True)

    print("[3/4] Probing AKShare valuation history (stock_value_em)...", flush=True)
    try:
        df_v = ak.stock_value_em(symbol="000001")
        print(f"   rows={len(df_v)}, range={df_v['数据日期'].min()}..{df_v['数据日期'].max()}", flush=True)
    except Exception as exc:
        print(f"   [FAIL] {exc}", flush=True)

    print("[4/4] Probing CSI 500 constituents (BaoStock 30s timeout vs CSIndex fallback)...", flush=True)
    try:
        res = subprocess.run(
            [sys.executable, "-B", "-m", "csi500_alpha.fetch_csi500", "probe_bs"],
            cwd=project_root(), timeout=30, check=False,
        )
        if res.returncode != 0:
            print("   [WARN] BaoStock PIT probe failed; CSIndex fallback available (--members-source csindex).", flush=True)
    except subprocess.TimeoutExpired:
        print("   [WARN] BaoStock PIT probe timed out (30s); CSIndex fallback available.", flush=True)
    try:
        df_c = ak.index_stock_cons_csindex(symbol="000905")
        print(f"   CSIndex fallback index_stock_cons_csindex('000905') OK: {len(df_c)} rows", flush=True)
    except Exception as exc:
        print(f"   [FAIL] CSIndex fallback failed: {exc}", flush=True)


def print_data_summary(raw_dir: Path) -> None:
    cfg = load_config()
    inc_root = Path(cfg["project"]["incremental_root"])
    print("\n=== CSI 500 Local Data Summary ===")
    for name in ["000905.parquet", "H00905.parquet", "calendar.parquet", "schedule.parquet"]:
        p = raw_dir / name
        print(f"  {name:18s}: {'OK (' + str(len(pd.read_parquet(p))) + ' rows)' if p.exists() else 'MISSING'}")
    mem_files = list((raw_dir / "members").glob("*.parquet"))
    meta = raw_dir / "members" / "metadata.json"
    print(f"  members           : {len(mem_files)} parquet files ({meta.read_text(encoding='utf-8').strip() if meta.exists() else 'no metadata'})")
    if mem_files:
        n_codes = len(collect_unique_member_codes(raw_dir))
        n_raw = len(list((raw_dir / "bars_raw").glob("*.parquet")))
        n_hfq = len(list((raw_dir / "bars_hfq").glob("*.parquet")))
        n_val = len(list((raw_dir / "valuations").glob("*.parquet")))
        print(f"  unique codes      : {n_codes}")
        print(f"  bars_raw / hfq    : {n_raw}/{n_codes} raw, {n_hfq}/{n_codes} hfq")
        print(f"  valuations        : {n_val}/{n_codes}")
    print(f"  shared industry   : {'OK' if (inc_root / 'industry_map.parquet').exists() else 'MISSING'}")
    print(f"  shared reports    : {len(list((inc_root / 'quarterly_reports').glob('*.parquet')))}/48 quarters")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "probe_bs":
        try:
            _probe_baostock_zz500()
        except Exception as exc:
            print(f"   [WARN] {exc}", flush=True)
            raise SystemExit(1) from None
