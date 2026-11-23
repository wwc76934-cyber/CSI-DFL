"""Resumable CSI 500 data fetcher (index, constituents, daily bars, valuations)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .config import load_config, project_root


DEFAULT_V3_CACHE_DIR = Path(r"D:\QuantProjects\regime-factor\data\raw\v3\development")

BS_RAW_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST"
)
BS_HFQ_FIELDS = "date,code,open,high,low,close,volume,amount,adjustflag,turn"


def _to_iso_date(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Expected YYYYMMDD or YYYY-MM-DD date string, got {value!r}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


class _BoundedSocket(socket.socket):
    def recv(self, *args):
        data = super().recv(*args)
        if not data:
            raise ConnectionError("BaoStock closed its connection")
        return data


class BaoSession:
    """Managed BaoStock session with bounded socket timeout, guarded send_msg, and automatic reconnect."""

    def __init__(self, timeout: float = 45.0):
        self.timeout = timeout
        self._orig_socket = None
        self._orig_timeout: float | None = None
        self._orig_send_msg = None
        self._sock_util = None
        self.api = None
        self.context = None
        self.connected = False

    def __enter__(self) -> "BaoSession":
        import baostock as bs
        import baostock.common.context as context
        import baostock.util.socketutil as sock_util

        self.api = bs
        self.context = context
        self._sock_util = sock_util
        self._orig_socket = socket.socket
        self._orig_timeout = socket.getdefaulttimeout()
        self._orig_send_msg = sock_util.send_msg
        socket.socket = _BoundedSocket
        socket.setdefaulttimeout(self.timeout)
        session = self
        orig_send_msg = self._orig_send_msg

        def _guarded_send_msg(msg: str):
            resp = orig_send_msg(msg)
            if resp is None or (isinstance(resp, str) and resp.strip() == ""):
                session.close()
                raise ConnectionError("BaoStock socket send_msg failed; closed socket for reconnect")
            return resp

        sock_util.send_msg = _guarded_send_msg
        return self

    def close(self) -> None:
        if self.context is not None:
            conn = getattr(self.context, "default_socket", None)
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
        self.connected = False

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.connected and self.api is not None:
                try:
                    self.api.logout()
                except Exception:
                    pass
            self.close()
        finally:
            if self._sock_util is not None and self._orig_send_msg is not None:
                self._sock_util.send_msg = self._orig_send_msg
            if self._orig_socket is not None:
                socket.socket = self._orig_socket
            socket.setdefaulttimeout(self._orig_timeout)

    def query(self, method: str, label: str, *args, empty: bool = False, **kwargs) -> pd.DataFrame:
        for attempt in range(4):
            try:
                if not self.connected:
                    res = self.api.login()
                    if res.error_code != "0":
                        raise ConnectionError(f"BaoStock login failed: {res.error_code} {res.error_msg}")
                    self.connected = True
                rs = getattr(self.api, method)(*args, **kwargs)
                if rs.error_code != "0":
                    raise ValueError(f"{label}: {rs.error_code} {rs.error_msg}")
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rs.error_code != "0":
                    raise ValueError(f"{label}: {rs.error_code} {rs.error_msg}")
                if not empty and not rows:
                    raise ValueError(f"{label}: empty response")
                return pd.DataFrame(rows, columns=rs.fields)
            except (ValueError, OSError, ConnectionError) as err:
                self.close()
                if attempt == 3:
                    raise RuntimeError(f"{label}: {err}") from err
                time.sleep(min(6, attempt + 1))


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


def code_to_tx_symbol(code: str) -> str:
    raw = str(code).strip().lower().replace(".", "")
    if raw.startswith(("sh", "sz", "bj")) and len(raw) >= 8:
        return raw[:2] + raw[2:].zfill(6)
    digits = code_to_digits(code)
    if digits.startswith(("6", "5")):
        return f"sh{digits}"
    if digits.startswith(("8", "4", "92")):
        return f"bj{digits}"
    return f"sz{digits}"


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

    # Default: BaoStock true Point-in-Time weekly snapshots matching schedule.parquet (2015-01 .. 2025-12)
    sched_path = raw_dir / "schedule.parquet"
    if not sched_path.exists():
        raise FileNotFoundError("Missing schedule.parquet; run --step index first.")
    sched = pd.read_parquet(sched_path, columns=["signal_date"])
    sig_days = pd.to_datetime(sched["signal_date"]).sort_values().unique()
    snap_dates = [
        pd.Timestamp(d) for d in sig_days if pd.Timestamp("2015-01-01") <= pd.Timestamp(d) <= pd.Timestamp("2025-12-31")
    ]

    pending = [d for d in snap_dates if force or not (mem_dir / f"{d.strftime('%Y-%m-%d')}.parquet").exists()]
    if pending:
        with BaoSession() as session:
            for idx, d in enumerate(pending, 1):
                tag = d.strftime("%Y-%m-%d")
                target = mem_dir / f"{tag}.parquet"
                df = session.query("query_zz500_stocks", f"zz500 {tag}", date=tag, empty=False)
                df["snapshot_date"] = d
                df["updateDate"] = pd.to_datetime(df["updateDate"], errors="coerce")
                df["code"] = df["code"].astype(str).map(normalize_a_share_code)
                _atomic_parquet(df, target)
                if idx % 50 == 0 or idx == len(pending):
                    print(f"[members] BaoStock weekly PIT snapshot {idx}/{len(pending)} ({tag})", flush=True)
                time.sleep(0.02)

    meta_path.write_text(
        json.dumps({"source": "baostock", "pit_members": True, "frequency": "weekly", "snapshots": len(snap_dates)}, indent=2),
        encoding="utf-8",
    )
    print(f"[members] Ready {len(snap_dates)} BaoStock weekly PIT CSI 500 snapshots (pit_members=True)", flush=True)
    return mem_dir


def collect_unique_member_codes(raw_dir: Path) -> list[str]:
    mem_dir = raw_dir / "members"
    meta_path = mem_dir / "metadata.json"
    files: list[Path] = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("pit_members") is True:
            files = sorted(mem_dir.glob("20*.parquet"))
        elif meta.get("pit_members") is False and (mem_dir / "csindex_latest.parquet").exists():
            files = [mem_dir / "csindex_latest.parquet"]
    if not files:
        files = sorted(mem_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No member parquet files in {mem_dir}; run --step members first.")
    codes: set[str] = set()
    for f in files:
        df = pd.read_parquet(f, columns=["code"])
        codes.update(df["code"].astype(str).tolist())
    return sorted(codes)


def collect_member_date_spans(
    raw_dir: Path,
    start_date: str = "2014-06-01",
    end_date: str = "2025-12-31",
    warmup_days: int = 420,
    tail_days: int = 60,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return exact [need_start, need_end] window per member code to avoid needless history fetches."""
    start_ts = pd.Timestamp(_to_iso_date(start_date))
    end_ts = pd.Timestamp(_to_iso_date(end_date))
    mem_dir = raw_dir / "members"
    meta_path = mem_dir / "metadata.json"
    pit_members = False
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pit_members = bool(meta.get("pit_members", False))

    if not pit_members:
        return {code: (start_ts, end_ts) for code in collect_unique_member_codes(raw_dir)}

    files = sorted(mem_dir.glob("20*.parquet"))
    if not files:
        raise FileNotFoundError(f"No PIT member parquet files in {mem_dir}; run --step members first.")

    min_snap: dict[str, pd.Timestamp] = {}
    max_snap: dict[str, pd.Timestamp] = {}
    for f in files:
        cols = pd.read_parquet(f)
        if "code" not in cols.columns or cols.empty:
            continue
        if "snapshot_date" in cols.columns:
            snap_vals = pd.to_datetime(cols["snapshot_date"], errors="coerce").fillna(pd.Timestamp(f.stem))
        else:
            snap_vals = pd.Series(pd.Timestamp(f.stem), index=cols.index)
        grouped = pd.DataFrame({"code": cols["code"].astype(str), "snap": snap_vals}).groupby("code")["snap"].agg(["min", "max"])
        for code, row in grouped.iterrows():
            lo = pd.Timestamp(row["min"])
            hi = pd.Timestamp(row["max"])
            if code not in min_snap or lo < min_snap[code]:
                min_snap[code] = lo
            if code not in max_snap or hi > max_snap[code]:
                max_snap[code] = hi

    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for code in sorted(min_snap):
        need_start = max(start_ts, min_snap[code] - pd.Timedelta(days=warmup_days))
        need_end = min(end_ts, max_snap[code] + pd.Timedelta(days=tail_days))
        if need_start > need_end:
            need_start = start_ts
        spans[code] = (need_start, need_end)
    return spans


def _group_missing_year_chunks(
    missing_years: list[int],
    need_start: pd.Timestamp,
    need_end: pd.Timestamp,
    max_years: int = 5,
) -> list[tuple[str, str]]:
    """Group sorted missing years into <=max_years contiguous ISO date ranges clipped to [need_start, need_end]."""
    if not missing_years:
        return []
    years = sorted(set(missing_years))
    runs: list[list[int]] = []
    cur: list[int] = []
    for yr in years:
        if not cur or (yr == cur[-1] + 1 and len(cur) < max_years):
            cur.append(yr)
        else:
            runs.append(cur)
            cur = [yr]
    if cur:
        runs.append(cur)

    chunks: list[tuple[str, str]] = []
    for run in runs:
        lo = max(need_start, pd.Timestamp(run[0], 1, 1))
        hi = min(need_end, pd.Timestamp(run[-1], 12, 31))
        if lo <= hi:
            chunks.append((lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")))
    return chunks


def _load_v3_cached_years(
    v3_cache_dir: Path | None,
    code: str,
    need_years: list[int],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], set[int]]:
    raw_parts: list[pd.DataFrame] = []
    hfq_parts: list[pd.DataFrame] = []
    cached_years: set[int] = set()
    if v3_cache_dir is None or not v3_cache_dir.exists():
        return raw_parts, hfq_parts, cached_years

    v3_raw = v3_cache_dir / "raw"
    v3_adj = v3_cache_dir / "adjusted"
    if not v3_raw.exists() or not v3_adj.exists():
        return raw_parts, hfq_parts, cached_years

    for yr in need_years:
        rf = v3_raw / str(yr) / f"{code}.parquet"
        af = v3_adj / str(yr) / f"{code}.parquet"
        if rf.exists() and af.exists():
            try:
                r_df = pd.read_parquet(rf)
                a_df = pd.read_parquet(af)
                if "date" in r_df.columns:
                    r_df["date"] = pd.to_datetime(r_df["date"], errors="coerce")
                if "date" in a_df.columns:
                    a_df["date"] = pd.to_datetime(a_df["date"], errors="coerce")
                raw_parts.append(r_df)
                hfq_parts.append(a_df)
                cached_years.add(yr)
            except Exception:
                continue
    return raw_parts, hfq_parts, cached_years


def _validate_raw_hfq_consistency(raw_df: pd.DataFrame, hfq_df: pd.DataFrame, code: str) -> None:
    n_raw, n_hfq = len(raw_df), len(hfq_df)
    if abs(n_raw - n_hfq) > 5:
        raise ValueError(f"{code}: raw/hfq row mismatch ({n_raw} vs {n_hfq})")
    if (n_raw == 0) != (n_hfq == 0) and max(n_raw, n_hfq) > 5:
        raise ValueError(f"{code}: empty vs non-empty raw/hfq mismatch ({n_raw} vs {n_hfq})")
    if n_raw > 0 and n_hfq > 0:
        gap_days = abs((pd.Timestamp(raw_df["date"].max()) - pd.Timestamp(hfq_df["date"].max())).days)
        if gap_days > 10:
            raise ValueError(f"{code}: raw/hfq max date mismatch ({raw_df['date'].max()} vs {hfq_df['date'].max()})")


def _is_cached_code_valid(rf: Path, hf: Path, vf: Path) -> bool:
    if not (rf.exists() and hf.exists() and vf.exists()):
        return False
    try:
        r = pd.read_parquet(rf, columns=["date"])
        h = pd.read_parquet(hf, columns=["date"])
        _validate_raw_hfq_consistency(r, h, rf.stem)
        return True
    except Exception:
        return False


def _standardize_bs_raw(
    df: pd.DataFrame,
    code: str,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_cols = [
        "date", "code", "open", "close", "high", "low", "preclose",
        "volume", "amount", "turn", "change_amt", "pct_chg", "tradestatus", "is_st",
    ]
    val_cols = ["date", "code", "pe_ttm", "pb_mrq", "ps_ttm", "float_mv"]
    if df is None or df.empty:
        return pd.DataFrame(columns=raw_cols), pd.DataFrame(columns=val_cols)

    work = df.copy()
    if "adjustflag" in work.columns:
        work = work.loc[work["adjustflag"].astype(str).str.strip().eq("3")].copy()
    if work.empty:
        return pd.DataFrame(columns=raw_cols), pd.DataFrame(columns=val_cols)

    work["_dt"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.loc[work["_dt"].notna()]
    if end_date is not None:
        work = work.loc[work["_dt"].le(pd.Timestamp(_to_iso_date(end_date)))]
    work = work.sort_values("_dt").drop_duplicates("_dt").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=raw_cols), pd.DataFrame(columns=val_cols)

    def num(col: str) -> pd.Series:
        if col not in work.columns:
            return pd.Series(np.nan, index=work.index, dtype="float64")
        return pd.to_numeric(work[col], errors="coerce")

    close_s = num("close")
    preclose_s = num("preclose")
    amount_s = num("amount")
    turn_s = num("turn")
    float_mv_s = pd.Series(
        np.where(turn_s.gt(0) & turn_s.le(100) & amount_s.gt(0), amount_s / (turn_s / 100.0), np.nan),
        index=work.index,
        dtype="float64",
    )

    raw_out = pd.DataFrame({
        "date": work["_dt"],
        "code": code,
        "open": num("open"),
        "close": close_s,
        "high": num("high"),
        "low": num("low"),
        "preclose": preclose_s,
        "volume": num("volume"),
        "amount": amount_s,
        "turn": turn_s,
        "change_amt": close_s - preclose_s,
        "pct_chg": num("pctChg"),
        "tradestatus": num("tradestatus"),
        "is_st": num("isST"),
    })[raw_cols]

    val_out = pd.DataFrame({
        "date": work["_dt"],
        "code": code,
        "pe_ttm": num("peTTM"),
        "pb_mrq": num("pbMRQ"),
        "ps_ttm": num("psTTM"),
        "float_mv": float_mv_s,
    })[val_cols]
    return raw_out, val_out


def _standardize_bs_hfq(
    df: pd.DataFrame,
    code: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    cols = ["date", "code", "open", "close", "high", "low", "volume", "amount", "turn", "change_amt", "pct_chg"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    if "adjustflag" in work.columns:
        work = work.loc[work["adjustflag"].astype(str).str.strip().eq("1")].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)

    work["_dt"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.loc[work["_dt"].notna()]
    if end_date is not None:
        work = work.loc[work["_dt"].le(pd.Timestamp(_to_iso_date(end_date)))]
    work = work.sort_values("_dt").drop_duplicates("_dt").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=cols)

    def num(col: str) -> pd.Series:
        if col not in work.columns:
            return pd.Series(np.nan, index=work.index, dtype="float64")
        return pd.to_numeric(work[col], errors="coerce")

    return pd.DataFrame({
        "date": work["_dt"],
        "code": code,
        "open": num("open"),
        "close": num("close"),
        "high": num("high"),
        "low": num("low"),
        "volume": num("volume"),
        "amount": num("amount"),
        "turn": num("turn"),
        "change_amt": float("nan"),
        "pct_chg": float("nan"),
    })[cols]


def _standardize_tx_bars(df: pd.DataFrame, code: str) -> pd.DataFrame:
    cols = ["date", "code", "open", "close", "high", "low", "volume", "amount", "turn", "change_amt", "pct_chg"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "code": code,
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
        "amount": pd.to_numeric(df["amount"], errors="coerce"),
        "turn": pd.to_numeric(df["turnover"], errors="coerce") * 100.0,
        "change_amt": float("nan"),
        "pct_chg": float("nan"),
    })
    return out[cols].dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _fetch_bs_bars_shard(
    shard_tasks: list[tuple[str, str, str, bool]],
    raw_dir: Path,
    end_iso: str,
    sleep_sec: float,
    v3_cache_dir: Path | None,
    worker_id: int = 0,
) -> list[dict[str, str]]:
    """Worker routine to fetch missing <=5y BaoStock bar chunks and splice with any v3 cache."""
    raw_out = raw_dir / "bars_raw"
    hfq_out = raw_dir / "bars_hfq"
    val_out = raw_dir / "valuations"
    failed: list[dict[str, str]] = []
    prefix = f"[bars:w{worker_id}]" if worker_id > 0 else "[bars]"

    with BaoSession(timeout=45.0) as session:
        for idx, (code, s_iso, e_iso, allow_v3) in enumerate(shard_tasks, 1):
            need_start = pd.Timestamp(s_iso)
            need_end = pd.Timestamp(e_iso)
            need_years = list(range(need_start.year, need_end.year + 1))

            raw_parts: list[pd.DataFrame] = []
            hfq_parts: list[pd.DataFrame] = []
            cached_years: set[int] = set()
            if allow_v3 and v3_cache_dir is not None:
                raw_parts, hfq_parts, cached_years = _load_v3_cached_years(v3_cache_dir, code, need_years)

            missing_years = [y for y in need_years if y not in cached_years]
            chunks = _group_missing_year_chunks(missing_years, need_start, need_end, max_years=5)
            try:
                for c_start, c_end in chunks:
                    df_bs_raw = session.query(
                        "query_history_k_data_plus",
                        f"{code} raw+val {c_start}..{c_end}",
                        code,
                        BS_RAW_FIELDS,
                        start_date=c_start,
                        end_date=c_end,
                        frequency="d",
                        adjustflag="3",
                        empty=True,
                    )
                    if not df_bs_raw.empty and "date" in df_bs_raw.columns:
                        df_bs_raw["date"] = pd.to_datetime(df_bs_raw["date"], errors="coerce")
                        raw_parts.append(df_bs_raw)

                    df_bs_hfq = session.query(
                        "query_history_k_data_plus",
                        f"{code} hfq {c_start}..{c_end}",
                        code,
                        BS_HFQ_FIELDS,
                        start_date=c_start,
                        end_date=c_end,
                        frequency="d",
                        adjustflag="1",
                        empty=True,
                    )
                    if not df_bs_hfq.empty and "date" in df_bs_hfq.columns:
                        df_bs_hfq["date"] = pd.to_datetime(df_bs_hfq["date"], errors="coerce")
                        hfq_parts.append(df_bs_hfq)

                    if sleep_sec > 0:
                        time.sleep(sleep_sec)

                raw_cat = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
                hfq_cat = pd.concat(hfq_parts, ignore_index=True) if hfq_parts else pd.DataFrame()
                raw_df, val_df = _standardize_bs_raw(raw_cat, code, end_date=end_iso)
                hfq_df = _standardize_bs_hfq(hfq_cat, code, end_date=end_iso)
                _validate_raw_hfq_consistency(raw_df, hfq_df, code)

                _atomic_parquet(raw_df, raw_out / f"{code}.parquet")
                _atomic_parquet(val_df, val_out / f"{code}.parquet")
                _atomic_parquet(hfq_df, hfq_out / f"{code}.parquet")
            except Exception as exc:
                failed.append({"code": code, "error": str(exc)})

            if idx % 25 == 0 or idx == len(shard_tasks):
                print(f"{prefix} BaoStock progress {idx}/{len(shard_tasks)} (failed={len(failed)})", flush=True)

    return failed


def fetch_bars(
    raw_dir: Path,
    start_date: str = "20140601",
    end_date: str = "20251231",
    source: str = "baostock",
    force: bool = False,
    sleep_sec: float = 0.02,
    v3_cache_dir: Path | None = DEFAULT_V3_CACHE_DIR,
    workers: int = 1,
) -> None:
    """Fetch raw and hfq daily bars (plus valuations when using BaoStock) for all CSI 500 constituent codes."""
    raw_out = raw_dir / "bars_raw"
    hfq_out = raw_dir / "bars_hfq"
    val_out = raw_dir / "valuations"
    raw_out.mkdir(parents=True, exist_ok=True)
    hfq_out.mkdir(parents=True, exist_ok=True)

    if source == "baostock":
        val_out.mkdir(parents=True, exist_ok=True)
        start_iso = _to_iso_date(start_date)
        end_iso = _to_iso_date(end_date)
        spans = collect_member_date_spans(raw_dir, start_date=start_iso, end_date=end_iso)
        codes = sorted(spans)

        bad_cached = [
            c
            for c in codes
            if not force
            and (raw_out / f"{c}.parquet").exists()
            and (hfq_out / f"{c}.parquet").exists()
            and (val_out / f"{c}.parquet").exists()
            and not _is_cached_code_valid(raw_out / f"{c}.parquet", hfq_out / f"{c}.parquet", val_out / f"{c}.parquet")
        ]
        if bad_cached:
            print(
                f"[bars] Re-queueing {len(bad_cached)} truncated/mismatched cached codes for repair: {bad_cached[:10]}",
                flush=True,
            )

        pending = [
            c
            for c in codes
            if force or not _is_cached_code_valid(raw_out / f"{c}.parquet", hfq_out / f"{c}.parquet", val_out / f"{c}.parquet")
        ]
        if not pending:
            print(f"[bars] All {len(codes)} codes already cached & verified (bars_raw, bars_hfq, valuations)!", flush=True)
            return

        n_v3_full = 0
        net_pending: list[tuple[str, str, str, bool]] = []
        for code in pending:
            need_start, need_end = spans[code]
            s_iso = need_start.strftime("%Y-%m-%d")
            e_iso = need_end.strftime("%Y-%m-%d")
            need_years = list(range(need_start.year, need_end.year + 1))

            if v3_cache_dir is not None:
                raw_parts, hfq_parts, cached_years = _load_v3_cached_years(v3_cache_dir, code, need_years)
                missing_years = [y for y in need_years if y not in cached_years]
                if not missing_years:
                    try:
                        raw_cat = pd.concat(raw_parts, ignore_index=True)
                        hfq_cat = pd.concat(hfq_parts, ignore_index=True)
                        raw_df, val_df = _standardize_bs_raw(raw_cat, code, end_date=end_iso)
                        hfq_df = _standardize_bs_hfq(hfq_cat, code, end_date=end_iso)
                        _validate_raw_hfq_consistency(raw_df, hfq_df, code)
                        _atomic_parquet(raw_df, raw_out / f"{code}.parquet")
                        _atomic_parquet(val_df, val_out / f"{code}.parquet")
                        _atomic_parquet(hfq_df, hfq_out / f"{code}.parquet")
                        n_v3_full += 1
                        continue
                    except Exception:
                        net_pending.append((code, s_iso, e_iso, False))
                        continue

            net_pending.append((code, s_iso, e_iso, True))

        if n_v3_full > 0:
            print(f"[bars] Imported {n_v3_full} fully-cached codes locally from v3 cache", flush=True)

        failed: list[dict[str, str]] = []
        if net_pending:
            n_workers = max(1, min(int(workers), len(net_pending)))
            print(f"[bars] Fetching {len(net_pending)} codes via BaoStock (chunks<=5y, workers={n_workers})...", flush=True)
            if n_workers == 1:
                failed = _fetch_bs_bars_shard(net_pending, raw_dir, end_iso, sleep_sec, v3_cache_dir, worker_id=0)
            else:
                shards = [net_pending[i::n_workers] for i in range(n_workers)]
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futs = [
                        pool.submit(_fetch_bs_bars_shard, shard, raw_dir, end_iso, sleep_sec, v3_cache_dir, i + 1)
                        for i, shard in enumerate(shards)
                    ]
                    for fut in futs:
                        failed.extend(fut.result())

        err_path = raw_dir / "bars_errors.csv"
        if failed:
            pd.DataFrame(failed).to_csv(err_path, index=False, encoding="utf-8-sig")
            print(f"[bars] Finished with {len(failed)} failed codes logged to {err_path}", flush=True)
        else:
            if err_path.exists():
                err_path.unlink()
            print(f"[bars] All {len(codes)} codes ready via BaoStock (bars_raw, bars_hfq, valuations)!", flush=True)
        return

    import akshare as ak

    codes = collect_unique_member_codes(raw_dir)
    failed = []
    for idx, code in enumerate(codes, 1):
        rf = raw_out / f"{code}.parquet"
        hf = hfq_out / f"{code}.parquet"
        if rf.exists() and hf.exists() and not force:
            continue

        tx_sym = code_to_tx_symbol(code)
        ok = False
        for attempt in range(1, 4):
            try:
                if not rf.exists() or force:
                    df_raw = ak.stock_zh_a_hist_tx(symbol=tx_sym, start_date=start_date, end_date=end_date, adjust="")
                    _atomic_parquet(_standardize_tx_bars(df_raw, code), rf)
                if not hf.exists() or force:
                    df_hfq = ak.stock_zh_a_hist_tx(symbol=tx_sym, start_date=start_date, end_date=end_date, adjust="hfq")
                    _atomic_parquet(_standardize_tx_bars(df_hfq, code), hf)
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
    source: str = "baostock",
    force: bool = False,
    sleep_sec: float = 0.02,
) -> None:
    """Fetch daily valuation metrics (PE_TTM, PB_MRQ, FloatMV) via BaoStock or AKShare fallback."""
    val_dir = raw_dir / "valuations"
    val_dir.mkdir(parents=True, exist_ok=True)

    if source == "baostock":
        end_iso = _to_iso_date(end_date)
        spans = collect_member_date_spans(raw_dir, start_date="2014-06-01", end_date=end_iso)
        codes = sorted(spans)
        pending = [c for c in codes if force or not (val_dir / f"{c}.parquet").exists()]
        if not pending:
            print(f"[valuations] All {len(codes)} codes already populated by BaoStock bars step!", flush=True)
            return

        failed = []
        with BaoSession(timeout=45.0) as session:
            for idx, code in enumerate(pending, 1):
                vf = val_dir / f"{code}.parquet"
                need_start, need_end = spans[code]
                need_years = list(range(need_start.year, need_end.year + 1))
                chunks = _group_missing_year_chunks(need_years, need_start, need_end, max_years=5)
                try:
                    parts: list[pd.DataFrame] = []
                    for c_start, c_end in chunks:
                        df_bs_raw = session.query(
                            "query_history_k_data_plus",
                            f"{code} val {c_start}..{c_end}",
                            code,
                            BS_RAW_FIELDS,
                            start_date=c_start,
                            end_date=c_end,
                            frequency="d",
                            adjustflag="3",
                            empty=True,
                        )
                        if not df_bs_raw.empty and "date" in df_bs_raw.columns:
                            df_bs_raw["date"] = pd.to_datetime(df_bs_raw["date"], errors="coerce")
                            parts.append(df_bs_raw)
                        if sleep_sec > 0:
                            time.sleep(sleep_sec)
                    cat = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                    _, val_df = _standardize_bs_raw(cat, code, end_date=end_iso)
                    _atomic_parquet(val_df, vf)
                except Exception as exc:
                    failed.append({"code": code, "error": str(exc)})

                if idx % 25 == 0 or idx == len(pending):
                    print(f"[valuations] BaoStock progress {idx}/{len(pending)} (total={len(codes)}, failed={len(failed)})", flush=True)

        err_path = raw_dir / "valuations_errors.csv"
        if failed:
            pd.DataFrame(failed).to_csv(err_path, index=False, encoding="utf-8-sig")
            print(f"[valuations] Finished with {len(failed)} failed codes logged to {err_path}", flush=True)
        else:
            if err_path.exists():
                err_path.unlink()
            print(f"[valuations] All {len(codes)} codes ready via BaoStock!", flush=True)
        return

    import akshare as ak

    codes = collect_unique_member_codes(raw_dir)
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
    with BaoSession(timeout=20.0) as session:
        df_2015 = session.query("query_zz500_stocks", "zz500 2015-01-09", date="2015-01-09", empty=False)
        df_2024 = session.query("query_zz500_stocks", "zz500 2024-01-05", date="2024-01-05", empty=False)
        print(
            f"   BaoStock PIT query_zz500_stocks OK: 2015-01-09={len(df_2015)} rows, 2024-01-05={len(df_2024)} rows",
            flush=True,
        )

        df_raw = session.query(
            "query_history_k_data_plus",
            "sz.000001 raw+val 2015-01",
            "sz.000001",
            BS_RAW_FIELDS,
            start_date="2015-01-05",
            end_date="2015-01-30",
            frequency="d",
            adjustflag="3",
            empty=False,
        )
        df_hfq = session.query(
            "query_history_k_data_plus",
            "sz.000001 hfq 2015-01",
            "sz.000001",
            BS_HFQ_FIELDS,
            start_date="2015-01-05",
            end_date="2015-01-30",
            frequency="d",
            adjustflag="1",
            empty=False,
        )
        raw_std, val_std = _standardize_bs_raw(df_raw, "sz.000001")
        hfq_std = _standardize_bs_hfq(df_hfq, "sz.000001")
        print(
            f"   BaoStock bars sz.000001 (2015-01): raw={len(raw_std)}, hfq={len(hfq_std)}, "
            f"preclose={int(raw_std['preclose'].notna().sum())}, pe_ttm={int(val_std['pe_ttm'].notna().sum())}, "
            f"pb_mrq={int(val_std['pb_mrq'].notna().sum())}, float_mv={int(val_std['float_mv'].notna().sum())}",
            flush=True,
        )


def probe_apis() -> None:
    import akshare as ak

    print("[1/3] Probing AKShare CSI 500 index (H00905 & 000905)...", flush=True)
    try:
        df_tr = ak.index_hist_csindex(symbol="H00905", start_date="20240101", end_date="20240131")
        df_px = ak.index_hist_csindex(symbol="000905", start_date="20240101", end_date="20240131")
        print(f"   H00905 rows={len(df_tr)}, 000905 rows={len(df_px)}", flush=True)
    except Exception as exc:
        print(f"   [FAIL] {exc}", flush=True)

    print("[2/3] Probing BaoStock PIT CSI 500 constituents & 2015 daily bars/valuations (30s timeout)...", flush=True)
    try:
        probe_env = {**os.environ, "PYTHONPATH": str(project_root() / "src")}
        res = subprocess.run(
            [sys.executable, "-B", "-m", "csi500_alpha.fetch_csi500", "probe_bs"],
            cwd=project_root(), env=probe_env, timeout=30, check=False,
        )
        if res.returncode != 0:
            print("   [WARN] BaoStock PIT probe failed; AKShare fallback available (--members-source csindex --data-source akshare).", flush=True)
    except subprocess.TimeoutExpired:
        print("   [WARN] BaoStock PIT probe timed out (30s); AKShare fallback available.", flush=True)

    print("[3/3] Probing AKShare fallback APIs (stock_zh_a_hist_tx & index_stock_cons_csindex)...", flush=True)
    try:
        df_b = ak.stock_zh_a_hist_tx(symbol="sz000001", start_date="20240101", end_date="20240131", adjust="")
        df_c = ak.index_stock_cons_csindex(symbol="000905")
        print(f"   TX bars rows={len(df_b)}, CSIndex members rows={len(df_c)}", flush=True)
    except Exception as exc:
        print(f"   [FAIL] {exc}", flush=True)


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
