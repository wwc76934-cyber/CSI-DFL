"""User-runnable CLI to probe and fetch CSI 500 raw market data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from csi500_alpha.fetch_csi500 import (
    DEFAULT_V3_CACHE_DIR,
    fetch_bars,
    fetch_index_and_calendar,
    fetch_members,
    fetch_valuations,
    print_data_summary,
    probe_apis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Probe AKShare and BaoStock CSI 500 APIs")
    parser.add_argument("--step", choices=["index", "members", "bars", "valuations", "all"], default="all")
    parser.add_argument("--members-source", choices=["baostock", "csindex"], default="baostock")
    parser.add_argument(
        "--data-source",
        choices=["baostock", "akshare"],
        default="baostock",
        help="Source for daily bars and valuations (default: baostock)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sleep", type=float, default=None, help="Sleep seconds between requests (default: 0.02 baostock, 0.08 akshare)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel BaoStock worker processes for bars step (default: 1; try 2 if desired)")
    parser.add_argument("--no-v3-cache", action="store_true", help="Disable reusing local regime-factor v3 BaoStock cache")
    args = parser.parse_args()

    if args.probe:
        probe_apis()
        return

    sleep_sec = args.sleep if args.sleep is not None else (0.02 if args.data_source == "baostock" else 0.08)
    v3_cache_dir = None if args.no_v3_cache else DEFAULT_V3_CACHE_DIR
    raw_dir = ROOT / "data" / "raw"
    if args.step in ("index", "all"):
        fetch_index_and_calendar(raw_dir, force=args.force)
    if args.step in ("members", "all"):
        fetch_members(raw_dir, source=args.members_source, force=args.force)
    if args.step in ("bars", "all"):
        fetch_bars(
            raw_dir,
            source=args.data_source,
            force=args.force,
            sleep_sec=sleep_sec,
            v3_cache_dir=v3_cache_dir,
            workers=max(1, args.workers),
        )
    if args.step in ("valuations", "all"):
        fetch_valuations(raw_dir, source=args.data_source, force=args.force, sleep_sec=sleep_sec)

    print_data_summary(raw_dir)


if __name__ == "__main__":
    main()
