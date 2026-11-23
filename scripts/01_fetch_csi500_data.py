"""User-runnable CLI to probe and fetch CSI 500 raw market data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from csi500_alpha.fetch_csi500 import (
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.08)
    args = parser.parse_args()

    if args.probe:
        probe_apis()
        return

    raw_dir = ROOT / "data" / "raw"
    if args.step in ("index", "all"):
        fetch_index_and_calendar(raw_dir, force=args.force)
    if args.step in ("members", "all"):
        fetch_members(raw_dir, source=args.members_source, force=args.force)
    if args.step in ("bars", "all"):
        fetch_bars(raw_dir, force=args.force, sleep_sec=args.sleep)
    if args.step in ("valuations", "all"):
        fetch_valuations(raw_dir, force=args.force, sleep_sec=args.sleep)

    print_data_summary(raw_dir)


if __name__ == "__main__":
    main()
