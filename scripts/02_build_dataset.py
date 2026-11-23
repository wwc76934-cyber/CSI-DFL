"""Step 2: Build 2014-2025 CSI 500 10-factor panel, benchmarks, and non-return audit."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from csi500_alpha.config import load_config
from csi500_alpha.factors import load_and_build_csi500_dataset


def main() -> None:
    cfg = load_config(ROOT / "configs" / "experiment.toml")
    raw_dir = ROOT / "data" / "raw"
    proc_dir = ROOT / "data" / "processed"
    rep_dir = ROOT / "reports"
    proc_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    print("[dataset] Building CSI 500 10-factor panel and benchmarks...", flush=True)
    panel, bars, audit_df, benchmarks, pit_members = load_and_build_csi500_dataset(cfg, raw_dir)

    panel.to_parquet(proc_dir / "panel_csi500_dev.parquet", index=False)
    bars.to_parquet(proc_dir / "daily_bars_csi500_dev.parquet", index=False)
    benchmarks.to_parquet(proc_dir / "benchmarks_csi500_dev.parquet", index=False)
    audit_df.to_csv(rep_dir / "factor_data_audit.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== Non-Return Factor Audit (pit_members={pit_members}) ===")
    print(audit_df.to_string(index=False))
    print(f"\n[dataset] Saved panel ({len(panel):,} rows, {panel['signal_date'].nunique()} weeks) to {proc_dir}")


if __name__ == "__main__":
    main()
