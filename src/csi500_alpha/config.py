"""Configuration loader for csi500-alpha."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (project_root() / "configs" / "experiment.toml")
    with cfg_path.open("rb") as fh:
        return tomllib.load(fh)
