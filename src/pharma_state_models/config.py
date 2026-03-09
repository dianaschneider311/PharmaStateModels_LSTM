from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PREDICTED_TIME_PERIOD = "202407"
TARGET_COLUMNS = {"ap_segment_trend" : "ap_segment", 
                  "visit1MonthCount" : "visit",
                  "emailOpen1MonthCount": "email"}
TEST_SIZE = 0.2
RANDOM_SEED = 42


@dataclass
class ProjectConfig:
    raw: dict[str, Any]


def load_config(config_path: str | Path | None) -> ProjectConfig:
    """Load YAML configuration file."""
    project_root = Path(__file__).resolve().parents[2]
    default_config_path = project_root / "configs" / "base.yaml"

    if config_path is None:
        path = default_config_path
    else:
        input_path = Path(config_path)
        path = (
            input_path
            if input_path.is_absolute()
            else (project_root / input_path)
        )

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(raw=raw)
