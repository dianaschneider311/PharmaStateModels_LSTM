from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    raw: dict[str, Any]


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load YAML configuration file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(raw=raw)

