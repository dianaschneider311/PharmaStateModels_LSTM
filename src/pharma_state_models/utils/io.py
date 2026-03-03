from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)

