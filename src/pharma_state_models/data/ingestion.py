from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

def load_raw_data():
    cj = pd.read_csv(RAW_DIR / "cj_2023.csv")
    adlx = pd.read_csv(RAW_DIR / "adlx.csv")
    adlh = pd.read_csv(RAW_DIR / "adlh.csv")
    return cj, adlx, adlh
 
