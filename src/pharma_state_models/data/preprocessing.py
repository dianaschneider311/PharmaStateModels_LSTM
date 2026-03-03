from __future__ import annotations

import pandas as pd

from pharma_state_models.data.preprocessing_preliminary import (
    run_preliminary_preprocessing,
)
from pharma_state_models.data.preprocessing_timeseries import run_timeseries_preprocessing


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible entry point.

    Keep this function as an orchestrator once you implement:
    1) preliminary preprocessing
    2) multivariate time-series preprocessing
    """
    _ = run_preliminary_preprocessing
    _ = run_timeseries_preprocessing
    raise NotImplementedError("Implement orchestrated preprocessing logic.")
