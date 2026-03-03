from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    id_column: str,
    time_column: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create many-to-one sequence windows:
    X shape: (n_samples, sequence_length, n_features)
    y shape: (n_samples,)
    """
    raise NotImplementedError("Implement sequence building logic.")

