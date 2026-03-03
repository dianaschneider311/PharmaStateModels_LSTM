from __future__ import annotations

import numpy as np
import pandas as pd
from keras import Model


def predict_next(model: Model, x_input: np.ndarray) -> np.ndarray:
    """Run next-period predictions."""
    raise NotImplementedError("Implement model inference.")


def format_predictions(hcp_ids: list[str], preds: np.ndarray) -> pd.DataFrame:
    """Return standard output table for downstream systems."""
    raise NotImplementedError("Implement prediction formatting.")

