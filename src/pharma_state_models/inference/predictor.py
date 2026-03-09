from __future__ import annotations

import numpy as np
import pandas as pd
from keras import Model


def predict_next(model: Model, x_input: np.ndarray) -> np.ndarray | list[np.ndarray]:
    """Run next-period predictions."""
    return model.predict(x_input, verbose=0)


def format_predictions(
    hcp_ids: list[str],
    preds: np.ndarray | list[np.ndarray],
    output_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return standard output table for downstream systems."""
    out = pd.DataFrame({"accountUid": hcp_ids})

    if isinstance(preds, list):
        names = output_names if output_names is not None else [f"channel_{i + 1}" for i in range(len(preds))]
        for name, pred in zip(names, preds):
            out[f"{name}_prob1"] = pred.ravel()
        return out

    if preds.ndim == 1:
        out["score"] = preds
        return out

    if preds.shape[1] == 1:
        out["score"] = preds.ravel()
        return out

    names = output_names if output_names is not None else [f"channel_{i + 1}" for i in range(preds.shape[1])]
    for idx, name in enumerate(names):
        out[f"{name}_prob1"] = preds[:, idx]
    return out
