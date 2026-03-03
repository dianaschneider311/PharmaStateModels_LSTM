from __future__ import annotations

import numpy as np


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Placeholder for MAE/RMSE/correlation metrics."""
    raise NotImplementedError("Implement regression evaluation metrics.")


def evaluate_classification(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Placeholder for ROC-AUC/PR-AUC/accuracy metrics."""
    raise NotImplementedError("Implement classification evaluation metrics.")

