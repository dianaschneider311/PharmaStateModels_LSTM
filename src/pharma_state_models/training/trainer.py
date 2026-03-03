from __future__ import annotations

import numpy as np
from keras import Model


def train_model(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    epochs: int,
    validation_split: float,
    early_stopping_patience: int,
):
    """Train model and return history object."""
    raise NotImplementedError("Implement training loop with early stopping.")

