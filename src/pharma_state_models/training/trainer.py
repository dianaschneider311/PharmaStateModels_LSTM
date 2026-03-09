from __future__ import annotations

import numpy as np
from collections.abc import Mapping
from keras import Model
from keras.callbacks import EarlyStopping


def train_model(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray | Mapping[str, np.ndarray],
    sample_weight: Mapping[str, np.ndarray] | None,
    batch_size: int,
    epochs: int,
    validation_split: float,
    early_stopping_patience: int,
    validation_data: tuple[np.ndarray, list[np.ndarray]] | None = None,
):
    """Train model and return history object."""
    fit_targets: np.ndarray | list[np.ndarray]
    if isinstance(y_train, Mapping):
        fit_targets = [y_train[name] for name in model.output_names]
    else:
        fit_targets = y_train

    fit_sample_weight: None | list[np.ndarray]
    if sample_weight is None:
        fit_sample_weight = None
    else:
        fit_sample_weight = [sample_weight[name] for name in model.output_names]

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        fit_targets,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split if validation_data is None else 0.0,
        validation_data=validation_data,
        sample_weight=fit_sample_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history
