from __future__ import annotations

from keras import Model


def build_lstm_model(
    n_features: int,
    sequence_length: int,
    task_type: str = "regression",
):
    """
    Placeholder for model architecture:
    Input -> LSTM -> Dropout -> Dense -> Head
    """
    raise NotImplementedError("Implement LSTM model creation.")


def compile_model(model: Model, task_type: str, learning_rate: float) -> Model:
    """Compile Keras model with task-appropriate loss/metrics."""
    raise NotImplementedError("Implement model compilation.")

