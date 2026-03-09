from __future__ import annotations

from keras import Input, Model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


def build_lstm_model(
    timesteps: int,
    n_features: int,
    output_names: list[str],
) -> Model:
    inputs = Input(shape=(timesteps, n_features))
    x = LSTM(64)(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)

    outputs = [Dense(1, activation="sigmoid", name=name)(x) for name in output_names]
    model = Model(inputs=inputs, outputs=outputs)
    return model



def compile_model(model: Model, output_names: list[str], learning_rate: float) -> Model:
    """Compile multi-head model with one binary head per target."""
    optimizer = Adam(learning_rate=learning_rate)
    losses = {name: "binary_crossentropy" for name in output_names}
    metrics = {name: ["accuracy"] for name in output_names}
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    return model
