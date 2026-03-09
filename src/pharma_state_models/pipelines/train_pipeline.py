from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pharma_state_models.config import load_config
from pharma_state_models.data.preprocessing_timeseries import run_timeseries_preprocessing
from pharma_state_models.models.lstm_model import build_lstm_model, compile_model
from pharma_state_models.training.trainer import train_model
from pharma_state_models.utils.io import write_dataframe


def build_per_head_sample_weights(y_train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute balanced sample weights independently for each binary head."""
    weights: dict[str, np.ndarray] = {}
    for head_name, target in y_train.items():
        y = target.ravel().astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        n_total = len(y)

        if n_pos == 0 or n_neg == 0:
            weights[head_name] = np.ones(n_total, dtype=np.float32)
            continue

        pos_weight = n_total / (2.0 * n_pos)
        neg_weight = n_total / (2.0 * n_neg)
        head_weights = np.where(y == 1, pos_weight, neg_weight).astype(np.float32)
        weights[head_name] = head_weights
    return weights


def tune_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold in [0.05, 0.95] that maximizes F1 on validation data."""
    candidates = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        y_hat = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_hat, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def run_train_pipeline(config_path: str) -> None:
    """
    Orchestrates:
    1) load data
    2) preprocess
    3) build sequences
    4) build+compile model
    5) train
    6) evaluate
    7) save artifacts
    """
    config = load_config(config_path).raw

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_output,
        train_output_table,
        test_output_table,
    ) = run_timeseries_preprocessing()

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    val_fraction = float(training_cfg.get("validation_split", 0.2))
    indices = np.arange(X_train.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=42,
        shuffle=True,
    )
    X_fit = X_train[train_idx]
    X_val = X_train[val_idx]
    y_fit = {name: target[train_idx] for name, target in y_train.items()}
    y_val = {name: target[val_idx] for name, target in y_train.items()}

    timesteps = X_fit.shape[1]
    n_features = X_fit.shape[2]
    output_names = list(y_fit.keys())
    learning_rate = model_cfg.get("learning_rate", 0.001)

    model = build_lstm_model(
        timesteps=timesteps,
        n_features=n_features,
        output_names=output_names,
    )
    model = compile_model(
        model=model,
        output_names=output_names,
        learning_rate=learning_rate,
    )
    sample_weight = build_per_head_sample_weights(y_fit)

    history = train_model(
        model=model,
        x_train=X_fit,
        y_train=y_fit,
        sample_weight=sample_weight,
        batch_size=training_cfg.get("batch_size", 32),
        epochs=training_cfg.get("epochs", 50),
        validation_split=0.0,
        early_stopping_patience=training_cfg.get("early_stopping_patience", 5),
        validation_data=(X_val, [y_val[name] for name in output_names]),
    )

    test_metrics = model.evaluate(X_test, [y_test[name] for name in output_names], verbose=0, return_dict=True)
    val_preds = model.predict(X_val, verbose=0)
    test_preds = model.predict(X_test, verbose=0)

    if isinstance(val_preds, np.ndarray):
        val_preds = [val_preds]
    if isinstance(test_preds, np.ndarray):
        test_preds = [test_preds]

    per_head_scores: list[tuple[str, float, float, float | None, float | None, float | None]] = []
    for head_name, val_pred, test_pred in zip(model.output_names, val_preds, test_preds):
        y_val_true = y_val[head_name].ravel()
        y_val_prob = val_pred.ravel()
        threshold = tune_threshold_for_f1(y_val_true, y_val_prob)

        y_true = y_test[head_name].ravel()
        y_prob = test_pred.ravel()
        y_hat_05 = (y_prob >= 0.5).astype(int)
        y_hat_tuned = (y_prob >= threshold).astype(int)
        f1_05 = float(f1_score(y_true, y_hat_05, zero_division=0))
        f1_tuned = float(f1_score(y_true, y_hat_tuned, zero_division=0))

        roc_auc: float | None
        pr_auc: float | None
        if np.unique(y_true).size < 2:
            roc_auc = None
            pr_auc = None
        else:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))

        per_head_scores.append((head_name, threshold, f1_05, f1_tuned, roc_auc, pr_auc))

    train_preds = model.predict(X_train_output, verbose=0)
    if isinstance(train_preds, np.ndarray):
        train_preds = [train_preds]

    train_with_probs = train_output_table.copy()
    test_with_probs = test_output_table.copy()
    for head_name, train_pred, test_pred in zip(model.output_names, train_preds, test_preds):
        train_with_probs[f"{head_name}_prob1"] = train_pred.ravel()
        test_with_probs[f"{head_name}_prob1"] = test_pred.ravel()

    train_with_probs["dataset_split"] = "train"
    test_with_probs["dataset_split"] = "test"
    full_output = pd.concat([train_with_probs, test_with_probs], ignore_index=True)

    project_root = Path(__file__).resolve().parents[3]
    output_path_cfg = config.get("data", {}).get("output_predictions_path", "data/processed/predictions.csv")
    output_path = Path(output_path_cfg)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    write_dataframe(full_output, output_path)

    tuned_thresholds = {head_name: float(threshold) for head_name, threshold, *_ in per_head_scores}
    thresholds_path_cfg = config.get("scoring", {}).get(
        "tuned_thresholds_path",
        "data/processed/tuned_thresholds.json",
    )
    thresholds_path = Path(thresholds_path_cfg)
    if not thresholds_path.is_absolute():
        thresholds_path = project_root / thresholds_path
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump(tuned_thresholds, f, indent=2)

    print("Training completed.")
    print("Final training loss:", history.history["loss"][-1])
    print(f"Saved predictions with features to: {output_path}")
    print(f"Saved tuned thresholds to: {thresholds_path}")
    print("Labeled test metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")

    print("Per-head classification quality:")
    for head_name, threshold, f1_05, f1_tuned, roc_auc, pr_auc in per_head_scores:
        roc_auc_text = "N/A (single class in y_true)" if roc_auc is None else f"{roc_auc:.6f}"
        pr_auc_text = "N/A (single class in y_true)" if pr_auc is None else f"{pr_auc:.6f}"
        print(
            f"  {head_name}: threshold={threshold:.2f}, "
            f"F1@0.50={f1_05:.6f}, F1@tuned={f1_tuned:.6f}, "
            f"ROC-AUC={roc_auc_text}, PR-AUC={pr_auc_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM pipeline.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    run_train_pipeline(args.config)


if __name__ == "__main__":
    main()
