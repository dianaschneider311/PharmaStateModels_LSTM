from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from pharma_state_models.data.preprocessing_preliminary import build_train_test_data
from pharma_state_models.config import PREDICTED_TIME_PERIOD, TARGET_COLUMNS


def infer_timeseries_shape(feature_columns: pd.Index) -> tuple[int, int]:
    """Infer (timesteps, n_features) from columns named like '<feature>_YYYYMM'."""
    month_tokens = []
    non_ts_columns = []
    for col in feature_columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            month_tokens.append(parts[1])
        else:
            non_ts_columns.append(col)

    if non_ts_columns:
        preview = ", ".join(non_ts_columns[:5])
        raise ValueError(
            f"Found non time-series feature columns after target removal: {preview}"
        )

    timesteps = len(set(month_tokens))
    total_columns = len(feature_columns)
    if timesteps == 0 or total_columns % timesteps != 0:
        raise ValueError(
            f"Cannot infer reshape dimensions: total_columns={total_columns}, "
            f"timesteps={timesteps}"
        )
    n_features = total_columns // timesteps
    return timesteps, n_features


def run_timeseries_preprocessing():
    df_train, df_test = build_train_test_data()

    print(df_train.shape)
    print(df_test.shape)

    X_train_original_full = df_train.drop(columns=["accountUid"]).copy()
    X_test_original_full = df_test.drop(columns=["accountUid"]).copy()

    email_base_col = next(
        base_col for base_col, output_name in TARGET_COLUMNS.items() if output_name == "email"
    )
    email_target_col = f"{email_base_col}_{PREDICTED_TIME_PERIOD}"
    y_email_train = X_train_original_full[email_target_col].astype(int)

    # Reapply SMOTE to improve the email-positive class representation in train.
    smote = SMOTE(random_state=42)
    X_train_fit_full, y_email_resampled = smote.fit_resample(X_train_original_full, y_email_train)
    X_train_fit_full[email_target_col] = y_email_resampled

    class_target_train = pd.DataFrame(index=X_train_fit_full.index)
    class_target_test = pd.DataFrame(index=X_test_original_full.index)

    X_train_output_features = X_train_original_full.copy()
    X_test_output_features = X_test_original_full.copy()

    for base_col in TARGET_COLUMNS.keys():
        col_name = f"{base_col}_{PREDICTED_TIME_PERIOD}"

        class_target_train[col_name] = X_train_fit_full[col_name]
        class_target_test[col_name] = X_test_original_full[col_name]
        X_train_fit_full = X_train_fit_full.drop(columns=[col_name])
        X_train_output_features = X_train_output_features.drop(columns=[col_name])
        X_test_original_full = X_test_original_full.drop(columns=[col_name])
        X_test_output_features = X_test_output_features.drop(columns=[col_name])

    scaler = StandardScaler()
    train_feature_columns = X_train_fit_full.columns
    test_feature_columns = X_test_original_full.columns

    X_train_fit = pd.DataFrame(
        scaler.fit_transform(X_train_fit_full),
        columns=train_feature_columns,
        index=X_train_fit_full.index,
    )
    X_train_output = pd.DataFrame(
        scaler.transform(X_train_output_features),
        columns=X_train_output_features.columns,
        index=X_train_output_features.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test_original_full),
        columns=test_feature_columns,
        index=X_test_original_full.index,
    )

    if not X_train_fit.columns.equals(X_test.columns):
        X_test = X_test.reindex(columns=X_train_fit.columns)
        if X_test.isna().any().any():
            raise ValueError("Train/Test feature columns are misaligned after reindex.")
    if not X_train_fit.columns.equals(X_train_output.columns):
        X_train_output = X_train_output.reindex(columns=X_train_fit.columns)
        if X_train_output.isna().any().any():
            raise ValueError("Train output features are misaligned with model input columns.")

    T, F = infer_timeseries_shape(X_train_fit.columns)
    X_train_3d = X_train_fit.to_numpy().reshape(len(X_train_fit), T, F)
    X_train_output_3d = X_train_output.to_numpy().reshape(len(X_train_output), T, F)
    X_test_3d = X_test.to_numpy().reshape(len(X_test), T, F)
    y_train = {}
    y_test = {}

    for base_col, output_name in TARGET_COLUMNS.items():
        col_name = f"{base_col}_{PREDICTED_TIME_PERIOD}"
        train_values = (class_target_train[col_name].to_numpy() >= 0.5).astype(int)
        test_values = (class_target_test[col_name].to_numpy() >= 0.5).astype(int)
        head_name = "ap_segment_trend" if base_col == "ap_segment_trend" else output_name
        y_train[head_name] = train_values.reshape(-1, 1)
        y_test[head_name] = test_values.reshape(-1, 1)

    train_output_table = pd.concat(
        [df_train[["accountUid"]].reset_index(drop=True), X_train_output_features.reset_index(drop=True)],
        axis=1,
    )
    test_output_table = pd.concat(
        [df_test[["accountUid"]].reset_index(drop=True), X_test_output_features.reset_index(drop=True)],
        axis=1,
    )

    return X_train_3d, y_train, X_test_3d, y_test, X_train_output_3d, train_output_table, test_output_table
    


def main():
    print("START", flush=True)
    X_train_3d, y_train, X_test_3d, y_test, X_train_output_3d, train_output_table, test_output_table = run_timeseries_preprocessing()
    print(X_train_3d.shape, {k: v.shape for k, v in y_train.items()})
    print(X_test_3d.shape, {k: v.shape for k, v in y_test.items()})
    print(X_train_output_3d.shape, train_output_table.shape, test_output_table.shape)
    print("END", flush=True)



if __name__ == "__main__":
    main()
 
