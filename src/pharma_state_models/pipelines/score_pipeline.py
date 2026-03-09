from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pharma_state_models.config import load_config
from pharma_state_models.utils.io import write_dataframe


def build_account_profiles(
    predictions_df: pd.DataFrame,
    id_column: str,
    channel_thresholds: dict[str, float],
) -> pd.DataFrame:
    prob_columns = [c for c in predictions_df.columns if c.endswith("_prob1")]
    if not prob_columns:
        raise ValueError("No probability columns found. Expected columns ending with '_prob1'.")

    channels = [c.removesuffix("_prob1") for c in prob_columns]
    scores = predictions_df[prob_columns].to_numpy()
    best_idx = np.argmax(scores, axis=1)
    top_channel = np.array([channels[i] for i in best_idx], dtype=object)
    top_score = scores[np.arange(len(predictions_df)), best_idx]

    out = predictions_df.copy()
    out["top_channel"] = top_channel
    out["top_channel_score"] = top_score

    for channel in channels:
        threshold = float(channel_thresholds.get(channel, 0.5))
        out[f"recommend_{channel}"] = out[f"{channel}_prob1"] >= threshold

    top_thresholds = np.array([float(channel_thresholds.get(channel, 0.5)) for channel in top_channel])
    out["recommended_channel"] = np.where(top_score >= top_thresholds, top_channel, "none")

    out["propensity_confidence"] = pd.cut(
        out["top_channel_score"],
        bins=[-np.inf, 0.40, 0.70, np.inf],
        labels=["low", "medium", "high"],
    )

    profile_columns = [id_column]
    if "dataset_split" in out.columns:
        profile_columns.append("dataset_split")
    profile_columns += prob_columns
    profile_columns += [
        "top_channel",
        "top_channel_score",
        "recommended_channel",
        "propensity_confidence",
    ]
    profile_columns += [f"recommend_{channel}" for channel in channels]

    return out[profile_columns]


def run_score_pipeline(config_path: str) -> None:
    """
    Orchestrates:
    1) load model
    2) load latest sequence input
    3) predict
    4) write output for downstream consumers
    """
    config = load_config(config_path).raw
    project_root = Path(__file__).resolve().parents[3]

    data_cfg = config.get("data", {})
    scoring_cfg = config.get("scoring", {})

    predictions_path = Path(data_cfg.get("output_predictions_path", "data/processed/predictions.csv"))
    if not predictions_path.is_absolute():
        predictions_path = project_root / predictions_path
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    output_profile_path = Path(scoring_cfg.get("output_profiles_path", "data/processed/account_profiles.csv"))
    if not output_profile_path.is_absolute():
        output_profile_path = project_root / output_profile_path

    thresholds = scoring_cfg.get("channel_thresholds", {})
    tuned_thresholds_path = Path(
        scoring_cfg.get("tuned_thresholds_path", "data/processed/tuned_thresholds.json")
    )
    if not tuned_thresholds_path.is_absolute():
        tuned_thresholds_path = project_root / tuned_thresholds_path

    use_tuned_thresholds = bool(scoring_cfg.get("use_tuned_thresholds", True))
    thresholds_source = "config.scoring.channel_thresholds"
    if use_tuned_thresholds and tuned_thresholds_path.exists():
        with tuned_thresholds_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        thresholds = {k: float(v) for k, v in loaded.items()}
        thresholds_source = str(tuned_thresholds_path)

    predictions_df = pd.read_csv(predictions_path)

    id_column = "accountUid" if "accountUid" in predictions_df.columns else data_cfg.get("id_column", "accountUid")
    if id_column not in predictions_df.columns:
        raise ValueError(f"ID column '{id_column}' was not found in predictions file.")

    profiles = build_account_profiles(
        predictions_df=predictions_df,
        id_column=id_column,
        channel_thresholds=thresholds,
    )
    write_dataframe(profiles, output_profile_path)

    print(f"Loaded predictions from: {predictions_path}")
    print(f"Using thresholds from: {thresholds_source}")
    print(f"Saved account profiles to: {output_profile_path}")
    print(f"Profile rows: {len(profiles)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score LSTM model.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    run_score_pipeline(args.config)


if __name__ == "__main__":
    main()
