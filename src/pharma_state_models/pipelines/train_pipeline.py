from __future__ import annotations

import argparse

from pharma_state_models.config import load_config


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
    _ = load_config(config_path)
    raise NotImplementedError("Implement training pipeline.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    run_train_pipeline(args.config)


if __name__ == "__main__":
    main()

