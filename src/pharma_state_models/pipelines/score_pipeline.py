from __future__ import annotations

import argparse

from pharma_state_models.config import load_config


def run_score_pipeline(config_path: str) -> None:
    """
    Orchestrates:
    1) load model
    2) load latest sequence input
    3) predict
    4) write output for downstream consumers
    """
    _ = load_config(config_path)
    raise NotImplementedError("Implement scoring pipeline.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score LSTM model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    run_score_pipeline(args.config)


if __name__ == "__main__":
    main()

