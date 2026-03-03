# PharmaStateModels_LSTM

Skeleton project for a multivariate many-to-one LSTM workflow that predicts next-period HCP behaviors and propensities across channels.

## Scope
- Build-only skeleton (no business logic implemented)
- Data preparation, sequence building, training, evaluation, and inference placeholders
- Folder layout compatible with incremental step-by-step implementation

## High-Level Workflow
1. Ingest multichannel monthly HCP data
2. Preprocess and normalize features
3. Build rolling sequence windows (11 months history -> t+1 target)
4. Train LSTM model
5. Evaluate by task type (regression or classification)
6. Export predictions and state features for downstream systems

## Project Structure
```text
PharmaStateModels_LSTM/
  configs/
    base.yaml
  data/
    raw/
    processed/
  docs/
    project_overview.md
  knime/
    workflow_notes.md
  src/pharma_state_models/
    data/
      ingestion.py
      preprocessing.py
      schemas.py
    features/
      sequence_builder.py
    models/
      lstm_model.py
    training/
      trainer.py
    evaluation/
      metrics.py
    inference/
      predictor.py
    pipelines/
      train_pipeline.py
      score_pipeline.py
    utils/
      io.py
      logging_utils.py
    config.py
  tests/
    test_smoke.py
  pyproject.toml
```

## Quick Start
```bash
python -m pharma_state_models.pipelines.train_pipeline --config configs/base.yaml
```

Current behavior is intentionally placeholder-only and will raise `NotImplementedError` in unimplemented modules.
