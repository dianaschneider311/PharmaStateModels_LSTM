from __future__ import annotations

import pandas as pd

from pharma_state_models.data.preprocessing_preliminary import build_train_test_data

def run_timeseries_preprocessing():
    df_train, df_test = build_train_test_data()

    print(df_train.shape)
    print(df_test.shape)
    """
    Placeholder for multivariate time-series preparation, for example:
    - Sort by entity/time
    - Handle missing timestamps and values
    - Encode/scale features (training-safe)
    - Prepare data for sequence/window builder
    """
    # continue here

def main():
    print("START", flush=True)
    run_timeseries_preprocessing()

    print("END", flush=True)


if __name__ == "__main__":
    main()
 