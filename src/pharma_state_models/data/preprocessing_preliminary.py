from __future__ import annotations

import pandas as pd
import numpy as np
from pharma_state_models.data.ingestion import load_raw_data

predicted_time_period = "202311"
target_columns = ["ap_segment_trend","visit1MonthCount","emailOpen1MonthCount"]
test_size = 0.2

state_columns = {
    "visit": [
        "yearMonth",
        "visit1MonthCount",
        "visit1MonthWeekCount",
        "visit1MonthDayCount",
        "meanVisitGapsOver1Month",
        "meanVisitGapsOver3Month",
        "meanVisitGapsOver6Month",
        "accountUid",
        "criVisitScore",
        "criSuggestionVisitScore"
    ],
    "email": [
        "yearMonth",
        "emailSent1MonthCount",
        "emailSent1MonthWeekCount",
        "emailSent1MonthDayCount",
        "meanEmailSentGapsOver1Month",
        "meanEmailSentGapsOver3Month",
        "meanEmailSentGapsOver6Month",
        "emailOpen1MonthCount",
        "emailOpen1MonthWeekCount",
        "emailOpen1MonthDayCount",
        "meanEmailOpenGapsOver1Month",
        "meanEmailOpenGapsOver3Month",
        "meanEmailOpenGapsOver6Month",
        "emailClick1MonthCount",
        "emailClick1MonthWeekCount",
        "emailClick1MonthDayCount",
        "meanEmailClickGapsOver1Month",
        "meanEmailClickGapsOver3Month",
        "meanEmailClickGapsOver6Month",
        "emailOpenRate",
        "emailClickRate",
        "emailOpenClickRate",
        "emailOpenRateOver3Months",
        "emailClickRateOver3Months",
        "emailOpenClickRateOver3Months",
        "emailOpenRateOver6Months",
        "emailClickRateOver6Months",
        "emailOpenClickRateOver6Months",
        "emailOpenRateOver12Months",
        "emailClickRateOver12Months",
        "emailOpenClickRateOver12Months",
        "accountUid",
        "criOpenScore",
        "criSuggestionEmailScore"
    ],
    "ap_segment_trend": [
        "yearMonth",
        "accountUid",
        "ap_segment_trend"
    ]
}

cj_df, adlx_df, adlh_df = load_raw_data()

def clean_merge_columns(df):
    df = df.loc[:, ~df.columns.str.endswith("_y")].copy()
    df.columns = df.columns.str.replace(r"_x$", "", regex=True)
    return df

def run_preliminary_preprocessing(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
) -> pd.DataFrame:
    """
    Placeholder for preliminary preprocessing, for example:
    - Merge/join two source tables
    - Filter rows by column values
    - Basic cleanup before time-series shaping
    """
    
    # keep all rows from df_left, match from df_right on 2 differently named columns
    result = df_left.merge(
         df_right,
         how="left",
         right_on=["accountId", "interactionYearMonth"],
         left_on=["accountId", "yearMonth"],
         indicator=False  # optional: shows match status
)    
    result = clean_merge_columns(result)
    result["yearMonth"] = result["yearMonth"].astype(str)
    result = result[result["yearMonth"] != "202312"]
    # filter in Ocrevus 
    result = result[result["productId"] == 1016]
    # sort by account + yearMonth
    result = result.sort_values(by=["accountId","yearMonth"],ascending=[True,True])
    # missing substitution 
    result['AP_priority_segment'] = result.groupby(['accountId'])['AP_priority_segment'].ffill().bfill().fillna("C")
    float_cols = result.select_dtypes(include=["float", "float64"]).columns
    result[float_cols] = result[float_cols].fillna(0.0)

    
    return result


segment_code_map =  {"A":4,"B":3,"C":2,"D":1,"E":0}
def ap_priority_segment_trend(df, priority_map = segment_code_map,
                              account_col="accountUid",
                              time_col="yearMonth",
                              seg_col="AP_priority_segment"):

    # Sort once
    #out = df.sort_values([account_col, time_col], kind="mergesort").copy()
    out = df.copy()

    # Map segment to custom weight
    out["ap_segment_weight"] = out[seg_col].map(priority_map)

    # Shift the numeric weight
    out["lagged_ap_segment_weight"] = (
        out.groupby(account_col)["ap_segment_weight"].shift(1)
    )

    # Compute trend
    out["ap_segment_trend"] = (
        out["ap_segment_weight"] - out["lagged_ap_segment_weight"]
    ).fillna(0.0)

    return out

def time_series_features_flattening(df, state_type, state_columns):
    if state_type not in state_columns:
        raise ValueError(f"Unknown state type: {state_type}")

    cols = state_columns[state_type]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input df for '{state_type}': {missing}")

    value_cols = [c for c in cols if c not in ["accountUid", "yearMonth"]]
    df_subset = df[cols]
    df_pivot = df_subset.pivot(index="accountUid", columns="yearMonth", values=value_cols)
    df_pivot.columns = [f"{feature}_{month}" for feature, month in df_pivot.columns]
    return df_pivot.reset_index()

def sort_key(col):
    parts = col.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit():
        feature = parts[0]
        yearmonth = int(parts[1])   # 202301 → numeric
        return (yearmonth, feature)
    else:
        # Non time-series columns first
        return (0, col)
    
def partitioning_train_test_accounts(
    df,
    account_col="accountUid",
    test_size=0.2,
    seed=42,
):
    unique_accounts = df[account_col].dropna().unique()

    rng = np.random.default_rng(seed)
    shuffled_accounts = rng.permutation(unique_accounts)

    n_test = int(len(shuffled_accounts) * test_size)
    test_accounts = shuffled_accounts[:n_test].tolist()
    train_accounts = shuffled_accounts[n_test:].tolist()

    df_train = df[df[account_col].isin(train_accounts)].copy()
    df_test = df[df[account_col].isin(test_accounts)].copy()

    return df_train,df_test


def build_train_test_data():
    df1 = run_preliminary_preprocessing(adlh_df, cj_df)

    df2 = ap_priority_segment_trend(df1, segment_code_map)

    df_ts_visit = time_series_features_flattening(df2, "visit", state_columns)
    df_ts_email = time_series_features_flattening(df2, "email", state_columns)
    df_ts_segment_trend = time_series_features_flattening(df2, "ap_segment_trend", state_columns)

    df_ts = (
        df_ts_visit
        .merge(df_ts_email, on="accountUid")
        .merge(df_ts_segment_trend, on="accountUid")
    )

    # binary targets for 3 models
    cols_exception = []
    for target in target_columns:
        col = f"{target}_{predicted_time_period}"
        df_ts[col] = np.where(df_ts[col] > 0, 1, 0)
        cols_exception.append(col)

    # remove the columns related to the predicted time period
    cols_to_remove = [
        c for c in df_ts.columns
        if c.endswith(predicted_time_period) and c not in cols_exception
    ]
    df_ts = df_ts.drop(columns=cols_to_remove)

    # columns sorted by year + month + col_name
    df_ts = df_ts[sorted(df_ts.columns, key=sort_key)]

    df_train, df_test = partitioning_train_test_accounts(df_ts)

    return df_train, df_test
def main():
    print("START", flush=True)
    df_train, df_test = build_train_test_data()
    print(df_train.shape, df_test.shape)


if __name__ == "__main__":
    main()
 