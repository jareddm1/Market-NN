"""this just loads data its not that deep"""

import pandas as pd
from config import (
    TRAIN_FILE, TEST_FILE, DATE_COL, TARGET_COL,
    NON_FEATURE_COLS, TRAIN_FRAC, TUNE_FRAC, ROLLING_WINDOWS,
)


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_FILE)
    if DATE_COL in df.columns:
        df = df.sort_values(DATE_COL).reset_index(drop=True)

    if TARGET_COL in df.columns:
        before = len(df)
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        if before - len(df):
            print(f"[load_data] Dropped {before - len(df)} rows with missing target.")

    print(f"[load_data] Train shape: {df.shape}")
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_FILE)
    if DATE_COL in df.columns:
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"[load_data] Test shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column that is not metadata or a label is a candidate feature."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def three_way_split(df: pd.DataFrame):
    """
    Chronological train / tune / test split, it fits the base model and stacking,
    then grid-searches the allocation mapping, and spits out an untouched final score
    """
    if embargo is None:
        embargo = max(ROLLING_WINDOWS)        # drop rows whose rolling window crosses the cut

    n = len(df)
    i_train = int(n * TRAIN_FRAC)
    i_tune  = int(n * (TRAIN_FRAC + TUNE_FRAC))

    train_df = df.iloc[:i_train].copy()
    tune_df  = df.iloc[i_train + embargo : i_tune].copy()
    test_df  = df.iloc[i_tune + embargo :].copy()

    print(f"[load_data] Split (embargo={embargo}) -> train {len(train_df)} | "
          f"tune {len(tune_df)} | test {len(test_df)}")
    return train_df, tune_df, test_df
