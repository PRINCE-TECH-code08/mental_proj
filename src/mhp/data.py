"""Data loading / cleaning. Raw data is never modified on disk; cleaning happens in memory and is logged."""
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import RAW_CSV, SEED, TEST_SIZE, TARGET


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def clean(df: pd.DataFrame, log: dict | None = None) -> pd.DataFrame:
    """Same two cleaning steps as the baseline notebook, made explicit and logged."""
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    neg = int((df["Physical_Activity_Hours"] < 0).sum())
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)
    if log is not None:
        log.update(rows_raw=n0, duplicates_dropped=n0 - len(df), negative_activity_clipped=neg, rows_clean=len(df))
    return df


def split(df: pd.DataFrame):
    """Identical split to baseline: train_test_split(test_size=0.30, random_state=42) on the cleaned frame.
    Returns X_train, X_test, y_train, y_test where X still contains RAW columns (incl. Country)."""
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
