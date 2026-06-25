"""
Preparation Module — cleaning.py

Responsibilities:
- Fix data types for known columns
- Handle missing values using simple, transparent rules
- Remove or flag clearly invalid rows (e.g., negative durations)
- Return a cleaned DataFrame ready for encoding and splitting
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def fix_data_types(
    df: pd.DataFrame,
    numeric_columns: Optional[list[str]] = None,
    datetime_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fix data types for numeric and datetime columns.

    - numeric_columns: columns that should be numeric (coerce errors to NaN)
    - datetime_columns: columns that should be datetime (coerce errors to NaT)
    """
    df = df.copy()

    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if datetime_columns:
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def handle_missing_values(
    df: pd.DataFrame,
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Handle missing values with simple rules:

    - Drop columns where more than `drop_threshold` fraction of values are missing.
    - Drop rows that are completely empty.
    - Leave remaining missing values for later (e.g., imputation in modeling).
    """
    df = df.copy()

    # Drop columns with too many missing values
    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > drop_threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Drop rows that are entirely missing
    df = df.dropna(how="all")

    return df


def remove_or_flag_invalid_values(
    df: pd.DataFrame,
    non_negative_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Remove rows with clearly invalid values:

    - For columns in `non_negative_columns`, drop rows where values are negative.
    """
    df = df.copy()

    if non_negative_columns:
        for col in non_negative_columns:
            if col in df.columns:
                df = df[df[col] >= 0]

    return df


def clean_dataset(
    df: pd.DataFrame,
    numeric_columns: Optional[list[str]] = None,
    datetime_columns: Optional[list[str]] = None,
    non_negative_columns: Optional[list[str]] = None,
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Orchestrate cleaning steps:

    1. Fix data types.
    2. Handle missing values.
    3. Remove rows with invalid values.
    """
    df = fix_data_types(df, numeric_columns=numeric_columns, datetime_columns=datetime_columns)
    df = handle_missing_values(df, drop_threshold=drop_threshold)
    df = remove_or_flag_invalid_values(df, non_negative_columns=non_negative_columns)
    return df
