"""
Preparation Module — splitter.py

Responsibilities:
- Split the dataset into train, validation, and test sets
- Support optional stratification by the target column
- Ensure reproducibility using a random_state
- Return all splits in a clean, structured format
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _is_stratify_compatibility_error(error: ValueError) -> bool:
    """Return True when sklearn rejects stratification due to class-frequency constraints."""
    message = str(error).lower()
    return (
        "least populated class" in message
        or "number of classes" in message
        or "train_size" in message
        or "test_size" in message
    )


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into train, validation, and test sets.

    Parameters:
    - df: The full cleaned DataFrame
    - target_column: The name of the target variable
    - test_size: Fraction of data to allocate to the test set
    - val_size: Fraction of the *remaining* training data to allocate to validation
    - random_state: Seed for reproducibility
    - stratify: Whether to stratify splits based on the target column

    Returns:
    - X_train, X_val, X_test
    - y_train, y_val, y_test
    """
    df = df.copy()

    if target_column not in df.columns:
        available_columns = ", ".join(map(str, df.columns.tolist()))
        raise KeyError(
            f"Target column '{target_column}' was not found in the dataset. "
            f"Available columns are: {available_columns}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_target = y if stratify else None

    # First split: train + temp_test
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )
    except ValueError as error:
        if stratify and _is_stratify_compatibility_error(error):
            print(
                "Warning: stratified split is not possible with the current target distribution; "
                "falling back to non-stratified split."
            )
            X_train, X_temp, y_train, y_temp = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
        else:
            raise

    # Second split: validation + test from the temporary set
    val_fraction_of_temp = val_size / (1 - test_size)

    stratify_temp = y_temp if stratify else None

    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_fraction_of_temp,
            random_state=random_state,
            stratify=stratify_temp,
        )
    except ValueError as error:
        if stratify and _is_stratify_compatibility_error(error):
            print(
                "Warning: stratified validation/test split is not possible with the current target "
                "distribution; falling back to non-stratified split."
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=1 - val_fraction_of_temp,
                random_state=random_state,
                stratify=None,
            )
        else:
            raise

    return X_train, X_val, X_test, y_train, y_val, y_test
