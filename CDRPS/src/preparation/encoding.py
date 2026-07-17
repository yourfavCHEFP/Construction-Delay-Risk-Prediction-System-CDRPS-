"""
Preparation Module — encoding.py

Responsibilities:
- Encode categorical features using Label Encoding
- Scale numerical features using StandardScaler
- Return both the transformed DataFrame and the fitted transformers
- Never overwrite the original DataFrame in-place
- Ensure encoding and scaling only apply to the correct column types
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler


def encode_categorical(
    df: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical columns using Label Encoding.

    Returns:
    - A new DataFrame with encoded categorical columns
    - A dictionary of fitted LabelEncoders for later inverse_transform
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = pd.Series(
                le.fit_transform(df[col].astype(str)),
                index=df.index
            )
            encoders[col] = le

    return df, encoders


def scale_numerical(
    df: pd.DataFrame,
    numeric_columns: List[str],
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical columns using StandardScaler.

    Returns:
    - A new DataFrame with scaled numerical columns
    - The fitted StandardScaler object
    """
    df = df.copy()
    scaler = StandardScaler()

    available_numeric_cols = [
        col for col in numeric_columns
        if col in df.columns
    ]

    if available_numeric_cols:
        df[available_numeric_cols] = scaler.fit_transform(
            df[available_numeric_cols]
        )

    return df, scaler


def transform_skewed_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    skew_threshold: float = 1.0,
) -> Tuple[pd.DataFrame, PowerTransformer | None, Dict[str, object]]:
    """
    Transform highly skewed numeric features using Yeo-Johnson.

    Returns:
    - A new DataFrame with transformed skewed columns
    - The fitted PowerTransformer (or None if no columns exceeded threshold)
    - Metadata with selected columns and pre/post skew values
    """
    df = df.copy()
    available_numeric_cols = [col for col in numeric_columns if col in df.columns]

    if not available_numeric_cols:
        return df, None, {"transformed_columns": [], "pre_skew": {}, "post_skew": {}}

    skew_series = df[available_numeric_cols].skew(numeric_only=True)
    skewed_cols = [
        col
        for col in available_numeric_cols
        if pd.notna(skew_series.get(col)) and abs(float(skew_series[col])) >= skew_threshold
    ]

    if not skewed_cols:
        return df, None, {
            "transformed_columns": [],
            "pre_skew": skew_series.to_dict(),
            "post_skew": skew_series.to_dict(),
        }

    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    transformed_values = transformer.fit_transform(df[skewed_cols])
    df[skewed_cols] = transformed_values

    post_skew = df[skewed_cols].skew(numeric_only=True).to_dict()
    metadata = {
        "transformed_columns": skewed_cols,
        "pre_skew": skew_series.to_dict(),
        "post_skew": post_skew,
        "method": "yeo-johnson",
        "skew_threshold": skew_threshold,
    }

    return df, transformer, metadata

