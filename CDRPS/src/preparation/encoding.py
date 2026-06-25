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
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
            df[col] = le.fit_transform(df[col].astype(str))
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

    # Fit only on numeric columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df, scaler
