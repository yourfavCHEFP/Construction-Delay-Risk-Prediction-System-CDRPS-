"""
Validation Module — schema_validator.py

Responsibilities:
- Validate that all required columns exist
- Summarize missing values per column
- Detect basic inconsistencies (negative numbers, empty strings)
- Detect duplicate rows
- Detect invalid data types
- Stop early if required columns are missing
- Return a structured validation report for the pipeline
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


# 1. Check required columns
def check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """Return a list of required columns that are missing from the DataFrame."""
    return [col for col in required_columns if col not in df.columns]


# 2. Summarize missing values
def summarize_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Return missing value counts per column (only columns with missing values)."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return {str(col): int(count) for col, count in missing.to_dict().items()}


# 3. Detect basic inconsistencies
def detect_basic_inconsistencies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect:
    - Negative numeric values
    - Empty strings in object columns
    - Duplicate rows
    """
    inconsistencies: Dict[str, Any] = {}

    # Negative numeric values
    numeric_cols = df.select_dtypes(include=["number"]).columns
    negative_cols = [col for col in numeric_cols if (df[col] < 0).any()]
    if negative_cols:
        inconsistencies["negative_values_detected"] = negative_cols

    # Empty strings
    object_cols = df.select_dtypes(include=["object"]).columns
    empty_string_cols = [
        col for col in object_cols if df[col].astype(str).str.strip().eq("").any()
    ]
    if empty_string_cols:
        inconsistencies["empty_strings_detected"] = empty_string_cols

    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        inconsistencies["duplicate_rows"] = int(duplicate_count)

    return inconsistencies


# 4. Detect invalid data types
def detect_invalid_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, str]:
    """
    Compare actual dtypes to expected dtypes.
    Return columns where the dtype does not match.
    """
    mismatches = {}

    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if expected_type not in actual_type:
                mismatches[col] = actual_type

    return mismatches


# 5. Orchestrator — run all validation steps
def run_validation(
    df: pd.DataFrame,
    required_columns: List[str],
    expected_types: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Run validation checks and return a structured report.

    Early stop:
    - If required columns are missing, skip all other checks.
    """
    # Step 1 — Required columns
    missing_columns = check_required_columns(df, required_columns)

    # Early stop if critical issue
    if missing_columns:
        return {
            "missing_columns": missing_columns,
            "missing_values_summary": {},
            "inconsistencies": {},
            "invalid_types": {},
            "has_critical_issues": True,
        }

    # Step 2 — Missing values
    missing_values_summary = summarize_missing_values(df)

    # Step 3 — Inconsistencies
    inconsistencies = detect_basic_inconsistencies(df)

    # Step 4 — Invalid types (optional)
    invalid_types = {}
    if expected_types:
        invalid_types = detect_invalid_types(df, expected_types)

    return {
        "missing_columns": missing_columns,
        "missing_values_summary": missing_values_summary,
        "inconsistencies": inconsistencies,
        "invalid_types": invalid_types,
        "has_critical_issues": False,
    }
