"""
Ingestion Module — file_loader.py

Responsibilities:
- Validate file extensions (.csv, .xlsx, .xls)
- Load CSV or Excel files safely
- Handle encoding issues gracefully
- Provide a clean preview of the dataset (returned, not printed)
- Never expose raw stack traces to the user
"""

import os
import pandas as pd


# 1. validate_file_extension(path: str)
def validate_file_extension(path: str):
    """Validate that the file has a supported extension."""
    allowed_extensions = (".csv", ".xlsx", ".xls")
    if not path.lower().endswith(allowed_extensions):
        raise ValueError(
            "Unsupported file format. Please upload a CSV or Excel file (.csv, .xlsx, .xls)."
        )


# 2. load_file(path: str) -> pd.DataFrame
def load_file(path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset after validating the file extension.
    Handles encoding issues and empty files gracefully.
    """
    # Check file extension
    validate_file_extension(path)

    # Check file existence
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path}")

    # Load CSV
    if path.lower().endswith(".csv"):
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")

    # Load Excel
    else:
        df = pd.read_excel(path)

    # Check for empty dataset
    if df.empty:
        raise ValueError("The uploaded file is empty. Please upload a file with data.")

    return df


# 3. preview_data(df: pd.DataFrame, n=5)
def preview_data(df: pd.DataFrame, n=5, num_rows=None):
    """
    Return the first n rows of the DataFrame.
    (No printing — the notebook decides how to display it.)
    """
    if num_rows is not None:
        n = num_rows

    return df.head(n)
