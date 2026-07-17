import pandas as pd
import numpy as np
from CDRPS.src.preparation.cleaning import handle_missing_values, remove_or_flag_invalid_values, fix_data_types
from CDRPS.src.ingestion.file_loader import preview_data


def test_fix_data_types_coercion():
    # Exercise 5: Text to NaN
    df = pd.DataFrame({"score": ["1.5", "abc", "3.0"]})
    cleaned = fix_data_types(df, numeric_columns=["score"])
    assert pd.isna(cleaned.loc[1, "score"])

def test_handle_missing_values(sample_survey_df):
    # Exercise 6: Drops fully empty rows
    df = pd.DataFrame({"A": [1, None, 3], "B": [None, None, None]})
    cleaned = handle_missing_values(df, drop_threshold=0.5)
    assert "B" not in cleaned.columns
    assert len(cleaned) == 2 # Empty row dropped

def test_remove_invalid_no_op():
    # Exercise 7: No-op check
    df = pd.DataFrame({"val": [1, 2]})
    pd.testing.assert_frame_equal(df, remove_or_flag_invalid_values(df, None))


def test_preview_data_limit():
    # Exercise 8: Preview row limits
    df = pd.DataFrame({"id": range(10)})
    # Assuming preview_data(df, n=5) is your function
    assert len(preview_data(df, num_rows=3)) == 3
    assert len(preview_data(df)) == 5 # default