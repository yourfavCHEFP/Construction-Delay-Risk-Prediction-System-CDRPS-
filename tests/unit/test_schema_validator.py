import pandas as pd
import pytest
from CDRPS.src.validation.schema_validator import (
    check_required_columns,
    summarize_missing_values,
    detect_basic_inconsistencies,
    run_validation,
    detect_invalid_types,
)

def test_check_required_columns():
    df = pd.DataFrame({"A": [1], "B": [2]})
    assert check_required_columns(df, ["A", "B", "C"]) == ["C"]
    assert check_required_columns(df, ["A", "B"]) == []


def test_summarize_missing_values():
    # Exercise 2: Multiple gappy columns
    df = pd.DataFrame({"A": [1, None, None], "B": [1, 2, 3], "C": [None, 2, 3]})
    result = summarize_missing_values(df)
    assert result == {"A": 2, "C": 1}


def test_detect_inconsistencies():
    df = pd.DataFrame({"score": [1, -2, 1], "note": ["ok", "", "ok"]})
    result = detect_basic_inconsistencies(df)
    assert "score" in result["negative_values_detected"]
    assert "note" in result["empty_strings_detected"]
    assert result["duplicate_rows"] == 1


def test_run_validation_logic():
    # Exercise 4: Happy path
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    report = run_validation(df, required_columns=["A", "B"])
    assert report["has_critical_issues"] is False


def test_detect_invalid_types_mismatch():
    # Exercise 3: Type mismatch detection
    df = pd.DataFrame({"age": ["25", "thirty", "22"]}) # strings
    expected = {"age": "int"}
    mismatches = detect_invalid_types(df, expected)
    assert "age" in mismatches