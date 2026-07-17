import pandas as pd
import pytest

from CDRPS.src.preparation.splitter import split_dataset


def test_split_dataset_sizes():
    df = pd.DataFrame({
        "feature1": range(100),
        "target": [0, 1] * 50
    })

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df,
        target_column="target",
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert len(y_train) + len(y_val) + len(y_test) == len(df)


def test_split_dataset_missing_target():
    df = pd.DataFrame({
        "feature1": [1, 2, 3]
    })

    with pytest.raises(KeyError):
        split_dataset(
            df,
            target_column="target"
        )


def test_split_dataset_no_stratify():
    df = pd.DataFrame({
        "feature1": range(100),
        "target": [0] * 100
    })

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df,
        target_column="target",
        stratify=False
    )

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0


def test_split_dataset_fallback_from_stratify():
    df = pd.DataFrame({
        "feature1": list(range(50)),
        "target": [0] * 49 + [1]
    })

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df,
        target_column="target",
        stratify=True
    )

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0


def test_split_dataset_reproducible():
    df = pd.DataFrame({
        "feature1": range(100),
        "target": [0, 1] * 50
    })

    result1 = split_dataset(
        df,
        target_column="target",
        random_state=42
    )

    result2 = split_dataset(
        df,
        target_column="target",
        random_state=42
    )

    assert result1[0].equals(result2[0])
    assert result1[1].equals(result2[1])
    assert result1[2].equals(result2[2])


def test_split_dataset_preserves_columns():
    df = pd.DataFrame({
        "a": range(50),
        "b": range(50, 100),
        "target": [0, 1] * 25
    })

    X_train, X_val, X_test, _, _, _ = split_dataset(
        df,
        target_column="target"
    )

    assert "target" not in X_train.columns
    assert "target" not in X_val.columns
    assert "target" not in X_test.columns

    assert list(X_train.columns) == ["a", "b"]