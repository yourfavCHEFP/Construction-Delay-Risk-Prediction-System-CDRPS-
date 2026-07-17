import pandas as pd

from CDRPS.src.preparation.encoding import (
    encode_categorical,
    scale_numerical,
    transform_skewed_features,
)


def test_encode_categorical_columns():
    df = pd.DataFrame({
        "risk": ["low", "medium", "high"]
    })

    encoded, encoders = encode_categorical(
        df,
        categorical_columns=["risk"]
    )

    assert len(encoded) == 3
    assert "risk" in encoded.columns
    assert encoded["risk"].dtype.kind in "iu"
    assert "risk" in encoders


def test_encode_missing_column():
    df = pd.DataFrame({
        "risk": ["low", "medium"]
    })

    encoded, encoders = encode_categorical(
        df,
        categorical_columns=["priority"]
    )

    assert encoded.equals(df)
    assert encoders == {}


def test_scale_numerical():
    df = pd.DataFrame({
        "cost": [100, 200, 300, 400]
    })

    scaled, scaler = scale_numerical(
        df,
        numeric_columns=["cost"]
    )

    assert scaled.shape == df.shape
    assert abs(scaled["cost"].mean()) < 1e-6
    assert scaler is not None


def test_transform_skewed_features():
    df = pd.DataFrame({
        "value": [1, 1, 1, 1, 1000]
    })

    transformed, transformer, metadata = transform_skewed_features(
        df,
        numeric_columns=["value"]
    )

    assert transformer is not None
    assert "value" in metadata["transformed_columns"] # type: ignore
    assert metadata["method"] == "yeo-johnson"
    assert transformed.shape == df.shape


def test_transform_skewed_features_no_skew():
    df = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    transformed, transformer, metadata = transform_skewed_features(
        df,
        numeric_columns=["value"]
    )

    assert transformer is None
    assert metadata["transformed_columns"] == []


def test_transform_skewed_features_missing_column():
    df = pd.DataFrame({
        "value": [1, 2, 3]
    })

    transformed, transformer, metadata = transform_skewed_features(
        df,
        numeric_columns=["unknown_column"]
    )

    assert transformer is None
    assert metadata["transformed_columns"] == []
    assert transformed.equals(df)