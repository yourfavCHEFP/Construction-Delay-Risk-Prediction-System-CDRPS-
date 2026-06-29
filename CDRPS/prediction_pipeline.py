import json 
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "CDRPS" / "models"

MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"

_model = None
_scaler = None
_feature_columns = None


def load_artifacts():
    global _model, _scaler, _feature_columns

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)

    if _feature_columns is None:
        with open(FEATURES_PATH, "r") as f:
            _feature_columns = json.load(f)

    return _model, _scaler, _feature_columns


def prepare_features(input_dict: Dict[str, float]) -> np.ndarray:
    _, scaler, feature_columns = load_artifacts()

    row = {col: float(input_dict.get(col, 0.0)) for col in feature_columns}
    df = pd.DataFrame([row])
    X = df.values
    X_scaled = scaler.transform(X)
    return X_scaled


def predict_delay_risk(input_dict: Dict[str, float]) -> float:
    model, _, _ = load_artifacts()
    X_scaled = prepare_features(input_dict)
    y_pred = model.predict(X_scaled)[0]
    return float(y_pred)


def categorize_risk(score: float) -> str:
    if score < 2.0:
        return "Low Risk"
    elif score < 3.5:
        return "Medium Risk"
    else:
        return "High Risk"