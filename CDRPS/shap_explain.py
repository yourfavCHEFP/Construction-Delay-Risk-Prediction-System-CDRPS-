from pathlib import Path
import json
from typing import Iterable, Sequence, Tuple

import joblib
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "CDRPS" / "models"

MODEL_PATH = MODELS_DIR / "model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"

_model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    _feature_columns = json.load(f)

_explainer = shap.TreeExplainer(_model)


def _normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[0]
    if hasattr(shap_values, "values"):
        return shap_values.values
    return shap_values


def shap_explain_single(input_dict: dict):
    row = {col: float(input_dict.get(col, 0.0)) for col in _feature_columns}
    df = pd.DataFrame([row])
    shap_values = _normalize_shap_values(_explainer.shap_values(df))
    return df, shap_values


def top_contributing_features(shap_values, feature_names, top_n=5):
    shap_values = _normalize_shap_values(shap_values)
    if getattr(shap_values, "ndim", 1) > 1:
        shap_values = shap_values[0]

    shap_df = pd.DataFrame(
        {
            "Feature": list(feature_names),
            "SHAP Value": shap_values,
        }
    ).sort_values(by="SHAP Value", ascending=False)

    return shap_df.head(top_n)


def top_contributors(input_dict: dict, top_n=5):
    df_row, shap_values = shap_explain_single(input_dict)
    return top_contributing_features(shap_values, df_row.columns.tolist(), top_n=top_n)


def shap_summary_plot(shap_values, feature_names):
    shap_df = top_contributing_features(shap_values, feature_names, top_n=len(feature_names))

    import plotly.express as px

    fig = px.bar(
        shap_df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title="SHAP Summary Plot",
        labels={"SHAP Value": "SHAP Value", "Feature": "Feature"},
    )
    return fig


def explanation_text(contributors: Iterable[Sequence], max_items: int = 5) -> str:
    lines = []
    count = 0
    for item in contributors:
        if count >= max_items:
            break
        if len(item) < 2:
            continue

        feat = str(item[0])
        val = float(item[1])
        direction = "increased" if val > 0 else "decreased"
        lines.append(
            f"- **{feat.replace('_', ' ')}** -> {direction} the predicted delay risk index by {abs(val):.4f}"
        )
        count += 1

    return "\n".join(lines) if lines else "No contributing factors available."