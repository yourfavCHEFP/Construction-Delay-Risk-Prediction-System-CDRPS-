from pathlib import Path
import json
from turtle import st
from turtle import st

import joblib
import shap
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "CDRPS" / "models"

MODEL_PATH = MODELS_DIR / "model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"

_model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    _feature_columns = json.load(f)

_explainer = shap.TreeExplainer(_model)


def shap_explain_single(input_dict: dict):
    row = {col: float(input_dict.get(col, 0.0)) for col in _feature_columns}
    df = pd.DataFrame([row])
    shap_values = _explainer.shap_values(df)
    return df, shap_values

# 
from CDRPS.shap_explain import shap_explain_single
import matplotlib.pyplot as plt
import shap
import streamlit as st
# after prediction:
if st.button("Explain this prediction"):
    df_row, shap_values = shap_explain_single(user_inputs)

    st.write("Feature contributions to this prediction:")

    shap.initjs()
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df_row, plot_type="bar", show=False)
    st.pyplot(fig)



#let's refine a waterfall plot for the shap values
def shap_waterfall_plot(shap_values, feature_names):
    shap_values = shap_values[0]  # Assuming binary classification, take the first class
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values
    }).sort_values(by='SHAP Value', ascending=False)

    # Create a waterfall plot using Plotly
    import plotly.graph_objects as go

    fig = go.Figure(go.Waterfall(
        name="SHAP Values",
        orientation="v",
        measure=["relative"] * len(shap_df),
        x=shap_df['Feature'],
        y=shap_df['SHAP Value'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="SHAP Waterfall Plot",
        xaxis_title="Features",
        yaxis_title="SHAP Value",
        showlegend=True
    )

    return fig