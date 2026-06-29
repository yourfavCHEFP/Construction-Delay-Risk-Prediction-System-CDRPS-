import io

import chardet
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from CDRPS.prediction_pipeline import categorize_risk, load_artifacts, predict_delay_risk
from CDRPS.src.validation.schema_validator import run_validation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Construction Delay Risk System", layout="wide")


@st.cache_data
def detect_encoding(file_bytes):
    result = chardet.detect(file_bytes)
    return result.get("encoding") or "utf-8"


@st.cache_data
def load_uploaded_file(file_bytes, file_name, encoding):
    file_name = (file_name or "").lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, engine="python")

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(file_bytes))

    raise ValueError("Unsupported file format. Please upload CSV or Excel files.")


@st.cache_data
def process_data(df):
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="ignore")

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("The uploaded file does not contain numeric columns to analyze.")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(numeric_df)

    non_constant_mask = X_imputed.var(axis=0) > 0
    X_imputed = X_imputed[:, non_constant_mask]
    numeric_df = numeric_df.loc[:, numeric_df.columns[non_constant_mask]]

    if numeric_df.shape[1] < 2:
        raise ValueError("At least two non-constant numeric columns are required for PCA.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)

    df["PC1"] = pca_components[:, 0]
    df["PC2"] = pca_components[:, 1]

    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    df["Cluster"] = kmeans.fit_predict(pca_components)

    feature_cols = numeric_df.columns
    df["Delay_Risk_Index"] = df[feature_cols].mean(axis=1)

    return df


st.title("🏗️ Construction Delay Risk Prediction System")
st.write("Interactive dashboard for analyzing construction delay factors.")
st.caption("Validation issues are shown immediately after upload.")

uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])

required_columns_input = st.text_input(
    "Required columns (comma-separated, optional)",
    value="",
    help="Example: Respondent Number, Planned_Duration, Actual_Duration",
)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "PCA Visualization",
        "Cluster Analysis",
        "Delay Risk Index",
        "Raw Data",
        "Delay Risk Prediction",
    ],
)

df = None

if uploaded_file is not None:
    encoding_choice = st.selectbox(
        "Select file encoding",
        ["auto-detect", "utf-8", "latin1", "ISO-8859-1", "cp1252"],
    )

    try:
        file_bytes = uploaded_file.getvalue()
        encoding = detect_encoding(file_bytes) if encoding_choice == "auto-detect" else encoding_choice
        required_columns = [
            col.strip() for col in required_columns_input.split(",") if col.strip()
        ]

        with st.spinner("Reading file..."):
            df_raw = load_uploaded_file(file_bytes, uploaded_file.name, encoding)

        with st.spinner("Running validation..."):
            validation_results = run_validation(
                df_raw,
                required_columns=required_columns,
            )

        with st.expander("Validation Summary", expanded=True):
            st.write(f"Rows: {df_raw.shape[0]}, Columns: {df_raw.shape[1]}")

            missing_columns = validation_results.get("missing_columns", [])
            if missing_columns:
                st.error("Missing required columns detected")
                st.write(missing_columns)
            else:
                st.success("No required-column violations found.")

            missing_values_summary = validation_results.get("missing_values_summary", {})
            if missing_values_summary:
                st.warning("Missing values detected")
                st.dataframe(
                    pd.DataFrame(
                        list(missing_values_summary.items()),
                        columns=["Column", "Missing Count"],
                    )
                )
            else:
                st.success("No missing values detected.")

            inconsistencies = validation_results.get("inconsistencies", {})
            if inconsistencies:
                st.warning("Inconsistencies detected")
                st.json(inconsistencies)
            else:
                st.success("No inconsistencies detected.")

            invalid_types = validation_results.get("invalid_types", {})
            if invalid_types:
                st.warning("Invalid data types detected")
                st.json(invalid_types)

        if validation_results.get("has_critical_issues"):
            st.error("Critical validation issues found. Please fix the file before processing.")
            st.stop()

        df = df_raw.copy()

        if df.shape[0] > 5000:
            df = df.sample(5000, random_state=42)

        with st.spinner("Processing data..."):
            df = process_data(df)

        st.success("✅ File uploaded and processed successfully!")

    except Exception as exc:
        st.error(f"❌ Failed to read CSV or process data: {exc}")
        st.stop()

if df is None:
    st.info("📁 Please upload a CSV or Excel file to get started.")

elif page == "PCA Visualization":
    st.subheader("PCA Scatter Plot")
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="PCA Scatter Plot (PC1 vs PC2)",
        hover_data=df.columns,
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Cluster Analysis":
    st.subheader("Cluster Profiles")
    cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
    fig = px.imshow(
        cluster_profiles,
        aspect="auto",
        color_continuous_scale="RdBu",
        title="Cluster Profiles Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Delay Risk Index":
    st.subheader("Delay Risk Index Distribution")
    fig = px.histogram(
        df,
        x="Delay_Risk_Index",
        nbins=20,
        marginal="box",
        title="Distribution of Delay Risk Index",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 High-Risk Respondents")
    st.dataframe(df.sort_values("Delay_Risk_Index", ascending=False).head(20))

elif page == "Raw Data":
    st.subheader("Uploaded Dataset")
    st.dataframe(df)

elif page == "Delay Risk Prediction":
    st.subheader("Delay Risk Prediction")
    st.write("Adjust the project factors to estimate the delay risk index.")

    try:
        load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load prediction artifacts: {exc}")
        st.stop()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Delay_Risk_Index" in numeric_cols:
        numeric_cols.remove("Delay_Risk_Index")

    user_inputs = {}
    cols = st.columns(2)

    for i, col_name in enumerate(numeric_cols):
        with cols[i % 2]:
            default_val = float(df[col_name].mean()) if col_name in df.columns else 0.0
            user_inputs[col_name] = st.number_input(
                col_name,
                value=default_val,
            )

    if st.button("Predict Delay Risk"):
        with st.spinner("Predicting delay risk..."):
            score = predict_delay_risk(user_inputs)
            category = categorize_risk(score)

        st.success(f"Predicted Delay Risk Index: **{score:.2f}**")
        st.info(f"Risk Category: **{category}**")
