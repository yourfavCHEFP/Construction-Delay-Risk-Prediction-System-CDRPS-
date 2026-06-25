import chardet
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
def load_csv(file_bytes, encoding):
    return pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding=encoding, engine="python")


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

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

df = None

if uploaded_file is not None:
    encoding_choice = st.selectbox(
        "Select file encoding",
        ["auto-detect", "utf-8", "latin1", "ISO-8859-1", "cp1252"],
    )

    try:
        file_bytes = uploaded_file.getvalue()
        encoding = detect_encoding(file_bytes) if encoding_choice == "auto-detect" else encoding_choice

        with st.spinner("Reading CSV..."):
            df = load_csv(file_bytes, encoding)

        if df.shape[0] > 5000:
            df = df.sample(5000, random_state=42)

        with st.spinner("Processing data..."):
            df = process_data(df)

        st.success("✅ File uploaded and processed successfully!")

    except Exception as exc:
        st.error(f"❌ Failed to read CSV or process data: {exc}")
        st.stop()

if df is not None:
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["PCA Visualization", "Cluster Analysis", "Delay Risk Index", "Raw Data"],
    )

    if page == "PCA Visualization":
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

else:
    st.info("📁 Please upload a CSV file to get started.")