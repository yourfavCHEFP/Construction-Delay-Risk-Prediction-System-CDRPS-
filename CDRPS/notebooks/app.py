# import chardet
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans


# # ENCODING DETECTION FUNCTION
# def detect_encoding(file):
#     raw = file.read()
#     result = chardet.detect(raw)
#     file.seek(0)
#     return result["encoding"]



# # STREAMLIT UI

# st.set_page_config(page_title="Construction Delay Risk System", layout="wide")

# st.title("🏗️ Construction Delay Risk Prediction System")
# st.write("Interactive dashboard for analyzing construction delay factors.")

# df = None
# uploaded_file = st.file_uploader("Upload CSV")


# # PROCESS FILE

# if uploaded_file is not None:

#     # --- Encoding selector ---
#     encoding_choice = st.selectbox(
#         "Select file encoding",
#         ["auto-detect", "utf-8", "latin1", "ISO-8859-1", "cp1252"]
#     )

#     try:
#         # --- Detect / choose encoding ---
#         if encoding_choice == "auto-detect":
#             encoding = detect_encoding(uploaded_file)
#         else:
#             encoding = encoding_choice

#         # --- Safe CSV read ---
#         df = pd.read_csv(uploaded_file, encoding=encoding, engine="python")

#         # Convert numeric-looking columns
#         df = df.apply(pd.to_numeric, errors="ignore")

#         # Select numeric columns (candidate features)
#         numeric_df = df.select_dtypes(include=["int64", "float64"])

#         if numeric_df.shape[1] < 2:
#             st.error("❌ Not enough numeric columns for PCA. At least 2 numeric columns are required.")
#             st.write("Numeric columns found:", list(numeric_df.columns))
#             st.stop()

#         # Impute missing values
#         imputer = SimpleImputer(strategy="mean")
#         X_imputed = imputer.fit_transform(numeric_df)

#         # Drop constant columns AFTER imputation
#         non_constant_mask = X_imputed.var(axis=0) > 0
#         X_imputed = X_imputed[:, non_constant_mask]
#         numeric_df = numeric_df.loc[:, numeric_df.columns[non_constant_mask]]

#         if X_imputed.shape[1] < 2:
#             st.error("❌ Not enough non-constant numeric columns for PCA after cleaning.")
#             st.write("Remaining numeric columns:", list(numeric_df.columns))
#             st.stop()

#         # Scale
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X_imputed)

#         # PCA on cleaned, scaled data
#         pca = PCA(n_components=2)
#         pca_components = pca.fit_transform(X_scaled)
#         df["PC1"] = pca_components[:, 0]
#         df["PC2"] = pca_components[:, 1]

#         # Clustering on PCA components (more stable)
#         kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
#         df["Cluster"] = kmeans.fit_predict(pca_components)

#         # Delay Risk Index: average only numeric feature columns (not PC1, PC2, Cluster)
#         feature_cols = numeric_df.columns
#         df["Delay_Risk_Index"] = df[feature_cols].mean(axis=1)

#         st.success("✅ File uploaded and processed successfully!")

#     except Exception as e:
#         st.error(f"❌ Failed to read CSV or process data: {e}")
#         st.stop()



# # SIDEBAR NAVIGATION

# if df is not None:
#     st.sidebar.header("Navigation")
#     page = st.sidebar.radio(
#         "Go to:",
#         ["PCA Visualization", "Cluster Analysis", "Delay Risk Index", "Raw Data"]
#     )

#     # PCA Page
#     if page == "PCA Visualization":
#         st.subheader("PCA Scatter Plot")
#         fig = px.scatter(
#             df,
#             x="PC1",
#             y="PC2",
#             color="Cluster",
#             title="PCA Scatter Plot (PC1 vs PC2)",
#             hover_data=df.columns
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     # Cluster Page
#     elif page == "Cluster Analysis":
#         st.subheader("Cluster Profiles")
#         cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
#         fig = px.imshow(
#             cluster_profiles,
#             aspect="auto",
#             color_continuous_scale="RdBu",
#             title="Cluster Profiles Heatmap"
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     # Risk Index Page
#     elif page == "Delay Risk Index":
#         st.subheader("Delay Risk Index Distribution")
#         fig = px.histogram(
#             df,
#             x="Delay_Risk_Index",
#             nbins=20,
#             marginal="box",
#             title="Distribution of Delay Risk Index"
#         )
#         st.plotly_chart(fig, use_container_width=True)

#         st.subheader("Top 20 High-Risk Respondents")
#         st.dataframe(df.sort_values("Delay_Risk_Index", ascending=False).head(20))

#     # Raw Data Page
#     elif page == "Raw Data":
#         st.subheader("Uploaded Dataset")
#         st.dataframe(df)

# else:
#     st.info("📁 Please upload a CSV file to get started.")





import chardet
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ENCODING DETECTION FUNCTION
def detect_encoding(file):
    raw = file.read()
    result = chardet.detect(raw)
    file.seek(0)
    return result["encoding"]

# CACHED CSV LOADER
@st.cache_data
def load_csv(file, encoding):
    return pd.read_csv(file, encoding=encoding, engine="python")



# CACHED PROCESSING FUNCTION
@st.cache_data
def process_data(df):
    # Convert numeric-looking columns
    df = df.apply(pd.to_numeric, errors="ignore")

    # Select numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(numeric_df)

    # Drop constant columns AFTER imputation
    non_constant_mask = X_imputed.var(axis=0) > 0
    X_imputed = X_imputed[:, non_constant_mask]
    numeric_df = numeric_df.loc[:, numeric_df.columns[non_constant_mask]]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)

    df["PC1"] = pca_components[:, 0]
    df["PC2"] = pca_components[:, 1]

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    df["Cluster"] = kmeans.fit_predict(pca_components)

    # Delay Risk Index
    feature_cols = numeric_df.columns
    df["Delay_Risk_Index"] = df[feature_cols].mean(axis=1)

    return df


# STREAMLIT UI
st.set_page_config(page_title="Construction Delay Risk System", layout="wide")

st.title("🏗️ Construction Delay Risk Prediction System")
st.write("Interactive dashboard for analyzing construction delay factors.")

df = None
uploaded_file = st.file_uploader("Upload CSV")


# PROCESS FILE
if uploaded_file is not None:

    encoding_choice = st.selectbox(
        "Select file encoding",
        ["auto-detect", "utf-8", "latin1", "ISO-8859-1", "cp1252"]
    )

    try:
        if encoding_choice == "auto-detect":
            encoding = detect_encoding(uploaded_file)
        else:
            encoding = encoding_choice

        # Show spinner while loading
        with st.spinner("Reading CSV..."):
            df = load_csv(uploaded_file, encoding)

        # Optional sampling for huge datasets
        if df.shape[0] > 5000:
            df = df.sample(5000, random_state=42)

        # Show spinner while processing
        with st.spinner("Processing data..."):
            df = process_data(df)

        st.success("✅ File uploaded and processed successfully!")

    except Exception as e:
        st.error(f"❌ Failed to read CSV or process data: {e}")
        st.stop()


# SIDEBAR NAVIGATION

if df is not None:
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["PCA Visualization", "Cluster Analysis", "Delay Risk Index", "Raw Data"]
    )

    if page == "PCA Visualization":
        st.subheader("PCA Scatter Plot")
        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title="PCA Scatter Plot (PC1 vs PC2)",
            hover_data=df.columns
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Cluster Analysis":
        st.subheader("Cluster Profiles")
        cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
        fig = px.imshow(
            cluster_profiles,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Cluster Profiles Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Delay Risk Index":
        st.subheader("Delay Risk Index Distribution")
        fig = px.histogram(
            df,
            x="Delay_Risk_Index",
            nbins=20,
            marginal="box",
            title="Distribution of Delay Risk Index"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 20 High-Risk Respondents")
        st.dataframe(df.sort_values("Delay_Risk_Index", ascending=False).head(20))

    elif page == "Raw Data":
        st.subheader("Uploaded Dataset")
        st.dataframe(df)

else:
    st.info("📁 Please upload a CSV file to get started.")
