import io
import logging
from datetime import datetime
from typing import cast

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from CDRPS.prediction_pipeline import (
    categorize_risk,
    load_artifacts,
    predict_delay_risk,
    prediction_confidence,
)
from CDRPS.shap_explain import explanation_text, top_contributors
from CDRPS.src.validation.schema_validator import run_validation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Construction Delay Risk System", layout="wide")

logger = logging.getLogger("cdrps.app")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Sprint 4 global sync placeholders (KAN-118, KAN-121, KAN-122, KAN-123).
st.session_state.setdefault("shared_data_object", None)
st.session_state.setdefault("shared_prediction_results", None)
st.session_state.setdefault("shared_metadata", None)
st.session_state.setdefault("shared_shap_values", None)
st.session_state.setdefault("shared_charts", None)
st.session_state.setdefault("shared_filters", None)

# Sprint 4 pipeline placeholders (KAN-119, KAN-120, KAN-124, KAN-125, KAN-126).
pipeline_placeholders = {
    "error_handling": {
        "input_validation": None,
        "missing_model_error": None,
        "missing_file_error": None,
        "shap_error": None,
        "chart_error": None,
    },
    "logging": {
        "prediction_logs": None,
        "shap_logs": None,
        "metadata_logs": None,
        "export_logs": None,
        "chart_logs": None,
    },
    "performance": {
        "faster_chart_rendering": None,
        "faster_shap_computation": None,
        "faster_metadata_extraction": None,
        "faster_predictions": None,
    },
    "qa": {
        "ui_consistency_checks": None,
        "backend_consistency_checks": None,
        "chart_consistency_checks": None,
        "shap_consistency_checks": None,
        "metadata_consistency_checks": None,
    },
    "bug_fix": {
        "broken_imports": None,
        "broken_paths": None,
        "broken_charts": None,
        "broken_shap_visuals": None,
        "broken_metadata_displays": None,
    },
}


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


def build_prediction_pdf(df_export: pd.DataFrame) -> bytes:
    """Build a compact PDF summary for prediction exports."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        preview = df_export.head(20)
        summary_lines = [
            "CDRPS Prediction Export Summary",
            f"Generated: {datetime.utcnow().isoformat()} UTC",
            f"Rows: {len(df_export)}",
            f"Columns: {', '.join(df_export.columns)}",
            "",
            "Top 20 Rows:",
            preview.to_string(index=False),
        ]

        ax.text(
            0.01,
            0.99,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            wrap=True,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return buffer.getvalue()


def ensure_prediction_history_state():
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []


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
        "Predictions",
        "Visual Insights",
        "SHAP Visualizations",
        "Metadata Visualizations",
        "Risk Summary Visualizations",
        "Risk Category Visualizations",
        "Risk Distribution Visualizations",
        "Factor Impact Visualizations",
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

elif page == "Predictions":
    st.subheader("Predictions")
    ensure_prediction_history_state()
    st.markdown("### Predictions Table")
    st.caption("Prediction history is captured from the Delay Risk Prediction page.")

    history_df = pd.DataFrame(st.session_state["prediction_history"])
    if history_df.empty:
        st.info("No predictions yet. Run at least one prediction to populate this table.")
    else:
        st.markdown("#### Sort and Filter Controls")
        all_columns = history_df.columns.tolist()

        c1, c2, c3 = st.columns([2, 1, 2])
        sort_column = c1.selectbox("Sort by", all_columns, index=0)
        sort_ascending = c2.checkbox("Ascending", value=False)
        search_query = c3.text_input("Search", value="", placeholder="Type to search")

        filter_cols = st.columns([2, 2, 2])
        category_options = sorted(history_df["risk_category"].dropna().unique().tolist()) if "risk_category" in history_df.columns else []
        selected_categories = filter_cols[0].multiselect(
            "Risk Category Filter",
            options=category_options,
            default=category_options,
        )

        selected_columns = filter_cols[1].multiselect(
            "Visible Columns",
            options=all_columns,
            default=all_columns,
        )
        page_size = int(filter_cols[2].selectbox("Rows per page", options=[5, 10, 20, 50], index=1))

        filtered_df = history_df.copy()
        if selected_categories and "risk_category" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["risk_category"].isin(selected_categories)]

        if search_query.strip():
            term = search_query.strip().lower()
            mask = filtered_df.astype(str).apply(lambda col: col.str.lower().str.contains(term, na=False))
            filtered_df = filtered_df[mask.any(axis=1)]

        filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending)

        if not selected_columns:
            selected_columns = all_columns

        filtered_display = filtered_df[selected_columns]
        total_rows = len(filtered_display)
        total_pages = max(1, int(np.ceil(total_rows / page_size)))
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

        start_idx = (int(page_number) - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered_display.iloc[start_idx:end_idx]

        st.caption(f"Showing {len(page_df)} of {total_rows} filtered rows.")
        st.dataframe(page_df, use_container_width=True)

        st.markdown("#### Export and Download Interface")
        csv_bytes = filtered_display.to_csv(index=False).encode("utf-8")

        excel_buffer = io.BytesIO()
        filtered_display.to_excel(excel_buffer, index=False)
        excel_bytes = excel_buffer.getvalue()

        pdf_bytes = build_prediction_pdf(filtered_display)

        e1, e2, e3 = st.columns(3)
        e1.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="predictions_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
        e2.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name="predictions_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        e3.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="predictions_export.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.session_state["shared_filters"] = {
            "sort_column": sort_column,
            "sort_ascending": sort_ascending,
            "search_query": search_query,
            "selected_categories": selected_categories,
            "selected_columns": selected_columns,
            "page_size": page_size,
            "page_number": int(page_number),
        }
        st.session_state["shared_data_object"] = filtered_display

elif page == "Visual Insights":
    st.subheader("Visual Insights")

    visual_insights_ui = st.container()
    with visual_insights_ui:
        st.markdown("### Visual Insights Summary")
        risk_distribution_chart_slot = st.empty()
        factor_impact_chart_slot = st.empty()
        additional_visual_summaries_slot = st.empty()

    with risk_distribution_chart_slot.container():
        fig_risk = px.histogram(
            df,
            x="Delay_Risk_Index",
            nbins=24,
            title="Risk Distribution",
            marginal="box",
            color_discrete_sequence=["#E76F51"],
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with factor_impact_chart_slot.container():
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        excluded_cols = {"Delay_Risk_Index", "PC1", "PC2", "Cluster"}
        candidate_cols = [c for c in numeric_cols if c not in excluded_cols]

        if candidate_cols:
            impact_df = (
                df[candidate_cols]
                .std(numeric_only=True)
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            impact_df.columns = ["Feature", "Impact Score"]

            fig_impact = px.bar(
                impact_df,
                x="Impact Score",
                y="Feature",
                orientation="h",
                title="Top Factor Impact Proxy (Std Dev)",
                color="Impact Score",
                color_continuous_scale="Tealgrn",
            )
            fig_impact.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_impact, use_container_width=True)
        else:
            st.info("No numeric factor columns available for impact chart.")

        kan78_factor_impact_chart_ui = st.container()
        with kan78_factor_impact_chart_ui:
            st.markdown("#### KAN-78 Factor Impact Chart UI")
            factor_impact_chart_title_slot = st.empty()
            factor_impact_chart_function_call_slot = st.empty()
            factor_impact_chart_render_slot = st.empty()
            factor_impact_chart_description_slot = st.empty()

    with additional_visual_summaries_slot.container():
        st.markdown("#### Additional Visual Summaries")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Average Risk", f"{df['Delay_Risk_Index'].mean():.2f}")
        col_b.metric("Median Risk", f"{df['Delay_Risk_Index'].median():.2f}")
        high_risk_count = int((df["Delay_Risk_Index"] >= 3.5).sum())
        col_c.metric("High-Risk Records", f"{high_risk_count}")

    visual_insights_pipeline_slot = st.container()
    with visual_insights_pipeline_slot:
        st.markdown("#### Visual Insights Pipeline Placeholders")
        visual_backend_data_loader_slot = st.empty()
        visual_chart_data_binding_slot = st.empty()
        visual_insights_generation_slot = st.empty()
        visual_components_update_slot = st.empty()

    kan57_risk_distribution_ui = st.container()
    with kan57_risk_distribution_ui:
        st.markdown("### KAN-57 Risk Distribution Charts")
        predicted_risk_distribution_slot = st.empty()
        risk_histogram_or_density_slot = st.empty()
        color_coded_risk_band_slot = st.empty()

        kan77_risk_distribution_chart_ui = st.container()
        with kan77_risk_distribution_chart_ui:
            st.markdown("#### KAN-77 Risk Distribution Chart UI")
            risk_distribution_chart_title_slot = st.empty()
            risk_distribution_chart_function_call_slot = st.empty()
            risk_distribution_chart_render_slot = st.empty()
            risk_distribution_chart_description_slot = st.empty()

elif page == "SHAP Visualizations":
    st.subheader("SHAP Visualizations")
    top5 = st.session_state.get("shared_shap_values")
    if top5 is None or len(top5) == 0:
        st.info("Run a prediction first to view SHAP visualizations.")
    else:
        st.markdown("### SHAP Summary")
        st.dataframe(top5, use_container_width=True)

        shap_chart_df = top5.copy()
        shap_chart_df["Abs Impact"] = shap_chart_df["SHAP Value"].abs()

        fig_shap = px.bar(
            shap_chart_df.sort_values("Abs Impact", ascending=True),
            x="Abs Impact",
            y="Feature",
            orientation="h",
            title="SHAP Feature Impact (Absolute)",
            color="SHAP Value",
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("Force and dependence plots can be added in a later SHAP-specific enhancement.")

elif page == "Metadata Visualizations":
    st.subheader("Metadata Visualizations")
    metadata = st.session_state.get("shared_metadata")
    if metadata is None:
        st.info("Run a prediction first to view metadata visualizations.")
    else:
        st.markdown("### Metadata Overview")
        st.json(metadata)

        metrics = metadata.get("metrics", {})
        valid_metrics = {k: v for k, v in metrics.items() if isinstance(v, (float, int))}
        if valid_metrics:
            metric_df = pd.DataFrame(
                {"Metric": list(valid_metrics.keys()), "Value": list(valid_metrics.values())}
            )
            fig_meta = px.bar(metric_df, x="Metric", y="Value", title="Model Metrics")
            st.plotly_chart(fig_meta, use_container_width=True)

elif page == "Risk Summary Visualizations":
    st.subheader("Risk Summary Visualizations")
    ensure_prediction_history_state()
    history_df = pd.DataFrame(st.session_state["prediction_history"])
    if history_df.empty:
        st.info("Run predictions first to build risk summary visualizations.")
    else:
        st.markdown("### Risk Summary")
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Predictions", len(history_df))
        s2.metric("Average Score", f"{history_df['risk_score'].mean():.2f}")
        s3.metric("High Risk Count", int((history_df["risk_category"] == "High Risk").sum()))

        trend_df = history_df.reset_index().rename(columns={"index": "Run"})
        fig_trend = px.line(trend_df, x="Run", y="risk_score", title="Risk Score Trend")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("Insight: use this trend to monitor whether risk signals are worsening or improving.")

elif page == "Risk Category Visualizations":
    st.subheader("Risk Category Visualizations")
    ensure_prediction_history_state()
    history_df = pd.DataFrame(st.session_state["prediction_history"])
    if history_df.empty:
        st.info("Run predictions first to view category visualizations.")
    else:
        cat_counts = history_df["risk_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        fig_cat_bar = px.bar(cat_counts, x="Category", y="Count", title="Risk Category Distribution")
        st.plotly_chart(fig_cat_bar, use_container_width=True)

        fig_cat_pie = px.pie(cat_counts, names="Category", values="Count", title="Category Share")
        st.plotly_chart(fig_cat_pie, use_container_width=True)
        st.caption("Insight: category balance helps prioritize mitigation resources.")

elif page == "Risk Distribution Visualizations":
    st.subheader("Risk Distribution Visualizations")
    ensure_prediction_history_state()
    history_df = pd.DataFrame(st.session_state["prediction_history"])
    if history_df.empty:
        st.info("Run predictions first to view distribution visualizations.")
    else:
        fig_hist = px.histogram(history_df, x="risk_score", nbins=15, title="Risk Score Histogram")
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_density = px.density_contour(history_df, x="risk_score", y="confidence")
        fig_density.update_layout(title="Risk Score vs Confidence Density")
        st.plotly_chart(fig_density, use_container_width=True)

elif page == "Factor Impact Visualizations":
    st.subheader("Factor Impact Visualizations")
    top5 = st.session_state.get("shared_shap_values")
    if top5 is None or len(top5) == 0:
        st.info("Run a prediction first to view factor impact visualizations.")
    else:
        impact_df = top5.copy()
        impact_df["Abs Impact"] = impact_df["SHAP Value"].abs()
        fig_factor = px.bar(
            impact_df.sort_values("Abs Impact", ascending=True),
            x="Abs Impact",
            y="Feature",
            orientation="h",
            title="Factor Impact and Contribution Breakdown",
            color="Abs Impact",
            color_continuous_scale="OrRd",
        )
        st.plotly_chart(fig_factor, use_container_width=True)
        st.dataframe(impact_df[["Feature", "SHAP Value", "Abs Impact"]], use_container_width=True)

elif page == "Delay Risk Prediction":
    st.subheader("Delay Risk Prediction")
    st.write("Adjust the project factors to estimate the delay risk index.")
    ensure_prediction_history_state()

    if df is None:
        st.error("Please upload and process a dataset before running predictions.")
        st.stop()
    assert df is not None
    prediction_df = cast(pd.DataFrame, df)

    try:
        load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load prediction artifacts: {exc}")
        st.stop()

    numeric_cols = prediction_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Delay_Risk_Index" in numeric_cols:
        numeric_cols.remove("Delay_Risk_Index")

    user_inputs = {}
    cols = st.columns(2)

    for i, col_name in enumerate(numeric_cols):
        with cols[i % 2]:
            default_val = (
                float(prediction_df[col_name].mean()) if col_name in prediction_df.columns else 0.0
            )
            user_inputs[col_name] = st.number_input(
                col_name,
                value=default_val,
            )

    if st.button("Predict Delay Risk"):
        backend_pipeline_ui = st.container()
        with backend_pipeline_ui:
            st.markdown("### Integration Pipeline Placeholders")
            backend_prediction_call_slot = st.empty()
            prediction_result_store_slot = st.empty()
            shap_pipeline_call_slot = st.empty()
            shap_values_store_slot = st.empty()
            shap_chart_reference_slot = st.empty()
            shap_explanation_display_slot = st.empty()
            metadata_extraction_call_slot = st.empty()
            metadata_store_slot = st.empty()
            metadata_display_slot = st.empty()
            risk_summary_call_slot = st.empty()
            risk_summary_output_store_slot = st.empty()
            risk_summary_display_slot = st.empty()
            risk_category_call_slot = st.empty()
            risk_category_output_store_slot = st.empty()
            risk_category_display_slot = st.empty()
            risk_distribution_call_slot = st.empty()
            risk_distribution_output_store_slot = st.empty()
            risk_distribution_chart_display_slot = st.empty()
            factor_impact_call_slot = st.empty()
            factor_impact_output_store_slot = st.empty()
            factor_impact_chart_display_slot = st.empty()

        result_display = st.container()
        with result_display:
            st.markdown("### Prediction Result")
            risk_score_slot = st.empty()
            risk_category_slot = st.empty()

        risk_summary_ui = st.container()
        with risk_summary_ui:
            st.markdown("### Risk Summary UI")
            risk_summary_score_slot = st.empty()
            risk_summary_category_slot = st.empty()
            risk_summary_confidence_slot = st.empty()
            risk_summary_factors_slot = st.empty()
            risk_summary_indicator_slot = st.empty()

        contributors_ui = st.container()
        with contributors_ui:
            st.markdown("### Top Contributing Factors")
            contributors_list_slot = st.empty()
            contributors_shap_values_slot = st.empty()
            contributors_visual_slot = st.empty()

        try:
            with st.spinner("Predicting delay risk..."):
                logger.info("Running prediction pipeline")
                score = predict_delay_risk(user_inputs)
                category = categorize_risk(score)
                confidence = prediction_confidence(user_inputs)
                top5 = top_contributors(user_inputs, top_n=5)
        except Exception as exc:
            logger.exception("Prediction flow failed")
            st.error(f"Prediction flow failed: {exc}")
            st.stop()

        risk_score_slot.success(f"Predicted Delay Risk Index: **{score:.2f}**")
        risk_category_slot.info(f"Risk Category: **{category}**")

        with risk_summary_score_slot.container():
            st.metric("Risk Score", f"{score:.2f}")

        with risk_summary_category_slot.container():
            st.metric("Risk Category", category)

        with risk_summary_confidence_slot.container():
            st.metric("Confidence Score", f"{confidence:.2f}")

        badge_color = "#2A9D8F"
        if category == "Medium Risk":
            badge_color = "#E9C46A"
        elif category == "High Risk":
            badge_color = "#E76F51"

        with risk_summary_indicator_slot.container():
            st.markdown(
                (
                    "<div style='padding:0.6rem 0.8rem;border-radius:0.5rem;"
                    f"background:{badge_color};color:#111;font-weight:700;width:fit-content;'>"
                    f"Risk Indicator: {category}</div>"
                ),
                unsafe_allow_html=True,
            )

        with risk_summary_factors_slot.container():
            st.markdown("Top factors are listed in the section below.")

        st.subheader("Top 5 Contributing Factors")
        with contributors_list_slot.container():
            st.dataframe(top5, use_container_width=True)

        with contributors_shap_values_slot.container():
            for _, row in top5.iterrows():
                feat = row["Feature"]
                val = row["SHAP Value"]
                st.write(f"- **{feat.replace('_', ' ')}** -> SHAP impact: {val:.4f}")

        with contributors_visual_slot.container():
            top5_chart = top5.copy()
            top5_chart["Abs Impact"] = top5_chart["SHAP Value"].abs()
            fig_top5 = px.bar(
                top5_chart.sort_values("Abs Impact", ascending=True),
                x="Abs Impact",
                y="Feature",
                orientation="h",
                title="Top Contributing Factors by Absolute SHAP Impact",
                color="Abs Impact",
                color_continuous_scale="OrRd",
            )
            st.plotly_chart(fig_top5, use_container_width=True)

        st.markdown("### Explanation Summary")
        contributor_pairs = list(top5[["Feature", "SHAP Value"]].itertuples(index=False, name=None))
        st.info(explanation_text(contributor_pairs))

        st.markdown("### Global Feature Importance")
        model, scaler, feature_columns = load_artifacts()

        background_df = prediction_df.reindex(columns=feature_columns).copy()
        background_df = background_df.apply(pd.to_numeric, errors="coerce")
        background_df = background_df.fillna(background_df.mean(numeric_only=True)).fillna(0.0)
        background_df = background_df.sample(min(200, len(background_df)), random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_vals_bg = explainer.shap_values(background_df)
        if isinstance(shap_vals_bg, list):
            shap_vals_bg = shap_vals_bg[0]

        fig2, _ = plt.subplots()
        shap.summary_plot(shap_vals_bg, background_df, show=False)
        st.pyplot(fig2)
        plt.close(fig2)

        metrics = {"MAE": None, "RMSE": None, "R2": None}
        if "Delay_Risk_Index" in prediction_df.columns:
            eval_X = prediction_df.reindex(columns=feature_columns).copy()
            eval_X = eval_X.apply(pd.to_numeric, errors="coerce")
            eval_X = eval_X.fillna(eval_X.mean(numeric_only=True)).fillna(0.0)

            y_true = pd.to_numeric(prediction_df["Delay_Risk_Index"], errors="coerce")
            valid_mask = ~y_true.isna()
            if valid_mask.any():
                X_scaled_eval = scaler.transform(eval_X.loc[valid_mask].values)
                y_pred_eval = model.predict(X_scaled_eval)
                y_true_eval = y_true.loc[valid_mask].values
                metrics = {
                    "MAE": float(mean_absolute_error(y_true_eval, y_pred_eval)),
                    "RMSE": float(np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))),
                    "R2": float(r2_score(y_true_eval, y_pred_eval)),
                }

        metadata = {
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, "n_estimators", None),
            "trained_on_rows": int(len(prediction_df)),
            "trained_on_features": feature_columns,
            "metrics": metrics,
        }
        st.markdown("### Model Metadata")
        st.json(metadata)

        prediction_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "risk_score": float(score),
            "risk_category": category,
            "confidence": float(confidence),
            "top_factor": str(top5.iloc[0]["Feature"]) if not top5.empty else None,
            "top_factor_shap": float(top5.iloc[0]["SHAP Value"]) if not top5.empty else None,
        }

        st.session_state["prediction_history"].append(prediction_result)
        st.session_state["shared_prediction_results"] = prediction_result
        st.session_state["shared_shap_values"] = top5
        st.session_state["shared_metadata"] = metadata
        st.session_state["shared_data_object"] = prediction_df
        st.session_state["shared_charts"] = {
            "top_factor_impact": top5[["Feature", "SHAP Value"]].to_dict("records") if not top5.empty else []
        }

        backend_prediction_call_slot.success("Prediction function call completed.")
        prediction_result_store_slot.info("Prediction result stored in shared session state.")
        shap_pipeline_call_slot.success("SHAP pipeline call completed.")
        shap_values_store_slot.info("SHAP values stored in shared session state.")
        shap_chart_reference_slot.info("SHAP chart data reference prepared.")
        shap_explanation_display_slot.info("SHAP explanation text prepared for display.")
        metadata_extraction_call_slot.success("Metadata extraction completed.")
        metadata_store_slot.info("Metadata stored in shared session state.")
        metadata_display_slot.info("Metadata display updated in UI.")
        risk_summary_call_slot.success("Risk summary pipeline completed.")
        risk_summary_output_store_slot.info("Risk summary output stored.")
        risk_summary_display_slot.info("Risk summary UI updated.")
        risk_category_call_slot.success("Risk category pipeline completed.")
        risk_category_output_store_slot.info("Risk category output stored.")
        risk_category_display_slot.info("Risk category UI updated.")
        risk_distribution_call_slot.success("Risk distribution pipeline completed.")
        risk_distribution_output_store_slot.info("Risk distribution output stored.")
        risk_distribution_chart_display_slot.info("Risk distribution chart references updated.")
        factor_impact_call_slot.success("Factor impact pipeline completed.")
        factor_impact_output_store_slot.info("Factor impact output stored.")
        factor_impact_chart_display_slot.info("Factor impact chart references updated.")