import io
import logging

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

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

    predictions_ui = st.container()
    with predictions_ui:
        st.markdown("### Predictions Table")
        predictions_description_slot = st.empty()
        predictions_sort_filter_controls_slot = st.empty()
        predictions_table_slot = st.empty()

        with predictions_sort_filter_controls_slot.container():
            st.markdown("#### KAN-60 Sort and Filter Controls")
            predictions_sort_controls_slot = st.empty()
            predictions_filter_controls_slot = st.empty()
            predictions_search_bar_slot = st.empty()
            predictions_column_selection_slot = st.empty()
            predictions_pagination_slot = st.empty()

        predictions_export_download_slot = st.empty()
        with predictions_export_download_slot.container():
            st.markdown("#### KAN-70 Export and Download Interface")
            predictions_csv_export_slot = st.empty()
            predictions_pdf_export_slot = st.empty()
            predictions_download_options_slot = st.empty()
            predictions_future_export_formats_slot = st.empty()

        predictions_pipeline_integration_slot = st.empty()
        with predictions_pipeline_integration_slot.container():
            st.markdown("#### Backend Integration Placeholders")
            predictions_list_loader_slot = st.empty()
            predictions_dataframe_store_slot = st.empty()
            predictions_table_population_slot = st.empty()
            predictions_filter_application_slot = st.empty()
            predictions_export_read_slot = st.empty()
            predictions_export_dataframe_pass_slot = st.empty()
            predictions_csv_trigger_slot = st.empty()
            predictions_excel_trigger_slot = st.empty()
            predictions_pdf_trigger_slot = st.empty()
            predictions_download_bytes_slot = st.empty()

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

    shap_visualizations_ui = st.container()
    with shap_visualizations_ui:
        st.markdown("### SHAP Visualization Pipeline Placeholders")
        shap_summary_plot_slot = st.empty()
        shap_bar_chart_slot = st.empty()
        shap_force_plot_slot = st.empty()
        shap_dependence_plot_slot = st.empty()
        shap_values_input_slot = st.empty()
        shap_plot_generation_slot = st.empty()
        shap_visual_components_slot = st.empty()
        shap_metadata_usage_slot = st.empty()

elif page == "Metadata Visualizations":
    st.subheader("Metadata Visualizations")

    metadata_visualizations_ui = st.container()
    with metadata_visualizations_ui:
        st.markdown("### Metadata Visualization Pipeline Placeholders")
        metadata_backend_loader_slot = st.empty()
        metadata_chart_binding_slot = st.empty()
        metadata_summary_generation_slot = st.empty()
        metadata_visual_update_slot = st.empty()

elif page == "Risk Summary Visualizations":
    st.subheader("Risk Summary Visualizations")

    risk_summary_visualizations_ui = st.container()
    with risk_summary_visualizations_ui:
        st.markdown("### Risk Summary Visualization Placeholders")
        risk_summary_charts_slot = st.empty()
        risk_summary_stats_slot = st.empty()
        risk_summary_insights_slot = st.empty()

elif page == "Risk Category Visualizations":
    st.subheader("Risk Category Visualizations")

    risk_category_visualizations_ui = st.container()
    with risk_category_visualizations_ui:
        st.markdown("### Risk Category Visualization Placeholders")
        category_distribution_charts_slot = st.empty()
        category_comparison_charts_slot = st.empty()
        category_insights_slot = st.empty()

elif page == "Risk Distribution Visualizations":
    st.subheader("Risk Distribution Visualizations")

    risk_distribution_visualizations_ui = st.container()
    with risk_distribution_visualizations_ui:
        st.markdown("### Risk Distribution Visualization Placeholders")
        risk_distribution_charts_slot = st.empty()
        risk_distribution_histograms_slot = st.empty()
        risk_distribution_density_plots_slot = st.empty()

elif page == "Factor Impact Visualizations":
    st.subheader("Factor Impact Visualizations")

    factor_impact_visualizations_ui = st.container()
    with factor_impact_visualizations_ui:
        st.markdown("### Factor Impact Visualization Placeholders")
        factor_impact_charts_slot = st.empty()
        feature_importance_visuals_slot = st.empty()
        contribution_breakdowns_slot = st.empty()

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

        with st.spinner("Predicting delay risk..."):
            score = predict_delay_risk(user_inputs)
            category = categorize_risk(score)
            confidence = prediction_confidence(user_inputs)
            top5 = top_contributors(user_inputs, top_n=5)

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

        background_df = df.reindex(columns=feature_columns).copy()
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
        if "Delay_Risk_Index" in df.columns:
            eval_X = df.reindex(columns=feature_columns).copy()
            eval_X = eval_X.apply(pd.to_numeric, errors="coerce")
            eval_X = eval_X.fillna(eval_X.mean(numeric_only=True)).fillna(0.0)

            y_true = pd.to_numeric(df["Delay_Risk_Index"], errors="coerce")
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
            "trained_on_rows": int(len(df)),
            "trained_on_features": feature_columns,
            "metrics": metrics,
        }
        st.markdown("### Model Metadata")
        st.json(metadata)