"""Factor impact chart placeholders for KAN-58."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _resolve_factor_source(
    shap_values: Optional[Iterable[float]],
    aggregated_importance: Optional[Sequence[Tuple[str, float]]],
) -> str:
    """Identify which input source should drive the factor impact chart."""
    if aggregated_importance is not None:
        return "aggregated_importance"
    if shap_values is not None:
        return "shap_values"
    raise ValueError("Provide shap_values or aggregated_importance.")


def _normalize_factor_input(
    source: str,
    shap_values: Optional[Iterable[float]],
    aggregated_importance: Optional[Sequence[Tuple[str, float]]],
) -> List[Tuple[str, float]]:
    """Prepare a normalized factor-impact table for charting.

    This is a structural placeholder; transformation rules will be implemented later.
    """
    raise NotImplementedError("KAN-62 placeholder: factor input normalization not implemented yet.")


def _prepare_chart_spec(normalized_factors: List[Tuple[str, float]]) -> Dict[str, object]:
    """Prepare chart-spec metadata for a future bar/horizontal impact chart."""
    raise NotImplementedError("KAN-62 placeholder: chart specification not implemented yet.")


def build_factor_impact_chart(
    shap_values: Optional[Iterable[float]] = None,
    aggregated_importance: Optional[Sequence[Tuple[str, float]]] = None,
):
    """Prepare a factor impact chart object from SHAP values or aggregated importance.

    Intended responsibilities (to be implemented later):
    - Accept SHAP values or aggregated factor importance.
    - Generate a bar chart or horizontal impact chart.
    - Return a chart object suitable for dashboard rendering.
    """
    source = _resolve_factor_source(shap_values, aggregated_importance)

    normalized_factors = _normalize_factor_input(
        source=source,
        shap_values=shap_values,
        aggregated_importance=aggregated_importance,
    )

    chart_spec = _prepare_chart_spec(normalized_factors)

    # Placeholder for the future chart object creation step.
    _ = chart_spec
    raise NotImplementedError("KAN-62 placeholder: factor impact chart creation not implemented yet.")
