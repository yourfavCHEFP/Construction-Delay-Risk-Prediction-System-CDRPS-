from pathlib import Path
import pandas as pd
import pytest
import numpy as np
from CDRPS.prediction_pipeline import predict_delay_risk, prepare_features
from CDRPS.shap_explain import top_contributors


MODEL = Path("CDRPS/models/model.pkl")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not MODEL.exists(), reason="No model")]

def test_top_contributors_shape():
    # Exercise 10: Shape & Order
    res = top_contributors({"f1": 3.0}, top_n=5)
    assert len(res) == 5
    assert res["SHAP Value"].iloc[0] >= res["SHAP Value"].iloc[1]
    

def test_full_pipeline(valid_input_dict):
    score = predict_delay_risk(valid_input_dict)
    assert isinstance(score, float)
    # Exercise 12: Mini E2E
    assert predict_delay_risk(valid_input_dict) == pytest.approx(
    predict_delay_risk(valid_input_dict),
    rel=1e-12
    )


def test_prepare_features_width():
    # Exercise 11: Full width check
    input_data = {"Materials.1": 4.0}
    features = prepare_features(input_data)
    assert features.shape[1] == 51 # Must be full width
    assert np.isfinite(features[0,0])# First feature set