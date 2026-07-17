from CDRPS.shap_explain import explanation_text

def test_explanation_text():
    contributors = [("Financing_2", 0.18), ("Labor_1", -0.05)]
    text = explanation_text(contributors)
    assert "increased" in text and "Financing 2" in text
    assert explanation_text([]) == "No contributing factors available."