import pytest

from CDRPS.prediction_pipeline import categorize_risk

@pytest.mark.parametrize(
    "score, expected",
    [
         (0.0, "Low Risk"),
        (1.99, "Low Risk"),
        (2.0, "Medium Risk"),
        (3.49, "Medium Risk"),
        (3.5, "High Risk"),
        (5.0, "High Risk"),
        (-1.0, "Low Risk"),   # Exercise 1: negative score
        (10.0, "High Risk"),  # Exercise 1: score greater than 5
    ],
)
def test_categorize_risk_buckets(score, expected):
    assert categorize_risk(score) == expected