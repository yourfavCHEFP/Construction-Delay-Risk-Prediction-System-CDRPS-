import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "CDRPS" / "models"

@pytest.fixture
def sample_survey_df():
    return pd.DataFrame(
        {
            "Materials.1": [4, 2, 5, 3],
            "Financing.2": [5, 1, 4, 2],
            "Schedule.1": [3, 3, 4, 1],
        }
    )

@pytest.fixture
def feature_columns():
    path = MODELS_DIR / "feature_columns.json"
    if path.exists():
        return json.loads(path.read_text())
    return [f"feature_{i}" for i in range(51)]  # Fallback if file doesn't exist
# The 51 feature names the model expects.

@pytest.fixture
def valid_input_dict(feature_columns):
    return {col: 3.0 for col in feature_columns} # A full, valid prediction input.