import numpy as np
import CDRPS.prediction_pipeline as pp

class _FakeTree:
    def __init__(self, val): self._val = val
    def predict(self, X): return np.array([self._val])

class _FakeForest:
    def __init__(self, vals): self.estimators_ = [_FakeTree(v) for v in vals]

def test_confidence_mocked(monkeypatch):
    forest = _FakeForest([3.0, 3.0, 3.0])
    monkeypatch.setattr(pp, "load_artifacts", lambda: (forest, None, ["f1"]))
    monkeypatch.setattr(pp, "prepare_features", lambda d: np.zeros((1, 1)))
    assert pp.prediction_confidence({"f1": 1.0}) == 1.0