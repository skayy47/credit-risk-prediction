import logging
from credit_risk.modeling import pipeline as pipe_mod
from credit_risk.modeling.pipeline import build_model_pipeline
import pytest


def test_fallback_when_lgbm_missing_drops_params_and_logs(caplog, monkeypatch):
    # Simulate LightGBM not installed
    monkeypatch.setattr(pipe_mod, "LGBMClassifier", None)
    caplog.set_level(logging.WARNING)

    # Config requests lightgbm but supplies LGBM-only params
    params = {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.05, "max_iter": 50}

    pipe = build_model_pipeline(numeric_cols=["f"], categorical_cols=[], model_type="lightgbm", model_params=params)
    assert pipe is not None
    # Should log a message about falling back
    assert any("falling to" in m.message.lower() or "falling back" in m.message.lower() or "lightgbm not available" in m.message.lower() for m in caplog.records)
    # Should log dropped params
    assert any("Dropping unsupported LogisticRegression params" in m.message for m in caplog.records)
    # classifier should be LogisticRegression and not have LGBM-only params
    clf = pipe.named_steps["classifier"]
    clf_params = clf.get_params()
    assert "n_estimators" not in clf_params
    assert "num_leaves" not in clf_params
    # Confirm accepted param max_iter is applied
    assert clf_params.get("max_iter") == 50
