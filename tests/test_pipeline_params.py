from credit_risk.modeling.pipeline import build_model_pipeline
import pytest
import logging


def test_logreg_drops_lgbm_params(caplog):
    # When providing LGBM params to explicit logreg, we should drop unsupported keys and warn
    caplog.set_level(logging.WARNING)
    pipe = build_model_pipeline(numeric_cols=["a"], categorical_cols=[], model_type="logreg", model_params={"n_estimators": 100, "max_iter": 200})
    assert pipe is not None
    # Ensure warning logged about dropping n_estimators
    assert any("Dropping unsupported LogisticRegression params" in m.message for m in caplog.records)
    clf = pipe.named_steps["classifier"]
    params = clf.get_params()
    assert "n_estimators" not in params
    assert params.get("max_iter") == 200


def test_build_pipeline_accepts_logreg_params():
    # Should accept legit LogisticRegression params like 'max_iter'
    pipe = build_model_pipeline(numeric_cols=["a"], categorical_cols=[], model_type="logreg", model_params={"max_iter": 200})
    assert pipe is not None
