import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.modeling.train import run_train_simulate
from credit_risk.modeling import pipeline as pipe_mod


def test_train_simulate_fallback_to_logreg(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(pipe_mod, "LGBMClassifier", None)

    n = 60
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "SK_ID_CURR": range(1000, 1000 + n),
        "TARGET": rng.integers(0, 2, size=n).astype(int),
        "feat1": rng.standard_normal(n),
        "feat2": rng.integers(0, 3, size=n).astype(int),
    })

    import credit_risk.modeling.train as train_mod

    monkeypatch.setattr(train_mod, "load_raw_data", lambda p: df)
    monkeypatch.setattr(train_mod, "validate_raw_data", lambda d: None)

    # Patch names in the train module's own namespace (not credit_risk.config)
    monkeypatch.setattr(train_mod, "get_reports_dir", lambda: tmp_path / "reports")
    monkeypatch.setattr(train_mod, "get_predictions_dir", lambda: tmp_path / "preds")
    monkeypatch.setattr(train_mod, "get_models_dir", lambda: tmp_path / "models")

    run_train_simulate()

    preds = tmp_path / "preds"
    reports = tmp_path / "reports"
    assert (preds / "predictions.csv").exists(), "predictions.csv must be written"
    assert (preds / "decision_simulation.csv").exists(), "decision_simulation.csv must be written"
    assert (reports / "metrics_champion.csv").exists(), "metrics_champion.csv must be written"
    assert any(
        "falling back" in r.message.lower() or "dropping unsupported" in r.message.lower()
        for r in caplog.records
    ), "Expected fallback warning in logs"
