"""Unit tests for Phase 2 calibration helpers (no heavy data required)."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from credit_risk.modeling import calibrate as C


def test_make_calibrator_returns_prefit_isotonic():
    cal = C._make_calibrator(LogisticRegression())
    assert isinstance(cal, CalibratedClassifierCV)
    assert cal.method == "isotonic"
    assert cal.cv == "prefit"


def test_reliability_rows_structure_and_labels():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=400)
    # Probabilities correlated with the label so both bins are populated.
    y_proba = np.clip(0.2 + 0.5 * y_true + rng.normal(0, 0.15, size=400), 0, 1)
    rows = C._reliability_rows("raw", y_true, y_proba, n_bins=5)
    assert len(rows) >= 1
    for r in rows:
        assert r["method"] == "raw"
        assert 0.0 <= r["mean_predicted"] <= 1.0
        assert 0.0 <= r["fraction_positive"] <= 1.0


def test_reliability_rows_method_label_passthrough():
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.25])
    rows = C._reliability_rows("calibrated", y_true, y_proba, n_bins=3)
    assert all(r["method"] == "calibrated" for r in rows)
