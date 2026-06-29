"""Phase 2 — probability calibration for the champion model.

`class_weight='balanced'` deliberately distorts the model's probability outputs
(it trades calibration for recall on the minority class). That's fine for ranking
metrics like AUC, but the business simulation prices loans off the *probability*
of default — so those probabilities must be calibrated, or the expected
profit/loss numbers are built on sand.

This fits an isotonic calibrator on a held-out slice of the training data
(prefit strategy — no leakage into the test set), then reports raw vs calibrated
Brier score and reliability curves. AUC is unchanged by isotonic calibration
(it is monotonic), which is exactly why we calibrate: same discrimination,
trustworthy probabilities.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.config import (
    get_models_dir,
    get_raw_data_path,
    get_reports_dir,
    load_model_config,
)
from credit_risk.data.ingest import load_raw_data, validate_raw_data
from credit_risk.modeling.pipeline import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_feature_lists,
    build_model_pipeline,
)

LOG = logging.getLogger(__name__)


def _make_calibrator(estimator):
    """CalibratedClassifierCV with prefit estimator, across sklearn versions."""
    try:  # sklearn >= 1.2
        return CalibratedClassifierCV(estimator=estimator, method="isotonic", cv="prefit")
    except TypeError:  # pragma: no cover - older sklearn
        return CalibratedClassifierCV(base_estimator=estimator, method="isotonic", cv="prefit")


def _reliability_rows(method: str, y_true, y_proba, n_bins: int = 10) -> list[dict]:
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return [
        {"method": method, "mean_predicted": round(float(mp), 6), "fraction_positive": round(float(fp), 6)}
        for mp, fp in zip(mean_pred, frac_pos)
    ]


def run_calibrate() -> dict:
    """Fit an isotonic calibrator and write calibration metrics + reliability curve."""
    model_config = load_model_config()
    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(get_raw_data_path())
    validate_raw_data(df)

    target_col = model_config.get("target", TARGET_COLUMN)
    id_col = model_config.get("id_column", ID_COLUMN)
    numeric_cols, categorical_cols = build_feature_lists(df, target_col=target_col, id_col=id_col)

    y = df[target_col].astype(int)
    X = df.drop(columns=[c for c in (target_col, id_col) if c in df.columns])

    split_cfg = model_config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    model_block = model_config.get("model", {})
    model_type = (model_block.get("type") or "lightgbm").lower()
    model_params = dict(model_block.get("params", {}) or {})
    model_params.setdefault("random_state", random_state)

    # Same outer split as train.py -> identical held-out test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Inner split: fit the model on one part, calibrate on the other (prefit).
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train
    )

    LOG.info("Fitting champion pipeline on %d rows, calibrating on %d", len(y_fit), len(y_calib))
    pipeline = build_model_pipeline(numeric_cols, categorical_cols, model_type, model_params)
    pipeline.fit(X_fit, y_fit)

    calibrator = _make_calibrator(pipeline)
    calibrator.fit(X_calib, y_calib)

    raw_proba = pipeline.predict_proba(X_test)[:, 1]
    cal_proba = calibrator.predict_proba(X_test)[:, 1]

    metrics = {
        "auc_roc": round(roc_auc_score(y_test, cal_proba), 6),  # unchanged by isotonic
        "raw_brier": round(brier_score_loss(y_test, raw_proba), 6),
        "calibrated_brier": round(brier_score_loss(y_test, cal_proba), 6),
        "raw_log_loss": round(log_loss(y_test, raw_proba), 6),
        "calibrated_log_loss": round(log_loss(y_test, cal_proba), 6),
    }
    metrics["brier_improvement"] = round(metrics["raw_brier"] - metrics["calibrated_brier"], 6)
    pd.DataFrame([metrics]).to_csv(reports_dir / "calibration_metrics.csv", index=False)
    LOG.info(
        "Calibration: Brier %.4f -> %.4f (AUC %.4f unchanged)",
        metrics["raw_brier"], metrics["calibrated_brier"], metrics["auc_roc"],
    )

    rows = _reliability_rows("raw", y_test, raw_proba) + _reliability_rows("calibrated", y_test, cal_proba)
    pd.DataFrame(rows).to_csv(reports_dir / "calibration_curve.csv", index=False)
    LOG.info("Wrote %s", reports_dir / "calibration_curve.csv")

    # Persist the calibrated model (gitignored models dir).
    try:
        import joblib

        models_dir = get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrator, models_dir / "credit_risk_pipeline_calibrated.joblib")
    except Exception as exc:  # pragma: no cover - serialization is best-effort
        LOG.warning("Could not save calibrated model: %s", exc)

    return metrics
