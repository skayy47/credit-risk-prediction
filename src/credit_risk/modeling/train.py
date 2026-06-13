"""Train baseline/champion model, evaluate on test set, save artifacts."""
import json
import logging
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from credit_risk.config import (
    get_raw_data_path,
    get_reports_dir,
    get_predictions_dir,
    get_models_dir,
    load_model_config,
)
from credit_risk.data.ingest import load_raw_data, validate_raw_data
from credit_risk.modeling.pipeline import TARGET_COLUMN, ID_COLUMN, build_feature_lists, build_model_pipeline
from credit_risk.modeling.simulate import (
    write_predictions_csv,
    write_decision_simulation_csv,
)

LOG = logging.getLogger(__name__)

_SHAP_SAMPLE_SIZE = 5_000  # rows used for SHAP computation (speed/memory trade-off)


def _ensure_required_columns(df: pd.DataFrame) -> None:
    for col in (TARGET_COLUMN, ID_COLUMN):
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")


def _get_X_y_ids(df: pd.DataFrame, target_col: str = TARGET_COLUMN, id_col: str = ID_COLUMN):
    y = df[target_col]
    ids = df[id_col]
    X = df.drop(columns=[target_col, id_col])
    return X, y, ids


def _evaluate_binary_proba(y_true: pd.Series, y_proba: pd.Series) -> dict:
    y_true = y_true.astype(int)
    return {
        "auc_roc": round(roc_auc_score(y_true, y_proba), 6),
        "average_precision": round(average_precision_score(y_true, y_proba), 6),
        "brier_score": round(brier_score_loss(y_true, y_proba), 6),
        "log_loss": round(log_loss(y_true, y_proba), 6),
    }


def _save_metrics(metrics: dict, out_path: Path) -> None:
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)


def _write_metrics_compare(reports_dir: Path) -> None:
    baseline_path = reports_dir / "metrics_baseline.csv"
    champion_path = reports_dir / "metrics_champion.csv"
    if not baseline_path.exists() or not champion_path.exists():
        return
    baseline_df = pd.read_csv(baseline_path)
    champion_df = pd.read_csv(champion_path)
    baseline_df["model"] = "Baseline (no class weight)"
    champion_df["model"] = "Champion (class_weight=balanced)"
    compare = pd.concat([baseline_df, champion_df], ignore_index=True)
    cols = ["model"] + [c for c in compare.columns if c != "model"]
    compare[cols].to_csv(reports_dir / "metrics_compare.csv", index=False)
    LOG.info("Wrote %s", reports_dir / "metrics_compare.csv")


def _write_confusion_matrix(y_true: pd.Series, y_proba: pd.Series, threshold: float, out_path: Path) -> None:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred)
    tn, fp, fn, tp = cm.ravel()
    df = pd.DataFrame([{
        "threshold": threshold,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 6),
        "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 6),
    }])
    df.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)


def _write_calibration_bins(y_true: pd.Series, y_proba: pd.Series, n_bins: int = 10, out_path: Path = None) -> None:
    bins = pd.cut(y_proba, bins=n_bins, include_lowest=True)
    result = y_true.astype(int).groupby(bins, observed=False).agg(
        n=("count"), actual_default_rate=("mean")
    ).reset_index()
    result.columns = ["prob_bin", "n", "actual_default_rate"]
    result["predicted_midpoint"] = result["prob_bin"].apply(lambda x: round((x.left + x.right) / 2, 4))
    result = result[["prob_bin", "predicted_midpoint", "n", "actual_default_rate"]]
    result.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)


def _compute_shap_importance(pipeline, X_sample: pd.DataFrame, out_path: Path) -> None:
    try:
        import shap
    except ImportError:
        LOG.warning("shap not installed — skipping SHAP feature importance. Install with: pip install shap")
        return

    LOG.info("Computing SHAP feature importance on %d rows...", len(X_sample))
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(preprocessor.transform(X_sample.head(1)).shape[1])]

    X_transformed = preprocessor.transform(X_sample)

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    except Exception as exc:
        LOG.warning("TreeExplainer failed (%s) — falling back to model feature_importances_", exc)
        if hasattr(classifier, "feature_importances_"):
            mean_abs_shap = classifier.feature_importances_
        else:
            LOG.warning("No feature_importances_ available — skipping SHAP")
            return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Clean up sklearn column transformer prefix (e.g. "num__AMT_INCOME" → "AMT_INCOME")
    importance_df["feature"] = importance_df["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    importance_df.to_csv(out_path, index=False)
    LOG.info("Wrote SHAP importance for %d features to %s", len(importance_df), out_path)


def _save_pipeline(pipeline, out_dir: Path, label: str) -> Path:
    try:
        import joblib
    except ImportError:
        LOG.warning("joblib not available — skipping model serialization")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"credit_risk_pipeline_{label}.joblib"
    import joblib
    joblib.dump(pipeline, out_path)
    LOG.info("Saved pipeline to %s", out_path)
    return out_path


def train_evaluate_and_save(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    reports_dir: Path,
) -> dict:
    LOG.info("Fitting pipeline on train set (%d rows)", len(y_train))
    pipeline.fit(X_train, y_train)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Trained pipeline does not support predict_proba")

    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    metrics = _evaluate_binary_proba(y_test, pd.Series(y_proba_test))
    LOG.info("Test metrics: auc_roc=%.4f, average_precision=%.4f", metrics["auc_roc"], metrics["average_precision"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    return metrics, pd.Series(y_proba_test, index=y_test.index)


def _save_model_metadata(
    *,
    out_dir: Path,
    run_id: str,
    created_at: str,
    model_type: str,
    numeric_features: list,
    categorical_features: list,
    metrics: dict,
    train_time_seconds: float,
    label: str = "baseline",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "label": label,
        "created_at": created_at,
        "model_type": model_type,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_features": len(numeric_features) + len(categorical_features),
        "metrics": metrics,
        "train_time_seconds": round(train_time_seconds, 3),
    }
    out_path = out_dir / f"model_metadata_{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Wrote %s", out_path)
    return out_path


def run_train_simulate() -> None:
    """Load data, train model, evaluate, write all artifacts."""
    raw_path = get_raw_data_path()
    reports_dir = get_reports_dir()
    predictions_dir = get_predictions_dir()
    model_config = load_model_config()
    from credit_risk.config import validate_model_config
    validate_model_config(model_config)

    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading raw data from %s", raw_path)
    df = load_raw_data(raw_path)
    validate_raw_data(df)
    _ensure_required_columns(df)

    target_col = model_config.get("target", TARGET_COLUMN)
    id_col = model_config.get("id_column", ID_COLUMN)
    model_label = model_config.get("label", "baseline")

    numeric_cols, categorical_cols = build_feature_lists(df, target_col=target_col, id_col=id_col)
    LOG.info("Features: %d numeric, %d categorical", len(numeric_cols), len(categorical_cols))

    model_block = model_config.get("model", {})
    model_type = (model_block.get("type") or "lightgbm").lower()
    model_params = dict(model_block.get("params", {}) or {})

    split_cfg = model_config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)
    if "random_state" not in model_params:
        model_params.setdefault("random_state", random_state)

    pipeline = build_model_pipeline(numeric_cols, categorical_cols, model_type, model_params)

    X, y, ids = _get_X_y_ids(df, target_col=target_col, id_col=id_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    start_time = time()
    effective_model_type = model_type
    try:
        metrics, y_proba_test = train_evaluate_and_save(
            pipeline, X_train, X_test, y_train, y_test, reports_dir
        )
    except Exception as exc:
        if model_type == "lightgbm":
            LOG.warning("LightGBM failed (%s). Falling back to LogisticRegression.", exc)
            from credit_risk.modeling.pipeline import _sanitize_model_params
            sanitized, dropped = _sanitize_model_params("logreg", model_params or {})
            if dropped:
                LOG.warning("Dropping unsupported LogisticRegression params: %s", dropped)
            pipeline = build_model_pipeline(numeric_cols, categorical_cols, "logreg", sanitized)
            effective_model_type = "logreg"
            try:
                metrics, y_proba_test = train_evaluate_and_save(
                    pipeline, X_train, X_test, y_train, y_test, reports_dir
                )
            except Exception as fallback_exc:
                raise RuntimeError("Both LightGBM and LogisticRegression failed") from fallback_exc
        else:
            raise

    # Write metrics file based on label
    metrics_filename = f"metrics_{model_label}.csv"
    _save_metrics(metrics, reports_dir / metrics_filename)

    # If this is the champion run, also compare with baseline
    if model_label == "champion":
        _write_metrics_compare(reports_dir)

    # Confusion matrix at business default threshold (0.2)
    business_threshold = float(model_config.get("business_threshold", 0.2))
    _write_confusion_matrix(y_test, y_proba_test, threshold=business_threshold, out_path=reports_dir / "confusion_matrix.csv")

    # Calibration bins
    _write_calibration_bins(y_test, y_proba_test, n_bins=10, out_path=reports_dir / "calibration_bins.csv")

    # SHAP feature importance (sample from test set)
    sample_size = min(_SHAP_SAMPLE_SIZE, len(X_test))
    X_shap_sample = X_test.sample(n=sample_size, random_state=42)
    _compute_shap_importance(pipeline, X_shap_sample, out_path=reports_dir / "feature_importance_shap.csv")

    # Save model pipeline
    models_dir = get_models_dir()
    _save_pipeline(pipeline, models_dir, model_label)

    # Full-dataset predictions for BI
    LOG.info("Generating predictions for full dataset (%d rows)", len(X))
    y_proba_full = pipeline.predict_proba(X)[:, 1]

    from credit_risk.modeling.simulate import _run_id, _created_at
    run_id = _run_id()
    created_at = _created_at()

    write_predictions_csv(
        application_id=ids,
        y_true=y,
        y_proba=pd.Series(y_proba_full),
        out_dir=predictions_dir,
        run_id=run_id,
        created_at=created_at,
    )

    thresholds_cfg = model_config.get("thresholds", {})
    business_cfg = model_config.get("business", {})
    write_decision_simulation_csv(
        y_true=y,
        y_proba=pd.Series(y_proba_full),
        out_dir=predictions_dir,
        threshold_start=thresholds_cfg.get("start", 0.05),
        threshold_stop=thresholds_cfg.get("stop", 0.95),
        threshold_step=thresholds_cfg.get("step", 0.05),
        profit_if_good=business_cfg.get("profit_if_good", 0.2),
        loss_given_default=business_cfg.get("loss_given_default", 1.0),
    )

    train_time = time() - start_time
    _save_model_metadata(
        out_dir=models_dir,
        run_id=run_id,
        created_at=created_at,
        model_type=effective_model_type,
        numeric_features=list(numeric_cols),
        categorical_features=list(categorical_cols),
        metrics=metrics,
        train_time_seconds=train_time,
        label=model_label,
    )

    LOG.info(
        "train-simulate completed [label=%s, auc_roc=%.4f, time=%.1fs]",
        model_label, metrics["auc_roc"], train_time,
    )
