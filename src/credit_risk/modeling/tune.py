"""Phase 2 — Optuna hyperparameter tuning for the LightGBM champion.

Runs a stratified k-fold CV study (objective = mean validation AUC) over the
LightGBM search space, using early stopping inside each fold to pick the number
of trees. The best params (plus the averaged best iteration count) are written
back into ``configs/model.yaml`` so a subsequent ``train-simulate`` uses them.

Tuning uses the SAME train/test split as train.py (test_size + random_state from
configs/model.yaml), and only ever sees the train portion — the test set stays
untouched for honest final evaluation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from credit_risk.config import (
    get_raw_data_path,
    get_reports_dir,
    load_model_config,
    project_root,
)
from credit_risk.data.ingest import load_raw_data, validate_raw_data
from credit_risk.modeling.pipeline import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_feature_lists,
    build_preprocessor,
)

LOG = logging.getLogger(__name__)

_DEFAULT_TUNING = {
    "n_trials": 60,
    "timeout_seconds": 1500,
    "cv_folds": 3,
    "early_stopping_rounds": 50,
    "max_estimators": 2000,
    "random_state": 42,
    "sample_size": None,  # if set, tune on a stratified subsample for speed
}


def _objective(trial, X, y, numeric_cols, categorical_cols, base_params, tuning):
    import lightgbm as lgb

    params = dict(base_params)
    params.update(
        {
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    )

    skf = StratifiedKFold(n_splits=tuning["cv_folds"], shuffle=True, random_state=tuning["random_state"])
    fold_aucs, fold_iters = [], []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Preprocessor is fit on the train fold ONLY (no leakage).
        pre = build_preprocessor(numeric_cols, categorical_cols)
        X_tr_t = pre.fit_transform(X_tr)
        X_va_t = pre.transform(X_va)

        model = lgb.LGBMClassifier(n_estimators=tuning["max_estimators"], **params)
        model.fit(
            X_tr_t,
            y_tr,
            eval_set=[(X_va_t, y_va)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(tuning["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        proba = model.predict_proba(X_va_t)[:, 1]
        fold_aucs.append(roc_auc_score(y_va, proba))
        fold_iters.append(model.best_iteration_ or tuning["max_estimators"])

    trial.set_user_attr("mean_best_iteration", int(np.mean(fold_iters)))
    return float(np.mean(fold_aucs))


def _write_tuned_params(best_params: dict, n_estimators: int, base_params: dict) -> Path:
    """Merge tuned params into configs/model.yaml -> model.params (preserving structure)."""
    root = project_root()
    path = root / "configs" / "model.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    merged = dict(base_params)
    merged.update(best_params)
    # Floor the tree count: tuning may early-stop on a subsample, but the final
    # retrain runs on all data and benefits from enough trees.
    merged["n_estimators"] = max(int(n_estimators), 300)
    # Round floats for readability.
    for k, v in list(merged.items()):
        if isinstance(v, float):
            merged[k] = round(v, 6)

    cfg.setdefault("model", {})["params"] = merged
    cfg["model"]["type"] = "lightgbm"
    cfg["model"]["tuned"] = True

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    LOG.info("Wrote tuned params to %s", path)
    return path


def run_tune() -> dict:
    """Run the Optuna study and persist the best hyperparameters."""
    import optuna

    model_config = load_model_config()
    tuning = dict(_DEFAULT_TUNING)
    tuning.update(model_config.get("tuning", {}) or {})

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
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Optional stratified subsample for tuning speed (final retrain still uses all data).
    sample_size = tuning.get("sample_size")
    if sample_size and sample_size < len(X_train):
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=int(sample_size), random_state=random_state, stratify=y_train
        )
        LOG.info("Subsampled tuning set to %d rows for speed", len(X_train))

    LOG.info("Tuning on %d train rows (%d numeric, %d categorical features)",
             len(y_train), len(numeric_cols), len(categorical_cols))

    base_params = {
        "objective": "binary",
        "class_weight": "balanced",
        "subsample_freq": 1,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": random_state,
    }

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="credit_risk_lgbm")
    study.optimize(
        lambda t: _objective(t, X_train, y_train, numeric_cols, categorical_cols, base_params, tuning),
        n_trials=tuning["n_trials"],
        timeout=tuning["timeout_seconds"],
        show_progress_bar=False,
    )

    best = study.best_trial
    n_estimators = best.user_attrs.get("mean_best_iteration", 500)
    LOG.info("Best CV AUC = %.5f over %d trials (n_estimators=%d)",
             best.value, len(study.trials), n_estimators)
    LOG.info("Best params: %s", best.params)

    _write_tuned_params(best.params, n_estimators, base_params)

    # Persist a tuning summary for the record / dashboard.
    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {"best_cv_auc": round(best.value, 6), "n_trials": len(study.trials), "n_estimators": int(n_estimators)}
    summary.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.params.items()})
    pd.DataFrame([summary]).to_csv(reports_dir / "tuning_results.csv", index=False)
    LOG.info("Wrote %s", reports_dir / "tuning_results.csv")
    return summary
