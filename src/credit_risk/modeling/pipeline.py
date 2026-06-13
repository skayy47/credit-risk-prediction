"""Build preprocessing + model pipeline.

Numeric: median impute. Categorical: OneHotEncoder.
Supports LightGBM when available and falls back to a sklearn-only model otherwise.
"""
import logging
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LOG = logging.getLogger(__name__)

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

try:  # LightGBM is optional – we can fall back to sklearn-only models.
    from lightgbm import LGBMClassifier  # type: ignore[import]
except Exception:  # pragma: no cover - import-time failures
    LGBMClassifier = None  # type: ignore[assignment]


def _get_numeric_and_categorical(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Return (numeric_columns, categorical_columns) excluding drop_cols."""
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    numeric = []
    categorical = []
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]) and feature_df[col].dtype.kind in "iufc":
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_feature_lists(df: pd.DataFrame, target_col: str = TARGET_COLUMN, id_col: str = ID_COLUMN) -> Tuple[List[str], List[str]]:
    """Return (numeric_columns, categorical_columns) for modeling. Drops target and id."""
    return _get_numeric_and_categorical(df, [target_col, id_col])


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """ColumnTransformer: median imputer for numeric, OneHotEncoder(handle_unknown='ignore') for categorical."""
    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        )
    if not transformers:
        raise ValueError("No numeric or categorical columns for preprocessor")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _get_init_param_names(cls) -> set:
    """Return set of parameter names accepted by cls.__init__ (excluding 'self' and **kwargs)."""
    import inspect

    sig = inspect.signature(cls.__init__)
    params = set(p.name for p in sig.parameters.values() if p.name not in ("self", "args", "kwargs"))
    return params


def _sanitize_model_params(model_type: str, params: dict) -> tuple[dict, list]:
    """Return (sanitized_params, dropped_keys).

    - For 'lightgbm': return params unchanged.
    - For 'logreg'/'logistic'/'logisticregression': keep only allowed LogisticRegression keys and drop the rest.
    """
    if not params:
        return {}, []
    mt = (model_type or "").lower()
    if mt.startswith("lightgbm"):
        return dict(params), []

    # Allowed LogisticRegression keys we support (explicit whitelist)
    allowed = {
        "C",
        "class_weight",
        "dual",
        "fit_intercept",
        "intercept_scaling",
        "l1_ratio",
        "max_iter",
        "n_jobs",
        "penalty",
        "random_state",
        "solver",
        "tol",
        "verbose",
        "warm_start",
    }
    sanitized = {k: v for k, v in params.items() if k in allowed}
    dropped = [k for k in params.keys() if k not in allowed]
    return sanitized, dropped


def _validate_estimator_params(estimator_cls, params: dict) -> None:
    """Raise ValueError if params contains keys not accepted by estimator_cls.__init__."""
    if not params:
        return
    allowed = _get_init_param_names(estimator_cls)
    invalid = [k for k in params.keys() if k not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported parameters for {estimator_cls.__name__}: {invalid}. "
            f"Allowed parameters: {sorted(list(allowed))}"
        )


def _build_sklearn_baseline(model_params: dict):
    """Baseline sklearn-only classifier used when LightGBM is unavailable."""
    from sklearn.linear_model import LogisticRegression

    params = {"max_iter": 1000, "n_jobs": -1}
    # Allow overriding defaults from config.
    if model_params:
        # Sanitize: drop keys not applicable for LogisticRegression and warn
        sanitized, dropped = _sanitize_model_params("logreg", model_params)
        if dropped:
            LOG.warning("Dropping unsupported LogisticRegression params: %s", dropped)
        # Validate remaining keys against sklearn estimator signature
        _validate_estimator_params(LogisticRegression, sanitized)
        params.update(sanitized)
    return LogisticRegression(**params)


def build_model_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    model_type: str,
    model_params: dict,
) -> Pipeline:
    """Pipeline: preprocessor (impute + OHE) -> classifier.

    - model_type == "lightgbm": try LGBMClassifier, fall back to sklearn baseline on failure.
    - any other value: use sklearn baseline only (no LightGBM dependency).
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    model_type_norm = (model_type or "lightgbm").lower()

    if model_type_norm == "lightgbm" and LGBMClassifier is not None:
        try:
            clf = LGBMClassifier(**(model_params or {}))  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception(
                "Failed to initialize LGBMClassifier (%s). Falling back to sklearn LogisticRegression.",
                exc,
            )
            # On exception, fallback to sklearn and pass sanitized params
            sanitized, dropped = _sanitize_model_params("logreg", model_params or {})
            if dropped:
                LOG.warning("Dropping unsupported LogisticRegression params during fallback: %s", dropped)
            clf = _build_sklearn_baseline(sanitized)
    else:
        if model_type_norm == "lightgbm" and LGBMClassifier is None:
            LOG.warning(
                "LightGBM not available -> falling back to LogisticRegression"
            )
            # sanitize params for logistic and warn
            sanitized, dropped = _sanitize_model_params("logreg", model_params or {})
            if dropped:
                LOG.warning("Dropping unsupported LogisticRegression params: %s", dropped)
            clf = _build_sklearn_baseline(sanitized)
        else:
            clf = _build_sklearn_baseline(model_params)

    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
