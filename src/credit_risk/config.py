"""Load paths from configs/paths.yaml. Paths are relative to project root."""
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_paths() -> dict:
    paths_file = _PROJECT_ROOT / "configs" / "paths.yaml"
    if not paths_file.exists():
        raise FileNotFoundError(f"Paths config not found: {paths_file}")
    with open(paths_file) as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError("configs/paths.yaml is empty")
    return data


def get_raw_data_path() -> Path:
    """Path to raw application train CSV. From configs/paths.yaml only."""
    paths = _load_paths()
    raw = paths.get("raw_data")
    if not raw:
        raise ValueError("configs/paths.yaml must define 'raw_data'")
    return _PROJECT_ROOT / raw


def get_reports_dir() -> Path:
    """Directory for report artifacts. From configs/paths.yaml only."""
    paths = _load_paths()
    reports = paths.get("reports_dir")
    if not reports:
        raise ValueError("configs/paths.yaml must define 'reports_dir'")
    return _PROJECT_ROOT / reports


def get_predictions_dir() -> Path:
    """Directory for dashboard/prediction outputs. From configs/paths.yaml only."""
    paths = _load_paths()
    pred = paths.get("predictions_dir")
    if not pred:
        raise ValueError("configs/paths.yaml must define 'predictions_dir'")
    return _PROJECT_ROOT / pred


def get_models_dir() -> Path:
    """Directory for trained model metadata. From configs/paths.yaml only."""
    paths = _load_paths()
    models = paths.get("models_dir")
    if not models:
        raise ValueError("configs/paths.yaml must define 'models_dir'")
    return _PROJECT_ROOT / models


def load_model_config() -> dict:
    """Load model config from configs/model.yaml."""
    path = _PROJECT_ROOT / "configs" / "model.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None or (isinstance(data, dict) and len(data) == 0):
        raise ValueError("configs/model.yaml is empty or invalid")
    return data


def validate_model_config(cfg: dict) -> None:
    """Basic validation for expected model config keys used by Phase 3.

    - Ensures 'model' is a mapping and 'type' is a string
    - Ensures 'split.test_size' is in (0,1) and 'random_state' if present is an int
    - Ensures thresholds (start, stop, step) are numeric and in [0,1] with start < stop
    - Ensures business params are numeric when present
    """
    if not isinstance(cfg, dict):
        raise ValueError("Model config must be a mapping (configs/model.yaml)")

    model_block = cfg.get("model")
    if model_block is None or not isinstance(model_block, dict):
        raise ValueError("configs/model.yaml must define a 'model' mapping")
    if "type" in model_block and not isinstance(model_block["type"], str):
        raise ValueError("configs/model.yaml: 'model.type' must be a string")
    # Enforce supported model types for Phase 3
    model_type = (model_block.get("type") or "lightgbm").lower()
    supported = {"lightgbm", "logreg"}
    if model_type not in supported:
        raise ValueError(f"configs/model.yaml: unsupported model.type '{model_type}'. Supported: {sorted(supported)}")

    split = cfg.get("split", {}) or {}
    if not isinstance(split, dict):
        raise ValueError("configs/model.yaml: 'split' must be a mapping")
    test_size = split.get("test_size", 0.2)
    if not isinstance(test_size, (int, float)) or not (0 < float(test_size) < 1):
        raise ValueError("configs/model.yaml: 'split.test_size' must be a number in (0, 1)")
    rs = split.get("random_state")
    if rs is not None and not isinstance(rs, int):
        raise ValueError("configs/model.yaml: 'split.random_state' must be an integer if provided")

    thresholds = cfg.get("thresholds") or {}
    if thresholds:
        try:
            start = float(thresholds.get("start", 0.05))
            stop = float(thresholds.get("stop", 0.95))
            step = float(thresholds.get("step", 0.05))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("configs/model.yaml: thresholds values must be numeric") from exc
        if not (0 <= start < stop <= 1):
            raise ValueError("configs/model.yaml: thresholds must satisfy 0 <= start < stop <= 1")
        if not (step > 0):
            raise ValueError("configs/model.yaml: 'thresholds.step' must be a positive number")

    business = cfg.get("business") or {}
    if business:
        if "profit_if_good" in business and not isinstance(business["profit_if_good"], (int, float)):
            raise ValueError("configs/model.yaml: 'business.profit_if_good' must be numeric")
        if "loss_given_default" in business and not isinstance(business["loss_given_default"], (int, float)):
            raise ValueError("configs/model.yaml: 'business.loss_given_default' must be numeric")
    # scenarios list is validated in modeling/scenarios.py when used
