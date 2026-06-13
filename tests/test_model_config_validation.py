from credit_risk.config import validate_model_config
import pytest


def test_validate_model_config_accepts_good_config():
    cfg = {
        "model": {"type": "lightgbm", "params": {}},
        "split": {"test_size": 0.2, "random_state": 42},
        "thresholds": {"start": 0.05, "stop": 0.95, "step": 0.05},
        "business": {"profit_if_good": 0.2, "loss_given_default": 1.0},
    }
    # Should not raise
    validate_model_config(cfg)


def test_validate_model_config_rejects_invalid_test_size():
    cfg = {
        "model": {"type": "lightgbm", "params": {}},
        "split": {"test_size": 1.2, "random_state": 42},
    }
    with pytest.raises(ValueError):
        validate_model_config(cfg)
