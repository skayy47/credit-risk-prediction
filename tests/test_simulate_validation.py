import pandas as pd
import pytest
from credit_risk.modeling.simulate import write_decision_simulation_csv
from pathlib import Path

def test_write_decision_simulation_rejects_invalid_probs(tmp_path):
    y_true = pd.Series([0, 1, 0])
    y_proba = pd.Series([0.1, 1.2, 0.3])  # 1.2 is invalid
    with pytest.raises(ValueError):
        write_decision_simulation_csv(y_true=y_true, y_proba=y_proba, out_dir=tmp_path)

def test_write_decision_simulation_rejects_mismatched_length(tmp_path):
    y_true = pd.Series([0, 1])
    y_proba = pd.Series([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        write_decision_simulation_csv(y_true=y_true, y_proba=y_proba, out_dir=tmp_path)
