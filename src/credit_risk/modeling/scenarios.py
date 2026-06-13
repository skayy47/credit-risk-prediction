"""Scenario-based what-if simulations using existing predictions.

Reads predictions from data/outputs/predictions/predictions.csv and scenario
definitions from configs/model.yaml, then writes scenario_results.csv with
one row per scenario (approval_rate, default_rate, expected_profit, etc.).
"""
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from credit_risk.config import get_predictions_dir, load_model_config

LOG = logging.getLogger(__name__)


REQUIRED_PRED_COLS = ["y_true", "y_proba"]


def _load_predictions() -> pd.DataFrame:
    """Load predictions.csv and validate required columns."""
    pred_dir = get_predictions_dir()
    pred_path = pred_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    df = pd.read_csv(pred_path)
    missing = [c for c in REQUIRED_PRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {missing}")
    return df


def _iter_scenarios(cfg: Dict) -> Iterable[Dict]:
    """Yield validated scenario dicts from model config."""
    scenarios = cfg.get("scenarios")
    if scenarios is None:
        raise ValueError("configs/model.yaml must define a 'scenarios' list for what-if simulation")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("'scenarios' must be a non-empty list in configs/model.yaml")

    for idx, sc in enumerate(scenarios):
        if not isinstance(sc, dict):
            raise ValueError(f"Scenario #{idx} in configs/model.yaml must be a mapping")
        name = sc.get("name") or f"scenario_{idx+1}"
        try:
            threshold = float(sc["threshold"])
            profit_if_good = float(sc["profit_if_good"])
            loss_given_default = float(sc["loss_given_default"])
        except KeyError as exc:
            raise ValueError(
                f"Scenario '{name}' is missing required key: {exc}"
            ) from exc
        yield {
            "name": name,
            "threshold": threshold,
            "profit_if_good": profit_if_good,
            "loss_given_default": loss_given_default,
        }


def _evaluate_scenario(
    *,
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float,
    profit_if_good: float,
    loss_given_default: float,
) -> Dict:
    """Compute approval, default rate and value metrics for a single scenario."""
    y_true = y_true.astype(int)
    approved = y_proba < threshold
    n = len(y_true)
    n_approved = int(approved.sum())
    approval_rate = n_approved / n if n else 0.0

    if n_approved > 0:
        default_rate = float(y_true[approved].mean())
        good_mask = (y_true == 0) & approved
        bad_mask = (y_true == 1) & approved
        good_approved = int(good_mask.sum())
        bad_approved = int(bad_mask.sum())
    else:
        default_rate = 0.0
        good_approved = 0
        bad_approved = 0

    expected_profit = good_approved * profit_if_good
    expected_loss = bad_approved * loss_given_default
    net_value = expected_profit - expected_loss

    return {
        "approval_rate": round(approval_rate, 6),
        "default_rate": round(default_rate, 6),
        "expected_profit": round(expected_profit, 6),
        "expected_loss": round(expected_loss, 6),
        "net_value": round(net_value, 6),
    }


def run_simulate_scenarios() -> None:
    """Entry point for CLI simulate-scenarios.

    Combines existing predictions with configured scenarios to produce
    scenario_results.csv (one row per scenario).
    """
    cfg = load_model_config()
    from credit_risk.config import validate_model_config
    validate_model_config(cfg)
    df_pred = _load_predictions()

    y_true = df_pred["y_true"]
    y_proba = df_pred["y_proba"]

    rows: List[Dict] = []
    for sc in _iter_scenarios(cfg):
        LOG.info(
            "Evaluating scenario '%s' (threshold=%.3f, profit_if_good=%.3f, loss_given_default=%.3f)",
            sc["name"],
            sc["threshold"],
            sc["profit_if_good"],
            sc["loss_given_default"],
        )
        metrics = _evaluate_scenario(
            y_true=y_true,
            y_proba=y_proba,
            threshold=sc["threshold"],
            profit_if_good=sc["profit_if_good"],
            loss_given_default=sc["loss_given_default"],
        )
        row = {
            "scenario_name": sc["name"],
            "threshold": sc["threshold"],
            "profit_if_good": sc["profit_if_good"],
            "loss_given_default": sc["loss_given_default"],
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise ValueError("No scenarios evaluated; check configs/model.yaml 'scenarios' list")

    out_df = pd.DataFrame(rows)
    preds_dir = get_predictions_dir()
    out_path = preds_dir / "scenario_results.csv"
    out_df.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)

