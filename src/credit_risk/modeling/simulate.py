"""Generate predictions table and decision simulation CSV for Power BI."""
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _risk_band_quantiles(y_proba: pd.Series, n_bands: int = 3) -> pd.Series:
    """Assign Low/Medium/High from quantiles of y_proba. 3 bands: 0-1/3, 1/3-2/3, 2/3-1."""
    q = y_proba.quantile([1/3, 2/3])
    bands = pd.Series(index=y_proba.index, dtype=object)
    bands[y_proba <= q.iloc[0]] = "Low"
    bands[(y_proba > q.iloc[0]) & (y_proba <= q.iloc[1])] = "Medium"
    bands[y_proba > q.iloc[1]] = "High"
    return bands


def write_predictions_csv(
    application_id: pd.Series,
    y_true: pd.Series,
    y_proba: pd.Series,
    out_dir: Path,
    run_id: str | None = None,
    created_at: str | None = None,
) -> Path:
    """Write predictions.csv: run_id, created_at, application_id, y_true, y_proba, risk_band."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = _run_id()
    if created_at is None:
        created_at = _created_at()
    risk_band = _risk_band_quantiles(y_proba)
    df = pd.DataFrame({
        "run_id": run_id,
        "created_at": created_at,
        "application_id": application_id.values,
        "y_true": y_true.values,
        "y_proba": y_proba.round(6).values,
        "risk_band": risk_band.values,
    })
    out_path = out_dir / "predictions.csv"
    df.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)
    return out_path


def write_decision_simulation_csv(
    y_true: pd.Series,
    y_proba: pd.Series,
    out_dir: Path,
    threshold_start: float = 0.05,
    threshold_stop: float = 0.95,
    threshold_step: float = 0.05,
    profit_if_good: float = 0.2,
    loss_given_default: float = 1.0,
) -> Path:
    """For each threshold: approved = (y_proba < threshold). Write approval_rate, default_rate_among_approved, expected_profit, expected_loss, expected_value."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate thresholds and inputs to fail fast and loudly in CI
    try:
        threshold_start = float(threshold_start)
        threshold_stop = float(threshold_stop)
        threshold_step = float(threshold_step)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("thresholds must be numeric") from exc
    if not (0 <= threshold_start < threshold_stop <= 1):
        raise ValueError("Invalid threshold range: require 0 <= start < stop <= 1")
    if threshold_step <= 0:
        raise ValueError("Invalid threshold step: must be a positive number")

    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length")

    # Ensure probabilities are in [0, 1]
    if not ((y_proba >= 0.0) & (y_proba <= 1.0)).all():
        raise ValueError("y_proba must contain probabilities in [0, 1]")

    thresholds = []
    t = threshold_start
    while t <= threshold_stop:
        thresholds.append(round(t, 4))
        t += threshold_step
    rows = []
    n = len(y_true)
    y_true = y_true.astype(int)
    for thresh in thresholds:
        approved = (y_proba < thresh).values
        n_approved = approved.sum()
        approval_rate = n_approved / n if n else 0.0
        if n_approved > 0:
            default_rate_approved = y_true[approved].mean()
            good_approved = ((y_true == 0) & approved).sum()
            bad_approved = ((y_true == 1) & approved).sum()
        else:
            default_rate_approved = 0.0
            good_approved = 0
            bad_approved = 0
        expected_profit = good_approved * profit_if_good
        expected_loss = bad_approved * loss_given_default
        expected_value = expected_profit - expected_loss
        rows.append({
            "threshold": thresh,
            "approval_rate": round(approval_rate, 6),
            "default_rate_among_approved": round(default_rate_approved, 6),
            "expected_profit": round(expected_profit, 6),
            "expected_loss": round(expected_loss, 6),
            "expected_value": round(expected_value, 6),
        })
    df = pd.DataFrame(rows)
    out_path = out_dir / "decision_simulation.csv"
    df.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)
    return out_path
