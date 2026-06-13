"""Build and write dashboard tables (KPI overview, missingness, segments, feature summary, correlation)."""
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from credit_risk.config import get_predictions_dir, get_raw_data_path
from credit_risk.data.ingest import load_raw_data, validate_raw_data
from credit_risk.features.groups import build_feature_group_mapping
from credit_risk.features.summary import build_feature_summary_df

TARGET_COLUMN = "TARGET"
REQUIRED_COLUMNS = ["AMT_INCOME_TOTAL", "DAYS_BIRTH", "AMT_CREDIT", "CODE_GENDER", TARGET_COLUMN]
LOG = logging.getLogger(__name__)


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}. Cannot build dashboard tables.")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_kpi_overview(df: pd.DataFrame, run_id: str, created_at_str: str) -> pd.DataFrame:
    """One row: run_id, created_at, n_rows, n_cols, default_rate, avg_missing_rate, pct_columns_missing_gt_40."""
    n_rows, n_cols = df.shape
    default_rate = round((df[TARGET_COLUMN].sum() / len(df)), 6)
    missing_rates = df.isna().mean()
    avg_missing_rate = round(missing_rates.mean(), 6)
    pct_columns_missing_gt_40 = round((missing_rates > 0.4).mean() * 100, 4)
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "created_at": created_at_str,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "default_rate": default_rate,
                "avg_missing_rate": avg_missing_rate,
                "pct_columns_missing_gt_40": pct_columns_missing_gt_40,
            }
        ]
    )


def build_missingness_by_feature(
    df: pd.DataFrame, feature_group_mapping: dict[str, str]
) -> pd.DataFrame:
    """One row per feature: feature, missing_rate, dtype, feature_group."""
    n_total = len(df)
    rows = []
    for col in df.columns:
        missing_rate = round(df[col].isna().sum() / n_total, 6)
        rows.append(
            {
                "feature": col,
                "missing_rate": missing_rate,
                "dtype": str(df[col].dtype),
                "feature_group": feature_group_mapping.get(col, "other"),
            }
        )
    return pd.DataFrame(rows)


def _age_from_days_birth(series: pd.Series) -> pd.Series:
    """Convert DAYS_BIRTH (negative days) to age in years."""
    return (-series / 365.25).round(1)


def _age_band_labels() -> list[tuple[float, float, str]]:
    """(min_age, max_age, label). 65+ means 65 to 120."""
    return [
        (0, 25, "<25"),
        (25, 35, "25-35"),
        (35, 45, "35-45"),
        (45, 55, "45-55"),
        (55, 65, "55-65"),
        (65, 120, "65+"),
    ]


def _assign_age_band(age: pd.Series) -> pd.Series:
    out = pd.Series(index=age.index, dtype=object)
    for low, high, label in _age_band_labels():
        out[(age >= low) & (age < high)] = label
    return out


def _quantile_bin_series(series: pd.Series, n_bins: int = 5) -> pd.Series:
    """Bin series into q1..q5 by quantiles. Returns series of labels."""
    q = series.quantile([i / n_bins for i in range(1, n_bins)])
    def bin_val(x):
        if pd.isna(x):
            return None
        for i, v in enumerate(q):
            if x <= v:
                return f"q{i + 1}"
        return f"q{n_bins}"
    return series.apply(bin_val)


def build_population_segments(df: pd.DataFrame) -> pd.DataFrame:
    """One row per segment value: segment_name, segment_value, n, default_rate."""
    rows = []
    target = df[TARGET_COLUMN]

    # income_band: 5 quantile bins on AMT_INCOME_TOTAL
    inc = df["AMT_INCOME_TOTAL"].dropna()
    if len(inc) > 0:
        work = df.copy()
        work["_band"] = _quantile_bin_series(df["AMT_INCOME_TOTAL"], 5)
        for val in ["q1", "q2", "q3", "q4", "q5"]:
            mask = work["_band"] == val
            n = mask.sum()
            if n > 0:
                dr = round(target[mask].mean(), 6)
                rows.append({"segment_name": "income_band", "segment_value": val, "n": int(n), "default_rate": dr})

    # age_band: from DAYS_BIRTH
    age = _age_from_days_birth(df["DAYS_BIRTH"])
    work = df.copy()
    work["_band"] = _assign_age_band(age)
    for _, _, label in _age_band_labels():
        mask = work["_band"] == label
        n = mask.sum()
        if n > 0:
            dr = round(target[mask].mean(), 6)
            rows.append({"segment_name": "age_band", "segment_value": label, "n": int(n), "default_rate": dr})

    # credit_amount_band: 5 quantile bins on AMT_CREDIT
    cred = df["AMT_CREDIT"].dropna()
    if len(cred) > 0:
        work = df.copy()
        work["_band"] = _quantile_bin_series(df["AMT_CREDIT"], 5)
        for val in ["q1", "q2", "q3", "q4", "q5"]:
            mask = work["_band"] == val
            n = mask.sum()
            if n > 0:
                dr = round(target[mask].mean(), 6)
                rows.append({"segment_name": "credit_amount_band", "segment_value": val, "n": int(n), "default_rate": dr})

    # gender: CODE_GENDER
    for val in df["CODE_GENDER"].dropna().unique().tolist():
        mask = df["CODE_GENDER"] == val
        n = mask.sum()
        if n > 0:
            dr = round(target[mask].mean(), 6)
            rows.append({"segment_name": "gender", "segment_value": str(val), "n": int(n), "default_rate": dr})

    return pd.DataFrame(rows)


def build_correlation_top(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features only, sorted by absolute correlation with TARGET desc. Columns: feature, corr_with_target."""
    numeric = df.select_dtypes(include=["number"])
    if TARGET_COLUMN not in numeric.columns:
        return pd.DataFrame(columns=["feature", "corr_with_target"])
    corrs = numeric.corr()[TARGET_COLUMN].drop(TARGET_COLUMN, errors="ignore")
    corrs = corrs.dropna()
    corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
    out = pd.DataFrame({"feature": corrs.index, "corr_with_target": corrs.values.round(6)})
    return out


def run_make_dashboard_tables() -> None:
    """Load raw data, validate, build all dashboard tables, write CSVs to predictions_dir. Fails loudly if required columns missing."""
    raw_path = get_raw_data_path()
    out_dir = get_predictions_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading raw data from %s", raw_path)
    df = load_raw_data(raw_path)
    validate_raw_data(df)
    _ensure_required_columns(df)

    run_id = _run_id()
    created_at_str = _created_at()
    feature_group_mapping = build_feature_group_mapping(df.columns.tolist())

    LOG.info("Building KPI overview")
    kpi = build_kpi_overview(df, run_id, created_at_str)
    kpi.to_csv(out_dir / "kpi_overview.csv", index=False)
    LOG.info("Wrote kpi_overview.csv")

    LOG.info("Building missingness by feature")
    miss = build_missingness_by_feature(df, feature_group_mapping)
    miss.to_csv(out_dir / "missingness_by_feature.csv", index=False)
    LOG.info("Wrote missingness_by_feature.csv")

    LOG.info("Building population segments")
    seg = build_population_segments(df)
    seg.to_csv(out_dir / "population_segments.csv", index=False)
    LOG.info("Wrote population_segments.csv")

    LOG.info("Building feature summary")
    summary = build_feature_summary_df(df, feature_group_mapping)
    summary.to_csv(out_dir / "feature_summary.csv", index=False)
    LOG.info("Wrote feature_summary.csv")

    LOG.info("Building correlation (top numeric vs TARGET)")
    corr = build_correlation_top(df)
    corr.to_csv(out_dir / "correlation_top.csv", index=False)
    LOG.info("Wrote correlation_top.csv")

    LOG.info("make-dashboard-tables completed successfully")