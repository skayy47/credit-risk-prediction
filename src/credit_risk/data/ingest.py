"""Ingest raw data and validate; export shape, missingness, dtypes, target stats."""
import logging
from pathlib import Path

import pandas as pd

from credit_risk.config import get_raw_data_path, get_reports_dir

TARGET_COLUMN = "TARGET"
LOG = logging.getLogger(__name__)


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def _ensure_target_exists(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"TARGET column missing. Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
        )


def _ensure_target_binary(df: pd.DataFrame) -> None:
    unique = df[TARGET_COLUMN].dropna().unique()
    if not (set(unique) <= {0, 1}):
        raise ValueError(
            f"TARGET must be binary (0/1). Found values: {sorted(unique.tolist())}"
        )


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate raw dataset: TARGET exists and is binary. Raises on failure."""
    _ensure_target_exists(df)
    _ensure_target_binary(df)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw CSV and return DataFrame."""
    _ensure_file_exists(path)
    LOG.info("Loading raw data from %s", path)
    return pd.read_csv(str(path))


def export_data_shape(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write rows and columns to data_shape.txt. Returns path written."""
    out_path = out_dir / "data_shape.txt"
    with open(out_path, "w") as f:
        f.write(f"rows,{df.shape[0]}\n")
        f.write(f"columns,{df.shape[1]}\n")
    LOG.info("Wrote %s", out_path)
    return out_path


def export_missingness(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write per-column missing ratio to missingness.csv. Returns path written."""
    out_path = out_dir / "missingness.csv"
    missing = df.isna().sum()
    total = len(df)
    ratio = (missing / total).round(6)
    report = pd.DataFrame({"column": missing.index, "missing_ratio": ratio.values})
    report.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)
    return out_path


def export_dtypes(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write column name and dtype to dtypes.csv. Returns path written."""
    out_path = out_dir / "dtypes.csv"
    report = pd.DataFrame({"column": df.columns, "dtype": [str(d) for d in df.dtypes]})
    report.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)
    return out_path


def export_target_stats(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write TARGET counts and percentages to target_stats.csv. Returns path written."""
    out_path = out_dir / "target_stats.csv"
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = len(df)
    pct = (counts / total * 100).round(4)
    report = pd.DataFrame(
        {"target_value": counts.index, "count": counts.values, "pct": pct.values}
    )
    report.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)
    return out_path


def run_ingest_validate() -> None:
    """Read raw data from config path, validate, and export reports. Fails loudly on validation errors."""
    raw_path = get_raw_data_path()
    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    validate_raw_data(df)

    export_data_shape(df, reports_dir)
    export_missingness(df, reports_dir)
    export_dtypes(df, reports_dir)
    export_target_stats(df, reports_dir)
    LOG.info("ingest-validate completed successfully")
