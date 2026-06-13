"""Build feature_summary table: numeric stats vs categorical top values."""
import logging

import pandas as pd

from credit_risk.features.groups import build_feature_group_mapping

LOG = logging.getLogger(__name__)


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and s.dtype.kind in "iufc"


def _numeric_stats(s: pd.Series) -> dict:
    out = {}
    valid = s.dropna()
    if len(valid) == 0:
        out["mean"] = out["std"] = out["p05"] = out["p50"] = out["p95"] = ""
        return out
    out["mean"] = round(valid.mean(), 6)
    out["std"] = round(valid.std(), 6) if valid.std() is not None else ""
    out["p05"] = round(valid.quantile(0.05), 6)
    out["p50"] = round(valid.quantile(0.50), 6)
    out["p95"] = round(valid.quantile(0.95), 6)
    return out


def _categorical_tops(s: pd.Series, top_n: int = 2) -> dict:
    out = {}
    counts = s.value_counts()
    if len(counts) == 0:
        out["top1"] = out["top1_rate"] = out["top2"] = out["top2_rate"] = ""
        return out
    top = counts.head(top_n)
    total = s.notna().sum()
    rates = (top / total * 100).round(4)
    out["top1"] = str(top.index[0])
    out["top1_rate"] = rates.iloc[0]
    out["top2"] = str(top.index[1]) if len(top) > 1 else ""
    out["top2_rate"] = rates.iloc[1] if len(top) > 1 else ""
    return out


def build_feature_summary_df(
    df: pd.DataFrame, feature_group_mapping: dict[str, str]
) -> pd.DataFrame:
    """One row per feature: dtype, n, missing_rate, numeric stats or categorical tops, feature_group."""
    rows = []
    n_total = len(df)
    for col in df.columns:
        s = df[col]
        n = s.notna().sum()
        missing_rate = round((s.isna().sum() / n_total), 6)
        dtype = str(s.dtype)
        fg = feature_group_mapping.get(col, "other")

        row = {
            "feature": col,
            "dtype": dtype,
            "n": int(n),
            "missing_rate": missing_rate,
            "mean": "",
            "std": "",
            "p05": "",
            "p50": "",
            "p95": "",
            "top1": "",
            "top1_rate": "",
            "top2": "",
            "top2_rate": "",
            "feature_group": fg,
        }

        if _is_numeric_series(s):
            row.update(_numeric_stats(s))
        else:
            row.update(_categorical_tops(s))

        rows.append(row)

    out = pd.DataFrame(rows)
    LOG.info("Built feature_summary with %d rows", len(out))
    return out
