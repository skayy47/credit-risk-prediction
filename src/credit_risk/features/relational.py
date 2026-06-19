"""Phase 1 — relational feature engineering over the auxiliary Home Credit tables.

The baseline model used only ``application_train``. The Home Credit dataset's
predictive signal lives in the relational tables (prior bureau credits, previous
applications, installment payment behaviour, POS/credit-card balances). This
module aggregates each of those to one row per ``SK_ID_CURR`` and left-joins the
results onto the application table.

Design notes:
- Every aggregator is a pure function of its input DataFrame(s), so the logic is
  unit-tested with small synthetic frames (see tests/test_relational.py) without
  needing the full ~690MB dataset.
- Aggregators are defensive: they only use columns that are present, so a slightly
  different schema degrades gracefully instead of raising KeyError.
- Count-style join misses are filled with 0; ratio/mean misses are left as NaN for
  the downstream imputer in the modelling pipeline to handle.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from credit_risk.config import get_feature_sources, project_root

LOG = logging.getLogger(__name__)

ID = "SK_ID_CURR"
BUREAU_ID = "SK_ID_BUREAU"


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN (not inf) where denominator is 0."""
    return numerator / denominator.replace(0, pd.NA)


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# ── bureau (+ optional bureau_balance) ──────────────────────────────────────────
def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per SK_ID_CURR summarising prior credits reported to the bureau."""
    if ID not in bureau.columns:
        raise ValueError(f"bureau must contain {ID}")
    df = bureau.copy()

    # Optionally fold in monthly status history (count of past-due months per credit).
    if bureau_balance is not None and not bureau_balance.empty and BUREAU_ID in df.columns:
        bb = bureau_balance.copy()
        if "STATUS" in bb.columns:
            bb["IS_DPD"] = bb["STATUS"].astype(str).isin(["1", "2", "3", "4", "5"]).astype(int)
            bb_agg = bb.groupby(BUREAU_ID).agg(
                BB_MONTHS=("STATUS", "size"),
                BB_DPD_MONTHS=("IS_DPD", "sum"),
            )
            df = df.merge(bb_agg, on=BUREAU_ID, how="left")

    g = df.groupby(ID)
    out = pd.DataFrame(index=g.size().index)
    out["BUREAU_N_CREDITS"] = g.size()
    if "CREDIT_ACTIVE" in df.columns:
        active = (df["CREDIT_ACTIVE"].astype(str) == "Active").astype(int)
        out["BUREAU_N_ACTIVE"] = active.groupby(df[ID]).sum()
        out["BUREAU_ACTIVE_RATIO"] = _safe_div(out["BUREAU_N_ACTIVE"], out["BUREAU_N_CREDITS"])
    for col, name in [
        ("AMT_CREDIT_SUM_DEBT", "BUREAU_DEBT"),
        ("AMT_CREDIT_SUM", "BUREAU_CREDIT"),
    ]:
        if col in df.columns:
            out[f"{name}_SUM"] = g[col].sum()
            out[f"{name}_MEAN"] = g[col].mean()
    if "CREDIT_DAY_OVERDUE" in df.columns:
        out["BUREAU_OVERDUE_MAX"] = g["CREDIT_DAY_OVERDUE"].max()
    if "AMT_CREDIT_SUM_OVERDUE" in df.columns:
        out["BUREAU_AMT_OVERDUE_SUM"] = g["AMT_CREDIT_SUM_OVERDUE"].sum()
    if "DAYS_CREDIT" in df.columns:
        out["BUREAU_DAYS_CREDIT_MEAN"] = g["DAYS_CREDIT"].mean()
    if "BB_DPD_MONTHS" in df.columns:
        out["BUREAU_BB_DPD_MONTHS_SUM"] = g["BB_DPD_MONTHS"].sum()
    return out.reset_index()


# ── previous_application ─────────────────────────────────────────────────────────
def aggregate_previous_apps(prev: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR summarising the applicant's prior Home Credit apps."""
    if ID not in prev.columns:
        raise ValueError(f"previous_application must contain {ID}")
    g = prev.groupby(ID)
    out = pd.DataFrame(index=g.size().index)
    out["PREV_N_APPS"] = g.size()
    if "NAME_CONTRACT_STATUS" in prev.columns:
        status = prev["NAME_CONTRACT_STATUS"].astype(str)
        out["PREV_APPROVED_RATIO"] = (status == "Approved").astype(int).groupby(prev[ID]).mean()
        out["PREV_REFUSED_RATIO"] = (status == "Refused").astype(int).groupby(prev[ID]).mean()
    if "AMT_APPLICATION" in prev.columns:
        out["PREV_AMT_APPLICATION_MEAN"] = g["AMT_APPLICATION"].mean()
    if "AMT_CREDIT" in prev.columns:
        out["PREV_AMT_CREDIT_MEAN"] = g["AMT_CREDIT"].mean()
    if {"AMT_CREDIT", "AMT_APPLICATION"}.issubset(prev.columns):
        ratio = _safe_div(prev["AMT_CREDIT"], prev["AMT_APPLICATION"])
        out["PREV_CREDIT_TO_APP_RATIO"] = ratio.groupby(prev[ID]).mean()
    if "DAYS_DECISION" in prev.columns:
        out["PREV_DAYS_DECISION_MEAN"] = g["DAYS_DECISION"].mean()
    return out.reset_index()


# ── installments_payments ────────────────────────────────────────────────────────
def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Payment-behaviour features: lateness and shortfall on scheduled installments."""
    if ID not in inst.columns:
        raise ValueError(f"installments_payments must contain {ID}")
    df = inst.copy()
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(df.columns):
        # Both are negative day-offsets; paid later than due => entry > instalment.
        df["_DPD"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)
        df["_IS_LATE"] = (df["_DPD"] > 0).astype(int)
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(df.columns):
        df["_SHORTFALL"] = (1 - _safe_div(df["AMT_PAYMENT"], df["AMT_INSTALMENT"])).clip(lower=0)
    g = df.groupby(ID)
    out = pd.DataFrame(index=g.size().index)
    out["INST_N_INSTALMENTS"] = g.size()
    if "_IS_LATE" in df.columns:
        out["INST_LATE_RATE"] = g["_IS_LATE"].mean()
        out["INST_DPD_MEAN"] = g["_DPD"].mean()
        out["INST_DPD_MAX"] = g["_DPD"].max()
    if "_SHORTFALL" in df.columns:
        out["INST_SHORTFALL_MEAN"] = g["_SHORTFALL"].mean()
    return out.reset_index()


# ── POS_CASH_balance ─────────────────────────────────────────────────────────────
def aggregate_pos(pos: pd.DataFrame) -> pd.DataFrame:
    """POS/cash-loan monthly balance behaviour: days-past-due signal."""
    if ID not in pos.columns:
        raise ValueError(f"POS_CASH_balance must contain {ID}")
    g = pos.groupby(ID)
    out = pd.DataFrame(index=g.size().index)
    out["POS_N_MONTHS"] = g.size()
    if "SK_DPD" in pos.columns:
        out["POS_DPD_MEAN"] = g["SK_DPD"].mean()
        out["POS_DPD_MAX"] = g["SK_DPD"].max()
        out["POS_N_LATE_MONTHS"] = (pos["SK_DPD"] > 0).astype(int).groupby(pos[ID]).sum()
    return out.reset_index()


# ── credit_card_balance ──────────────────────────────────────────────────────────
def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Credit-card balance behaviour: utilisation and days-past-due."""
    if ID not in cc.columns:
        raise ValueError(f"credit_card_balance must contain {ID}")
    df = cc.copy()
    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(df.columns):
        df["_UTIL"] = _safe_div(df["AMT_BALANCE"], df["AMT_CREDIT_LIMIT_ACTUAL"])
    g = df.groupby(ID)
    out = pd.DataFrame(index=g.size().index)
    out["CC_N_MONTHS"] = g.size()
    if "AMT_BALANCE" in df.columns:
        out["CC_BALANCE_MEAN"] = g["AMT_BALANCE"].mean()
    if "_UTIL" in df.columns:
        out["CC_UTILIZATION_MEAN"] = g["_UTIL"].mean()
    if "SK_DPD" in df.columns:
        out["CC_DPD_MEAN"] = g["SK_DPD"].mean()
    return out.reset_index()


# ── orchestration ────────────────────────────────────────────────────────────────
_COUNT_PREFIXES = ("BUREAU_N", "BUREAU_BB", "PREV_N", "INST_N", "POS_N", "CC_N")


def build_relational_features(app: pd.DataFrame, aggregates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join each aggregate table (keyed by SK_ID_CURR) onto the application frame.

    ``aggregates`` maps a source name to its already-aggregated DataFrame. Count
    columns are filled with 0 on join-miss; everything else is left as NaN.
    """
    if ID not in app.columns:
        raise ValueError(f"application table must contain {ID}")
    out = app.copy()
    added: list[str] = []
    for name, agg in aggregates.items():
        if agg is None or agg.empty:
            continue
        new_cols = [c for c in agg.columns if c != ID]
        out = out.merge(agg, on=ID, how="left")
        added.extend(new_cols)
        LOG.info("Merged %d features from '%s'", len(new_cols), name)
    count_cols = [c for c in added if c.startswith(_COUNT_PREFIXES)]
    if count_cols:
        out[count_cols] = out[count_cols].fillna(0)
    LOG.info("Enriched application table: %d base -> %d total columns (+%d)",
             app.shape[1], out.shape[1], len(added))
    return out


_AGGREGATORS = {
    "bureau": lambda data: aggregate_bureau(data["bureau"], data.get("bureau_balance")),
    "previous_application": lambda data: aggregate_previous_apps(data["previous_application"]),
    "installments_payments": lambda data: aggregate_installments(data["installments_payments"]),
    "pos_cash_balance": lambda data: aggregate_pos(data["pos_cash_balance"]),
    "credit_card_balance": lambda data: aggregate_credit_card(data["credit_card_balance"]),
}


def run_build_features() -> Path:
    """CLI entry: read base + auxiliary tables per feature_sources.yaml, build the
    enriched application table, and write it to the configured output path."""
    cfg = get_feature_sources()
    root = project_root()
    sources = cfg.get("sources", {}) or {}

    base_rel = cfg.get("base_table", "data/raw/application_train.csv")
    base_path = root / base_rel
    if not base_path.exists():
        raise FileNotFoundError(f"Base table not found: {base_path}")
    LOG.info("Loading base table %s", base_path)
    app = pd.read_csv(base_path)

    # Load every enabled, present source once.
    loaded: dict[str, pd.DataFrame] = {}
    for name, spec in sources.items():
        if not (spec or {}).get("enabled", False):
            continue
        path = root / spec["path"]
        if not path.exists():
            LOG.warning("Source '%s' enabled but file missing (%s) — skipping", name, path)
            continue
        LOG.info("Loading source '%s' from %s", name, path)
        loaded[name] = pd.read_csv(path)

    # bureau_balance is folded into the bureau aggregate, not aggregated on its own.
    aggregates: dict[str, pd.DataFrame] = {}
    for name, fn in _AGGREGATORS.items():
        if name == "bureau":
            if "bureau" in loaded:
                aggregates["bureau"] = aggregate_bureau(loaded["bureau"], loaded.get("bureau_balance"))
            continue
        if name in loaded:
            aggregates[name] = fn(loaded)

    enriched = build_relational_features(app, aggregates)

    out_rel = cfg.get("output", "data/raw/application_train_enriched.csv")
    out_path = root / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    LOG.info("Wrote enriched table (%d rows, %d cols) to %s",
             enriched.shape[0], enriched.shape[1], out_path)
    return out_path
