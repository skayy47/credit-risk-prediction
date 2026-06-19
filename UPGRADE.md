# UPGRADE PLAN — Credit Risk Engine → 9/10

**Goal:** move this from a *software-engineering* showcase (8–9) with *shallow modeling* (5–6) to a defensible end-to-end credit-risk project (overall 9/10).
**Status legend:** ⬜ todo · 🔄 in progress · ✅ done · ⛔ blocked

Current baseline: single LightGBM on `application_train` only · AUC **0.7594** · Gini 0.519 · KS 0.578 · no tuning · no calibration · 8 tests.
Target: AUC **≥ 0.78** with relational features · calibrated PD · Optuna-tuned · ~15 tests · zero overclaims.

---

## Phase 0 — Credibility fixes (no data needed) — IN PROGRESS

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 0.1 | Soften "regulatory-grade" → "audit-ready, per-prediction" | `README.md:23` | ✅ |
| 0.2 | Untrack model artifacts + stray theme files; add to `.gitignore` | `.gitignore`, git index | 🔄 |
| 0.3 | Reframe "Basel III scorecard" → "credit scorecard (Gini/KS discrimination)" | **portfolio** `content.ts`, LinkedIn About | ⬜ (separate repo) |
| 0.4 | Resolve the dead demo URL — pick one slug, make README badge + repo homepage match | `README.md`, GitHub homepage | ⬜ (needs SKAY click-test) |

**Acceptance:** repo has no committed `.joblib`/metadata; every claim is defensible; one working demo link.

---

## Phase 1 — Relational feature engineering (THE AUC lever) — ⛔ needs Kaggle data

Add features from the auxiliary Home Credit tables. This is the single highest-impact change and the thing interviewers ask about.

New module: `src/credit_risk/features/relational.py`

```python
def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Per SK_ID_CURR: n_prior_credits, n_active, sum/mean AMT_CREDIT_SUM_DEBT,
       max CREDIT_DAY_OVERDUE, active/closed ratio, mean DAYS_CREDIT."""

def aggregate_previous_apps(prev: pd.DataFrame) -> pd.DataFrame:
    """n_prior_apps, approval_rate, mean(AMT_APPLICATION vs AMT_CREDIT),
       mean DAYS_DECISION, refused_ratio."""

def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """late_payment_rate, mean DAYS_PAST_DUE, mean payment_shortfall_ratio."""

def aggregate_pos_and_cc(pos: pd.DataFrame, cc: pd.DataFrame) -> pd.DataFrame:
    """SK_DPD stats, balance utilization, count of late months."""

def build_relational_features(paths: PathsConfig) -> pd.DataFrame:
    """Left-join all aggregates onto application table by SK_ID_CURR."""
```

- Keep it config-driven: add `configs/feature_sources.yaml` (table paths + which agg groups to enable).
- Wire into `data/ingest.py` so `train` consumes the enriched frame.
- Watch memory: read aux tables with explicit dtypes; aggregate then drop raw.

**Acceptance:** AUC ≥ 0.78 on the same stratified split; feature count documented; joins reproducible from config.

---

## Phase 2 — Tuning + calibration — ⛔ needs Kaggle data

| # | Task | Detail |
|---|------|--------|
| 2.1 | **Optuna tuning** | New `tune` CLI subcommand. Objective = stratified 5-fold CV AUC with early stopping. Search: `num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `reg_alpha`, `reg_lambda`, `n_estimators`. Persist `study.db` + best params → `configs/model.yaml`. |
| 2.2 | **Probability calibration** | Wrap champion in `CalibratedClassifierCV(method="isotonic", cv=5)`. Refit on train, evaluate on held-out test. |
| 2.3 | **Report champion Brier + reliability curve** | Fill the currently-empty Brier cell; export `calibration_curve.csv` + PNG. |
| 2.4 | **README note** | One line: "profit/loss simulation assumes *calibrated* PD — here's the reliability curve." Signals real maturity. |

**Acceptance:** champion Brier reported; reliability curve near-diagonal; Optuna study committed (params, not the model).

---

## Phase 3 — Defense polish — partially data-dependent

| # | Task | Status |
|---|------|--------|
| 3.1 | Tests 8 → ~15: cover `relational.py` aggregations + a calibration sanity test | ⬜ |
| 3.2 | README "Modeling decisions" section: class_weight vs SMOTE, isotonic rationale, AUC progression 0.628 → 0.759 → 0.78+ | ⬜ |
| 3.3 | Add reliability curve + SHAP summary to the Streamlit dashboard | ⬜ |

---

## Critical path & sequencing

1. **Phase 0** — today, no data required.
2. **Phase 1** before Phase 2 — never tune/calibrate a shallow model first.
3. Phase 3 last.

## ⛔ BLOCKER
Phases 1–2 require the full **Home Credit Default Risk** dataset (the auxiliary CSVs: `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`) placed in `data/raw/`. Source: https://www.kaggle.com/c/home-credit-default-risk/data

**Effort:** ~5–7 focused days. **Outcome:** modeling 5–6 → 8–9, overall → 9/10, every current weak point becomes defensible.
