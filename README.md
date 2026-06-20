# Credit Risk Prediction Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-informational)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/explainability-SHAP-green)](https://shap.readthedocs.io/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-prediction-skay.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade ML pipeline that predicts probability of default (PD) on 307,511 real loan applications — with SHAP explainability, business scenario simulation, Basel III–aligned metrics (Gini / KS), and BI-ready outputs.

**[Live Demo →](https://credit-risk-prediction-skay.streamlit.app)**

---

## What this is

Most ML credit-risk projects are notebooks. This is a **config-driven CLI pipeline** built as a software engineer would build it:

- `src/` layout with a real Python package
- YAML-configured — zero hardcoded paths
- CLI entry point (`python -m credit_risk.cli <command>`)
- Every output is a contract: stable CSVs for BI tools
- SHAP TreeExplainer for audit-ready, per-prediction explainability
- Business simulation: threshold sweep + named scenarios
- Basel III–aligned dashboard: AUC, Gini coefficient, KS statistic
- 8 pytest tests including LightGBM → LogReg fallback path

---

## Results

| Model | AUC-ROC | Gini | Avg Precision | Brier Score |
|---|---|---|---|---|
| Baseline (no weighting) | 0.6280 | 0.256 | 0.1305 | 0.0728 |
| **Champion (class_weight=balanced)** | **0.7594** | **0.519** | **0.2492** | 0.1653 |

Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — 307,511 rows, 122 features, 8.1% default rate.

> **Note on Brier score:** The champion model's higher Brier score (0.165 vs 0.073) is expected — `class_weight='balanced'` shifts the model's probability outputs toward the minority class, which is the intended behavior for recall-focused lending decisions. AUC-ROC and Average Precision are the metrics that matter here.

---

## Architecture

```
credit-risk/
├── src/credit_risk/
│   ├── cli.py                 ← Entry point (argparse subcommands)
│   ├── config.py              ← YAML loaders + strict validation
│   ├── data/
│   │   └── ingest.py          ← Load + validate raw CSV
│   ├── features/
│   │   ├── dashboard_tables.py ← KPI, segments, missingness → BI CSVs
│   │   ├── groups.py          ← Regex-based feature group tagging
│   │   └── summary.py         ← Per-feature stats
│   └── modeling/
│       ├── pipeline.py        ← ColumnTransformer → LGBMClassifier
│       ├── train.py           ← Fit, evaluate, SHAP, save artifacts
│       ├── simulate.py        ← Threshold sweep → decision_simulation.csv
│       └── scenarios.py       ← Named scenario evaluation
├── configs/
│   ├── model.yaml             ← Model type, params, thresholds, scenarios
│   ├── paths.yaml             ← All I/O paths (never hardcoded)
│   └── feature_groups.yaml   ← Regex → feature group mapping
├── data/outputs/
│   ├── reports/               ← Metrics, SHAP, calibration, confusion matrix
│   └── predictions/           ← Decision simulation, scenarios, segments
├── tests/                     ← 8 pytest tests
├── app.py                     ← Streamlit dashboard (v2)
└── Makefile
```

---

## Engineering highlights

### 1. Config-driven, zero hardcoded paths
Every path, model parameter, and business rule lives in `configs/*.yaml`. The code reads them at runtime via a `validate_model_config()` gate that fails loudly on invalid input — before loading 307K rows.

### 2. SHAP TreeExplainer — production-grade explainability
After training, `shap.TreeExplainer` computes feature attributions on a 5,000-row sample from the test set. Results are exported as `feature_importance_shap.csv` — not model internals, but actual Shapley values for each prediction.

```
EXT_SOURCE_3       ████████████████████ 0.381  ← external credit bureau score
EXT_SOURCE_2       ██████████████████   0.345  ← external credit bureau score
EXT_SOURCE_1       ███████████          0.200  ← external credit bureau score
AMT_GOODS_PRICE    ██████████           0.189  ← loan purpose amount
AMT_CREDIT         █████████            0.168  ← total credit amount
DAYS_EMPLOYED      ██████               0.121  ← employment length
DAYS_BIRTH         ██████               0.114  ← applicant age
...
```

### 3. Business simulation layer — bridging ML to decisions
The pipeline runs 19 threshold sweeps (0.05–0.95) and computes approval rate, default rate, expected profit/loss, and net value at each point. Three named scenarios (Base / Conservative / Aggressive) model different lending strategies.

```python
# configs/model.yaml
scenarios:
  - name: Base
    threshold: 0.2
    profit_if_good: 0.20
    loss_given_default: 1.0
  - name: Conservative
    threshold: 0.1
    ...
```

### 4. Basel III–aligned metrics
The dashboard computes **Gini coefficient** (2 × AUC − 1) and **KS statistic** (max |CDF_default − CDF_good|) — the two discriminatory power metrics required by Basel III credit risk models, alongside standard AUC-ROC and Average Precision.

### 5. LightGBM → LogReg fallback chain
`build_model_pipeline()` gracefully degrades if LightGBM is unavailable — silently swapping in `LogisticRegression` and dropping incompatible params with a warning. This is tested in CI.

### 6. Stratified split + class_weight handling
The champion model uses `class_weight='balanced'` to address the 91.9% / 8.1% class imbalance, and the train/test split is stratified to preserve the imbalance ratio.

---

## Quickstart

```bash
git clone https://github.com/skayy47/credit-risk-prediction
cd credit-risk-prediction

# Install
pip install -e ".[dev,app]"

# Place data
# → data/raw/application_train.csv  (from Kaggle: home-credit-default-risk)

# Run full pipeline
python -m credit_risk.cli ingest-validate
python -m credit_risk.cli make-dashboard-tables
python -m credit_risk.cli train-simulate      # ~3-5 min on CPU
python -m credit_risk.cli simulate-scenarios

# Launch dashboard
streamlit run app.py
```

Or via Makefile:
```bash
make all    # full pipeline
make serve  # Streamlit
make test   # pytest
```

---

## CLI reference

| Command | What it does | Key outputs |
|---|---|---|
| `ingest-validate` | Load CSV, validate schema, export QA reports | `data_shape.txt`, `missingness.csv`, `dtypes.csv`, `target_stats.csv` |
| `make-dashboard-tables` | Feature stats, segments, KPIs → BI CSVs | `kpi_overview.csv`, `population_segments.csv`, `feature_summary.csv` |
| `train-simulate` | Full train + eval + SHAP + threshold sweep | `metrics_champion.csv`, `feature_importance_shap.csv`, `confusion_matrix.csv`, `calibration_bins.csv`, `predictions.csv`, `decision_simulation.csv` |
| `simulate-scenarios` | Evaluate named lending scenarios | `scenario_results.csv` |

---

## Dashboard (v2)

The Streamlit app reads pre-generated CSVs from `data/outputs/` — no retraining on deploy.

| Tab | Content |
|---|---|
| 📊 Model Intelligence | AUC comparison, ROC curve, confusion matrix (Precision/Recall/F1), risk score distribution, calibration curve |
| 🔍 Feature Insights | SHAP top-N bar (color-coded by feature group), feature-group donut, correlation-with-target chart, missingness overview |
| 💼 Business Simulator | Threshold slider (pre-set to optimal), approval/default/net chart, annual portfolio projection |
| 📋 Scenario Analysis | Scenario cards (Base/Conservative/Aggressive), profit/loss chart, risk vs return scatter |
| 🧩 Population Segments | All-segment overview grid, detailed analysis by income/age/credit band/gender |

**KPI row** — 6 live metrics: Dataset · Features · Default Rate · Champion AUC · **Gini Coefficient** · **KS Statistic**

---

## Output contracts (BI-ready)

All outputs are stable CSV contracts for Power BI, Tableau, or any BI tool:

```
data/outputs/
├── reports/
│   ├── metrics_baseline.csv          → auc_roc, average_precision, brier_score, log_loss
│   ├── metrics_champion.csv          → same, for class_weight=balanced model
│   ├── metrics_compare.csv           → both models side by side
│   ├── feature_importance_shap.csv   → feature, mean_abs_shap, rank
│   ├── confusion_matrix.csv          → tn, fp, fn, tp, precision, recall
│   └── calibration_bins.csv          → predicted_midpoint, actual_default_rate, n
└── predictions/
    ├── kpi_overview.csv              → n_rows, default_rate, avg_missing_rate
    ├── decision_simulation.csv       → 19 thresholds × (approval_rate, default_rate, net_value)
    ├── scenario_results.csv          → 3 scenarios × business metrics
    ├── population_segments.csv       → segment_name, segment_value, n, default_rate
    ├── feature_summary.csv           → per-feature stats + group tag
    └── correlation_top.csv           → feature × pearson correlation with TARGET
```

---

## Tests

```bash
pytest tests/ -v
```

```
tests/test_model_config_validation.py   - config schema validation
tests/test_pipeline_fallback.py         - LightGBM → LogReg fallback
tests/test_pipeline_params.py           - param sanitization
tests/test_simulate_validation.py       - threshold edge cases
tests/test_train_fallback.py            - full pipeline fallback integration
```

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — `application_train.csv`

| Attribute | Value |
|---|---|
| Rows | 307,511 |
| Features | 122 (incl. TARGET, SK_ID_CURR) |
| Target | `TARGET` = 1 (default), 0 (good payer) |
| Class distribution | 91.9% good / 8.1% default |
| Features used | 103 numeric + 16 categorical |

The raw CSV is not committed (large binary). Download from Kaggle and place at `data/raw/application_train.csv`.

---

## Author

**Oussama Skia (SKAY)** — AI/Data Engineer, Casablanca  
[GitHub](https://github.com/skayy47) · [LinkedIn](https://linkedin.com/in/oussama-skia)

> Not a vibe-coded notebook — every artifact is verifiable, every output is a contract.
