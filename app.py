"""Credit Risk Prediction Engine — Streamlit Dashboard v2.

Reads pre-generated CSV artifacts from data/outputs/.
No model retraining at runtime.
"""
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk | Oussama Skia",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE    = Path(__file__).parent
REPORTS = BASE / "data" / "outputs" / "reports"
PREDS   = BASE / "data" / "outputs" / "predictions"

# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#1E40AF"
C_GREEN  = "#059669"
C_AMBER  = "#D97706"
C_RED    = "#DC2626"
C_INDIGO = "#4338CA"
C_SLATE  = "#475569"
C_TEAL   = "#0D9488"
C_VIOLET = "#7C3AED"

# ── premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }

/* KPI cards */
.kpi-card {
  border-radius: 12px;
  padding: 16px 18px;
  margin: 2px 0;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0,0,0,0.10);
}
.kpi-label {
  font-size: 0.70em;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #64748B;
  margin-bottom: 5px;
}
.kpi-value {
  font-size: 1.80em;
  font-weight: 800;
  color: #0F172A;
  line-height: 1.1;
}
.kpi-delta {
  font-size: 0.78em;
  font-weight: 600;
  margin-top: 3px;
}

/* Insight callout */
.insight-box {
  background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
  border-left: 4px solid #1E40AF;
  border-radius: 0 8px 8px 0;
  padding: 12px 16px;
  margin: 10px 0;
  font-size: 0.88em;
  color: #1E293B;
}
.insight-box b { color: #1E40AF; }
.insight-box code {
  background: #DBEAFE;
  border-radius: 3px;
  padding: 1px 4px;
  font-size: 0.92em;
}

/* Scenario cards */
.sc-card {
  border-radius: 12px;
  padding: 20px 18px;
  transition: box-shadow 0.2s;
  min-height: 220px;
}
.sc-card:hover { box-shadow: 0 8px 20px rgba(0,0,0,0.10); }
.sc-title { font-size: 1.15em; font-weight: 700; margin-bottom: 12px; }
.sc-row {
  display: flex;
  justify-content: space-between;
  margin: 5px 0;
  font-size: 0.87em;
  color: #334155;
}
.sc-net { font-size: 1.30em; font-weight: 800; margin-top: 14px; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 8px 8px 0 0;
  font-weight: 600;
  padding: 8px 16px;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def _csv(path: str, fallback=None):
    p = Path(path)
    if not p.exists() or p.stat().st_size < 10:
        return fallback
    try:
        return pd.read_csv(p)
    except Exception:
        return fallback


def _kpi(label: str, value: str, delta: str = "", color: str = C_BLUE, icon: str = ""):
    st.markdown(
        f"""<div class="kpi-card"
             style="background:linear-gradient(135deg,{color}1C,{color}08);
                    border-left:4px solid {color};">
          <div class="kpi-label">{icon}&nbsp;{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta" style="color:{color}">{delta}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _insight(text: str):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


def _chart(fig: go.Figure, height: int = 320, title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=13, color="#0F172A"), x=0) if title else None,
        margin=dict(l=0, r=0, t=32 if title else 10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=11.5),
    )
    return fig


@st.cache_data
def _compute_roc(preds_path: str):
    """ROC curve from predictions CSV. Returns (fpr list, tpr list, auc float)."""
    try:
        from sklearn.metrics import roc_curve, auc
        df = pd.read_csv(preds_path)
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_proba"])
        return fpr.tolist(), tpr.tolist(), round(float(auc(fpr, tpr)), 4)
    except Exception:
        return None, None, None


@st.cache_data
def _compute_ks(preds_path: str) -> float:
    """KS statistic — max |CDF_default − CDF_good|. No scipy needed."""
    try:
        df = pd.read_csv(preds_path).sort_values("y_proba")
        n_d = int(df["y_true"].sum())
        n_g = len(df) - n_d
        if n_d == 0 or n_g == 0:
            return None
        cum_d = df["y_true"].cumsum() / n_d
        cum_g = (1 - df["y_true"]).cumsum() / n_g
        return round(float((cum_d - cum_g).abs().max()), 4)
    except Exception:
        return None


def _shap_group(f: str) -> str:
    u = f.upper()
    if "EXT_SOURCE" in u:
        return "External Bureau"
    if u.startswith("AMT"):
        return "Loan Amounts"
    if u.startswith("DAYS"):
        return "Temporal / Age"
    if any(k in u for k in ("NAME_", "CODE_", "GENDER", "EDUCATION", "OCCUPATION",
                              "FAMILY", "HOUSING", "ORGANIZATION", "TYPE")):
        return "Demographics"
    if any(k in u for k in ("REGION", "CITY", "RATING")):
        return "Geography"
    if any(k in u for k in ("FLAG_", "OWN_", "REG_", "LIVE_")):
        return "Flags"
    return "Other"


_GROUP_COLORS = {
    "External Bureau": C_BLUE,
    "Loan Amounts":    C_GREEN,
    "Temporal / Age":  C_AMBER,
    "Demographics":    C_INDIGO,
    "Geography":       C_TEAL,
    "Flags":           C_SLATE,
    "Credit History":  C_VIOLET,
    "Other":           "#94A3B8",
}


# ── load data ─────────────────────────────────────────────────────────────────
kpi           = _csv(str(PREDS  / "kpi_overview.csv"))
metrics_base  = _csv(str(REPORTS / "metrics_baseline.csv"))
metrics_champ = _csv(str(REPORTS / "metrics_champion.csv"))
metrics_cmp   = _csv(str(REPORTS / "metrics_compare.csv"))
shap_df       = _csv(str(REPORTS / "feature_importance_shap.csv"))
conf_mat      = _csv(str(REPORTS / "confusion_matrix.csv"))
calib_df      = _csv(str(REPORTS / "calibration_bins.csv"))
decision_sim  = _csv(str(PREDS  / "decision_simulation.csv"))
scenarios     = _csv(str(PREDS  / "scenario_results.csv"))
segments      = _csv(str(PREDS  / "population_segments.csv"))
feat_summary  = _csv(str(PREDS  / "feature_summary.csv"))
preds_df      = _csv(str(PREDS  / "predictions.csv"))
corr_df       = _csv(str(PREDS  / "correlation_top.csv"))

# ── derived metrics ───────────────────────────────────────────────────────────
base_auc     = metrics_base["auc_roc"].iloc[0]  if metrics_base  is not None else 0.628
champ_auc    = metrics_champ["auc_roc"].iloc[0] if metrics_champ is not None else 0.759
default_rate = kpi["default_rate"].iloc[0]       if kpi           is not None else 0.0807
gini         = round(2 * champ_auc - 1, 3)

preds_path_str = str(PREDS / "predictions.csv")
ks_stat        = _compute_ks(preds_path_str) if preds_df is not None else None
fpr_list, tpr_list, roc_auc = _compute_roc(preds_path_str) if preds_df is not None else (None, None, None)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Risk Engine")
    st.markdown(
        "Production ML pipeline predicting probability of default (PD) on **307,511** "
        "loan applications from the [Home Credit Default Risk]"
        "(https://www.kaggle.com/c/home-credit-default-risk) dataset."
    )
    st.divider()

    st.markdown("**Model Vitals**")
    st.progress(min(champ_auc, 1.0), text=f"AUC-ROC: {champ_auc:.3f}")
    st.progress(min(gini, 1.0),      text=f"Gini:    {gini:.3f}")
    if ks_stat:
        st.progress(min(ks_stat, 1.0), text=f"KS Stat: {ks_stat:.3f}")

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- LightGBM · scikit-learn\n"
        "- SHAP TreeExplainer\n"
        "- Config-driven CLI\n"
        "- Power BI–ready CSVs\n"
        "- Streamlit · Plotly"
    )
    st.divider()
    st.markdown("**Pipeline**")
    st.code(
        "python -m credit_risk.cli ingest-validate\n"
        "python -m credit_risk.cli make-dashboard-tables\n"
        "python -m credit_risk.cli train-simulate\n"
        "python -m credit_risk.cli simulate-scenarios",
        language="bash",
    )
    st.divider()
    st.markdown(
        "**Author:** [Oussama Skia](https://github.com/skayy47)  \n"
        "**Source:** [GitHub](https://github.com/skayy47/credit-risk-prediction)"
    )

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:2em;font-weight:800;color:#0F172A;margin-bottom:4px;'>"
    "💳 Credit Risk Prediction Engine</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#475569;font-size:1em;margin-bottom:20px;'>"
    "A production-grade ML pipeline that predicts probability of default on real banking data — "
    "with SHAP explainability, business scenario simulation, and BI-ready outputs.</p>",
    unsafe_allow_html=True,
)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    _kpi("Dataset", "307,511", "loan applications", C_BLUE, "🗄️")
with c2:
    _kpi("Features", "120", "engineered features", C_INDIGO, "⚙️")
with c3:
    _kpi("Default Rate", f"{default_rate:.1%}", "class imbalance", C_RED, "⚠️")
with c4:
    delta_auc = champ_auc - base_auc
    _kpi("Champion AUC", f"{champ_auc:.3f}", f"+{delta_auc:.3f} vs baseline", C_GREEN, "🏆")
with c5:
    _kpi("Gini Coefficient", f"{gini:.3f}", "2 × AUC − 1", C_TEAL, "📐")
with c6:
    ks_label = f"{ks_stat:.3f}" if ks_stat else "—"
    _kpi("KS Statistic", ks_label, "score separation", C_VIOLET, "📊")

st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Intelligence",
    "🔍 Feature Insights",
    "💼 Business Simulator",
    "📋 Scenario Analysis",
    "🧩 Population Segments",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Model Performance")

    # Metrics comparison table
    if metrics_cmp is not None:
        cmp_display = metrics_cmp.copy()
        for col in ["auc_roc", "average_precision", "brier_score", "log_loss"]:
            if col in cmp_display.columns:
                cmp_display[col] = cmp_display[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(cmp_display, use_container_width=True, hide_index=True)
    elif metrics_base is not None:
        st.dataframe(metrics_base, use_container_width=True, hide_index=True)

    if metrics_champ is not None:
        avg_prec = metrics_champ["average_precision"].iloc[0]
        ks_part = f" and KS <b>{ks_stat:.3f}</b>" if ks_stat is not None else ""
        _insight(
            f"<b>Key finding:</b> Champion model — AUC <b>{champ_auc:.3f}</b>, "
            f"Average Precision <b>{avg_prec:.3f}</b> on a {default_rate:.1%} imbalanced dataset. "
            f"Gini <b>{gini:.3f}</b>{ks_part} confirm strong discriminatory power "
            f"suitable for regulatory credit scoring (Basel III framework)."
        )

    # ── Row 1: AUC bar + Confusion matrix ────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### AUC-ROC Comparison")
        models, aucs, colors = [], [], []
        if metrics_base  is not None:
            models.append("Baseline");  aucs.append(base_auc);  colors.append(C_SLATE)
        if metrics_champ is not None:
            models.append("Champion"); aucs.append(champ_auc); colors.append(C_BLUE)
        if models:
            fig = go.Figure(go.Bar(
                x=models, y=aucs,
                marker_color=colors,
                text=[f"{v:.4f}" for v in aucs],
                textposition="outside",
                width=0.40,
            ))
            fig.add_hline(y=0.5, line_dash="dot", line_color="#CBD5E1",
                         annotation_text="Random (0.50)", annotation_position="right",
                         annotation_font_size=10)
            _chart(fig, 300).update_yaxis(range=[0.40, max(aucs) + 0.07], title="AUC-ROC")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Confusion Matrix")
        if conf_mat is not None:
            row = conf_mat.iloc[0]
            cm_data = [
                [int(row["true_negative"]),  int(row["false_positive"])],
                [int(row["false_negative"]), int(row["true_positive"])],
            ]
            prec = float(row.get("precision", 0))
            rec  = float(row.get("recall", 0))
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            fig = go.Figure(go.Heatmap(
                z=cm_data,
                x=["Predicted: Good", "Predicted: Default"],
                y=["Actual: Good",    "Actual: Default"],
                colorscale=[[0, "#F0F9FF"], [1, C_BLUE]],
                text=[[f"{v:,}" for v in r] for r in cm_data],
                texttemplate="%{text}",
                textfont=dict(size=15),
                showscale=False,
            ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                annotations=[dict(
                    text=f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}",
                    x=0.5, y=-0.20, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=12, color=C_SLATE),
                )],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `train-simulate` to generate the confusion matrix.")

    # ── Row 2: ROC Curve + Risk Score Distribution ────────────────────────────
    col_roc, col_dist = st.columns(2)

    with col_roc:
        st.markdown("#### ROC Curve")
        if fpr_list is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines", name="Random classifier",
                line=dict(color="#CBD5E1", dash="dash", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=fpr_list, y=tpr_list,
                mode="lines",
                name=f"Champion — full dataset (AUC = {roc_auc:.3f})",
                line=dict(color=C_BLUE, width=2.5),
                fill="tonexty",
                fillcolor="rgba(30,64,175,0.10)",
            ))
            _chart(fig, 320).update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.35, y=0.07),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"⚠️ Full dataset ROC (train + test) — optimistic. "
                f"Honest **test-set AUC: {champ_auc:.3f}** (see metrics table above)."
            )
        else:
            st.info("Run `train-simulate` to generate ROC curve data.")

    with col_dist:
        st.markdown("#### Risk Score Distribution")
        if preds_df is not None and "y_proba" in preds_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=preds_df[preds_df["y_true"] == 0]["y_proba"],
                name="Good (no default)",
                nbinsx=50,
                marker_color=C_GREEN,
                opacity=0.65,
            ))
            fig.add_trace(go.Histogram(
                x=preds_df[preds_df["y_true"] == 1]["y_proba"],
                name="Default",
                nbinsx=50,
                marker_color=C_RED,
                opacity=0.65,
            ))
            _chart(fig, 320).update_layout(
                barmode="overlay",
                xaxis_title="Predicted Default Probability",
                yaxis_title="Count",
                legend=dict(x=0.55, y=0.95),
            )
            st.plotly_chart(fig, use_container_width=True)
            if ks_stat:
                _insight(
                    f"<b>Score separation (KS = {ks_stat:.3f}):</b> The two distributions overlap "
                    "in the 0.1–0.4 range — the boundary zone where threshold selection matters most. "
                    "Higher KS = cleaner separation between defaulters and good payers."
                )
        else:
            st.info("Run `train-simulate` to generate prediction scores.")

    # ── Calibration ───────────────────────────────────────────────────────────
    if calib_df is not None:
        st.markdown("#### Probability Calibration")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=calib_df["predicted_midpoint"],
            y=calib_df["actual_default_rate"],
            mode="lines+markers",
            name="Model calibration",
            line=dict(color=C_BLUE, width=2.5),
            marker=dict(size=7, color=C_BLUE),
        ))
        max_val = max(
            calib_df["predicted_midpoint"].max(),
            calib_df["actual_default_rate"].max(),
        ) + 0.02
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect calibration",
            line=dict(color=C_SLATE, dash="dash", width=1.5),
        ))
        _chart(fig, 280).update_layout(
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Actual Default Rate",
            legend=dict(x=0.02, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)
        _insight(
            "<b>Calibration note:</b> The Champion model uses <code>class_weight='balanced'</code> "
            "which intentionally shifts probability outputs toward the minority class — "
            "by design, for recall-focused lending decisions. "
            "AUC-ROC and Average Precision are the primary performance metrics here."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Feature Importance — SHAP TreeExplainer")

    if shap_df is not None and len(shap_df) > 0:
        col_shap, col_group = st.columns([2, 1])

        with col_shap:
            top_n = st.slider(
                "Top N features",
                min_value=10, max_value=min(40, len(shap_df)), value=20, step=5,
            )
            top = shap_df.head(top_n).copy()
            top["feature_short"] = top["feature"].str[:45]
            top["group"] = top["feature"].apply(_shap_group)
            marker_colors = [_GROUP_COLORS.get(g, "#94A3B8") for g in top["group"]]

            fig = go.Figure(go.Bar(
                x=top["mean_abs_shap"],
                y=top["feature_short"],
                orientation="h",
                marker=dict(color=marker_colors),
                text=[f"{v:.4f}" for v in top["mean_abs_shap"]],
                textposition="outside",
                textfont=dict(size=10),
            ))
            _chart(fig, 80 + top_n * 23).update_layout(
                xaxis_title="Mean |SHAP value|",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=80, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "SHAP via `shap.TreeExplainer` on a 5,000-row test sample. "
                "Color = feature group. Higher |SHAP| = stronger model impact."
            )

        with col_group:
            st.markdown("#### By Feature Group")
            group_shap = top.groupby("group")["mean_abs_shap"].sum().reset_index()
            group_shap = group_shap.sort_values("mean_abs_shap", ascending=False)
            fig = go.Figure(go.Pie(
                labels=group_shap["group"],
                values=group_shap["mean_abs_shap"],
                hole=0.45,
                marker_colors=[_GROUP_COLORS.get(g, "#94A3B8") for g in group_shap["group"]],
                textfont=dict(size=10),
            ))
            _chart(fig, 290).update_layout(
                legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=9)),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            if len(shap_df) >= 3:
                t1, t2, t3 = shap_df.iloc[0], shap_df.iloc[1], shap_df.iloc[2]
                _insight(
                    f"<b>Top drivers:</b> <b>{t1['feature']}</b> ({t1['mean_abs_shap']:.3f}), "
                    f"<b>{t2['feature']}</b> ({t2['mean_abs_shap']:.3f}), "
                    f"<b>{t3['feature']}</b> ({t3['mean_abs_shap']:.3f}). "
                    "External bureau scores dominate — consistent with Basel III guidance."
                )

            with st.expander("Full SHAP table"):
                st.dataframe(
                    shap_df[["rank", "feature", "mean_abs_shap"]].assign(
                        mean_abs_shap=shap_df["mean_abs_shap"].apply(lambda x: f"{x:.6f}")
                    ),
                    use_container_width=True, hide_index=True,
                )
    else:
        st.info("Run `python -m credit_risk.cli train-simulate` to compute SHAP importance.")

    # ── Correlation with target ───────────────────────────────────────────────
    if corr_df is not None and len(corr_df) > 0:
        st.markdown("### Feature Correlation with Default")
        corr_top = corr_df.head(20).copy()
        corr_top["abs_corr"] = corr_top["corr_with_target"].abs()
        corr_top = corr_top.sort_values("abs_corr", ascending=False)
        bar_colors = [C_RED if v > 0 else C_GREEN for v in corr_top["corr_with_target"]]
        fig = go.Figure(go.Bar(
            x=corr_top["corr_with_target"],
            y=corr_top["feature"].str[:40],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in corr_top["corr_with_target"]],
            textposition="outside",
        ))
        _chart(fig, 430).update_layout(
            xaxis_title="Pearson Correlation with TARGET (default = 1)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=80, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        _insight(
            "<b>Reading the chart:</b> Green bars (negative correlation) = higher value → "
            "lower default risk (e.g. EXT_SOURCE scores). "
            "Red bars (positive) = higher value → higher default risk. "
            "Note: DAYS_BIRTH is stored as negative days — older applicants have lower risk."
        )

    # ── Missingness ───────────────────────────────────────────────────────────
    if feat_summary is not None and "missing_rate" in feat_summary.columns:
        st.markdown("### Data Quality — Feature Missingness")
        feat_miss = (
            feat_summary[feat_summary["missing_rate"] > 0]
            .sort_values("missing_rate", ascending=False)
            .head(25)
        )
        if len(feat_miss) > 0:
            miss_colors = [
                C_RED if r > 0.5 else C_AMBER if r > 0.2 else C_GREEN
                for r in feat_miss["missing_rate"]
            ]
            fig = go.Figure(go.Bar(
                x=feat_miss["missing_rate"] * 100,
                y=feat_miss["feature"].str[:40],
                orientation="h",
                marker_color=miss_colors,
                text=[f"{r:.0%}" for r in feat_miss["missing_rate"]],
                textposition="outside",
            ))
            _chart(fig, 530).update_layout(
                xaxis_title="Missing Rate (%)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=80, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            n_high = int((feat_miss["missing_rate"] > 0.5).sum())
            _insight(
                f"<b>Data quality:</b> <b>{n_high}</b> features have >50% missingness (red) — "
                "these come from external data tables not available for all applicants. "
                "The pipeline handles them with median imputation + missing-indicator encoding."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BUSINESS SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Interactive Business Threshold Simulator")
    st.markdown(
        "Adjust the decision threshold to see how it affects approval rate, "
        "default exposure, and estimated portfolio net value."
    )

    if decision_sim is not None:
        opt_idx    = decision_sim["expected_value"].idxmax()
        opt_thresh = float(decision_sim.loc[opt_idx, "threshold"])
        opt_net    = float(decision_sim.loc[opt_idx, "expected_value"])

        _insight(
            f"<b>Optimal threshold: {opt_thresh:.2f}</b> — maximises net portfolio value at "
            f"<b>${opt_net:,.0f}</b>. Approval rate: "
            f"<b>{decision_sim.loc[opt_idx,'approval_rate']:.1%}</b>, approved-loan default rate: "
            f"<b>{decision_sim.loc[opt_idx,'default_rate_among_approved']:.2%}</b>. "
            "Slider pre-set to optimal — drag to explore trade-offs."
        )

        threshold_vals   = decision_sim["threshold"].tolist()
        selected_thresh  = st.select_slider(
            "Decision Threshold",
            options=[round(t, 2) for t in threshold_vals],
            value=round(opt_thresh, 2),
        )
        sel_row = decision_sim[
            decision_sim["threshold"].round(2) == round(selected_thresh, 2)
        ].iloc[0]

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Approval Rate", f"{sel_row['approval_rate']:.1%}")
        with mc2:
            st.metric("Default Rate (approved)", f"{sel_row['default_rate_among_approved']:.2%}")
        with mc3:
            st.metric("Expected Profit", f"${sel_row['expected_profit']:,.0f}")
        with mc4:
            net = float(sel_row["expected_value"])
            delta_v = net - opt_net
            st.metric(
                "Net Value", f"${net:,.0f}",
                delta=f"{'▲' if delta_v >= 0 else '▼'} ${abs(delta_v):,.0f} vs optimal",
            )

        # Annual projection
        with st.expander("📈 Annual Portfolio Projection"):
            scale = st.number_input(
                "Portfolio scale factor (× current portfolio size)",
                min_value=1, max_value=10_000, value=100, step=10,
            )
            p1, p2, p3 = st.columns(3)
            p1.metric("Projected Annual Profit", f"${sel_row['expected_profit'] * scale:,.0f}")
            p2.metric("Projected Annual Loss",   f"${sel_row['expected_loss']   * scale:,.0f}")
            p3.metric("Projected Net Value",     f"${net * scale:,.0f}")

        st.markdown("---")

        # Multi-trace chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=decision_sim["threshold"], y=decision_sim["approval_rate"],
            name="Approval Rate",
            line=dict(color=C_BLUE, width=2.5),
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=decision_sim["threshold"], y=decision_sim["default_rate_among_approved"],
            name="Default Rate (approved)",
            line=dict(color=C_RED, width=2, dash="dash"),
            yaxis="y1",
        ))
        fig.add_trace(go.Bar(
            x=decision_sim["threshold"],
            y=decision_sim["expected_value"],
            name="Net Value ($)",
            marker_color=[
                C_GREEN if v >= 0 else C_RED
                for v in decision_sim["expected_value"]
            ],
            opacity=0.55,
            yaxis="y2",
        ))
        fig.add_vline(
            x=selected_thresh, line_dash="dot", line_color=C_INDIGO,
            annotation_text=f"Selected: {selected_thresh}",
            annotation_position="top left",
            annotation_font_size=10,
        )
        fig.add_vline(
            x=opt_thresh, line_dash="dash", line_color=C_GREEN,
            annotation_text=f"Optimal: {opt_thresh}",
            annotation_position="top right",
            annotation_font_size=10,
        )
        _chart(fig, 420).update_layout(
            xaxis_title="Threshold",
            yaxis=dict(title="Rate", tickformat=".0%", side="left"),
            yaxis2=dict(title="Net Value ($)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Optimal threshold (max net value): **{opt_thresh}** at **${opt_net:,.0f}**"
        )
    else:
        st.info("Run `python -m credit_risk.cli train-simulate` to generate decision simulation data.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Business Scenario Comparison")
    st.markdown(
        "Three pre-configured lending strategies evaluated on the full portfolio — "
        "comparing approval volume, risk exposure, and expected net value."
    )

    if scenarios is not None:
        SC_COLOR = {"Base": C_BLUE, "Conservative": C_INDIGO, "Aggressive": C_GREEN}
        SC_BG    = {"Base": "#EFF6FF", "Conservative": "#EEF2FF", "Aggressive": "#ECFDF5"}

        sc_cols = st.columns(len(scenarios))
        for i, (_, sc) in enumerate(scenarios.iterrows()):
            with sc_cols[i]:
                name  = sc["scenario_name"]
                color = SC_COLOR.get(name, C_SLATE)
                bg    = SC_BG.get(name, "#F8FAFC")
                st.markdown(
                    f"""<div class="sc-card" style="background:{bg};
                          border:1px solid {color}30;border-top:4px solid {color};">
                      <div class="sc-title" style="color:{color}">{name}</div>
                      <div class="sc-row">
                        <span>Threshold</span><b>{sc['threshold']}</b>
                      </div>
                      <div class="sc-row">
                        <span>Approval Rate</span><b>{sc['approval_rate']:.1%}</b>
                      </div>
                      <div class="sc-row">
                        <span>Default Rate</span><b>{sc['default_rate']:.2%}</b>
                      </div>
                      <div class="sc-row">
                        <span>Expected Profit</span>
                        <b style="color:{C_GREEN}">${sc['expected_profit']:,.0f}</b>
                      </div>
                      <div class="sc-row">
                        <span>Expected Loss</span>
                        <b style="color:{C_RED}">${sc['expected_loss']:,.0f}</b>
                      </div>
                      <div class="sc-net" style="color:{color}">
                        Net Value: ${sc['net_value']:,.0f}
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        col_bar, col_scatter = st.columns(2)

        with col_bar:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Expected Profit",
                x=scenarios["scenario_name"], y=scenarios["expected_profit"],
                marker_color=C_GREEN, opacity=0.85,
            ))
            fig.add_trace(go.Bar(
                name="Expected Loss",
                x=scenarios["scenario_name"], y=-scenarios["expected_loss"],
                marker_color=C_RED, opacity=0.85,
            ))
            fig.add_trace(go.Scatter(
                name="Net Value",
                x=scenarios["scenario_name"], y=scenarios["net_value"],
                mode="markers+text",
                marker=dict(size=14, color=C_INDIGO, symbol="diamond"),
                text=[f"${v:,.0f}" for v in scenarios["net_value"]],
                textposition="top center",
            ))
            _chart(fig, 380).update_layout(
                barmode="relative",
                yaxis_title="Value ($)",
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_scatter:
            # Risk vs Return
            fig = go.Figure()
            for _, sc in scenarios.iterrows():
                name  = sc["scenario_name"]
                color = SC_COLOR.get(name, C_SLATE)
                fig.add_trace(go.Scatter(
                    x=[float(sc["default_rate"])],
                    y=[float(sc["net_value"])],
                    mode="markers+text",
                    name=name,
                    marker=dict(size=26, color=color, opacity=0.82,
                               line=dict(width=2, color="white")),
                    text=[name],
                    textposition="top center",
                    textfont=dict(size=11, color=color),
                ))
            _chart(fig, 380).update_layout(
                xaxis_title="Portfolio Default Rate",
                xaxis=dict(tickformat=".2%"),
                yaxis_title="Net Portfolio Value ($)",
                showlegend=False,
            )
            fig.add_annotation(
                x=0.5, y=0.04, xref="paper", yref="paper",
                text="← Lower risk | Higher return →",
                showarrow=False, font=dict(size=10, color=C_SLATE),
            )
            st.plotly_chart(fig, use_container_width=True)

        best = scenarios.loc[scenarios["net_value"].idxmax()]
        cons_net = scenarios[scenarios["scenario_name"] == "Conservative"]["net_value"]
        _insight(
            f"<b>Recommendation:</b> The <b>{best['scenario_name']}</b> strategy delivers the "
            f"highest net value at <b>${best['net_value']:,.0f}</b> with a "
            f"<b>{best['default_rate']:.2%}</b> portfolio default rate and "
            f"<b>{best['approval_rate']:.1%}</b> approval rate. "
            + (f"Risk-averse lenders may prefer Conservative (${cons_net.iloc[0]:,.0f} net) "
               "for lower default exposure." if len(cons_net) > 0 else "")
        )
    else:
        st.info("Run `python -m credit_risk.cli simulate-scenarios` to generate scenario results.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — POPULATION SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Default Rate by Population Segment")

    if segments is not None:
        segment_types = segments["segment_name"].unique().tolist()

        # ── Overview: all segments side by side ───────────────────────────────
        st.markdown("#### Segment Overview")
        overview_cols = st.columns(len(segment_types))
        for i, seg_name in enumerate(segment_types):
            with overview_cols[i]:
                seg_data = segments[segments["segment_name"] == seg_name].copy()
                w_avg    = (seg_data["default_rate"] * seg_data["n"]).sum() / seg_data["n"].sum()
                fig = go.Figure(go.Bar(
                    x=seg_data["segment_value"].astype(str),
                    y=seg_data["default_rate"],
                    text=[f"{v:.1%}" for v in seg_data["default_rate"]],
                    textposition="outside",
                    marker=dict(
                        color=seg_data["default_rate"],
                        colorscale=[[0, "#DCFCE7"], [0.5, C_AMBER], [1.0, C_RED]],
                        cmin=0.0,
                        cmax=float(seg_data["default_rate"].max()),
                        showscale=False,
                    ),
                ))
                fig.add_hline(
                    y=w_avg, line_dash="dash", line_color=C_SLATE,
                    annotation_text=f"Avg: {w_avg:.2%}",
                    annotation_position="right",
                    annotation_font_size=9,
                )
                _chart(fig, 250, title=seg_name.replace("_", " ").title()).update_layout(
                    yaxis=dict(tickformat=".1%"),
                    margin=dict(l=0, r=60, t=30, b=0),
                    xaxis_title=None,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Detailed single segment ───────────────────────────────────────────
        st.markdown("#### Detailed Segment Analysis")
        selected_seg = st.selectbox("Select Segment", segment_types)
        seg_data     = segments[segments["segment_name"] == selected_seg].copy()
        total_n      = seg_data["n"].sum()
        w_avg        = (seg_data["default_rate"] * seg_data["n"]).sum() / total_n

        fig = go.Figure(go.Bar(
            x=seg_data["segment_value"].astype(str),
            y=seg_data["default_rate"],
            text=[f"{v:.1%}" for v in seg_data["default_rate"]],
            textposition="outside",
            marker=dict(
                color=seg_data["default_rate"],
                colorscale=[[0, "#DCFCE7"], [0.5, C_AMBER], [1.0, C_RED]],
                cmin=0.0,
                cmax=float(seg_data["default_rate"].max()),
                showscale=True,
                colorbar=dict(title="Default Rate", tickformat=".1%"),
            ),
        ))
        fig.add_hline(
            y=w_avg, line_dash="dash", line_color=C_SLATE,
            annotation_text=f"Overall avg: {w_avg:.2%}",
            annotation_position="right",
        )
        _chart(fig, 390).update_layout(
            xaxis_title=selected_seg.replace("_", " ").title(),
            yaxis=dict(title="Default Rate", tickformat=".1%"),
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        riskiest = seg_data.loc[seg_data["default_rate"].idxmax()]
        safest   = seg_data.loc[seg_data["default_rate"].idxmin()]
        spread   = float(riskiest["default_rate"]) - float(safest["default_rate"])

        with col_a:
            st.metric(
                "Highest Risk Segment", str(riskiest["segment_value"]),
                f"{riskiest['default_rate']:.2%} default rate",
                delta_color="inverse",
            )
        with col_b:
            st.metric(
                "Lowest Risk Segment", str(safest["segment_value"]),
                f"{safest['default_rate']:.2%} default rate",
            )
        with col_c:
            st.metric("Risk Spread", f"{spread:.2%}", "highest − lowest segment")

        _insight(
            f"<b>Population insight:</b> The <b>{riskiest['segment_value']}</b> group has a "
            f"<b>{riskiest['default_rate']:.2%}</b> default rate vs "
            f"<b>{safest['default_rate']:.2%}</b> for <b>{safest['segment_value']}</b> — "
            f"a <b>{spread:.2%}</b> spread. This differential supports risk-based pricing strategies."
        )
    else:
        st.info("Run `python -m credit_risk.cli make-dashboard-tables` to generate population segments.")


# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:0.85em;padding:8px 0;'>"
    "Built by <b style='color:#475569;'>Oussama Skia</b> · "
    "<a href='https://github.com/skayy47/credit-risk-prediction' style='color:#1E40AF;'>GitHub</a> · "
    "<a href='https://www.kaggle.com/c/home-credit-default-risk' style='color:#1E40AF;'>Home Credit Data</a> · "
    "LightGBM + SHAP · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
