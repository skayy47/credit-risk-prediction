"""Credit Risk Prediction Engine — Streamlit Dashboard.

Reads pre-generated CSV artifacts from data/outputs/.
No model retraining at runtime.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk | Oussama Skia",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
REPORTS = BASE / "data" / "outputs" / "reports"
PREDS   = BASE / "data" / "outputs" / "predictions"

# ── colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#1E40AF"
C_GREEN  = "#059669"
C_AMBER  = "#D97706"
C_RED    = "#DC2626"
C_INDIGO = "#4338CA"
C_SLATE  = "#475569"
C_BG     = "#F8FAFC"

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def _csv(path: Path, fallback=None):
    p = Path(path)
    if not p.exists() or p.stat().st_size < 10:
        return fallback
    return pd.read_csv(p)


def _metric_card(label: str, value: str, delta: str = "", delta_color: str = "normal"):
    st.metric(label=label, value=value, delta=delta or None, delta_color=delta_color)


# ── load data ─────────────────────────────────────────────────────────────────
kpi          = _csv(PREDS / "kpi_overview.csv")
metrics_base = _csv(REPORTS / "metrics_baseline.csv")
metrics_champ = _csv(REPORTS / "metrics_champion.csv")
metrics_cmp  = _csv(REPORTS / "metrics_compare.csv")
shap_df      = _csv(REPORTS / "feature_importance_shap.csv")
conf_mat     = _csv(REPORTS / "confusion_matrix.csv")
calib_df     = _csv(REPORTS / "calibration_bins.csv")
decision_sim = _csv(PREDS  / "decision_simulation.csv")
scenarios    = _csv(PREDS  / "scenario_results.csv")
segments     = _csv(PREDS  / "population_segments.csv")
feat_summary = _csv(PREDS  / "feature_summary.csv")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Risk Engine")
    st.markdown(
        "Production ML pipeline predicting probability of default (PD) on **307,511** "
        "loan applications from the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset."
    )
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
st.markdown("## 💳 Credit Risk Prediction Engine")
st.markdown(
    "A production-grade ML pipeline that predicts probability of default on real banking data — "
    "with SHAP explainability, business scenario simulation, and BI-ready outputs."
)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    _metric_card("Dataset", "307,511", "loan applications")
with c2:
    _metric_card("Features", "120", "engineered")
with c3:
    default_rate = kpi["default_rate"].iloc[0] if kpi is not None else 0.0807
    _metric_card("Default Rate", f"{default_rate:.1%}", "class imbalance")
with c4:
    base_auc = metrics_base["auc_roc"].iloc[0] if metrics_base is not None else 0.628
    _metric_card("Baseline AUC", f"{base_auc:.3f}", "LightGBM, no weighting")
with c5:
    if metrics_champ is not None:
        champ_auc = metrics_champ["auc_roc"].iloc[0]
        delta = champ_auc - base_auc
        _metric_card("Champion AUC", f"{champ_auc:.3f}", f"+{delta:.3f} vs baseline", "normal")
    elif metrics_base is not None:
        _metric_card("Champion AUC", f"{base_auc:.3f}", "class_weight=balanced")
    else:
        _metric_card("Champion AUC", "—", "")

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

    # ── metrics comparison table ──────────────────────────────────────────────
    if metrics_cmp is not None:
        cmp_display = metrics_cmp.copy()
        for col in ["auc_roc", "average_precision", "brier_score", "log_loss"]:
            if col in cmp_display.columns:
                cmp_display[col] = cmp_display[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(
            cmp_display,
            use_container_width=True,
            hide_index=True,
        )
    elif metrics_base is not None:
        st.dataframe(metrics_base, use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)

    # ── AUC comparison bar ────────────────────────────────────────────────────
    with col_l:
        st.markdown("#### AUC-ROC Comparison")
        models, aucs, colors = [], [], []
        if metrics_base is not None:
            models.append("Baseline")
            aucs.append(metrics_base["auc_roc"].iloc[0])
            colors.append(C_SLATE)
        if metrics_champ is not None:
            models.append("Champion")
            aucs.append(metrics_champ["auc_roc"].iloc[0])
            colors.append(C_BLUE)

        if models:
            fig = go.Figure(go.Bar(
                x=models, y=aucs,
                marker_color=colors,
                text=[f"{v:.4f}" for v in aucs],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis=dict(range=[0.5, max(aucs) + 0.05], title="AUC-ROC"),
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── confusion matrix ──────────────────────────────────────────────────────
    with col_r:
        st.markdown("#### Confusion Matrix")
        if conf_mat is not None:
            row = conf_mat.iloc[0]
            cm_data = [
                [int(row["true_negative"]), int(row["false_positive"])],
                [int(row["false_negative"]), int(row["true_positive"])],
            ]
            fig = go.Figure(go.Heatmap(
                z=cm_data,
                x=["Predicted: Good", "Predicted: Default"],
                y=["Actual: Good", "Actual: Default"],
                colorscale=[[0, "#F0F9FF"], [1, C_BLUE]],
                text=[[f"{v:,}" for v in row_] for row_ in cm_data],
                texttemplate="%{text}",
                showscale=False,
            ))
            prec = row.get("precision", 0)
            rec  = row.get("recall", 0)
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                annotations=[dict(
                    text=f"Precision: {prec:.3f} | Recall: {rec:.3f}",
                    x=0.5, y=-0.15, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=12, color=C_SLATE),
                )],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `train-simulate` to generate the confusion matrix.")

    # ── calibration curve ─────────────────────────────────────────────────────
    if calib_df is not None:
        st.markdown("#### Probability Calibration")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=calib_df["predicted_midpoint"],
            y=calib_df["actual_default_rate"],
            mode="lines+markers",
            name="Model calibration",
            line=dict(color=C_BLUE, width=2),
            marker=dict(size=6),
        ))
        max_val = max(calib_df["predicted_midpoint"].max(), calib_df["actual_default_rate"].max()) + 0.02
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect calibration",
            line=dict(color=C_SLATE, dash="dash", width=1),
        ))
        fig.update_layout(
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Actual Default Rate",
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Feature Importance — SHAP TreeExplainer")

    if shap_df is not None and len(shap_df) > 0:
        top_n = st.slider("Top N features", min_value=10, max_value=min(40, len(shap_df)), value=20, step=5)
        top = shap_df.head(top_n).copy()
        top["feature_short"] = top["feature"].str[:45]

        fig = go.Figure(go.Bar(
            x=top["mean_abs_shap"],
            y=top["feature_short"],
            orientation="h",
            marker=dict(
                color=top["mean_abs_shap"],
                colorscale=[[0, "#DBEAFE"], [1, C_BLUE]],
                showscale=False,
            ),
        ))
        fig.update_layout(
            xaxis_title="Mean |SHAP value|",
            yaxis=dict(autorange="reversed"),
            height=60 + top_n * 22,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "SHAP values computed via `shap.TreeExplainer` on a 5,000-row sample from the test set. "
            "Higher |SHAP| = stronger impact on model output."
        )

        with st.expander("Full SHAP table"):
            st.dataframe(
                shap_df[["rank", "feature", "mean_abs_shap"]].assign(
                    mean_abs_shap=shap_df["mean_abs_shap"].apply(lambda x: f"{x:.6f}")
                ),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("SHAP importance not yet generated. Run `python -m credit_risk.cli train-simulate` to compute it.")

    # ── missingness ───────────────────────────────────────────────────────────
    if feat_summary is not None and "missing_rate" in feat_summary.columns:
        st.markdown("### Data Quality — Feature Missingness")
        feat_miss = feat_summary[feat_summary["missing_rate"] > 0].sort_values("missing_rate", ascending=False).head(25)
        if len(feat_miss) > 0:
            fig = go.Figure(go.Bar(
                x=feat_miss["missing_rate"] * 100,
                y=feat_miss["feature"].str[:40],
                orientation="h",
                marker_color=C_AMBER,
            ))
            fig.update_layout(
                xaxis_title="Missing Rate (%)",
                yaxis=dict(autorange="reversed"),
                height=500, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

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
        threshold_vals = decision_sim["threshold"].tolist()
        selected_thresh = st.select_slider(
            "Decision Threshold",
            options=[round(t, 2) for t in threshold_vals],
            value=0.2,
        )
        row = decision_sim[decision_sim["threshold"].round(2) == round(selected_thresh, 2)].iloc[0]

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Approval Rate", f"{row['approval_rate']:.1%}")
        with mc2:
            st.metric("Default Rate (approved)", f"{row['default_rate_among_approved']:.2%}")
        with mc3:
            st.metric("Expected Profit", f"${row['expected_profit']:,.0f}")
        with mc4:
            net = row["expected_value"]
            st.metric("Net Value", f"${net:,.0f}", delta=f"{'▲' if net > 0 else '▼'} {abs(net):,.0f}")

        st.markdown("---")

        # Multi-trace chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=decision_sim["threshold"], y=decision_sim["approval_rate"],
            name="Approval Rate", line=dict(color=C_BLUE, width=2),
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=decision_sim["threshold"], y=decision_sim["default_rate_among_approved"],
            name="Default Rate (approved)", line=dict(color=C_RED, width=2, dash="dash"),
            yaxis="y1",
        ))
        fig.add_trace(go.Bar(
            x=decision_sim["threshold"],
            y=decision_sim["expected_value"],
            name="Net Value ($)",
            marker_color=[C_GREEN if v >= 0 else C_RED for v in decision_sim["expected_value"]],
            opacity=0.5,
            yaxis="y2",
        ))
        fig.add_vline(x=selected_thresh, line_dash="dot", line_color=C_INDIGO, annotation_text=f"threshold={selected_thresh}")

        # Find optimal threshold
        opt_idx = decision_sim["expected_value"].idxmax()
        opt_thresh = decision_sim.loc[opt_idx, "threshold"]
        fig.add_vline(x=opt_thresh, line_dash="dash", line_color=C_GREEN,
                      annotation_text=f"optimal={opt_thresh}", annotation_position="top right")

        fig.update_layout(
            xaxis_title="Threshold",
            yaxis=dict(title="Rate", tickformat=".0%", side="left"),
            yaxis2=dict(title="Net Value ($)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            height=420, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Optimal threshold (max net value): **{opt_thresh}** at net value **${decision_sim.loc[opt_idx, 'expected_value']:,.0f}**")
    else:
        st.info("Run `python -m credit_risk.cli train-simulate` to generate the decision simulation table.")

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
        cols_sc = st.columns(len(scenarios))
        for i, (_, sc) in enumerate(scenarios.iterrows()):
            with cols_sc[i]:
                name = sc["scenario_name"]
                color = {"Base": C_BLUE, "Conservative": C_INDIGO, "Aggressive": C_GREEN}.get(name, C_SLATE)
                st.markdown(
                    f"<div style='border:2px solid {color};border-radius:10px;padding:16px;'>"
                    f"<h4 style='color:{color};margin-top:0'>{name}</h4>"
                    f"<p><b>Threshold:</b> {sc['threshold']}</p>"
                    f"<p><b>Approval Rate:</b> {sc['approval_rate']:.1%}</p>"
                    f"<p><b>Default Rate:</b> {sc['default_rate']:.2%}</p>"
                    f"<p><b>Expected Profit:</b> ${sc['expected_profit']:,.0f}</p>"
                    f"<p><b>Expected Loss:</b> ${sc['expected_loss']:,.0f}</p>"
                    f"<p style='font-size:1.2em;font-weight:bold;color:{color};'>"
                    f"Net Value: ${sc['net_value']:,.0f}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        fig = go.Figure()
        scenario_colors = [C_BLUE, C_INDIGO, C_GREEN][:len(scenarios)]
        fig.add_trace(go.Bar(
            name="Expected Profit",
            x=scenarios["scenario_name"],
            y=scenarios["expected_profit"],
            marker_color=C_GREEN, opacity=0.8,
        ))
        fig.add_trace(go.Bar(
            name="Expected Loss",
            x=scenarios["scenario_name"],
            y=-scenarios["expected_loss"],
            marker_color=C_RED, opacity=0.8,
        ))
        fig.add_trace(go.Scatter(
            name="Net Value",
            x=scenarios["scenario_name"],
            y=scenarios["net_value"],
            mode="markers+text",
            marker=dict(size=14, color=C_INDIGO, symbol="diamond"),
            text=[f"${v:,.0f}" for v in scenarios["net_value"]],
            textposition="top center",
        ))
        fig.update_layout(
            barmode="relative",
            yaxis_title="Value ($)",
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python -m credit_risk.cli simulate-scenarios` to generate scenario results.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — POPULATION SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Default Rate by Population Segment")

    if segments is not None:
        segment_types = segments["segment_name"].unique().tolist()
        selected_seg = st.selectbox("Segment", segment_types)
        seg_data = segments[segments["segment_name"] == selected_seg].copy()

        fig = go.Figure(go.Bar(
            x=seg_data["segment_value"].astype(str),
            y=seg_data["default_rate"],
            text=[f"{v:.1%}" for v in seg_data["default_rate"]],
            textposition="outside",
            marker=dict(
                color=seg_data["default_rate"],
                colorscale=[[0, "#DCFCE7"], [1, C_RED]],
                showscale=False,
            ),
        ))
        population_line = seg_data["n"].sum()
        overall_default = (seg_data["default_rate"] * seg_data["n"]).sum() / seg_data["n"].sum()
        fig.add_hline(
            y=overall_default,
            line_dash="dash", line_color=C_SLATE,
            annotation_text=f"Overall: {overall_default:.2%}",
        )
        fig.update_layout(
            xaxis_title=selected_seg.replace("_", " ").title(),
            yaxis=dict(title="Default Rate", tickformat=".1%"),
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            riskiest = seg_data.loc[seg_data["default_rate"].idxmax()]
            st.metric(
                "Highest Risk Segment",
                str(riskiest["segment_value"]),
                f"{riskiest['default_rate']:.2%} default rate",
                delta_color="inverse",
            )
        with col_b:
            safest = seg_data.loc[seg_data["default_rate"].idxmin()]
            st.metric(
                "Lowest Risk Segment",
                str(safest["segment_value"]),
                f"{safest['default_rate']:.2%} default rate",
            )
    else:
        st.info("Run `python -m credit_risk.cli make-dashboard-tables` to generate population segments.")

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:0.85em;'>"
    "Built by <b>Oussama Skia</b> · "
    "<a href='https://github.com/skayy47/credit-risk-prediction' style='color:#1E40AF;'>GitHub</a> · "
    "<a href='https://www.kaggle.com/c/home-credit-default-risk' style='color:#1E40AF;'>Home Credit Data</a> · "
    "LightGBM + SHAP"
    "</div>",
    unsafe_allow_html=True,
)
