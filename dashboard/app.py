"""
VaxAlert Dashboard
Run: streamlit run dashboard/app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# Force light theme for all Plotly charts in the app
pio.templates.default = "plotly_white"

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import (
    get_connection, get_facilities, get_vaccines, get_clusters,
    get_shocks_for_facility, get_delivery_log, get_session_log,
    get_target_population,
)
from utils.features import display_dates, display_date

st.set_page_config(
    page_title="VaxAlert - Ethiopia EPI",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Design System (CSS) ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    [data-testid="stAppViewContainer"] {
        background-color: #f9fafb;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    /* KPI Card styling override */
    [data-testid="stMetric"] {
        background-color: white !important;
        padding: 1.25rem !important;
        border-radius: 16px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
        border: 1px solid #f3f4f6 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.875rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #4b5563 !important;
        font-size: 0.875rem !important;
    }

    /* Sidebar clean-up */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e5e7eb;
    }
    
    .st-emotion-cache-16idsys p {
        font-weight: 600;
        color: #374151;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #f3f4f6;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* Chart containers */
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #f3f4f6;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_facilities():
    return get_facilities()

@st.cache_data(ttl=300)
def load_vaccines():
    return get_vaccines()

@st.cache_data(ttl=300)
def load_clusters():
    return get_clusters()

@st.cache_data(ttl=300)
def load_stock_ledger():
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM stock_ledger ORDER BY facility_id, antigen, week", conn)

@st.cache_data(ttl=300)
def load_target_population():
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM target_population", conn)

@st.cache_data(ttl=300)
def load_forecast_output():
    with get_connection() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "forecast_output" not in tables:
            return pd.DataFrame()
        return pd.read_sql("SELECT * FROM forecast_output", conn)

@st.cache_data(ttl=300)
def load_model_metrics():
    with get_connection() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "model_metrics" not in tables:
            return pd.DataFrame()
        return pd.read_sql("SELECT * FROM model_metrics", conn)

@st.cache_data(ttl=300)
def load_feature_importance():
    with get_connection() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "feature_importance" not in tables:
            return pd.DataFrame()
        return pd.read_sql("SELECT * FROM feature_importance", conn)


@st.cache_data(ttl=300)
def load_shocks(facility_id):
    return get_shocks_for_facility(facility_id)

@st.cache_data(ttl=300)
def load_delivery(facility_id, antigen):
    return get_delivery_log(facility_id, antigen)

@st.cache_data(ttl=300)
def load_session(facility_id, antigen):
    return get_session_log(facility_id, antigen)

# ── Load all data at startup ─────────────────────────────────────────────────

facilities = load_facilities()
vaccines = load_vaccines()
clusters = load_clusters()
stock_ledger = load_stock_ledger()
target_population = load_target_population()
forecast_output = load_forecast_output()
model_metrics = load_model_metrics()
feature_importance = load_feature_importance()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 1.5rem 0.5rem; background: #f9fafb; border-radius: 16px; margin-bottom: 2rem; border: 1px solid #e5e7eb;">
            <img src="https://raw.githubusercontent.com/stevenrskelton/flag-icon/master/png/75/country-4x3/et.png" width="45" style="margin-bottom: 1rem; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="margin:0; color:#1e3a8a; font-size: 1.5rem; font-weight: 700;">VaxAlert</h2>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.25rem; font-weight: 500;">Ethiopia EPI System</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    view = st.radio(
        "Navigation",
        ["🌍 National Overview", "🏥 Facility Drill-Down",
         "🔗 Cascade View", "📊 Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()

    # Global filters
    st.subheader("Filters")
    all_antigens = sorted(vaccines["antigen_code"].tolist())
    selected_antigens = st.multiselect("Antigen", all_antigens, default=all_antigens)

    all_tiers = sorted(facilities["access_tier"].unique().tolist())
    selected_tiers = st.multiselect("Access Tier", all_tiers, default=all_tiers)

    selected_alert = st.multiselect(
        "Alert Status", ["critical", "warning", "ok"],
        default=["critical", "warning", "ok"]
    )

    forecast_horizon = st.slider("Forecast Horizon (weeks)", 1, 8, 8)

    st.divider()
    st.caption("Data: synthetic · 30 facilities · 7 antigens · 7 years (364 weeks)")

# Filter forecast_output based on sidebar selections
def filter_forecast(fo):
    """Filter forecast_output by sidebar selections (antigen, tier, alert).
    Horizon-week selection is handled inside each component, not here, so the
    full 8-week forecast remains available for downstream lookups."""
    if fo.empty:
        return fo
    mask = (
        fo["antigen"].isin(selected_antigens) &
        fo["alert_status"].isin(selected_alert)
    )
    fids_in_tier = facilities[facilities["access_tier"].isin(selected_tiers)]["facility_id"]
    mask &= fo["facility_id"].isin(fids_in_tier)
    return fo[mask]

fo_filtered = filter_forecast(forecast_output)
fac_filtered = facilities[facilities["access_tier"].isin(selected_tiers)]

# ── Helpers ──────────────────────────────────────────────────────────────────

def no_forecast_warning():
    st.warning(
        "No forecast data found. Run the pipeline first:\n\n"
        "```\npython evaluation/walk_forward_cv.py --sample 10\n"
        "python forecast/generate_forecasts.py\n```"
    )

# ════════════════════════════════════════════════════════════════════════════
# VIEW 1: National Overview
# ════════════════════════════════════════════════════════════════════════════

if view == "🌍 National Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🌍 National Overview</h1>
        <p>Real-time vaccine stockout risk and predictive analytics across Ethiopia's EPI network.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Real-time vaccine stockout risk across Ethiopia's EPI network")

    if forecast_output.empty:
        no_forecast_warning()
    else:
        from dashboard.components.kpi_cards import render_kpi_cards
        from dashboard.components.facility_map import render_facility_map
        from dashboard.components.alert_table import render_alert_table

        render_kpi_cards(stock_ledger, fo_filtered, target_population, fac_filtered,
                         forecast_horizon=forecast_horizon)

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Facility Alert Map")
        # Show the user which forecast week the map represents
        if not forecast_output.empty:
            first_wk = int(forecast_output["forecast_week"].min())
            target_wk = first_wk + (forecast_horizon - 1)
            target_row = forecast_output[forecast_output["forecast_week"] == target_wk]
            if not target_row.empty:
                target_date_str = display_date(target_row["forecast_date"].iloc[0]).strftime("%Y-%m-%d")
                st.caption(
                    f"📅 Showing forecast for **week {target_wk}** ({target_date_str}). "
                    f"Adjust the **Forecast Horizon** slider in the sidebar (1 = next week, 8 = 8 weeks ahead)."
                )

        render_facility_map(fac_filtered, fo_filtered, forecast_horizon=forecast_horizon)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Active Alerts & Risk Summary")
        render_alert_table(fo_filtered, fac_filtered, clusters,
                           forecast_horizon=forecast_horizon)
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# VIEW 2: Facility Drill-Down
# ════════════════════════════════════════════════════════════════════════════

elif view == "🏥 Facility Drill-Down":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Facility Drill-Down</h1>
        <p>Deep-dive into facility-level inventory trends, AI explainability, and historical shocks.</p>
    </div>
    """, unsafe_allow_html=True)

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        fac_options = facilities[["facility_id", "name", "region"]].copy()
        fac_options["label"] = fac_options["name"] + " (" + fac_options["region"] + ")"
        fac_labels = fac_options["label"].tolist()
        fac_ids = fac_options["facility_id"].tolist()
        sel_idx = st.selectbox("Select Facility", range(len(fac_labels)),
                               format_func=lambda i: fac_labels[i])
        sel_fid = fac_ids[sel_idx]

    with col_sel2:
        sel_ant = st.radio("Antigen", all_antigens, horizontal=True)

    sel_fac = facilities[facilities["facility_id"] == sel_fid].iloc[0]
    st.caption(f"**{sel_fac['name']}** · {sel_fac['type']} · {sel_fac['access_tier']} · {sel_fac['region']}")

    series = stock_ledger[
        (stock_ledger["facility_id"] == sel_fid) &
        (stock_ledger["antigen"] == sel_ant)
    ].reset_index(drop=True)

    fcast = forecast_output[
        (forecast_output["facility_id"] == sel_fid) &
        (forecast_output["antigen"] == sel_ant)
    ] if not forecast_output.empty else pd.DataFrame()

    shocks = load_shocks(sel_fid)
    tp = get_target_population(sel_fid, sel_ant)
    weekly_consumption = float(tp.get("weekly_consumption_baseline", 1.0)) if tp else 1.0
    lead_time = float(sel_fac["lead_time_days_mean"])
    reorder_point = lead_time * weekly_consumption / 7.0

    # Ensemble weights for this series
    ens_weights = {"w_sarimax": 0.5, "w_prophet": 0.5}
    val_mae = 0.0
    if not model_metrics.empty:
        ens_row = model_metrics[
            (model_metrics["facility_id"] == sel_fid) &
            (model_metrics["antigen"] == sel_ant) &
            (model_metrics["model"] == "ensemble") &
            (model_metrics["fold"] == "final")
        ]
        if not ens_row.empty:
            ens_weights["w_sarimax"] = float(ens_row.iloc[0]["w_sarimax"] or 0.5)
            ens_weights["w_prophet"] = float(ens_row.iloc[0]["w_prophet"] or 0.5)
            val_mae = float(ens_row.iloc[0]["mae"] or 0.0)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    if not series.empty:
        from dashboard.components.stock_chart import render_stock_chart
        render_stock_chart(
            series=series,
            forecast=fcast,
            facility_name=sel_fac["name"],
            antigen=sel_ant,
            ensemble_weights=ens_weights,
            val_mae=val_mae,
            shocks=shocks,
            reorder_point=reorder_point,
        )
    else:
        st.warning("No stock data for this facility/antigen combination.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-facility feature importance ──────────────────────────────────────
    if not feature_importance.empty:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        from dashboard.components.feature_importance_panel import render_feature_importance_panel
        render_feature_importance_panel(sel_fid, sel_fac["name"], feature_importance)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Delivery timeline ────────────────────────────────────────────────────
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Resupply Delivery History")
    deliveries = load_delivery(sel_fid, sel_ant)
    if not deliveries.empty:
        deliveries = deliveries.copy()
        deliveries["x"] = display_dates(deliveries["week_date"])
        fig_del = go.Figure()
        fig_del.add_trace(go.Bar(
            x=deliveries["x"],
            y=deliveries["quantity_ordered"],
            name="Ordered",
            marker_color="#95a5a6",
            opacity=0.6,
        ))
        fig_del.add_trace(go.Bar(
            x=deliveries["x"],
            y=deliveries["quantity_received"],
            name="Received",
            marker_color=deliveries["emergency_order"].map({1: "#ef4444", 0: "#3b82f6"}),
        ))
        fig_del.update_layout(
            barmode="overlay",
            height=280,
            title="Doses Ordered vs Received (red bars = emergency orders)",
            xaxis_title="Date", yaxis_title="Doses",
            margin=dict(l=50, r=20, t=50, b=40),
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_del, use_container_width=True)
    else:
        st.info("No delivery records for this selection.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Session performance ──────────────────────────────────────────────────
    st.subheader("Session Performance")
    sessions = load_session(sel_fid, sel_ant)
    if not sessions.empty:
        sessions = sessions.copy()
        sessions["x"] = display_dates(sessions["week_date"])
        fig_sess = go.Figure()
        fig_sess.add_trace(go.Scatter(
            x=sessions["x"], y=sessions["children_reached"],
            mode="lines", name="Children Reached",
            line=dict(color="#27ae60", width=2),
            fill="tozeroy", fillcolor="rgba(39,174,96,0.1)",
        ))
        fig_sess.add_trace(go.Scatter(
            x=sessions["x"], y=sessions["children_missed"],
            mode="lines", name="Children Missed",
            line=dict(color="#e74c3c", width=2),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
        ))
        fig_sess.update_layout(
            height=260,
            title="Weekly Children Reached vs Missed",
            xaxis_title="Date", yaxis_title="Children",
            margin=dict(l=50, r=20, t=50, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_sess, use_container_width=True)
    else:
        st.info("No session records for this selection.")

# ════════════════════════════════════════════════════════════════════════════
# VIEW 3: Cascade View
# ════════════════════════════════════════════════════════════════════════════

elif view == "🔗 Cascade View":
    st.markdown("""
    <div class="main-header">
        <h1>🔗 Cascade Network View</h1>
        <p>Visualize supply chain dependencies and simulate the impact of early resupply interventions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Stockout propagation from Health Centers to satellite Health Posts")

    hc_facilities = facilities[facilities["type"] == "Health Center"]
    hc_options = hc_facilities[["facility_id", "name"]].copy()
    hc_labels = hc_options["name"].tolist()
    hc_ids = hc_options["facility_id"].tolist()

    if not hc_ids:
        st.warning("No Health Centers found in the database.")
    else:
        sel_hc_idx = st.selectbox("Select Health Center", range(len(hc_labels)),
                                   format_func=lambda i: hc_labels[i])
        sel_hc = hc_ids[sel_hc_idx]

        from dashboard.components.cascade_view import render_cascade_view
        render_cascade_view(
            hc_id=sel_hc,
            clusters=clusters,
            facilities=facilities,
            forecast_output=fo_filtered if not fo_filtered.empty else forecast_output,
            stock_ledger=stock_ledger,
        )

# ════════════════════════════════════════════════════════════════════════════
# VIEW 4: Model Performance
# ════════════════════════════════════════════════════════════════════════════

elif view == "📊 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Performance</h1>
        <p>Transparency into the accuracy, reliability, and error metrics of our ensemble forecasting engine.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_metrics.empty:
        st.warning("No model metrics found. Run walk_forward_cv.py first.")
    else:
        MODEL_DISPLAY = {
            "naive_last_value": "Naive (Last Value)",
            "naive_seasonal_naive": "Naive (Seasonal)",
            "holt_winters": "Holt-Winters",
            "xgboost": "XGBoost",
            "prophet": "Prophet",
            "ensemble": "Ensemble",
        }

        # ── Per-model summary table ──────────────────────────────────────────
        st.subheader("Model Summary - Final Test (weeks 340-363)")
        final_df = model_metrics[model_metrics["fold"] == "final"].copy()
        if not final_df.empty:
            # Removed RMSE column - MAPE is more interpretable for non-technical readers
            summary_cols = ["mae", "mape", "interval_coverage",
                            "stockout_detection_rate", "false_alert_rate"]
            summary = final_df.groupby("model")[summary_cols].mean().round(3)
            summary.index = summary.index.map(lambda x: MODEL_DISPLAY.get(x, x))
            summary.columns = ["MAE (doses)", "MAPE (%)",
                                "Interval Cov. (%)", "SDR (%)", "False Alert (%)"]
            summary["Interval Cov. (%)"] = (summary["Interval Cov. (%)"] * 100).round(1)
            summary["SDR (%)"] = (summary["SDR (%)"] * 100).round(1)
            summary["False Alert (%)"] = (summary["False Alert (%)"] * 100).round(1)
            st.dataframe(summary, use_container_width=True)
            
            with st.expander("📖 Metric Definitions & How to Interpret This Table"):
                st.markdown("""
                - **MAE (Mean Absolute Error)**: The average 'miss' in doses. For example, an MAE of 10.5 means the AI's prediction is typically off by ~10 doses. **Lower is better.**
                - **MAPE (Mean Absolute Percentage Error)**: The average error as a percentage of the actual stock. This helps compare accuracy across facilities with very different volumes. **Lower is better.**
                - **Interval Cov. (Interval Coverage)**: Measures the reliability of the 'shaded range' you see in charts. We aim for 80%—if it's 80%, the actual stock stays inside our predicted range 8 out of 10 times.
                - **SDR (Stockout Detection Rate)**: **The most important operational metric.** It shows the % of actual stockouts that the AI correctly 'saw coming' at least 1 week in advance. **Higher is better.**
                - **False Alert (%)**: The % of time the AI fires a 'Critical' alert but the facility does *not* actually run out. We keep this low to prevent 'alert fatigue' for supply chain officers. **Lower is better.**
                """)

        cv_df = model_metrics[model_metrics["fold"].isin(["1", "2", "3"])].copy()
        if not cv_df.empty:
            st.subheader("CV Summary - Mean MAE across Folds 1-3")
            cv_summary = cv_df.groupby("model")["mae"].agg(["mean", "std"]).round(2)
            cv_summary.index = cv_summary.index.map(lambda x: MODEL_DISPLAY.get(x, x))
            cv_summary.columns = ["Mean MAE", "Std MAE"]
            st.dataframe(cv_summary, use_container_width=True)

        # ── MAE by access tier ───────────────────────────────────────────────
        st.subheader("MAE by Model × Access Tier (Final Test)")
        if not final_df.empty:
            tier_merged = final_df.merge(
                facilities[["facility_id", "access_tier"]], on="facility_id", how="left"
            )
            tier_mae = (
                tier_merged.groupby(["model", "access_tier"])["mae"]
                .mean().reset_index()
            )
            tier_mae["model_label"] = tier_mae["model"].map(lambda x: MODEL_DISPLAY.get(x, x))

            tiers = sorted(tier_mae["access_tier"].unique())
            models_in_data = tier_mae["model_label"].unique()
            colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12", "#9b59b6", "#1abc9c"]

            fig_tier = go.Figure()
            for i, m in enumerate(models_in_data):
                sub = tier_mae[tier_mae["model_label"] == m]
                fig_tier.add_trace(go.Bar(
                    name=m,
                    x=sub["access_tier"],
                    y=sub["mae"].round(2),
                    marker_color=colors[i % len(colors)],
                ))
            fig_tier.update_layout(
                barmode="group",
                height=360,
                xaxis_title="Access Tier",
                yaxis_title="Mean MAE (doses/week)",
                legend=dict(orientation="h", y=1.05),
                margin=dict(l=50, r=20, t=60, b=50),
            )
            st.plotly_chart(fig_tier, use_container_width=True)

        # ── Stockout detection breakdown ─────────────────────────────────────
        st.subheader("Stockout Detection Rate (Final Test) - Which model saves the most children")
        if not final_df.empty:
            sdr_data = final_df.groupby("model").agg(
                detected=("stockout_detection_rate", "mean"),
                false_alert=("false_alert_rate", "mean"),
                n_stockouts=("n_stockout_events", "sum"),
            ).reset_index()
            sdr_data["missed"] = 1.0 - sdr_data["detected"]
            sdr_data["model_label"] = sdr_data["model"].map(lambda x: MODEL_DISPLAY.get(x, x))

            fig_sdr = go.Figure()
            fig_sdr.add_trace(go.Bar(
                name="Detected",
                x=sdr_data["model_label"],
                y=(sdr_data["detected"] * 100).round(1),
                marker_color="#27ae60",
            ))
            fig_sdr.add_trace(go.Bar(
                name="Missed",
                x=sdr_data["model_label"],
                y=(sdr_data["missed"] * 100).round(1),
                marker_color="#e74c3c",
            ))
            fig_sdr.add_trace(go.Bar(
                name="False Alerts (%)",
                x=sdr_data["model_label"],
                y=(sdr_data["false_alert"] * 100).round(1),
                marker_color="#f39c12",
            ))
            fig_sdr.update_layout(
                barmode="group",
                height=360,
                yaxis_title="Percentage (%)",
                legend=dict(orientation="h", y=1.05),
                margin=dict(l=50, r=20, t=60, b=50),
            )
            st.plotly_chart(fig_sdr, use_container_width=True)

        # ── Tier × Feature Importance Heatmap ─────────────────────────────────
        if not feature_importance.empty:
            st.subheader("Feature Importance × Access Tier (XGBoost)")
            st.caption(
                "Mean XGBoost feature importance per access tier. "
                "Tells you which signals matter most where - pastoral facilities tend to "
                "depend on supply-chain features; urban facilities tend to depend on seasonality."
            )

            from dashboard.components.feature_importance_panel import PRETTY_NAMES
            fi_tier = feature_importance.merge(
                facilities[["facility_id", "access_tier"]], on="facility_id", how="left"
            )
            # Top-10 features overall
            top_feats = (fi_tier.groupby("feature")["importance"].mean()
                         .sort_values(ascending=False).head(10).index.tolist())
            sub = fi_tier[fi_tier["feature"].isin(top_feats)]
            pivot = (sub.groupby(["feature", "access_tier"])["importance"].mean()
                     .reset_index()
                     .pivot(index="feature", columns="access_tier", values="importance")
                     .fillna(0)
                     .reindex(top_feats))
            tier_order = ["urban", "rural_road", "rural_remote", "pastoral"]
            pivot = pivot.reindex(columns=[t for t in tier_order if t in pivot.columns])
            y_labels = [PRETTY_NAMES.get(f, f) for f in pivot.index]

            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=y_labels,
                colorscale="Viridis",
                colorbar=dict(title="Mean<br>importance"),
                hovertemplate="<b>%{y}</b><br>Tier: %{x}<br>Importance: %{z:.4f}<extra></extra>",
            ))
            fig_heat.update_layout(
                height=420,
                margin=dict(l=240, r=40, t=20, b=40),
                xaxis_title="Access tier",
                yaxis_title=None,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(color="#1a202c"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Individual series explorer ───────────────────────────────────────
        st.subheader("Individual Series Explorer")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fac_labels_e = (facilities["name"] + " (" + facilities["region"] + ")").tolist()
            fac_ids_e = facilities["facility_id"].tolist()
            sel_e_idx = st.selectbox("Facility", range(len(fac_labels_e)),
                                      format_func=lambda i: fac_labels_e[i],
                                      key="explorer_fac")
            sel_e_fid = fac_ids_e[sel_e_idx]
        with col_e2:
            sel_e_ant = st.selectbox("Antigen", all_antigens, key="explorer_ant")

        series_e = stock_ledger[
            (stock_ledger["facility_id"] == sel_e_fid) &
            (stock_ledger["antigen"] == sel_e_ant)
        ].reset_index(drop=True)

        fcast_e = forecast_output[
            (forecast_output["facility_id"] == sel_e_fid) &
            (forecast_output["antigen"] == sel_e_ant)
        ] if not forecast_output.empty else pd.DataFrame()

        model_metrics_e = model_metrics[
            (model_metrics["facility_id"] == sel_e_fid) &
            (model_metrics["antigen"] == sel_e_ant) &
            (model_metrics["fold"] == "final")
        ] if not model_metrics.empty else pd.DataFrame()

        fig_exp = go.Figure()
        # Actual stock from the held-out final test window only
        if not series_e.empty:
            from evaluation.walk_forward_cv import FINAL_TEST_START
            test_series = series_e[series_e["week"] >= FINAL_TEST_START].copy()
            test_series["x"] = display_dates(test_series["week_date"])
            fig_exp.add_trace(go.Scatter(
                x=test_series["x"],
                y=test_series["closing_stock"],
                mode="lines",
                name="Actual (test)",
                line=dict(color="#2c3e50", width=3),
            ))

        # All 5 model forecasts (weeks 156-163)
        model_colors = {
            "naive": "#95a5a6",
            "holt_winters": "#3498db",
            "xgboost": "#e74c3c",
            "prophet": "#27ae60",
            "ensemble": "#8e44ad",
        }
        if not fcast_e.empty:
            fcast_e = fcast_e.copy()
            fcast_e["x"] = display_dates(fcast_e["forecast_date"])
            for model_name, color in model_colors.items():
                m_rows = fcast_e[fcast_e["model"] == model_name].sort_values("forecast_date")
                if m_rows.empty:
                    continue
                label = MODEL_DISPLAY.get(model_name, model_name)
                mae_val = None
                if not model_metrics_e.empty:
                    m_metric = model_metrics_e[model_metrics_e["model"] == model_name]
                    if not m_metric.empty:
                        mae_val = m_metric.iloc[0]["mae"]

                trace_name = f"{label}"
                if mae_val is not None:
                    trace_name += f" (MAE={mae_val:.1f})"

                # Ensemble gets PI band
                if model_name == "ensemble":
                    fig_exp.add_trace(go.Scatter(
                        x=list(m_rows["x"]) + list(m_rows["x"])[::-1],
                        y=list(m_rows["yhat_upper"]) + list(m_rows["yhat_lower"])[::-1],
                        fill="toself",
                        fillcolor="rgba(142,68,173,0.1)",
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        name="Ensemble 80% PI",
                        showlegend=True,
                    ))
                    line_width = 3
                else:
                    line_width = 1.5

                fig_exp.add_trace(go.Scatter(
                    x=m_rows["x"],
                    y=m_rows["yhat"],
                    mode="lines",
                    name=trace_name,
                    line=dict(color=color, width=line_width,
                               dash="dash" if model_name != "ensemble" else "solid"),
                ))

        fac_name_e = facilities[facilities["facility_id"] == sel_e_fid]["name"].values[0]
        fig_exp.update_layout(
            title=f"{fac_name_e} - {sel_e_ant} | Test period + 8-week forecast",
            height=420,
            xaxis_title="Date", yaxis_title="Closing Stock (doses)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=50, r=20, t=60, b=80),
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", rangemode="tozero"),
        )
        st.plotly_chart(fig_exp, use_container_width=True)
