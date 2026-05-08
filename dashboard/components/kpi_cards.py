import numpy as np
import pandas as pd
import streamlit as st


def render_kpi_cards(
    stock_ledger: pd.DataFrame,
    forecast_output: pd.DataFrame,
    target_population: pd.DataFrame,
    facilities: pd.DataFrame,
    forecast_horizon: int = 1,
):
    """Render all 5 KPI cards. forecast_horizon (1..8) selects which forecast week to display."""

    # Slider semantics: horizon=1 → first forecast week; horizon=8 → last.
    if forecast_output.empty:
        latest_week = 364
    else:
        first_week = int(forecast_output["forecast_week"].min())
        latest_week = first_week + (forecast_horizon - 1)
    ens_latest = forecast_output[
        (forecast_output["model"] == "ensemble") &
        (forecast_output["forecast_week"] == latest_week)
    ]
    ens_prev_week = forecast_output[
        (forecast_output["model"] == "ensemble") &
        (forecast_output["forecast_week"] == latest_week - 1)
    ]

    # ── KPI 1: Stockout alerts ──────────────────────────────────────────────
    critical_now = int((ens_latest["alert_status"] == "critical").sum())
    critical_prev = int((ens_prev_week["alert_status"] == "critical").sum())
    delta_critical = critical_now - critical_prev

    # ── KPI 2: DTP dropout rate ─────────────────────────────────────────────
    last_52_start = max(0, int(stock_ledger["week"].max()) - 51)
    penta_recent = stock_ledger[
        (stock_ledger["antigen"] == "PENTA") &
        (stock_ledger["week"] >= last_52_start)
    ]
    actual_penta = penta_recent["doses_administered"].sum()
    penta_tp = target_population[target_population["antigen"] == "PENTA"]
    total_target_infants = penta_tp["target_infants"].sum()
    doses_in_series = 3
    if total_target_infants > 0:
        completion_rate = actual_penta / (total_target_infants * doses_in_series)
        dropout_rate = max(0.0, 1.0 - completion_rate)
    else:
        dropout_rate = float("nan")

    # ── KPI 3: Children at risk this week ───────────────────────────────────
    latest_ledger_week = int(stock_ledger["week"].max())
    children_at_risk = int(
        stock_ledger[stock_ledger["week"] == latest_ledger_week]["children_missed"].sum()
    )

    # ── KPI 4: Wastage rate (last 12 weeks) ─────────────────────────────────
    from utils.db import get_connection
    with get_connection() as conn:
        session_df = pd.read_sql("SELECT * FROM session_log", conn)
    recent_start = max(0, int(session_df["week"].max()) - 11)
    recent_sessions = session_df[session_df["week"] >= recent_start]
    total_admin = recent_sessions["doses_administered"].sum()
    total_wasted = recent_sessions["doses_wasted"].sum()
    wastage_rate = total_wasted / max(total_admin + total_wasted, 1)

    # ── KPI 5: Resupply urgency score ────────────────────────────────────────
    at_risk = ens_latest[ens_latest["alert_status"].isin(["critical", "warning"])].copy()
    if not at_risk.empty:
        merged = at_risk.merge(
            facilities[["facility_id", "lead_time_days_mean", "target_infants_annual"]],
            on="facility_id", how="left",
        )
        dts_safe = merged["predicted_days_to_stockout"].clip(lower=1)
        lt = merged["lead_time_days_mean"].fillna(14.0)
        ti = merged["target_infants_annual"].fillna(5000)
        raw_scores = (1.0 / dts_safe) * (lt / 7.0) * (ti / 100.0)
        weekly_sum = float(raw_scores.sum())

        # Compute historical weekly sums for normalisation
        all_ens = forecast_output[forecast_output["model"] == "ensemble"].copy()
        all_at_risk = all_ens[all_ens["alert_status"].isin(["critical", "warning"])].copy()
        if not all_at_risk.empty:
            grp = all_at_risk.merge(
                facilities[["facility_id", "lead_time_days_mean", "target_infants_annual"]],
                on="facility_id", how="left"
            )
            dts2 = grp["predicted_days_to_stockout"].clip(lower=1)
            lt2 = grp["lead_time_days_mean"].fillna(14.0)
            ti2 = grp["target_infants_annual"].fillna(5000)
            raw2 = (1.0 / dts2) * (lt2 / 7.0) * (ti2 / 100.0)
            hist_weekly = raw2.groupby(all_at_risk["forecast_week"].values).sum()
            p95 = float(np.percentile(hist_weekly.values, 95)) if len(hist_weekly) > 0 else max(weekly_sum, 1)
        else:
            p95 = max(weekly_sum, 1)

        urgency_score = min(100.0, (weekly_sum / max(p95, 1e-9)) * 100.0)
    else:
        urgency_score = 0.0

    # ── Render ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="🚨 Critical Stockout Alerts",
            value=critical_now,
            delta=f"{delta_critical:+d} vs prev week",
            delta_color="inverse",
        )

    with col2:
        if np.isnan(dropout_rate):
            dtp_display = "N/A"
            dtp_delta = None
        else:
            dtp_display = f"{dropout_rate * 100:.1f}%"
            dtp_delta = "WHO threshold: 10%" if dropout_rate > 0.10 else "Within WHO threshold"
        st.metric(
            label="💉 DTP3 Dropout Rate",
            value=dtp_display,
            delta=dtp_delta,
            delta_color="off",
        )
        if not np.isnan(dropout_rate):
            color = "🔴" if dropout_rate > 0.10 else ("🟡" if dropout_rate > 0.05 else "🟢")
            st.caption(f"{color} WHO acceptable < 10%")

    with col3:
        st.metric(
            label="👶 Children Missed This Week",
            value=f"{children_at_risk:,}",
        )

    with col4:
        wrate_pct = wastage_rate * 100
        color = "🔴" if wrate_pct > 10 else ("🟡" if wrate_pct > 7 else "🟢")
        st.metric(
            label="🗑️ Vaccine Wastage Rate",
            value=f"{wrate_pct:.1f}%",
            delta="WHO benchmark: 10%",
            delta_color="off",
        )
        st.caption(f"{color} Last 12 weeks")

    with col5:
        st.metric(
            label="⚡ Resupply Urgency Score",
            value=f"{urgency_score:.0f} / 100",
            delta="Higher = more urgent",
            delta_color="off",
        )
