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

    # ── KPI 3: Predicted Children at Risk (Future) ──────────────────────────
    # If a facility is predicted to be in 'critical' status, the children at 
    # risk are those served by that facility/antigen pair in that week.
    if ens_latest.empty:
        children_at_risk = 0
    else:
        critical_pairs = ens_latest[ens_latest["alert_status"] == "critical"]
        if critical_pairs.empty:
            children_at_risk = 0
        else:
            # Merge with target_population to get weekly demand
            at_risk_merged = critical_pairs.merge(
                target_population[["facility_id", "antigen", "weekly_consumption_baseline"]],
                on=["facility_id", "antigen"],
                how="left"
            )
            children_at_risk = int(at_risk_merged["weekly_consumption_baseline"].sum())

    # Calculate delta for children at risk
    if forecast_horizon == 1:
        # Compare against the very last week of actuals (week 363)
        last_actual_week = int(stock_ledger["week"].max())
        prev_actuals = stock_ledger[stock_ledger["week"] == last_actual_week]
        
        # Merge with target_population to get baseline demand for those that were critical/warning
        # (Though 'is_stockout' is the actual alert in the ledger)
        prev_critical = prev_actuals[prev_actuals["is_stockout"] == True]
        prev_merged = prev_critical.merge(
            target_population[["facility_id", "antigen", "weekly_consumption_baseline"]],
            on=["facility_id", "antigen"],
            how="left"
        )
        prev_risk = int(prev_merged["weekly_consumption_baseline"].sum())
        
        # Also compute prev_critical_count for KPI 1
        critical_prev = int(prev_actuals["is_stockout"].sum())
    else:
        # Standard forecast-to-forecast comparison
        if ens_prev_week.empty:
            prev_risk = 0
            critical_prev = 0
        else:
            prev_critical = ens_prev_week[ens_prev_week["alert_status"] == "critical"]
            prev_merged = prev_critical.merge(
                target_population[["facility_id", "antigen", "weekly_consumption_baseline"]],
                on=["facility_id", "antigen"],
                how="left"
            )
            prev_risk = int(prev_merged["weekly_consumption_baseline"].sum())
            critical_prev = int((ens_prev_week["alert_status"] == "critical").sum())
    
    delta_risk = children_at_risk - prev_risk
    delta_critical = critical_now - critical_prev

    # ── KPI 4: Wastage rate (last 12 weeks) ─────────────────────────────────
    from utils.db import get_connection
    with get_connection() as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "session_log" in tables:
            session_df = pd.read_sql("SELECT * FROM session_log", conn)
            recent_start = max(0, int(session_df["week"].max()) - 11)
            recent_sessions = session_df[session_df["week"] >= recent_start]
            total_admin = recent_sessions["doses_administered"].sum()
            total_wasted = recent_sessions["doses_wasted"].sum()
            wastage_rate = total_wasted / max(total_admin + total_wasted, 1)
        else:
            wastage_rate = 0.0

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
            label="🚨 Predicted Critical Alerts",
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
            st.caption(f"{color} WHO acceptable < 10% (Last 12m Actual)")

    with col3:
        st.metric(
            label="👶 Predicted Children at Risk",
            value=f"{children_at_risk:,}",
            delta=f"{delta_risk:+d} vs prev week",
            delta_color="inverse",
        )
        st.caption("Based on future critical alerts")

    with col4:
        wrate_pct = wastage_rate * 100
        color = "🔴" if wrate_pct > 10 else ("🟡" if wrate_pct > 7 else "🟢")
        st.metric(
            label="🗑️ Vaccine Wastage Rate",
            value=f"{wrate_pct:.1f}%",
            delta="WHO benchmark: 10%",
            delta_color="off",
        )
        st.caption(f"{color} Last 12 weeks Actual (Static)")

    with col5:
        st.metric(
            label="⚡ Resupply Urgency Score",
            value=f"{urgency_score:.0f} / 100",
            delta="Higher = more urgent",
            delta_color="off",
        )
        st.caption("Future priority index")
