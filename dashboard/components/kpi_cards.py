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
            ag = recent_sessions.groupby("antigen")[["doses_administered", "doses_wasted"]].sum()
            ag["Wastage %"] = (ag["doses_wasted"] / (ag["doses_administered"] + ag["doses_wasted"]).clip(lower=1) * 100).round(1)
            antigen_wastage_df = ag[["Wastage %"]].reset_index().rename(columns={"antigen": "Antigen"})
            antigen_wastage_df = antigen_wastage_df.sort_values("Wastage %", ascending=False).reset_index(drop=True)
        else:
            wastage_rate = 0.0
            antigen_wastage_df = pd.DataFrame()

    # ── KPI 5: Resupply urgency score ────────────────────────────────────────
    # Raw score for the current forecast week
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

        # ── Normalise against last 52 weeks of actual stock ledger ──────────
        # This anchors the score to real operational history rather than just
        # the 8-week forecast horizon (which produced an artificially inflated
        # score because p95 of 8 values ≈ the maximum of those 8 values).
        #
        # Historical DTS is estimated from the stock ledger as:
        #   hist_dts_days = (closing_stock / weekly_consumption_baseline) * 7
        # Then we apply the same raw score formula and take p95 of the
        # resulting 52 weekly sums — giving a stable, seasonally-grounded
        # baseline that only includes normal operational conditions (post-conflict,
        # post-pandemic weeks are not in the last 52).
        hist_start = max(0, int(stock_ledger["week"].max()) - 51)
        hist_sl = stock_ledger[stock_ledger["week"] >= hist_start].copy()

        # Merge lead time and target infants onto the stock ledger
        hist_sl = hist_sl.merge(
            facilities[["facility_id", "lead_time_days_mean", "target_infants_annual"]],
            on="facility_id", how="left"
        )
        # Merge weekly_consumption_baseline from target_population
        hist_sl = hist_sl.merge(
            target_population[["facility_id", "antigen", "weekly_consumption_baseline"]],
            on=["facility_id", "antigen"], how="left"
        )

        # Compute retrospective DTS and alert classification
        hist_sl["hist_dts_days"] = (
            hist_sl["closing_stock"] /
            hist_sl["weekly_consumption_baseline"].clip(lower=0.1)
        ) * 7

        # Classify historical rows using the same threshold logic as forecasts:
        # critical threshold = lead_time + safety_buffer; warning = critical × 1.5
        # Use a simplified fixed safety buffer of 14 days for historical rows
        # (cv_mae is not available per-week historically)
        # Mirror generate_forecasts.py: threshold T = lead_time + safety_buffer
        # critical → DTS ≤ T×0.5 / warning → DTS ≤ T / ok → DTS > T
        hist_sl["thresh"] = hist_sl["lead_time_days_mean"].fillna(14.0) + 14.0
        hist_sl["hist_alert"] = "ok"
        hist_sl.loc[hist_sl["hist_dts_days"] <= hist_sl["thresh"],       "hist_alert"] = "warning"
        hist_sl.loc[hist_sl["hist_dts_days"] <= hist_sl["thresh"] * 0.5, "hist_alert"] = "critical"

        # Only use WARNING rows (not critical) for p95.
        # Historical critical rows have closing_stock ≈ 0 → hist_dts clipped to
        # 1 day → raw scores 10-100× larger than any forecast warning row (which
        # has DTS ≥ thresh_critical ≈ 35+ days). Including them would make p95
        # so high that every forecast warning week scores near zero.
        # The right question is: "how do current warnings compare to typical
        # warning situations in recent history?" — not against actual past stockouts.
        hist_at_risk = hist_sl[hist_sl["hist_alert"] == "warning"].copy()

        if not hist_at_risk.empty:
            h_dts = hist_at_risk["hist_dts_days"].clip(lower=1)
            h_lt  = hist_at_risk["lead_time_days_mean"].fillna(14.0)
            h_ti  = hist_at_risk["target_infants_annual"].fillna(5000)
            hist_raw = (1.0 / h_dts) * (h_lt / 7.0) * (h_ti / 100.0)
            hist_weekly_sums = hist_raw.groupby(hist_at_risk["week"].values).sum()
            # Fill weeks where no facility was in warning-only state with 0
            all_hist_weeks = pd.Series(0.0, index=range(hist_start, int(stock_ledger["week"].max()) + 1))
            all_hist_weeks.update(hist_weekly_sums)
            p95 = float(np.percentile(all_hist_weeks.values, 95))
            p95 = max(p95, 1e-9)
        else:
            p95 = max(weekly_sum, 1e-9)

        urgency_score = min(100.0, (weekly_sum / p95) * 100.0)
    else:
        urgency_score = 0.0

    # ── Render ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Predicted Critical Alerts",
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
            label="DTP3 Dropout Rate",
            value=dtp_display,
            delta=dtp_delta,
            delta_color="off",
        )
        if not np.isnan(dropout_rate):
            color = "🔴" if dropout_rate > 0.10 else ("🟡" if dropout_rate > 0.05 else "🟢")
            st.caption(f"{color} WHO acceptable < 10% (Last 12m Actual)")

    with col3:
        st.metric(
            label="Predicted Children at Risk",
            value=f"{children_at_risk:,}",
            delta=f"{delta_risk:+d} vs prev week",
            delta_color="inverse",
        )
        st.caption("Based on future critical alerts")

    with col4:
        wrate_pct = wastage_rate * 100
        color = "🔴" if wrate_pct > 10 else ("🟡" if wrate_pct > 7 else "🟢")
        st.metric(
            label="Vaccine Wastage Rate",
            value=f"{wrate_pct:.1f}%",
            delta="WHO benchmark: 10%",
            delta_color="off",
        )
        st.caption(f"{color} Last 12 weeks · by antigen below")
        if not antigen_wastage_df.empty:
            def _color_wrate(val):
                if val > 10: return "color: #e74c3c; font-weight: 600"
                if val > 7:  return "color: #e67e22; font-weight: 600"
                return "color: #27ae60; font-weight: 600"
            with st.expander("By antigen"):
                st.dataframe(
                    antigen_wastage_df.style.map(_color_wrate, subset=["Wastage %"]),
                    use_container_width=True, hide_index=True, height=264,
                )

    with col5:
        st.metric(
            label="Resupply Urgency Score",
            value=f"{urgency_score:.0f} / 100",
            delta="Higher = more urgent",
            delta_color="off",
        )
        st.caption("vs. last 52 weeks · warning + critical facilities")
