import pandas as pd
import streamlit as st


def _build_display(merged: pd.DataFrame, hp_ids: set) -> pd.DataFrame:
    merged = merged.copy()
    merged["Cascades"] = merged["facility_id"].apply(lambda x: "Yes" if x in hp_ids else "No")
    display = merged[[
        "name", "type", "region", "antigen",
        "predicted_days_to_stockout", "lead_time_days_mean", "Cascades",
    ]].copy()
    display.columns = [
        "Facility", "Type", "Region", "Antigen",
        "Days to Stockout", "Lead Time (days)", "Cascades",
    ]
    return display.sort_values("Days to Stockout", ascending=True).reset_index(drop=True)


def render_alert_table(
    forecast_output: pd.DataFrame,
    facilities: pd.DataFrame,
    clusters: pd.DataFrame,
    forecast_horizon: int = 1,
):
    """Two stacked tables - Critical first (most urgent), then Warning.
    forecast_horizon (1..8) selects which forecast week to display."""

    if forecast_output.empty:
        target_week = 364
    else:
        first_week = int(forecast_output["forecast_week"].min())
        target_week = first_week + (forecast_horizon - 1)
    ens_latest = forecast_output[
        (forecast_output["model"] == "ensemble") &
        (forecast_output["forecast_week"] == target_week)
    ].copy()

    if ens_latest.empty:
        st.info("No alerts at this time.")
        return

    merged = ens_latest.merge(
        facilities[["facility_id", "name", "type", "region", "access_tier",
                    "lead_time_days_mean"]],
        on="facility_id", how="left",
    )
    hp_ids = set(clusters["hp_id"].unique())

    critical = merged[merged["alert_status"] == "critical"]
    warning = merged[merged["alert_status"] == "warning"]

    # ── Critical ────────────────────────────────────────────────────────────
    st.markdown(
        f"#### 🚨 Critical Alerts <span style='color:#c0392b; font-weight:600'>"
        f"({len(critical)})</span>",
        unsafe_allow_html=True,
    )
    if critical.empty:
        st.caption("No critical alerts.")
    else:
        crit_display = _build_display(critical, hp_ids)
        styled_crit = crit_display.style.set_properties(
            **{"background-color": "#fde8e8", "color": "#c0392b", "font-weight": "600"},
            subset=["Days to Stockout"],
        )
        st.dataframe(styled_crit, use_container_width=True, hide_index=True, height=240)
        st.caption(
            f"{len(crit_display)} critical alert(s) - facility×antigen pairs at "
            "or near stockout. Sorted by days to stockout (most urgent first)."
        )

    st.markdown(" ")

    # ── Warning ─────────────────────────────────────────────────────────────
    st.markdown(
        f"#### ⚠️ Warning Alerts <span style='color:#e67e22; font-weight:600'>"
        f"({len(warning)})</span>",
        unsafe_allow_html=True,
    )
    if warning.empty:
        st.caption("No warning-level alerts.")
    else:
        warn_display = _build_display(warning, hp_ids)
        styled_warn = warn_display.style.set_properties(
            **{"background-color": "#fef9e7", "color": "#e67e22", "font-weight": "600"},
            subset=["Days to Stockout"],
        )
        st.dataframe(styled_warn, use_container_width=True, hide_index=True, height=240)
        st.caption(
            f"{len(warn_display)} warning alert(s) - supply will fall below safety threshold "
            "within the lead-time window. Action recommended."
        )
