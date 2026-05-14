import math

import pandas as pd
import streamlit as st

from utils.db import get_connection


def _load_target_population() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT facility_id, antigen, weekly_consumption_baseline, vial_size FROM target_population",
            conn,
        )


def _build_display(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["Margin (days)"] = merged["predicted_days_to_stockout"] - merged["alert_threshold_days"]
    cols_src = [
        "name", "type", "region", "antigen",
        "predicted_days_to_stockout", "lead_time_days_mean",
        "alert_threshold_days", "Margin (days)", "restock_qty",
    ]
    cols_dst = [
        "Facility", "Type", "Region", "Antigen",
        "Days to Stockout", "Lead Time (days)",
        "Threshold (days)", "Margin (days)", "Suggested Order (doses)",
    ]
    if "restock_qty" not in merged.columns:
        cols_src = cols_src[:-1]
        cols_dst = cols_dst[:-1]
    display = merged[cols_src].copy()
    display.columns = cols_dst
    return display.sort_values("Days to Stockout", ascending=True).reset_index(drop=True)


def render_alert_table(
    forecast_output: pd.DataFrame,
    facilities: pd.DataFrame,
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

    # Load target population data and compute restock quantities
    tp_df = _load_target_population()
    merged = merged.merge(tp_df, on=["facility_id", "antigen"], how="left")

    BUFFER_BY_TIER = {
        "Health Post":   {"urban": 2, "rural": 3, "pastoral": 6},
        "Health Center": {"urban": 4, "rural": 5, "pastoral": 7},
        "Hospital":      {"urban": 4, "rural": 5, "pastoral": 7},
    }

    def _yhat_at_lead_time(row):
        lt_weeks = max(1, round(row["lead_time_days_mean"] / 7))
        horizon_week = target_week + lt_weeks - 1
        candidate = forecast_output[
            (forecast_output["model"] == "ensemble") &
            (forecast_output["facility_id"] == row["facility_id"]) &
            (forecast_output["antigen"] == row["antigen"])
        ]
        if candidate.empty:
            return row.get("closing_stock", 0) or 0
        avail = candidate["forecast_week"].values
        closest = avail[abs(avail - horizon_week).argmin()]
        yhat_row = candidate[candidate["forecast_week"] == closest]
        return float(yhat_row["yhat"].iloc[0]) if not yhat_row.empty else 0.0

    merged["yhat_at_delivery"] = merged.apply(_yhat_at_lead_time, axis=1)

    def _restock_qty(row):
        lt_weeks = row["lead_time_days_mean"] / 7
        tier = str(row.get("access_tier", "rural") or "rural").lower()
        buf = BUFFER_BY_TIER.get(row["type"], {}).get(tier, 4)
        wc = row.get("weekly_consumption_baseline", 0) or 0
        vs = max(1, row.get("vial_size", 1) or 1)
        target = wc * (lt_weeks + buf)
        raw = max(0.0, target - row["yhat_at_delivery"])
        return int(math.ceil(raw / vs) * vs)

    merged["restock_qty"] = merged.apply(_restock_qty, axis=1)

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
        crit_display = _build_display(critical)
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
        warn_display = _build_display(warning)
        styled_warn = warn_display.style.set_properties(
            **{"background-color": "#fef9e7", "color": "#e67e22", "font-weight": "600"},
            subset=["Days to Stockout"],
        )
        st.dataframe(styled_warn, use_container_width=True, hide_index=True, height=240)
        st.caption(
            f"{len(warn_display)} warning alert(s) - supply will fall below safety threshold "
            "within the lead-time window. Action recommended."
        )
