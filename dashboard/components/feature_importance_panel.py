import pandas as pd
import streamlit as st
import plotly.graph_objects as go


PRETTY_NAMES = {
    "lag_1": "Stock 1 week ago",
    "lag_2": "Stock 2 weeks ago",
    "lag_4": "Stock 4 weeks ago",
    "lag_8": "Stock 8 weeks ago",
    "lag_12": "Stock 12 weeks ago",
    "lag_26": "Stock 6 months ago",
    "lag_52": "Stock 1 year ago",
    "rmean_4": "4-week average stock",
    "rmean_12": "12-week average stock",
    "rmean_26": "6-month average stock",
    "rstd_4": "4-week stock volatility",
    "rstd_12": "12-week stock volatility",
    "weeks_since_last_resupply": "Weeks since last delivery",
    "log_weeks_since_resupply": "Weeks since delivery (log scale)",
    "birth_seasonality_factor": "Birth seasonality",
    "is_rainy_season": "Rainy season (Jun-Sep)",
    "is_measles_sia": "Active measles campaign",
    "is_polio_snid": "Active polio campaign",
    "is_conflict_period": "Conflict-disruption period",
    "is_pandemic_period": "Pandemic-disruption period",
    "demand_shock_multiplier": "Demand-shock multiplier",
    "supply_shock_multiplier": "Supply-shock multiplier",
    "lead_time_multiplier": "Lead-time-shock multiplier",
    "hc_stock_lag_4": "Supervising HC's recent stock (cascade signal)",
    "recent_stockout_4w": "Recent stockout (last 4 weeks)",
    "lead_time_var_6": "Lead-time variability",
    "month_sin": "Month-of-year (sine encoding)",
    "month_cos": "Month-of-year (cosine encoding)",
    "woy_sin": "Week-of-year (sine encoding)",
    "woy_cos": "Week-of-year (cosine encoding)",
}


def _pretty(name: str) -> str:
    return PRETTY_NAMES.get(name, name)


def render_feature_importance_panel(facility_id: str, facility_name: str,
                                     importance_df: pd.DataFrame):
    """Render top-5 features for the selected facility (mean across antigens)."""
    fac_imp = (
        importance_df[importance_df["facility_id"] == facility_id]
        .sort_values("rank", ascending=True)
        .head(5)
    )
    if fac_imp.empty:
        st.info(
            "Feature importance not yet available for this facility. "
            "Run `python forecast/generate_forecasts.py` to populate."
        )
        return

    st.subheader("What drives this facility's forecast?")
    st.caption(
        f"Mean feature importance across all 7 antigens at **{facility_name}** "
        "- from the XGBoost component of the ensemble."
    )

    # Reverse for horizontal bar (highest importance at top)
    fac_imp = fac_imp.iloc[::-1].reset_index(drop=True)
    pretty_labels = [_pretty(f) for f in fac_imp["feature"]]
    importance_pct = (fac_imp["importance"] / fac_imp["importance"].sum() * 100).round(1)

    fig = go.Figure(go.Bar(
        x=fac_imp["importance"].values,
        y=pretty_labels,
        orientation="h",
        marker=dict(color="#3498db"),
        text=[f"{p}%" for p in importance_pct],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=80, t=20, b=30),
        xaxis_title="Relative importance (XGBoost gain)",
        yaxis_title=None,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1a202c"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Expandable legend
    with st.expander("What do these features mean?"):
        legend_rows = [{
            "Feature": _pretty(f),
            "Description": _describe(f),
        } for f in fac_imp["feature"]]
        st.table(pd.DataFrame(legend_rows))


def _describe(feature_name: str) -> str:
    """One-line plain-English description of why this feature matters."""
    descriptions = {
        "lag_1": "What was the stock level last week? Recent persistence is usually the strongest signal.",
        "lag_4": "Stock from 4 weeks ago - captures monthly resupply rhythm.",
        "lag_12": "Stock from 12 weeks ago - picks up quarterly patterns.",
        "lag_26": "Stock from 6 months ago - captures half-year cycles.",
        "lag_52": "Stock from exactly 1 year ago - anchors annual seasonality.",
        "rmean_4": "Average stock over the last 4 weeks - short-term trend.",
        "rmean_12": "Average stock over the last 12 weeks - medium-term trend.",
        "rmean_26": "Average stock over the last 6 months - long-term level.",
        "rstd_4": "How much stock has fluctuated recently - captures volatility.",
        "rstd_12": "Stock volatility over the last 12 weeks.",
        "weeks_since_last_resupply": "How long since the last delivery? Long gaps signal supply chain trouble.",
        "log_weeks_since_resupply": "Log-scaled version of weeks since delivery - handles very long gaps.",
        "birth_seasonality_factor": "Estimated birth rate this week (1.0 ± 15% over the year). Higher in March-April, lower in Sept-Oct.",
        "is_rainy_season": "Is it the kiremt rainy season (June-September)? Roads close, deliveries delayed.",
        "is_measles_sia": "Is a measles SIA campaign active? Demand spikes 4-8×.",
        "is_polio_snid": "Is a polio SNID campaign active? Demand spikes 3-6×.",
        "is_conflict_period": "Is there an active conflict disruption? Supply collapses, demand drops sharply.",
        "is_pandemic_period": "Was the simulation in a pandemic disruption window?",
        "demand_shock_multiplier": "Combined demand multiplier from all active shocks this week.",
        "supply_shock_multiplier": "Combined supply multiplier from all active shocks (0 = no delivery).",
        "lead_time_multiplier": "Lead-time stretch factor due to disruption (5× during conflict).",
        "hc_stock_lag_4": "Supervising Health Center's average stock over the last 4 weeks - when HC runs low, HP cascade-stockouts follow.",
        "recent_stockout_4w": "Did this facility have any stockout in the last 4 weeks? Stockouts cluster in time.",
        "lead_time_var_6": "How variable have the last 6 deliveries been? High variance = unpredictable supply chain.",
        "month_sin": "Cyclical encoding of month-of-year (sine). Captures smooth annual seasonality.",
        "month_cos": "Cyclical encoding of month-of-year (cosine).",
        "woy_sin": "Cyclical encoding of week-of-year (sine).",
        "woy_cos": "Cyclical encoding of week-of-year (cosine).",
    }
    return descriptions.get(feature_name, "Engineered feature.")
