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
    "week": "Long-term time trend",
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

    st.subheader("🔍 AI Decision Transparency")
    st.markdown(
        f"This chart shows exactly which 'signals' the **XGBoost AI** used to build the forecast for **{facility_name}**. "
        "Understanding these drivers helps you know *why* an alert was triggered."
    )

    # Highlight the Primary Driver
    top_feature = fac_imp.iloc[0]["feature"]
    top_feature_pct = (fac_imp.iloc[0]["importance"] / fac_imp["importance"].sum() * 100).round(1)
    
    # Get facility metadata for better explanation context
    from utils.db import get_connection
    with get_connection() as conn:
        meta = conn.execute("SELECT region, access_tier, type FROM facilities WHERE facility_id=?", (facility_id,)).fetchone()
        region, tier, ftype = meta if meta else ("Unknown", "Unknown", "Unknown")

    st.success(
        f"**Primary Driver:** '{_pretty(top_feature)}' accounts for **{top_feature_pct}%** of the AI's decision. "
        f"{_describe(top_feature, region, tier, ftype)}"
    )

    # Reverse for horizontal bar (highest importance at top)
    fac_imp_sorted = fac_imp.iloc[::-1].reset_index(drop=True)
    pretty_labels = [_pretty(f) for f in fac_imp_sorted["feature"]]
    importance_pct = (fac_imp_sorted["importance"] / fac_imp_sorted["importance"].sum() * 100).round(1)

    fig = go.Figure(go.Bar(
        x=fac_imp_sorted["importance"].values,
        y=pretty_labels,
        orientation="h",
        marker=dict(color="#3498db"),
        text=[f"{p}%" for p in importance_pct],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=80, t=20, b=30),
        xaxis_title="Relative signal strength (Gain)",
        yaxis_title=None,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1a202c"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Expandable legend with more detailed insights
    with st.expander("📖 Detailed Feature Glossary"):
        legend_rows = [{
            "Signal Category": _get_category(f),
            "Technical Name": f,
            "Plain-English Description": _describe(f, region, tier, ftype),
        } for f in fac_imp["feature"]]
        st.table(pd.DataFrame(legend_rows))


def _get_category(feature_name: str) -> str:
    """Group features into logical categories for the user."""
    if "lag" in feature_name or "rmean" in feature_name or "rstd" in feature_name:
        return "📦 Historical Patterns"
    if "resupply" in feature_name or "lead_time" in feature_name or "supply" in feature_name:
        return "🚚 Supply Chain"
    if "is_rainy" in feature_name or "month" in feature_name or "birth" in feature_name:
        return "🗓️ Seasonal Cycles"
    if "is_measles" in feature_name or "is_polio" in feature_name or "demand" in feature_name:
        return "📢 Health Campaigns"
    if "conflict" in feature_name or "pandemic" in feature_name:
        return "⚠️ External Shocks"
    return "🛠️ Other Signal"


def _describe(feature_name: str, region: str = "Unknown", tier: str = "Unknown", ftype: str = "Unknown") -> str:
    """One-line plain-English description of why this feature matters, with facility context."""
    
    # Context-aware flavor text
    is_remote = tier in ["rural_remote", "pastoral"]
    loc_context = f"In {region}," if region != "Unknown" else "For this facility,"
    
    descriptions = {
        "lag_1": f"{loc_context} the AI anchors the forecast to last week's actual stock. This prevents 'drift' and ensures the prediction remains grounded in your current reality.",
        "lag_4": f"The AI identifies a monthly resupply rhythm. Since this is a {ftype}, it recognizes the 4-week delivery cycle common in this network.",
        "lag_12": "Identifies quarterly patterns. The AI has noticed that vaccine usage often shifts every 3 months due to local health outreach cycles.",
        "lag_26": "Captures half-year trends. This is often linked to seasonal migration patterns or major bi-annual health screenings.",
        "lag_52": "Anchors the forecast to the same time last year. It remembers if you typically run low during this specific month.",
        "rmean_4": "Filters out 'weekly noise' (random spikes) to find the true underlying demand trend over the last month.",
        "rmean_12": "Sets a baseline using the average consumption from the last quarter, providing a stable target for future stock levels.",
        "rmean_26": "Tracks long-term growth. If your catchment population is increasing, this feature captures that 6-month upward trend.",
        "rstd_4": f"Measures recent uncertainty. {'Because this is a remote facility,' if is_remote else ''} high volatility here makes the AI more conservative with its stockout warnings.",
        "rstd_12": "Tracks supply chain stability over the last quarter. Higher values indicate an unreliable delivery history.",
        "weeks_since_last_resupply": "The 'Stockout Clock.' The more weeks that pass without a delivery, the higher the AI scores the risk of running out.",
        "log_weeks_since_resupply": "A mathematical adjustment for long delivery gaps, common during road closures or major disruptions.",
        "birth_seasonality_factor": f"Adjusts for Ethiopia's birth peak. Demand for infant vaccines like BCG and Polio naturally spikes in {region} during high-birth months.",
        "is_rainy_season": f"Detects the Kiremt season (Jun-Sep). {'In remote areas like this,' if is_remote else ''} the AI expects significant delivery delays due to road conditions.",
        "is_measles_sia": "The 'Campaign Spike.' Tells the AI that a national measles campaign is active, which can jump demand by up to 800%.",
        "is_polio_snid": "Prepares for a Polio National Immunization Day, signaling a sudden, large-scale drawdown of doses.",
        "is_conflict_period": "Alerts the AI that standard supply chains in this region are compromised. It expects zero incoming stock and broken logistics.",
        "is_pandemic_period": "Adjusts for global-scale disruptions that historically affected vaccine delivery and attendance.",
        "demand_shock_multiplier": "A combined factor tracking how external events (campaigns, shocks) are collectively pushing demand higher.",
        "supply_shock_multiplier": "The 'Supply Gate.' If this is zero, the AI knows that no matter how much you order, nothing will arrive.",
        "lead_time_multiplier": "Estimates the 'stretch' in delivery times. Higher values mean the delivery truck is expected to be late.",
        "hc_stock_lag_4": f"The 'Cascade Signal.' {'Since this is a Health Post,' if ftype == 'Health Post' else 'For this facility,'} the AI monitors if the supervising center is low on stock.",
        "recent_stockout_4w": "The 'History of Risk.' Facilities that ran out recently are statistically more likely to experience another gap soon.",
        "lead_time_var_6": "Measures the reliability of the delivery truck. If the truck arrival time is erratic, the AI builds in a larger safety buffer.",
        "month_sin": "Helps the AI understand smooth, repeating annual cycles in weather and disease patterns.",
        "month_cos": "Works with Month (Sine) to help the AI pinpoint the exact time of year for its seasonal adjustments.",
        "woy_sin": "Tracks very fine-grained weekly patterns throughout the Ethiopian calendar year.",
        "woy_cos": "Works with Week (Sine) to ensure the AI knows exactly which week of the year it is forecasting for.",
        "week": "Captures the long-term trend over the years, accounting for general population growth in this region.",
    }
    return descriptions.get(feature_name, "A specialized data signal that helps the AI refine its stockout predictions based on historical patterns.")
