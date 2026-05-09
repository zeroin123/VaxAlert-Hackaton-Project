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
    
    st.success(
        f"**Primary Driver:** '{_pretty(top_feature)}' accounts for **{top_feature_pct}%** of the AI's decision. "
        f"{_describe(top_feature)}"
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
            "Plain-English Description": _describe(f),
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


def _describe(feature_name: str) -> str:
    """One-line plain-English description of why this feature matters."""
    descriptions = {
        "lag_1": "Uses last week's stock to anchor the prediction. This ensures the forecast stays realistic based on current supply.",
        "lag_4": "Identifies the monthly resupply rhythm—the AI 'remembers' if deliveries usually arrive every 4 weeks.",
        "lag_12": "Looks back 3 months to see if there are quarterly spikes or dips in vaccine usage.",
        "lag_26": "Captures half-year patterns, like seasonal movements between urban and rural areas.",
        "lag_52": "Anchors annual cycles—for example, if this facility always runs low in December, the AI picks it up here.",
        "rmean_4": "Smooths out weekly noise to find the 'short-term trend' in how much vaccine is being used.",
        "rmean_12": "Provides the 'average consumption level' over the last 3 months to set a baseline for future needs.",
        "rmean_26": "Captures long-term population growth or shifts in facility demand over the half-year.",
        "rstd_4": "Measures how 'unpredictable' the stock has been recently. High volatility makes the AI more cautious.",
        "rstd_12": "Tracks how stable the supply chain has been over the last quarter.",
        "weeks_since_last_resupply": "The 'Hunger' signal. The longer since the last delivery, the more the AI expects stockout risk to climb.",
        "log_weeks_since_resupply": "Handles the mathematical impact of extreme delivery gaps (e.g., during long road closures).",
        "birth_seasonality_factor": "Adjusts the forecast for the 'Birth Peak' in Ethiopia—demand for BCG and Polio spikes when more babies are born.",
        "is_rainy_season": "Detects the Kiremt season (Jun-Sep). It tells the AI to expect road delays and missed collection trips.",
        "is_measles_sia": "The 'Demand Spike' signal. Signals the AI that vaccination demand will jump up to 8x normal levels.",
        "is_polio_snid": "Prepares the AI for the sudden increase in doses needed during national Polio campaign days.",
        "is_conflict_period": "Tells the AI that standard supply chains are broken. It expects zero deliveries and reduced attendance.",
        "is_pandemic_period": "Accounts for the unique disruptions seen during global health emergencies.",
        "demand_shock_multiplier": "A combined factor that helps the AI understand how multiple events are pushing demand up or down.",
        "supply_shock_multiplier": "The 'Supply Block' signal. If this is zero, the AI knows no new stock can enter the building.",
        "lead_time_multiplier": "Estimates how much 'stretch' to add to delivery times due to local disruptions.",
        "hc_stock_lag_4": "The 'Early Warning' signal from the Health Center. If the supervisor is low, this Health Post will likely run out soon.",
        "recent_stockout_4w": "Identifies a 'Cycle of Poverty'—facilities that ran out recently are statistically more likely to run out again.",
        "lead_time_var_6": "Measures how unreliable the delivery truck is. High variance means the AI can't trust the delivery to arrive on time.",
        "month_sin": "Helps the AI understand smooth, repeating annual cycles in weather and disease patterns.",
        "month_cos": "A mathematical pair to Month (Sine) that helps the AI pinpoint the exact time of year.",
        "woy_sin": "Picks up very fine-grained weekly patterns throughout the Ethiopian calendar year.",
        "woy_cos": "Works with Week (Sine) to ensure the AI knows exactly which week of the year it is forecasting.",
        "week": "Captures the long-term trend in vaccine usage over the years. It helps the AI account for population growth.",
    }
    return descriptions.get(feature_name, "An engineered signal that helps the AI refine its stockout predictions.")
