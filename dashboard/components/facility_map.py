import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


def render_facility_map(
    facilities: pd.DataFrame,
    forecast_output: pd.DataFrame,
    forecast_horizon: int = 1,
):
    """Choropleth map of all 30 facilities coloured by worst alert status.
    forecast_horizon (1..8) selects which forecast week to display."""

    if forecast_output.empty:
        latest_week = 364
    else:
        first_week = int(forecast_output["forecast_week"].min())
        latest_week = first_week + (forecast_horizon - 1)
    ens_latest = forecast_output[
        (forecast_output["model"] == "ensemble") &
        (forecast_output["forecast_week"] == latest_week)
    ]

    # Worst alert per facility
    alert_rank = {"critical": 2, "warning": 1, "ok": 0}
    fac_alert = (
        ens_latest.assign(rank=ens_latest["alert_status"].map(alert_rank))
        .sort_values("rank", ascending=False)
        .drop_duplicates("facility_id")[["facility_id", "alert_status", "predicted_days_to_stockout", "antigen"]]
    )
    fac_data = facilities.merge(fac_alert, on="facility_id", how="inner")

    # Center on Ethiopia
    m = folium.Map(location=[9.0, 40.0], zoom_start=6, tiles="CartoDB positron")

    color_map = {"critical": "#e74c3c", "warning": "#f39c12", "ok": "#27ae60"}
    max_pop = fac_data["catchment_pop"].max()

    for _, row in fac_data.iterrows():
        color = color_map.get(row["alert_status"], "#27ae60")
        radius = 6 + 12 * (row["catchment_pop"] / max_pop)

        dts = row.get("predicted_days_to_stockout", "N/A")
        worst_ant = row.get("antigen", "-")
        popup_html = f"""
        <b>{row['name']}</b><br>
        Type: {row['type']}<br>
        Tier: {row['access_tier']}<br>
        Region: {row['region']}<br>
        Alert: <b style='color:{color}'>{row['alert_status'].upper()}</b><br>
        Worst antigen: {worst_ant}<br>
        Days to stockout: {dts}
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['name']} - {row['alert_status'].upper()}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;">
    <b>Alert Status</b><br>
    <span style="color:#e74c3c">● Critical</span><br>
    <span style="color:#f39c12">● Warning</span><br>
    <span style="color:#27ae60">● OK</span><br>
    <small>Circle size ∝ catchment population</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(
        m,
        width=None,
        height=480,
        returned_objects=[],
        key=f"facility_map_{forecast_horizon}_{len(fac_data)}"
    )
