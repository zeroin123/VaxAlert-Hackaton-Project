import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.features import display_dates


def render_stock_chart(
    series: pd.DataFrame,
    forecast: pd.DataFrame,
    facility_name: str,
    antigen: str,
    ensemble_weights: dict,
    val_mae: float,
    shocks: pd.DataFrame,
    reorder_point: float,
):
    """
    Combined actual stock history + 8-week ensemble forecast chart.

    Display layer applies a 4-year backward date offset (DISPLAY_OFFSET_WEEKS)
    so a 7-year DB history (2023-2029) renders as 2019-2025, with forecasts
    appearing in early 2026. Models, metrics, and stored data are unchanged.

    The full history is shown (all 7 years) so the operator can see long-run
    patterns, shocks, and seasonality across multiple cycles.
    """
    series = series.copy()
    series["week_date"] = pd.to_datetime(series["week_date"])

    # Apply display-only date offset
    series["x"] = display_dates(series["week_date"])

    if not forecast.empty:
        forecast = forecast.copy()
        forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"])
        forecast["x"] = display_dates(forecast["forecast_date"])

    fig = go.Figure()

    # Training-period shading: shade the entire visible history region in light grey
    if not series.empty:
        x_start = series["x"].iloc[0]
        x_end = series["x"].iloc[-1]
        fig.add_shape(
            type="rect",
            x0=x_start, x1=x_end,
            y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor="rgba(200,200,200,0.10)",
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=x_start, y=1, xref="x", yref="paper",
            text="Training history", showarrow=False,
            font=dict(size=9, color="gray"),
            xanchor="left", yanchor="top",
        )

    # Actual stock history
    fig.add_trace(go.Scatter(
        x=series["x"],
        y=series["closing_stock"],
        mode="lines",
        name="Actual Stock",
        line=dict(color="#2c3e50", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Stock: %{y:.0f} doses<extra></extra>",
    ))

    # Ensemble forecast
    ens = forecast[forecast["model"] == "ensemble"].sort_values("forecast_date") if not forecast.empty else pd.DataFrame()
    if not ens.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([ens["x"], ens["x"].iloc[::-1]]),
            y=pd.concat([ens["yhat_upper"], ens["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="80% Prediction Interval",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=ens["x"],
            y=ens["yhat"],
            mode="lines",
            name="Ensemble Forecast",
            line=dict(color="#3498db", width=2, dash="dash"),
            hovertemplate="Forecast: %{y:.0f} doses<extra></extra>",
        ))

    # Reorder-point line (full-width horizontal)
    if reorder_point > 0:
        fig.add_shape(
            type="line",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=reorder_point, y1=reorder_point,
            line=dict(color="red", width=1.5, dash="dot"),
        )
        fig.add_annotation(
            xref="paper", x=0.99, yref="y", y=reorder_point,
            text=f"Reorder point ({reorder_point:.0f})",
            showarrow=False,
            font=dict(size=9, color="red"),
            xanchor="right", yanchor="bottom",
        )

    # Shock events that fall inside the visible window
    if shocks is not None and not shocks.empty:
        visible_weeks = set(series["week"].astype(int).tolist()) if not series.empty else set()
        plotted_weeks = set()
        for _, shock in shocks.iterrows():
            w = int(shock["week"])
            if w in plotted_weeks or w not in visible_weeks:
                continue
            plotted_weeks.add(w)
            shock_rows = series[series["week"] == w]
            if shock_rows.empty:
                continue
            shock_ts = shock_rows["x"].iloc[0]
            short_label = shock["shock_type"].replace("_", " ").title()[:12]

            fig.add_shape(
                type="line",
                x0=shock_ts, x1=shock_ts,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="orange", width=1, dash="dot"),
            )
            fig.add_annotation(
                x=shock_ts, y=0.95, xref="x", yref="paper",
                text=short_label, showarrow=False,
                font=dict(size=7, color="darkorange"),
                textangle=-90,
                xanchor="center", yanchor="top",
            )

    w_x = ensemble_weights.get("w_xgb", ensemble_weights.get("w_sarimax", 0.5))
    w_p = ensemble_weights.get("w_prophet", 0.5)
    subtitle = (
        f"XGBoost {w_x*100:.0f}% / Prophet {w_p*100:.0f}%  |  "
        f"Typical error: ±{val_mae:.1f} doses/week"
    )

    fig.update_layout(
        title=dict(
            text=f"{facility_name} : {antigen} Stock Forecast<br><sup>{subtitle}</sup>",
            font_size=14,
        ),
        xaxis_title="Date",
        yaxis_title="Closing Stock (doses)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=440,
        margin=dict(l=50, r=20, t=80, b=50),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", type="date"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", rangemode="tozero"),
    )

    st.plotly_chart(fig, use_container_width=True)
