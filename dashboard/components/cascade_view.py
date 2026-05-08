import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.features import display_date, display_dates


def _worst_alert(group):
    rank = {"critical": 2, "warning": 1, "ok": 0}
    return group.sort_values(key=lambda s: s.map(rank), ascending=False).iloc[0]


def render_cascade_view(
    hc_id: str,
    clusters: pd.DataFrame,
    facilities: pd.DataFrame,
    forecast_output: pd.DataFrame,
    stock_ledger: pd.DataFrame,
):
    hp_rows = clusters[clusters["hc_id"] == hc_id]
    hp_ids = hp_rows["hp_id"].tolist()
    hc_name = hp_rows["hc_name"].iloc[0] if not hp_rows.empty else hc_id

    latest_week = int(forecast_output["forecast_week"].min()) if not forecast_output.empty else 156
    ens_latest = forecast_output[
        (forecast_output["model"] == "ensemble") &
        (forecast_output["forecast_week"] == latest_week)
    ]

    def get_worst(fid):
        rows = ens_latest[ens_latest["facility_id"] == fid]
        if rows.empty:
            return "ok", 999
        rank = {"critical": 2, "warning": 1, "ok": 0}
        worst = rows.loc[rows["alert_status"].map(rank).idxmax()]
        return worst["alert_status"], int(worst["predicted_days_to_stockout"])

    alert_color = {"critical": "#e74c3c", "warning": "#f39c12", "ok": "#27ae60"}

    # ── Network diagram ──────────────────────────────────────────────────────
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    edge_x, edge_y = [], []

    hc_info = facilities[facilities["facility_id"] == hc_id]
    hc_pop = int(hc_info["catchment_pop"].values[0]) if not hc_info.empty else 25000
    hc_alert, hc_dts = get_worst(hc_id)

    # HC node at top center
    node_x.append(0.5)
    node_y.append(0.9)
    node_text.append(f"<b>{hc_name}</b><br>HC<br>Alert: {hc_alert.upper()}<br>DTS: {hc_dts}d")
    node_color.append(alert_color[hc_alert])
    node_size.append(30 + hc_pop / 2000)

    n_hp = len(hp_ids)
    for i, hp_id in enumerate(hp_ids):
        hp_info = facilities[facilities["facility_id"] == hp_id]
        hp_name = hp_info["name"].values[0] if not hp_info.empty else hp_id
        hp_pop = int(hp_info["catchment_pop"].values[0]) if not hp_info.empty else 5000
        hp_alert, hp_dts = get_worst(hp_id)

        x = (i + 1) / (n_hp + 1)
        y = 0.15

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{hp_name}</b><br>HP<br>Alert: {hp_alert.upper()}<br>DTS: {hp_dts}d")
        node_color.append(alert_color[hp_alert])
        node_size.append(15 + hp_pop / 500)

        # HC stockout weeks = cascade risk = line thickness
        hc_so_weeks = int(stock_ledger[
            (stock_ledger["facility_id"] == hc_id)
        ]["is_stockout"].sum())
        line_width = max(1, min(8, hc_so_weeks / 5))

        edge_x += [0.5, x, None]
        edge_y += [0.9, 0.15, None]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2, color="#bdc3c7"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Nodes
    short_labels = [t.split("<br>")[0].replace("<b>", "").replace("</b>", "") for t in node_text]
    # Truncate long names for legibility
    short_labels = [n if len(n) <= 28 else n[:25] + "…" for n in short_labels]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=2, color="#2c3e50")),
        text=short_labels,
        textposition=["top center"] + ["bottom center"] * n_hp,
        textfont=dict(color="#1a202c", size=11, family="sans-serif"),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=f"Cascade Network: {hc_name}",
                   font=dict(color="#1a202c", size=15)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.15]),
        height=460,
        margin=dict(l=20, r=20, t=60, b=80),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1a202c"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Counterfactual: "what if HC had been resupplied N weeks earlier?" ────
    st.divider()
    st.subheader("Counterfactual: Earlier Resupply Impact")

    hc_history = stock_ledger[stock_ledger["facility_id"] == hc_id].copy()
    if not hc_history.empty:
        # Find the longest consecutive HC stockout period (across antigens combined)
        hc_weekly = hc_history.groupby("week")["is_stockout"].max().reset_index()
        hc_weekly = hc_weekly.sort_values("week").reset_index(drop=True)
        longest_start, longest_end, longest_len = 0, 0, 0
        cur_start, cur_len = None, 0
        for _, r in hc_weekly.iterrows():
            if r["is_stockout"] == 1:
                if cur_start is None:
                    cur_start = int(r["week"])
                cur_len += 1
                if cur_len > longest_len:
                    longest_len = cur_len
                    longest_start = cur_start
                    longest_end = int(r["week"])
            else:
                cur_start, cur_len = None, 0

        if longest_len >= 2:
            # Sum children missed across HC + all dependent HPs during this period
            cluster_ids = [hc_id] + hp_ids
            window = stock_ledger[
                stock_ledger["facility_id"].isin(cluster_ids)
                & (stock_ledger["week"] >= longest_start)
                & (stock_ledger["week"] <= longest_end)
            ]
            children_missed = int(window["children_missed"].sum())

            slider_max = max(2, longest_len)
            earlier_n = st.slider(
                f"What if **{hc_name}** had been resupplied this many weeks earlier?",
                min_value=1, max_value=slider_max, value=min(2, slider_max),
                key=f"counterfactual_{hc_id}",
            )
            ratio = min(1.0, earlier_n / longest_len)
            saved = int(children_missed * ratio)

            # Look up approximate dates for human readability (display offset applied)
            start_date_row = stock_ledger[stock_ledger["week"] == longest_start].iloc[:1]
            end_date_row = stock_ledger[stock_ledger["week"] == longest_end].iloc[:1]
            if not start_date_row.empty:
                start_date = display_date(start_date_row["week_date"].iloc[0]).strftime("%Y-%m-%d")
            else:
                start_date = f"week {longest_start}"
            if not end_date_row.empty:
                end_date = display_date(end_date_row["week_date"].iloc[0]).strftime("%Y-%m-%d")
            else:
                end_date = f"week {longest_end}"

            cols = st.columns([2, 1])
            with cols[0]:
                st.markdown(
                    f"""
**Cascade insight (descriptive replay):**

During **{start_date}** to **{end_date}**, **{hc_name}** was stocked out for
**{longest_len} consecutive weeks**. Across the HC and its **{len(hp_ids)}**
satellite Health Posts, **{children_missed:,} children** went unvaccinated
during that period.

**If the HC had been resupplied {earlier_n} week(s) earlier**, an estimated
**~{saved:,} additional children** would have been vaccinated.
                    """
                )
            with cols[1]:
                st.metric(
                    label="Additional children vaccinated",
                    value=f"{saved:,}",
                    delta=f"{ratio * 100:.0f}% of cascade window",
                    delta_color="off",
                )

            st.caption(
                "Note: this is a descriptive replay. It scales actual historical "
                "impact by the proportion of the stockout window covered by an "
                "earlier intervention. Not a re-simulation."
            )
        else:
            st.caption(
                "This Health Center has no significant stockout periods on record - "
                "no counterfactual to compute."
            )
    else:
        st.caption("No historical stock data for this Health Center.")

    # ── Cascade impact table ─────────────────────────────────────────────────
    st.subheader("Cascade Impact by Antigen")
    all_fids = [hc_id] + hp_ids
    impact_rows = []
    vaccines = ens_latest["antigen"].unique()
    for fid in all_fids:
        fac_info = facilities[facilities["facility_id"] == fid]
        fac_name = fac_info["name"].values[0] if not fac_info.empty else fid
        fac_type = fac_info["type"].values[0] if not fac_info.empty else "-"
        for ant in vaccines:
            row = ens_latest[
                (ens_latest["facility_id"] == fid) &
                (ens_latest["antigen"] == ant)
            ]
            if row.empty:
                continue
            cascade_count = int(stock_ledger[
                (stock_ledger["facility_id"] == fid) &
                (stock_ledger["antigen"] == ant)
            ]["cascade_affected"].sum())
            children_missed = int(stock_ledger[
                (stock_ledger["facility_id"] == fid) &
                (stock_ledger["antigen"] == ant)
            ]["children_missed"].sum())
            impact_rows.append({
                "Facility": fac_name,
                "Type": fac_type,
                "Antigen": ant,
                "Alert": row.iloc[0]["alert_status"].upper(),
                "DTS (days)": int(row.iloc[0]["predicted_days_to_stockout"]),
                "Cascade Weeks": cascade_count,
                "Children Missed": children_missed,
            })

    if impact_rows:
        impact_df = pd.DataFrame(impact_rows)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

    # ── Cascade heatmap ──────────────────────────────────────────────────────
    st.subheader("Cascade Timeline (by Health Post × Week)")
    if hp_ids:
        cascade_data = []
        for hp_id in hp_ids:
            hp_info = facilities[facilities["facility_id"] == hp_id]
            hp_name = hp_info["name"].values[0] if not hp_info.empty else hp_id
            hp_series = stock_ledger[stock_ledger["facility_id"] == hp_id]
            if hp_series.empty:
                continue
            # Average cascade_affected across antigens per week
            weekly_cascade = hp_series.groupby("week")["cascade_affected"].max().reset_index()
            weekly_cascade["hp"] = hp_name
            cascade_data.append(weekly_cascade)

        if cascade_data:
            cascade_df = pd.concat(cascade_data)
            pivot = cascade_df.pivot(index="hp", columns="week", values="cascade_affected").fillna(0)

            # Map week numbers to displayed dates
            week_to_date = (
                stock_ledger[["week", "week_date"]]
                .drop_duplicates("week")
                .set_index("week")["week_date"]
            )
            x_labels = [
                display_date(week_to_date.get(c, None)).strftime("%Y-%m") if c in week_to_date.index else f"W{c}"
                for c in pivot.columns
            ]

            fig2 = go.Figure(go.Heatmap(
                z=pivot.values,
                x=x_labels,
                y=pivot.index.tolist(),
                colorscale=[[0, "#27ae60"], [1, "#e74c3c"]],
                showscale=False,
                hovertemplate="HP: %{y}<br>Date: %{x}<br>Cascade: %{z}<extra></extra>",
            ))
            fig2.update_layout(
                title=dict(text="Cascade-Affected Weeks (red = HP cut off from HC supply)",
                           font=dict(color="#1a202c", size=13)),
                height=max(150, 40 * len(pivot)),
                margin=dict(l=170, r=20, t=50, b=40),
                xaxis=dict(showticklabels=len(pivot.columns) <= 52,
                           color="#1a202c"),
                yaxis=dict(color="#1a202c"),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(color="#1a202c"),
            )
            st.plotly_chart(fig2, use_container_width=True)
