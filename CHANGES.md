# VaxAlert — Change Log

> Each entry records what changed, which file, what the original code was, and how to revert.

---

## [2026-05-10] Wastage KPI — collapsible per-antigen breakdown (update)

**File:** `dashboard/components/kpi_cards.py`

**What changed:** Wrapped the per-antigen dataframe in `st.expander("By antigen")` so it is collapsed by default.

**To revert:** Remove the `with st.expander("By antigen"):` line and unindent the `st.dataframe(...)` call back one level.

---

## [2026-05-10] Wastage KPI — per-antigen breakdown

**File:** `dashboard/components/kpi_cards.py`

**What changed:** The Vaccine Wastage Rate KPI card now shows a per-antigen breakdown table (sorted descending, colour-coded red/amber/green against WHO 10% threshold) below the existing overall metric. Previously it showed only one aggregate number.

**To revert — replace these two sections:**

### Section 1 (computation, inside `with get_connection()` block)

Replace:
```python
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
```

With:
```python
        if "session_log" in tables:
            session_df = pd.read_sql("SELECT * FROM session_log", conn)
            recent_start = max(0, int(session_df["week"].max()) - 11)
            recent_sessions = session_df[session_df["week"] >= recent_start]
            total_admin = recent_sessions["doses_administered"].sum()
            total_wasted = recent_sessions["doses_wasted"].sum()
            wastage_rate = total_wasted / max(total_admin + total_wasted, 1)
        else:
            wastage_rate = 0.0
```

### Section 2 (render, inside `with col4:` block)

Replace:
```python
    with col4:
        wrate_pct = wastage_rate * 100
        color = "🔴" if wrate_pct > 10 else ("🟡" if wrate_pct > 7 else "🟢")
        st.metric(
            label="🗑️ Vaccine Wastage Rate",
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
            st.dataframe(
                antigen_wastage_df.style.map(_color_wrate, subset=["Wastage %"]),
                use_container_width=True, hide_index=True, height=264,
            )
```

With:
```python
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
```
