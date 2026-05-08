import numpy as np
import pandas as pd
from datetime import date, timedelta

START_DATE = date(2023, 1, 2)  # Monday of simulation week 0


# Display-only date shift. The simulation uses week 0 = 2023-01-02, so a 7-year
# run ends in 2029. For demo realism we want the 7 years of history to end in
# late 2025 (so forecasts appear in 2026, matching "today" framing). Subtract
# 4 years (208 weeks) at display time. This affects ONLY visual labels — model
# features, training, metrics, and stored DB dates are untouched.
DISPLAY_OFFSET_WEEKS = 208
DISPLAY_OFFSET = pd.Timedelta(weeks=DISPLAY_OFFSET_WEEKS)


def display_date(stored_date) -> pd.Timestamp:
    """Convert a stored DB week_date / forecast_date into the user-facing date.
    Accepts strings, datetimes, or pd.Timestamps; returns pd.Timestamp."""
    return pd.to_datetime(stored_date) - DISPLAY_OFFSET


def display_dates(series_or_array):
    """Vectorised version of display_date for pd.Series / arrays."""
    return pd.to_datetime(series_or_array) - DISPLAY_OFFSET


def _week_to_date(week: int) -> date:
    return START_DATE + timedelta(weeks=week)


def _birth_seasonality_factor(week_date: date) -> float:
    month = week_date.month
    phase = 2 * np.pi * (month - 9.5) / 12
    return 1.0 + 0.15 * np.cos(phase)


def _is_rainy_season(week_date: date) -> int:
    return 1 if week_date.month in (6, 7, 8, 9) else 0


def build_exog_features(facility_id: str, n_weeks: int = 364,
                          antigen: str = None) -> pd.DataFrame:
    """
    Returns feature matrix for weeks 0..(n_weeks-1).
    Future-week features (after end of training) are computed from calendar
    functions, not historical lookups — except weeks_since_last_resupply
    which is extrapolated forward as a constant (last known value).

    When `antigen` is provided, three series-specific features are appended:
      - hc_stock_lag_4:    supervising HC's 4-week rolling mean stock for the
                            same antigen (cascade signal). For HC/Hospital
                            (no upstream HC), uses the facility's own rolling
                            mean instead.
      - recent_stockout_4w: binary indicator if the (facility, antigen)
                            series had any stockout in the last 4 weeks.
      - lead_time_var_6:   std-dev of the last 6 deliveries' lead_time_actual.
    """
    from utils.db import get_shocks_for_facility, get_delivery_log

    shocks = get_shocks_for_facility(facility_id)

    # Pre-build shock lookup: week -> (demand_mult, supply_mult, lead_mult)
    demand_mult = {}
    supply_mult = {}
    lead_mult = {}
    campaign_mcv = set()   # weeks with active measles SIA
    campaign_opv = set()   # weeks with active polio SNID
    conflict_weeks = set()
    pandemic_weeks = set()

    for _, shock in shocks.iterrows():
        start_w = int(shock["week"])
        dur = int(shock["duration_weeks"])
        stype = shock["shock_type"]
        antigens = shock["affected_antigens"]

        for w in range(start_w, start_w + dur):
            if w >= n_weeks:
                continue
            demand_mult[w] = demand_mult.get(w, 1.0) * float(shock["demand_multiplier"])
            supply_mult[w] = supply_mult.get(w, 1.0) * float(shock["supply_multiplier"])
            lead_mult[w] = lead_mult.get(w, 1.0) * float(shock["lead_time_multiplier"])

            if stype == "measles_sia_campaign" and (antigens == "ALL" or antigens == "MCV"):
                campaign_mcv.add(w)
            if stype == "polio_snid_campaign" and (antigens == "ALL" or antigens == "OPV"):
                campaign_opv.add(w)
            if stype == "conflict_disruption":
                conflict_weeks.add(w)
            if stype == "pandemic_disruption":
                pandemic_weeks.add(w)

    # weeks_since_last_resupply from delivery_log — compute for all antigens combined
    # (facility-level, not antigen-level, since features are per facility)
    try:
        deliveries = get_delivery_log(facility_id, antigen=None)
    except Exception:
        deliveries = pd.DataFrame(columns=["week", "quantity_received"])

    # Build resupply weeks set
    resupply_weeks = set()
    if not deliveries.empty and "week" in deliveries.columns:
        mask = deliveries["quantity_received"] > 0
        resupply_weeks = set(deliveries.loc[mask, "week"].astype(int).tolist())

    rows = []
    last_resupply = -1
    for w in range(n_weeks):
        wd = _week_to_date(w)
        if w in resupply_weeks:
            last_resupply = w
        weeks_since = w - last_resupply if last_resupply >= 0 else w + 1
        log_weeks = np.log1p(weeks_since)

        rows.append({
            "week": w,
            "week_date": pd.Timestamp(wd),
            "is_rainy_season": _is_rainy_season(wd),
            "is_measles_sia": 1 if w in campaign_mcv else 0,
            "is_polio_snid": 1 if w in campaign_opv else 0,
            "is_conflict_period": 1 if w in conflict_weeks else 0,
            "is_pandemic_period": 1 if w in pandemic_weeks else 0,
            "demand_shock_multiplier": demand_mult.get(w, 1.0),
            "supply_shock_multiplier": supply_mult.get(w, 1.0),
            "lead_time_multiplier": lead_mult.get(w, 1.0),
            "birth_seasonality_factor": _birth_seasonality_factor(wd),
            "weeks_since_last_resupply": weeks_since,
            "log_weeks_since_resupply": log_weeks,
        })

    df = pd.DataFrame(rows)

    # ── Three high-ROI series-specific features (require antigen) ──────────
    if antigen is not None:
        df = _attach_series_features(df, facility_id, antigen, n_weeks)

    return df


def _attach_series_features(df: pd.DataFrame, facility_id: str,
                              antigen: str, n_weeks: int) -> pd.DataFrame:
    """Compute hc_stock_lag_4, recent_stockout_4w, lead_time_var_6 and
    attach them to the feature matrix."""
    from utils.db import get_connection

    with get_connection() as conn:
        sup = conn.execute(
            "SELECT supervising_hc_id, type FROM facilities WHERE facility_id=?",
            (facility_id,)
        ).fetchone()
        sup_hc_id = sup[0] if sup else None
        ftype = sup[1] if sup else "Health Post"

        own_series = pd.read_sql(
            "SELECT week, closing_stock, is_stockout FROM stock_ledger "
            "WHERE facility_id=? AND antigen=? ORDER BY week",
            conn, params=(facility_id, antigen),
        )

        if sup_hc_id and ftype == "Health Post":
            hc_series = pd.read_sql(
                "SELECT week, closing_stock FROM stock_ledger "
                "WHERE facility_id=? AND antigen=? ORDER BY week",
                conn, params=(sup_hc_id, antigen),
            )
        else:
            hc_series = own_series[["week", "closing_stock"]].copy()

        deliveries = pd.read_sql(
            "SELECT week, lead_time_actual FROM delivery_log "
            "WHERE facility_id=? AND antigen=? ORDER BY week",
            conn, params=(facility_id, antigen),
        )

    # 1. hc_stock_lag_4: shifted 4-week rolling mean of HC's stock
    if not hc_series.empty:
        hc_series = hc_series.set_index("week").reindex(range(n_weeks)).ffill().fillna(0)
        hc_rolling = (hc_series["closing_stock"]
                      .shift(1).rolling(window=4, min_periods=1).mean()
                      .fillna(0))
    else:
        hc_rolling = pd.Series([0.0] * n_weeks, index=range(n_weeks))
    df["hc_stock_lag_4"] = [float(hc_rolling.get(w, 0.0)) for w in df["week"]]

    # 2. recent_stockout_4w: did facility have ANY stockout in last 4 weeks?
    if not own_series.empty:
        own = own_series.set_index("week").reindex(range(n_weeks)).fillna(0)
        recent = (own["is_stockout"].shift(1)
                   .rolling(window=4, min_periods=1).max()
                   .fillna(0).astype(int))
    else:
        recent = pd.Series([0] * n_weeks, index=range(n_weeks))
    df["recent_stockout_4w"] = [int(recent.get(w, 0)) for w in df["week"]]

    # 3. lead_time_var_6: std of last 6 deliveries' actual lead time at week w
    lead_times = []
    if not deliveries.empty:
        d = deliveries.sort_values("week").reset_index(drop=True)
        for w in df["week"]:
            past = d[d["week"] < w].tail(6)
            if len(past) >= 2:
                lead_times.append(float(past["lead_time_actual"].std(ddof=0)))
            else:
                lead_times.append(0.0)
    else:
        lead_times = [0.0] * len(df)
    df["lead_time_var_6"] = lead_times

    return df


def build_exog_future(facility_id: str, start_week: int = 156, n_weeks: int = 8,
                       antigen: str = None) -> pd.DataFrame:
    """
    Build exog feature matrix for future weeks (e.g. 156-163).
    Shock multipliers default to 1.0 (no active disruption assumed).
    Calendar-derived features are computed from actual future dates.
    """
    from utils.db import get_delivery_log

    # Get last resupply from historical delivery log to seed weeks_since_last_resupply
    try:
        deliveries = get_delivery_log(facility_id, antigen=None)
    except Exception:
        deliveries = pd.DataFrame(columns=["week", "quantity_received"])

    last_resupply = -1
    if not deliveries.empty and "week" in deliveries.columns:
        mask = deliveries["quantity_received"] > 0
        if mask.any():
            last_resupply = int(deliveries.loc[mask, "week"].max())

    rows = []
    for offset in range(n_weeks):
        w = start_week + offset
        wd = _week_to_date(w)
        weeks_since = w - last_resupply if last_resupply >= 0 else w + 1
        log_weeks = np.log1p(weeks_since)

        rows.append({
            "week": w,
            "week_date": pd.Timestamp(wd),
            "is_rainy_season": _is_rainy_season(wd),
            "is_measles_sia": 0,
            "is_polio_snid": 0,
            "is_conflict_period": 0,
            "is_pandemic_period": 0,
            "demand_shock_multiplier": 1.0,
            "supply_shock_multiplier": 1.0,
            "lead_time_multiplier": 1.0,
            "birth_seasonality_factor": _birth_seasonality_factor(wd),
            "weeks_since_last_resupply": weeks_since,
            "log_weeks_since_resupply": log_weeks,
        })

    df = pd.DataFrame(rows)

    # Append series-specific features for forecast weeks. We use the LAST
    # observed values from history as constants over the forecast window —
    # the HC stock state and stockout indicator are unknown for the future,
    # so propagate the most recent observation.
    if antigen is not None:
        from utils.db import get_connection
        with get_connection() as conn:
            sup = conn.execute(
                "SELECT supervising_hc_id, type FROM facilities WHERE facility_id=?",
                (facility_id,)
            ).fetchone()
            sup_hc_id = sup[0] if sup else None
            ftype = sup[1] if sup else "Health Post"

            target_fid = sup_hc_id if (sup_hc_id and ftype == "Health Post") else facility_id
            hc_recent = conn.execute(
                "SELECT closing_stock FROM stock_ledger "
                "WHERE facility_id=? AND antigen=? ORDER BY week DESC LIMIT 4",
                (target_fid, antigen)
            ).fetchall()
            hc_lag_4 = float(np.mean([r[0] for r in hc_recent])) if hc_recent else 0.0

            so_recent = conn.execute(
                "SELECT MAX(is_stockout) FROM stock_ledger WHERE facility_id=? "
                "AND antigen=? AND week >= (SELECT MAX(week) FROM stock_ledger) - 3",
                (facility_id, antigen)
            ).fetchone()
            recent_so = int(so_recent[0]) if so_recent and so_recent[0] is not None else 0

            lt_recent = conn.execute(
                "SELECT lead_time_actual FROM delivery_log WHERE facility_id=? "
                "AND antigen=? ORDER BY week DESC LIMIT 6",
                (facility_id, antigen)
            ).fetchall()
            lt_vals = [r[0] for r in lt_recent if r[0] is not None]
            lt_var = float(pd.Series(lt_vals).std(ddof=0)) if len(lt_vals) >= 2 else 0.0

        df["hc_stock_lag_4"] = hc_lag_4
        df["recent_stockout_4w"] = recent_so
        df["lead_time_var_6"] = lt_var

    return df


def build_prophet_events(facility_id: str) -> pd.DataFrame:
    """
    Returns DataFrame in Prophet holiday format.
    Conflict-affected facilities are derived from shock_events table.
    """
    from utils.db import get_connection

    rows = []

    # Simulation spans 7 years starting 2023-01-02
    years = [2023, 2024, 2025, 2026, 2027, 2028, 2029]

    # measles_sia_campaign: Oct-Nov, biennial (years 1, 3, 5, 7 = 2023, 2025, 2027, 2029)
    for yr in [2023, 2025, 2027, 2029]:
        for month in [10, 11]:
            for day in range(1, 29, 7):
                try:
                    ds = pd.Timestamp(yr, month, day)
                    rows.append({"holiday": "measles_sia_campaign", "ds": ds,
                                 "lower_window": -2, "upper_window": 2})
                except ValueError:
                    pass

    # polio_snid_campaign: April each year
    for yr in years:
        for day in range(1, 29, 7):
            try:
                ds = pd.Timestamp(yr, 4, day)
                rows.append({"holiday": "polio_snid_campaign", "ds": ds,
                             "lower_window": -1, "upper_window": 1})
            except ValueError:
                pass

    # kiremt_rainy_season: June-September each year
    for yr in years:
        for month in [6, 7, 8, 9]:
            for day in range(1, 29, 7):
                try:
                    ds = pd.Timestamp(yr, month, day)
                    rows.append({"holiday": "kiremt_rainy_season", "ds": ds,
                                 "lower_window": 0, "upper_window": 0})
                except ValueError:
                    pass

    # pandemic_disruption: weeks 8-30 from START_DATE
    for w in range(8, 31):
        ds = pd.Timestamp(START_DATE + timedelta(weeks=w))
        rows.append({"holiday": "pandemic_disruption", "ds": ds,
                     "lower_window": 0, "upper_window": 0})

    # conflict_disruption: facility-specific, derived from shock_events
    with get_connection() as conn:
        affected = conn.execute(
            "SELECT DISTINCT facility_id FROM shock_events WHERE shock_type='conflict_disruption'"
        ).fetchall()
        affected_ids = {r[0] for r in affected}

        if facility_id in affected_ids:
            conflict_rows = conn.execute(
                """
                SELECT week, duration_weeks FROM shock_events
                WHERE facility_id = ? AND shock_type = 'conflict_disruption'
                """,
                (facility_id,),
            ).fetchall()
            for cr in conflict_rows:
                start_w, dur = int(cr[0]), int(cr[1])
                for w in range(start_w, start_w + dur):
                    ds = pd.Timestamp(START_DATE + timedelta(weeks=w))
                    rows.append({"holiday": "conflict_disruption", "ds": ds,
                                 "lower_window": 0, "upper_window": 0})
                # post_disruption_catchup: 4 weeks after disruption ends
                for w in range(start_w + dur, start_w + dur + 4):
                    ds = pd.Timestamp(START_DATE + timedelta(weeks=w))
                    rows.append({"holiday": "post_disruption_catchup", "ds": ds,
                                 "lower_window": 0, "upper_window": 0})

    if not rows:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["holiday", "ds"])
    return df.reset_index(drop=True)
