"""
Production forecast generator (XGBoost + Prophet ensemble).

Prerequisites: walk_forward_cv.py must have run.

Training window: weeks 0-139 (excludes locked test set 140-155).
NOTE: In a real deployment you would retrain on all 156 weeks.
Forecast horizon: 8 weeks (weeks 156-163).
"""

import os
import sys
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import (
    get_facilities, get_vaccines, get_stock_series,
    get_target_population, write_forecasts, get_connection,
)
from utils.features import build_exog_features, build_exog_future
from models.baseline import NaiveModel, HoltWintersModel
from models.xgboost_model import XGBoostForecaster
from models.prophet_model import ProphetModel
from models.stacking import StackingEnsemble, build_meta_features

# Production retraining: use ALL 156 weeks (vs. CV which holds out 140-155).
# In a real deployment you'd always train on the latest data — this matches
# operational reality. CV/test-set integrity has already been measured and
# stored in model_metrics; the locked test was only for honest evaluation.
TRAIN_END = 364
FORECAST_START = 364
FORECAST_HORIZON = 8
FORECAST_END = FORECAST_START + FORECAST_HORIZON

# Load global stacking meta-model once at module load
_META_MODEL = StackingEnsemble.load()


def _load_ensemble_weights(facility_id, antigen) -> tuple:
    """Returns (w_xgb, w_prophet). Stored in legacy w_sarimax column for compat."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT w_sarimax, w_prophet FROM model_metrics
            WHERE facility_id=? AND antigen=? AND model='ensemble' AND fold='final'
            """,
            (facility_id, antigen),
        ).fetchone()
    if row and row[0] is not None:
        return float(row[0]), float(row[1])
    return 0.5, 0.5


def _is_zero_inflated(facility_id, antigen) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT zero_inflated FROM model_metrics
            WHERE facility_id=? AND antigen=? AND fold='final'
            LIMIT 1
            """,
            (facility_id, antigen),
        ).fetchone()
    return bool(row[0]) if row else False


def _get_cv_mae(facility_id, antigen, model) -> float:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT mae FROM model_metrics
            WHERE facility_id=? AND antigen=? AND model=? AND fold IN ('1','2','3')
            """,
            (facility_id, antigen, model),
        ).fetchall()
    if rows:
        vals = [r[0] for r in rows if r[0] is not None]
        return float(np.mean(vals)) if vals else float("nan")
    return float("nan")


def _compute_alert_status(predicted_dts: float, lead_time: float, cv_mae_weeks: float) -> str:
    safety_buffer = max(cv_mae_weeks * 7, 5.0)
    threshold = lead_time + safety_buffer
    if predicted_dts <= threshold * 0.5:
        return "critical"
    if predicted_dts <= threshold:
        return "warning"
    return "ok"


def _make_forecast_rows(
    facility_id, antigen, model_name, yhat, yhat_lower, yhat_upper,
    current_stock, weekly_burn, forecast_dates, lead_time,
    cv_mae_doses, weekly_consumption,
    w_sarimax=None, w_prophet=None, generated_at=None,
) -> list:
    # Convert MAE from doses to weeks (PRD-correct unit) and cap at 3 weeks.
    # For low-volume series (e.g. pastoral HPs at <1 dose/week), an MAE of a
    # few doses translates to many weeks, blowing up the safety buffer past
    # operational usefulness. Cap at 3 weeks — beyond that you wouldn't
    # pre-order even with high uncertainty.
    if weekly_consumption and weekly_consumption > 0.1:
        cv_mae_weeks = min(3.0, cv_mae_doses / weekly_consumption)
    else:
        cv_mae_weeks = 1.0
    rows = []
    if generated_at is None:
        generated_at = datetime.utcnow().isoformat()

    # The forecast yhat[i] is the PREDICTED CLOSING STOCK at week 156+i.
    # DTS = "if burn continues at the historical rate, how many days of
    # supply does that predicted stock represent?" Burn rate is the
    # weekly_consumption_baseline (a true rate), not the forecast yhat
    # (which is a stock level — different unit).
    burn_rate = max(weekly_consumption, 0.1)  # doses per week
    for i in range(len(yhat)):
        if burn_rate > 0:
            predicted_dts = max(0, int(float(yhat[i]) / burn_rate * 7))
        else:
            predicted_dts = 999

        alert = _compute_alert_status(predicted_dts, lead_time, cv_mae_weeks)
        rows.append({
            "facility_id": facility_id,
            "antigen": antigen,
            "forecast_week": FORECAST_START + i,
            "forecast_date": forecast_dates[i].strftime("%Y-%m-%d"),
            "model": model_name,
            "yhat": round(float(yhat[i]), 2),
            "yhat_lower": round(float(yhat_lower[i]), 2),
            "yhat_upper": round(float(yhat_upper[i]), 2),
            "predicted_days_to_stockout": predicted_dts,
            "alert_status": alert,
            "ensemble_w_sarimax": w_sarimax,    # repurposed: stores w_xgb
            "ensemble_w_prophet": w_prophet,
            "generated_at": generated_at,
        })
    return rows


def forecast_series(facility_id, antigen):
    """Returns (forecast_rows, importance_series_or_none)."""
    series_full = get_stock_series(facility_id, antigen)
    if len(series_full) < TRAIN_END:
        return [], None

    tp = get_target_population(facility_id, antigen)
    weekly_ceiling = None
    if tp and tp.get("stock_needed_annual"):
        weekly_ceiling = float(tp["stock_needed_annual"]) / 52 * 1.5
    weekly_consumption = float(tp.get("weekly_consumption_baseline", 1.0)) if tp else 1.0

    # Use the latest observed stock as the "current" stock for alert math.
    # This is what an operator sees on Monday of forecast week 156.
    current_stock = float(series_full["closing_stock"].iloc[len(series_full) - 1])

    exog_hist = build_exog_features(facility_id, n_weeks=TRAIN_END, antigen=antigen)
    exog_train = exog_hist.iloc[:TRAIN_END]
    exog_future = build_exog_future(facility_id, start_week=FORECAST_START,
                                       n_weeks=FORECAST_HORIZON, antigen=antigen)
    forecast_dates = [pd.Timestamp(exog_future.iloc[i]["week_date"]) for i in range(FORECAST_HORIZON)]

    y_train = series_full["closing_stock"].iloc[:TRAIN_END].astype(float)
    dates_train = pd.to_datetime(series_full["week_date"].iloc[:TRAIN_END])

    with get_connection() as conn:
        row = conn.execute(
            "SELECT lead_time_days_mean FROM facilities WHERE facility_id=?", (facility_id,)
        ).fetchone()
    lead_time = float(row[0]) if row else 14.0

    zero_inf = _is_zero_inflated(facility_id, antigen)
    cv_mae_naive = _get_cv_mae(facility_id, antigen, "naive_last_value")
    cv_mae_hw = _get_cv_mae(facility_id, antigen, "holt_winters")
    cv_mae_xgb = _get_cv_mae(facility_id, antigen, "xgboost")
    cv_mae_prophet = _get_cv_mae(facility_id, antigen, "prophet")
    cv_mae_ensemble = _get_cv_mae(facility_id, antigen, "ensemble")

    generated_at = datetime.utcnow().isoformat()
    all_rows = []

    # Naive
    nv = NaiveModel("last_value").fit(y_train)
    nv_int = nv.predict_interval(FORECAST_HORIZON)
    weekly_burn = max(nv_int["yhat"].mean(), 0.1)
    all_rows.extend(_make_forecast_rows(
        facility_id, antigen, "naive",
        nv_int["yhat"].values, nv_int["yhat_lower"].values, nv_int["yhat_upper"].values,
        current_stock, weekly_burn, forecast_dates, lead_time,
        cv_mae_naive if not np.isnan(cv_mae_naive) else 7.0,
        weekly_consumption,
        generated_at=generated_at,
    ))

    # Holt-Winters
    hw = HoltWintersModel().fit(y_train)
    hw_int = hw.predict_interval(FORECAST_HORIZON)
    weekly_burn = max(hw_int["yhat"].mean(), 0.1)
    all_rows.extend(_make_forecast_rows(
        facility_id, antigen, "holt_winters",
        hw_int["yhat"].values, hw_int["yhat_lower"].values, hw_int["yhat_upper"].values,
        current_stock, weekly_burn, forecast_dates, lead_time,
        cv_mae_hw if not np.isnan(cv_mae_hw) else 7.0,
        weekly_consumption,
        generated_at=generated_at,
    ))

    xgb_model = None
    prophet_model = None

    if not zero_inf:
        # XGBoost
        xgb_model = XGBoostForecaster(ceiling=weekly_ceiling)
        xgb_model.fit(y_train, dates_train, exog_train)
        x_int = xgb_model.predict_interval(FORECAST_HORIZON, exog_future)
        weekly_burn = max(x_int["yhat"].mean(), 0.1)
        all_rows.extend(_make_forecast_rows(
            facility_id, antigen, "xgboost",
            x_int["yhat"].values, x_int["yhat_lower"].values, x_int["yhat_upper"].values,
            current_stock, weekly_burn, forecast_dates, lead_time,
            cv_mae_xgb if not np.isnan(cv_mae_xgb) else 7.0,
            weekly_consumption,
            generated_at=generated_at,
        ))

        # Prophet
        train_ds = pd.DataFrame({
            "ds": dates_train, "y": y_train.values,
        })
        prophet_model = ProphetModel(facility_id=facility_id, ceiling=weekly_ceiling)
        prophet_model.fit(train_ds)

        future_ds = pd.DataFrame({"ds": [pd.Timestamp(d) for d in forecast_dates]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if prophet_model._model:
                p_fc = prophet_model._model.predict(future_ds)
                p_yhat = np.maximum(0, p_fc["yhat"].values)
                p_lower = np.maximum(0, p_fc["yhat_lower"].values)
                p_upper = np.maximum(0, p_fc["yhat_upper"].values)
            else:
                p_yhat = np.zeros(FORECAST_HORIZON)
                p_lower = np.zeros(FORECAST_HORIZON)
                p_upper = np.zeros(FORECAST_HORIZON)

        weekly_burn = max(p_yhat.mean(), 0.1)
        all_rows.extend(_make_forecast_rows(
            facility_id, antigen, "prophet",
            p_yhat, p_lower, p_upper,
            current_stock, weekly_burn, forecast_dates, lead_time,
            cv_mae_prophet if not np.isnan(cv_mae_prophet) else 7.0,
            weekly_consumption,
            generated_at=generated_at,
        ))

        # Stacked ensemble — uses the global meta-model trained during CV
        meta = _META_MODEL
        if meta is not None:
            # Need naive_seasonal predictions for the meta-feature row
            nv_sn = NaiveModel("seasonal_naive").fit(y_train)
            nv_sn_yhat = nv_sn.predict(FORECAST_HORIZON).values

            base_preds_dict = {
                "naive_last_value": nv_int["yhat"].values,
                "naive_seasonal":   nv_sn_yhat,
                "holt_winters":     hw_int["yhat"].values,
                "xgboost":          x_int["yhat"].values,
                "prophet":          p_yhat,
            }

            # Look up access_tier
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT access_tier FROM facilities WHERE facility_id=?", (facility_id,)
                ).fetchone()
            access_tier = row[0] if row else "rural_road"
            weekly_consumption = float(tp.get("weekly_consumption_baseline", 1.0)) if tp else 1.0

            meta_X = build_meta_features(
                base_preds_dict, current_stock, weekly_consumption,
                access_tier, list(range(1, FORECAST_HORIZON + 1)),
            )
            ens_int = meta.predict_interval(meta_X)
            e_yhat = ens_int["yhat"].values
            e_lower = ens_int["yhat_lower"].values
            e_upper = ens_int["yhat_upper"].values
            weekly_burn = max(float(e_yhat.mean()), 0.1)

            all_rows.extend(_make_forecast_rows(
                facility_id, antigen, "ensemble",
                e_yhat, e_lower, e_upper,
                current_stock, weekly_burn, forecast_dates, lead_time,
                cv_mae_ensemble if not np.isnan(cv_mae_ensemble) else 7.0,
                weekly_consumption,
                w_sarimax=None, w_prophet=None,
                generated_at=generated_at,
            ))

    importance = None
    if xgb_model is not None:
        try:
            importance = xgb_model.feature_importance()
        except Exception:
            importance = None
    return all_rows, importance


def main():
    with get_connection() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
    if "model_metrics" not in tables:
        print("ERROR: model_metrics table not found. Run walk_forward_cv.py first.")
        sys.exit(1)

    facilities = get_facilities()
    vaccines = get_vaccines()
    pairs = [(fid, ant) for fid in facilities["facility_id"] for ant in vaccines["antigen_code"]]

    print(f"Generating 8-week forecasts for {len(pairs)} series (XGBoost + Prophet ensemble)")

    with get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS forecast_output")
        conn.execute("DROP TABLE IF EXISTS feature_importance")
        conn.commit()

    all_rows = []
    importance_records = {}  # facility_id -> list of pd.Series across antigens

    for idx, (fid, ant) in enumerate(pairs):
        try:
            rows, importance = forecast_series(fid, ant)
            all_rows.extend(rows)
            if importance is not None and len(importance) > 0:
                importance_records.setdefault(fid, []).append(importance)
        except Exception as e:
            logger.error("Failed %s/%s: %s", fid, ant, e)
            continue

        if (idx + 1) % 20 == 0 or idx == len(pairs) - 1:
            print(f"  {idx + 1}/{len(pairs)} | {fid} {ant}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        write_forecasts(df)
        print(f"\nWrote {len(df)} forecast rows to forecast_output table.")

        ens_rows = df[df["model"] == "ensemble"]
        critical = (ens_rows["alert_status"] == "critical").sum()
        warning = (ens_rows["alert_status"] == "warning").sum()
        ok = (ens_rows["alert_status"] == "ok").sum()
        print(f"Alert summary (ensemble, all forecast weeks):")
        print(f"  Critical: {critical}  Warning: {warning}  OK: {ok}")

    # ── Aggregate + persist feature importance (facility-level mean) ───────
    if importance_records:
        from utils.db import write_feature_importance
        importance_rows = []
        gen_at = datetime.utcnow().isoformat()
        for fid, importance_list in importance_records.items():
            # Mean across antigens for this facility
            combined = pd.concat(importance_list, axis=1).mean(axis=1)
            top = combined.sort_values(ascending=False).head(15)
            for rank, (feat, imp) in enumerate(top.items(), start=1):
                importance_rows.append({
                    "facility_id": fid,
                    "feature": feat,
                    "importance": float(imp),
                    "rank": int(rank),
                    "generated_at": gen_at,
                })
        if importance_rows:
            imp_df = pd.DataFrame(importance_rows)
            write_feature_importance(imp_df)
            print(f"Wrote {len(imp_df)} feature_importance rows ({len(importance_records)} facilities × top-15).")


if __name__ == "__main__":
    main()
