"""
Walk-forward CV pipeline (XGBoost + Prophet + Naive + Holt-Winters)
with GLOBAL STACKING ensemble.

DATA SPLITS:
  Fold 1: Train 0-79   | Test 80-91
  Fold 2: Train 0-91   | Test 92-103
  Fold 3: Train 0-103  | Test 104-115
  Stacking dataset: train 0-127, predict 128-139 (used to train the meta-learner)
  Final test: weeks 140-155 (LOCKED — evaluated once with the trained meta-model)

Two passes:
  Pass 1 (per series):
    - Fit + score Naive, HW, XGBoost, Prophet on each CV fold
    - Fit on weeks 0-127, predict 128-139 with each base model
    - Persist those predictions in memory for the stacking dataset
    - Fit on weeks 0-139, predict 140-155 with each base model (held aside)

  Pass 2 (across all series):
    - Train ONE global StackingEnsemble meta-model on the pooled weight-window data
    - Apply it to each series' final-test base predictions to compute the
      stacked ensemble forecast on weeks 140-155
    - Score the ensemble against the locked final test
    - Save the meta-model to disk for production use
"""

import argparse
import logging
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import (
    get_facilities, get_vaccines, get_stock_series,
    write_model_metrics, get_connection, get_target_population,
)
from utils.features import build_exog_features
from models.baseline import NaiveModel, HoltWintersModel
from models.xgboost_model import XGBoostForecaster
from models.prophet_model import ProphetModel
from models.stacking import StackingEnsemble, build_meta_features, BASE_MODELS
from evaluation.metrics import compute_all_metrics

FOLDS = [
    {"fold": "1", "train_end": 180, "test_start": 180, "test_end": 200},
    {"fold": "2", "train_end": 220, "test_start": 220, "test_end": 240},
    {"fold": "3", "train_end": 260, "test_start": 260, "test_end": 280},
]
WEIGHT_WINDOW_TRAIN_END = 310
WEIGHT_WINDOW_TEST_START = 310
WEIGHT_WINDOW_TEST_END = 332
FINAL_TEST_START = 340
FINAL_TEST_END = 364
ZERO_INFLATION_THRESHOLD = 0.60


def is_zero_inflated(series: pd.Series) -> bool:
    return (series == 0).sum() / len(series) > ZERO_INFLATION_THRESHOLD


def _prophet_df(series_full: pd.DataFrame, train_end: int) -> pd.DataFrame:
    sub = series_full.iloc[:train_end]
    return pd.DataFrame({
        "ds": pd.to_datetime(sub["week_date"]),
        "y": sub["closing_stock"].values.astype(float),
    })


def _make_future_ds(series_full: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    sub = series_full.iloc[start:end]
    return pd.DataFrame({"ds": pd.to_datetime(sub["week_date"])})


def _get_lead_time(facility_id: str) -> float:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT lead_time_days_mean FROM facilities WHERE facility_id=?",
            (facility_id,)
        ).fetchone()
    return float(row[0]) if row else 14.0


def _predict_all_base(
    series_full, exog_full, train_end, test_start, test_end,
    facility_id, antigen, weekly_ceiling
) -> dict:
    """Train every base model on weeks 0..train_end and return predictions
    on weeks test_start..test_end. Returns dict[model_name] -> np.array."""
    h = test_end - test_start
    y_train = series_full["closing_stock"].iloc[:train_end].astype(float)
    dates_train = pd.to_datetime(series_full["week_date"].iloc[:train_end])
    exog_train = exog_full.iloc[:train_end]
    exog_test = exog_full.iloc[test_start:test_end]
    test_ds = _make_future_ds(series_full, test_start, test_end)

    preds = {}

    nv_lv = NaiveModel("last_value").fit(y_train)
    preds["naive_last_value"] = nv_lv.predict(h).values

    nv_sn = NaiveModel("seasonal_naive").fit(y_train)
    preds["naive_seasonal"] = nv_sn.predict(h).values

    hw = HoltWintersModel().fit(y_train)
    preds["holt_winters"] = hw.predict(h).values

    if is_zero_inflated(y_train):
        # XGBoost and Prophet skipped on zero-inflated series
        preds["xgboost"] = np.full(h, float(y_train.iloc[-1]))
        preds["prophet"] = np.full(h, float(y_train.iloc[-1]))
        return preds

    try:
        xgb_m = XGBoostForecaster(ceiling=weekly_ceiling, max_horizon=max(h, 12))
        xgb_m.fit(y_train, dates_train, exog_train, horizon=h)
        preds["xgboost"] = xgb_m.predict(h, exog_test).values
    except Exception as e:
        logger.warning("%s/%s XGBoost fit failed: %s", facility_id, antigen, e)
        preds["xgboost"] = np.full(h, float(y_train.iloc[-1]))

    try:
        ptrain = _prophet_df(series_full, train_end)
        prophet = ProphetModel(facility_id=facility_id, ceiling=weekly_ceiling)
        prophet.fit(ptrain)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if prophet._model:
                pf = prophet._model.predict(test_ds[["ds"]])
                preds["prophet"] = np.maximum(0, pf["yhat"].values)
            else:
                preds["prophet"] = np.full(h, float(y_train.iloc[-1]))
    except Exception as e:
        logger.warning("%s/%s Prophet fit failed: %s", facility_id, antigen, e)
        preds["prophet"] = np.full(h, float(y_train.iloc[-1]))

    return preds


def _metric_row(facility_id, antigen, model, fold, y_true, y_pred,
                lower, upper, stockouts, predicted_dts, lead_time, zero_inf):
    m = compute_all_metrics(y_true, y_pred, lower, upper, stockouts,
                             predicted_dts, lead_time)
    m.update({
        "facility_id": facility_id, "antigen": antigen,
        "model": model, "fold": fold,
        "zero_inflated": int(zero_inf), "sarimax_order": None,
        "fallback_used": 0, "w_sarimax": None, "w_prophet": None,
        "n_stockout_events": int(stockouts.sum()),
    })
    return m


def run_series_pass1(facility_id, antigen, series_full, exog_full,
                       all_metrics, stacking_X_rows, stacking_y, final_test_cache,
                       facilities_df):
    """First pass — base model CV + collect stacking rows + cache final-test preds."""
    y_full = series_full["closing_stock"].astype(float)
    week_dates_full = pd.to_datetime(series_full["week_date"])
    lead_time = _get_lead_time(facility_id)

    tp = get_target_population(facility_id, antigen)
    weekly_ceiling = None
    weekly_consumption = 1.0
    if tp:
        if tp.get("stock_needed_annual"):
            weekly_ceiling = float(tp["stock_needed_annual"]) / 52 * 1.5
        weekly_consumption = float(tp.get("weekly_consumption_baseline", 1.0))

    fac_row = facilities_df[facilities_df["facility_id"] == facility_id]
    access_tier = fac_row["access_tier"].iloc[0] if not fac_row.empty else "rural_road"
    zero_inf = is_zero_inflated(y_full)

    # ── CV folds 1, 2, 3 ────────────────────────────────────────────────────
    for fold_def in FOLDS:
        ts_start = fold_def["test_start"]
        ts_end = fold_def["test_end"]
        h = ts_end - ts_start
        y_test = y_full.iloc[ts_start:ts_end]
        actual_so = (series_full["is_stockout"].iloc[ts_start:ts_end].astype(bool)
                     .reset_index(drop=True))

        preds = _predict_all_base(
            series_full, exog_full, fold_def["train_end"],
            ts_start, ts_end, facility_id, antigen, weekly_ceiling,
        )

        for model_name, yhat in preds.items():
            yhat = np.asarray(yhat)
            # Simple bounds for interval (1-sigma proxy until conformal kicks in)
            std_proxy = float(np.std(y_full.iloc[:fold_def["train_end"]].values)) * 1.28
            lower = np.maximum(0, yhat - std_proxy)
            upper = yhat + std_proxy
            predicted_dts = pd.Series(
                [float(y_test.iloc[0]) / max(yhat.mean(), 0.1) * 7] * h
            )
            all_metrics.append(_metric_row(
                facility_id, antigen, model_name, fold_def["fold"],
                y_test.values, yhat, lower, upper, actual_so, predicted_dts,
                lead_time, zero_inf,
            ))

    # ── Stacking dataset: train 0-127, predict 128-139 ──────────────────────
    h_w = WEIGHT_WINDOW_TEST_END - WEIGHT_WINDOW_TEST_START
    y_w_test = y_full.iloc[WEIGHT_WINDOW_TEST_START:WEIGHT_WINDOW_TEST_END]
    base_w_preds = _predict_all_base(
        series_full, exog_full, WEIGHT_WINDOW_TRAIN_END,
        WEIGHT_WINDOW_TEST_START, WEIGHT_WINDOW_TEST_END,
        facility_id, antigen, weekly_ceiling,
    )
    current_stock_at_w = float(y_full.iloc[WEIGHT_WINDOW_TRAIN_END - 1])
    horizon_steps = list(range(1, h_w + 1))
    meta_X = build_meta_features(
        base_w_preds, current_stock_at_w, weekly_consumption,
        access_tier, horizon_steps,
    )
    stacking_X_rows.append(meta_X)
    stacking_y.extend(y_w_test.values.tolist())

    # ── Final test (weeks 140-155) — base models only here, ensemble in pass 2
    h_f = FINAL_TEST_END - FINAL_TEST_START
    y_f_test = y_full.iloc[FINAL_TEST_START:FINAL_TEST_END]
    actual_so_f = (series_full["is_stockout"].iloc[FINAL_TEST_START:FINAL_TEST_END]
                   .astype(bool).reset_index(drop=True))
    base_f_preds = _predict_all_base(
        series_full, exog_full, WEIGHT_WINDOW_TEST_END,  # 0-139
        FINAL_TEST_START, FINAL_TEST_END,
        facility_id, antigen, weekly_ceiling,
    )
    current_stock_at_f = float(y_full.iloc[WEIGHT_WINDOW_TEST_END - 1])

    # Score each base model on final test
    for model_name, yhat in base_f_preds.items():
        yhat = np.asarray(yhat)
        std_proxy = float(np.std(y_full.iloc[:WEIGHT_WINDOW_TEST_END].values)) * 1.28
        lower = np.maximum(0, yhat - std_proxy)
        upper = yhat + std_proxy
        predicted_dts = pd.Series(
            [float(y_f_test.iloc[0]) / max(yhat.mean(), 0.1) * 7] * h_f
        )
        # Naive last_value uses old name for consistency with dashboard
        out_name = "naive_last_value" if model_name == "naive_last_value" else \
                   "naive_seasonal_naive" if model_name == "naive_seasonal" else \
                   model_name
        all_metrics.append(_metric_row(
            facility_id, antigen, out_name, "final",
            y_f_test.values, yhat, lower, upper, actual_so_f, predicted_dts,
            lead_time, zero_inf,
        ))

    # Cache for pass 2
    final_test_cache[(facility_id, antigen)] = {
        "y_true": y_f_test.values,
        "actual_so": actual_so_f,
        "base_preds": base_f_preds,
        "current_stock": current_stock_at_f,
        "weekly_consumption": weekly_consumption,
        "access_tier": access_tier,
        "lead_time": lead_time,
        "zero_inf": zero_inf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    facilities = get_facilities()
    vaccines = get_vaccines()

    pairs = [(fid, ant) for fid in facilities["facility_id"] for ant in vaccines["antigen_code"]]
    if args.sample:
        pairs = pairs[:args.sample]

    print(f"=== Pass 1: base CV + collect stacking dataset ({len(pairs)} series) ===")
    all_metrics = []
    stacking_X_rows = []
    stacking_y = []
    final_test_cache = {}

    for idx, (fid, ant) in enumerate(pairs):
        try:
            series_full = get_stock_series(fid, ant)
            exog_full = build_exog_features(fid, n_weeks=FINAL_TEST_END, antigen=ant)
            if len(series_full) < FINAL_TEST_END:
                continue
            run_series_pass1(
                fid, ant, series_full, exog_full,
                all_metrics, stacking_X_rows, stacking_y, final_test_cache,
                facilities,
            )
        except Exception as e:
            logger.error("Failed %s/%s: %s", fid, ant, e)
            continue

        if (idx + 1) % 10 == 0 or idx == len(pairs) - 1:
            print(f"  {idx + 1}/{len(pairs)} | {fid} {ant}")

    print(f"\n=== Pass 2: train global stacking meta-learner ===")
    if not stacking_X_rows:
        print("No stacking data collected — exiting.")
        return

    stacking_X = pd.concat(stacking_X_rows, ignore_index=True)
    stacking_y_arr = np.array(stacking_y, dtype=float)
    print(f"  Stacking dataset: {stacking_X.shape[0]} rows × {stacking_X.shape[1]} features")

    meta = StackingEnsemble().fit(stacking_X, stacking_y_arr)
    meta.save()
    print(f"  Meta-model trained. Conformal half-width = {meta._conformal_width:.2f} doses")
    fi = meta.feature_importance().head(8)
    print(f"  Top meta-features:\n{fi.to_string()}")

    print(f"\n=== Pass 3: apply stacked ensemble to final test ===")
    h_f = FINAL_TEST_END - FINAL_TEST_START
    horizon_steps = list(range(1, h_f + 1))

    for (fid, ant), cache in final_test_cache.items():
        meta_X = build_meta_features(
            cache["base_preds"], cache["current_stock"],
            cache["weekly_consumption"], cache["access_tier"], horizon_steps,
        )
        ens_int = meta.predict_interval(meta_X)
        yhat = ens_int["yhat"].values
        lower = ens_int["yhat_lower"].values
        upper = ens_int["yhat_upper"].values

        predicted_dts = pd.Series(
            [float(cache["y_true"][0]) / max(yhat.mean(), 0.1) * 7] * h_f
        )
        all_metrics.append(_metric_row(
            fid, ant, "ensemble", "final",
            cache["y_true"], yhat, lower, upper, cache["actual_so"],
            predicted_dts, cache["lead_time"], cache["zero_inf"],
        ))

    print(f"\n=== Writing metrics ===")
    metrics_df = pd.DataFrame(all_metrics)
    with get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS model_metrics")
        conn.commit()
    write_model_metrics(metrics_df)
    print(f"Wrote {len(metrics_df)} metric rows.\n")

    # Summary
    df = pd.DataFrame(all_metrics)
    cv_df = df[df["fold"].isin(["1", "2", "3"])]
    final_df = df[df["fold"] == "final"]
    print("=" * 70)
    print("CV SUMMARY — Mean MAE (folds 1-3)")
    print("=" * 70)
    print(cv_df.groupby("model")["mae"].agg(["mean", "std"]).round(2).to_string())
    print()
    print("=" * 70)
    print("FINAL TEST (weeks 140-155)")
    print("=" * 70)
    print(final_df.groupby("model")[["mae", "interval_coverage",
                                      "stockout_detection_rate"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
