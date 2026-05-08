"""
XGBoost time-series model with DIRECT MULTI-STEP forecasting.

Replaces the previous recursive XGBoost. Direct multi-step is the
textbook-correct approach for ML time-series — it avoids error compounding
because each horizon h has its own dedicated model trained on actual lag
values (never on its own previous predictions).

Design (one trained estimator per horizon step h ∈ 1..H):
  Training pair (X_t, y_{t+h}) for each historical t where:
    X_t = lag features {y_{t-1..t-52}} + rolling stats + calendar + exog_t
    y_{t+h} = the actual stock h weeks later
  At forecast time:
    X_T uses only OBSERVED y values up to T plus the future exog_T+h.
    Each horizon-h model directly outputs the prediction for week T+h.

This produces H models per series. With H=12 (CV) or H=8 (production),
we have 12 × 210 series = 2,520 quick fits per CV pass.

Quantile-regression intervals: a 0.10 and 0.90 quantile model is fit per
horizon, giving an 80% prediction interval at every step.

Feature engineering ref: Time Series Forecasting in Python, Ch. 18-19
(lag features, rolling stats, cyclical calendar encodings).
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


LAG_STEPS = [1, 2, 4, 8, 12, 26, 52]
ROLL_MEAN_WINDOWS = [4, 12, 26]
ROLL_STD_WINDOWS = [4, 12]


def _calendar_features(week_dates: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(week_dates)
    month = dt.dt.month
    woy = dt.dt.isocalendar().week.astype(int)
    return pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "woy_sin": np.sin(2 * np.pi * woy / 52),
        "woy_cos": np.cos(2 * np.pi * woy / 52),
    }, index=dt.index)


def _build_lag_and_roll_features(y: pd.Series) -> pd.DataFrame:
    """Lag and rolling statistics — using only past values (shift by 1)."""
    feats = pd.DataFrame(index=y.index)
    for lag in LAG_STEPS:
        feats[f"lag_{lag}"] = y.shift(lag)
    for w in ROLL_MEAN_WINDOWS:
        feats[f"rmean_{w}"] = y.shift(1).rolling(window=w, min_periods=1).mean()
    for w in ROLL_STD_WINDOWS:
        feats[f"rstd_{w}"] = y.shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    return feats


def build_feature_matrix(
    y: pd.Series, week_dates: pd.Series, exog: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature matrix for direct training."""
    y = y.reset_index(drop=True).astype(float)
    week_dates = pd.to_datetime(week_dates).reset_index(drop=True)
    exog = exog.reset_index(drop=True).select_dtypes(include=[np.number])

    lag_feats = _build_lag_and_roll_features(y)
    cal = _calendar_features(week_dates)
    return pd.concat([lag_feats, cal, exog], axis=1)


class XGBoostForecaster:
    """
    Per-series XGBoost forecaster using DIRECT MULTI-STEP forecasting.

    For a forecast horizon H:
      - Fit H point-estimate models (squared error)
      - Fit H lower-quantile models (alpha=0.10) for PI lower bound
      - Fit H upper-quantile models (alpha=0.90) for PI upper bound
      Total: 3 × H models. With H=12 → 36 small models per series.
    """

    def __init__(self, ceiling: float = None, max_horizon: int = 16,
                 n_estimators: int = 500, max_depth: int = 3,
                 learning_rate: float = 0.01, subsample: float = 0.8,
                 early_stopping_rounds: int = 50,
                 outlier_clip_pct: float = 0.99):
        self.ceiling = ceiling
        self.max_horizon = max_horizon
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.outlier_clip_pct = outlier_clip_pct

        # Dict: horizon h -> point estimator (squared-error). Quantile models
        # replaced with conformal prediction intervals (calibrated on OOF residuals).
        self._models = {}
        # Dict: horizon h -> conformal half-width (80% PI)
        self._conformal_widths = {}
        self._feature_cols = None
        self._y_max = None
        self._train_y = None
        self._train_dates = None
        self._train_exog = None

    def _make_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def _bound(self, arr: np.ndarray) -> np.ndarray:
        arr = np.maximum(0, arr)
        if self.ceiling is not None:
            arr = np.minimum(arr, self.ceiling)
        # Hard cap at 2× historical max — prevents extrapolation blow-ups
        if self._y_max is not None:
            arr = np.minimum(arr, self._y_max * 2.0)
        return arr

    def fit(self, y: pd.Series, week_dates: pd.Series, exog: pd.DataFrame,
            horizon: int = None) -> "XGBoostForecaster":
        """
        Train one POINT-ESTIMATE XGBoost model per horizon h ∈ {1..horizon}.
        Prediction intervals come from conformal calibration: residuals on a
        validation slice define the 80% PI half-width per horizon. This
        replaces the unstable quantile-XGBoost (which gave 8.9% coverage)
        with a calibrated nonparametric interval (~80% by construction).
        """
        H = horizon or self.max_horizon
        y_clean = y.reset_index(drop=True).astype(float)

        # Outlier clipping (Rob Mulla-style) — Poisson noise creates extreme
        # weeks that drag tree splits. Cap at 99th percentile of training data.
        if self.outlier_clip_pct < 1.0 and len(y_clean) >= 20:
            cap = float(np.percentile(y_clean, self.outlier_clip_pct * 100))
            y_clean = y_clean.clip(upper=cap)

        self._train_y = y_clean
        self._train_dates = pd.to_datetime(week_dates).reset_index(drop=True)
        self._train_exog = exog.reset_index(drop=True).copy()
        self._y_max = float(self._train_y.max())

        feats_full = build_feature_matrix(self._train_y, self._train_dates, self._train_exog)

        for h in range(1, H + 1):
            target = self._train_y.shift(-h)
            valid_idx = feats_full.dropna().index.intersection(target.dropna().index)
            if len(valid_idx) < 25:
                continue

            X = feats_full.loc[valid_idx]
            y_h = target.loc[valid_idx]
            self._feature_cols = list(X.columns)

            # 80/20 chronological split — last 20% is BOTH the early-stopping
            # set AND the conformal calibration set
            split = int(len(X) * 0.80)
            X_tr, X_val = X.iloc[:split], X.iloc[split:]
            y_tr, y_val = y_h.iloc[:split], y_h.iloc[split:]

            model = self._make_model()
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            except Exception as e:
                logger.warning("XGB h=%d fit failed: %s", h, e)
                self._models[h] = None
                self._conformal_widths[h] = None
                continue

            # Conformal interval: 80th-percentile of |residual| on val set
            try:
                val_pred = model.predict(X_val)
                residuals = np.abs(y_val.values - val_pred)
                if len(residuals) >= 3:
                    width = float(np.percentile(residuals, 80))
                else:
                    width = float(np.std(y_tr.values)) * 1.28
            except Exception:
                width = float(np.std(y_tr.values)) * 1.28

            self._models[h] = model
            self._conformal_widths[h] = max(width, 0.5)

        return self

    def _build_forecast_feature_row(self, future_exog_row: pd.Series,
                                     future_date: pd.Timestamp) -> pd.DataFrame:
        """
        At forecast time T+h, lag features still use the LAST observed values
        from the training history (positions ..., T-1, T). Each horizon model
        was trained to map this set of features directly to y_{T+h}, so the
        same input row is used for ALL horizons.
        """
        y = self._train_y
        n = len(y)
        row = {}
        # Lag features point back to the last observed week T
        for lag in LAG_STEPS:
            row[f"lag_{lag}"] = float(y.iloc[-lag]) if n >= lag else float(y.iloc[0])
        for w in ROLL_MEAN_WINDOWS:
            row[f"rmean_{w}"] = float(y.iloc[-w:].mean()) if n >= 1 else 0.0
        for w in ROLL_STD_WINDOWS:
            tail = y.iloc[-w:] if n >= 2 else pd.Series([0.0, 0.0])
            row[f"rstd_{w}"] = float(tail.std()) if len(tail) > 1 else 0.0

        # Calendar — based on the future date for THIS horizon step
        row["month_sin"] = float(np.sin(2 * np.pi * future_date.month / 12))
        row["month_cos"] = float(np.cos(2 * np.pi * future_date.month / 12))
        woy = int(future_date.isocalendar().week)
        row["woy_sin"] = float(np.sin(2 * np.pi * woy / 52))
        row["woy_cos"] = float(np.cos(2 * np.pi * woy / 52))

        for col in future_exog_row.index:
            val = future_exog_row[col]
            if isinstance(val, (int, float, np.integer, np.floating)):
                row[col] = float(val)

        if self._feature_cols:
            for col in self._feature_cols:
                row.setdefault(col, 0.0)
            return pd.DataFrame([row])[self._feature_cols]
        return pd.DataFrame([row])

    def _predict_one_horizon(self, model, h: int, future_exog_row: pd.Series,
                              future_date: pd.Timestamp) -> float:
        if model is None:
            return float(self._train_y.iloc[-1]) if self._train_y is not None else 0.0
        X = self._build_forecast_feature_row(future_exog_row, future_date)
        try:
            return float(model.predict(X)[0])
        except Exception as e:
            logger.warning("XGB predict h=%d failed: %s", h, e)
            return float(self._train_y.iloc[-1]) if self._train_y is not None else 0.0

    def predict(self, h: int, future_exog: pd.DataFrame) -> pd.Series:
        if not self._models or self._train_y is None:
            return pd.Series([0.0] * h)
        future_exog = future_exog.reset_index(drop=True)
        last_date = self._train_dates.iloc[-1]
        preds = []
        for step in range(1, h + 1):
            future_date = pd.Timestamp(last_date) + pd.Timedelta(weeks=step)
            exog_row = future_exog.iloc[step - 1] if step - 1 < len(future_exog) else future_exog.iloc[-1]
            model = self._models.get(step)
            if model is None:
                preds.append(float(self._train_y.iloc[-1]))
                continue
            preds.append(self._predict_one_horizon(model, step, exog_row, future_date))
        return pd.Series(self._bound(np.array(preds)))

    def predict_interval(self, h: int, future_exog: pd.DataFrame,
                          alpha: float = 0.80) -> pd.DataFrame:
        """80% PI from conformal calibration: ŷ ± conformal_width_h."""
        if not self._models or self._train_y is None:
            zeros = np.zeros(h)
            return pd.DataFrame({"yhat": zeros, "yhat_lower": zeros, "yhat_upper": zeros})
        future_exog = future_exog.reset_index(drop=True)
        last_date = self._train_dates.iloc[-1]
        means, widths = [], []
        for step in range(1, h + 1):
            future_date = pd.Timestamp(last_date) + pd.Timedelta(weeks=step)
            exog_row = future_exog.iloc[step - 1] if step - 1 < len(future_exog) else future_exog.iloc[-1]
            model = self._models.get(step)
            if model is None:
                last_val = float(self._train_y.iloc[-1])
                means.append(last_val)
                widths.append(max(last_val * 0.20, 1.0))
                continue
            yhat = self._predict_one_horizon(model, step, exog_row, future_date)
            means.append(yhat)
            widths.append(self._conformal_widths.get(step, 1.0))

        means = self._bound(np.array(means))
        widths = np.array(widths)
        lowers = np.maximum(0, means - widths)
        uppers = means + widths
        if self.ceiling is not None:
            uppers = np.minimum(uppers, self.ceiling)
        if self._y_max is not None:
            uppers = np.minimum(uppers, self._y_max * 2.0)

        return pd.DataFrame({"yhat": means, "yhat_lower": lowers, "yhat_upper": uppers})

    def feature_importance(self) -> pd.Series:
        if not self._models:
            return pd.Series(dtype=float)
        importances = []
        for h, m in self._models.items():
            if m is not None:
                try:
                    importances.append(pd.Series(m.feature_importances_, index=self._feature_cols))
                except Exception:
                    pass
        if not importances:
            return pd.Series(dtype=float)
        return pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
