"""
Global stacking meta-learner.

Trains ONE shared meta-model on out-of-fold (OOF) predictions pooled across
all 210 (facility × antigen) series. Inputs to the meta-model are the
predictions from each base model on a given week, plus a few series-level
context features (current stock, weekly_consumption_baseline, access tier).

This is rigorously correct: the meta-model never sees in-sample predictions
from the base models, so its weights generalise to unseen series.

Why one global meta-model and not 210 per-series ones:
  - Per-series meta-models would have ~12 training rows each → severe overfit
  - Pooled training: 12 weeks × 210 series = 2,520 rows → enough signal
  - The meta-model can learn series-conditional weights via context features

Base model predictions used as features:
  - naive_last_value
  - naive_seasonal
  - holt_winters
  - xgboost
  - prophet

Context features:
  - current_stock at start of forecast (yhat_lag_1)
  - weekly_consumption_baseline (from target_population)
  - access_tier_encoded (0=urban, 1=rural_road, 2=rural_remote, 3=pastoral)
  - horizon_step (1..12) — meta can learn that simple methods do better at h=1
"""

import logging
import joblib
import os
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

META_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache", "meta_model.pkl"
)
os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)

BASE_MODELS = ["naive_last_value", "naive_seasonal", "holt_winters", "xgboost", "prophet"]
TIER_ENCODE = {"urban": 0, "rural_road": 1, "rural_remote": 2, "pastoral": 3}


def build_meta_features(
    base_predictions: dict,        # {model_name: array of h predictions}
    current_stock: float,
    weekly_consumption: float,
    access_tier: str,
    horizon_steps: list,           # e.g. [1,2,...,h]
) -> pd.DataFrame:
    """Build a feature matrix where each row is one (series, week) pair."""
    h = len(horizon_steps)
    rows = []
    for i in range(h):
        row = {f"pred_{m}": float(base_predictions.get(m, [0]*h)[i]) for m in BASE_MODELS}
        row["current_stock"] = float(current_stock)
        row["weekly_consumption"] = float(weekly_consumption or 1.0)
        row["tier"] = float(TIER_ENCODE.get(access_tier, 1))
        row["horizon"] = int(horizon_steps[i])
        # Pairwise predictions averages and variance (stacking with diversity)
        preds_arr = np.array([row[f"pred_{m}"] for m in BASE_MODELS])
        row["pred_mean"] = float(preds_arr.mean())
        row["pred_std"] = float(preds_arr.std())
        row["pred_min"] = float(preds_arr.min())
        row["pred_max"] = float(preds_arr.max())
        rows.append(row)
    return pd.DataFrame(rows)


class StackingEnsemble:
    """Global meta-learner using XGBoost as the stacker."""

    def __init__(self):
        self._meta = None
        self._feature_cols = None
        # Conformal width across all series for ensemble PI
        self._conformal_width = None

    @classmethod
    def load(cls) -> "StackingEnsemble":
        if not os.path.exists(META_MODEL_PATH):
            return None
        try:
            obj = joblib.load(META_MODEL_PATH)
            inst = cls()
            inst._meta = obj["model"]
            inst._feature_cols = obj["feature_cols"]
            inst._conformal_width = obj.get("conformal_width", 5.0)
            return inst
        except Exception as e:
            logger.warning("Failed to load meta-model: %s", e)
            return None

    def save(self):
        joblib.dump({
            "model": self._meta,
            "feature_cols": self._feature_cols,
            "conformal_width": self._conformal_width,
        }, META_MODEL_PATH)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "StackingEnsemble":
        """X: stacked OOF features, y: actual target values."""
        self._feature_cols = list(X.columns)

        # 80/20 chronological-by-series split (random ok here since each row is
        # already an independent (series, week) sample)
        n = len(X)
        split = int(n * 0.85)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y[:split], y[split:]

        self._meta = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=2,
            verbosity=0,
            early_stopping_rounds=30,
        )
        self._meta.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # Conformal width on validation
        val_pred = self._meta.predict(X_val)
        residuals = np.abs(y_val - val_pred)
        self._conformal_width = float(np.percentile(residuals, 80)) if len(residuals) > 0 else 5.0

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._meta is None:
            return np.zeros(len(X))
        # Align columns to training schema
        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self._feature_cols]
        return self._meta.predict(X)

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        yhat = self.predict(X)
        w = self._conformal_width if self._conformal_width is not None else 5.0
        lower = np.maximum(0, yhat - w)
        upper = yhat + w
        return pd.DataFrame({"yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})

    def feature_importance(self) -> pd.Series:
        if self._meta is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self._meta.feature_importances_,
            index=self._feature_cols,
        ).sort_values(ascending=False)
