"""
Inverse-MAE weighted ensemble of XGBoost + Prophet.
Replaces the previous SARIMAX+Prophet ensemble.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """
    Inverse-MAE weighted combination of XGBoost and Prophet.
    Weights are computed from the ensemble weight window (weeks 128-139).
    """

    def __init__(self):
        self._xgb = None
        self._prophet = None
        self.w_xgb = 0.5
        self.w_prophet = 0.5
        self.ceiling = None

    # Backwards-compatible aliases (kept so old code calling .w_sarimax doesn't break)
    @property
    def w_sarimax(self): return self.w_xgb
    @w_sarimax.setter
    def w_sarimax(self, v): self.w_xgb = float(v)

    def fit(self, xgb_model, prophet_model,
            val_series: pd.Series, val_exog: pd.DataFrame,
            val_ds: pd.DataFrame = None) -> "WeightedEnsemble":
        """
        Compute ensemble weights from validation window (weeks 128-139).
        """
        self._xgb = xgb_model
        self._prophet = prophet_model
        h = len(val_series)
        y_true = val_series.values.astype(float)

        mae_x = float("inf")
        mae_p = float("inf")

        if xgb_model is not None:
            try:
                x_preds = xgb_model.predict(h, val_exog).values
                mae_x = float(np.mean(np.abs(y_true - x_preds)))
            except Exception as e:
                logger.warning("Ensemble: XGBoost predict failed: %s", e)

        if prophet_model is not None and val_ds is not None:
            try:
                p_df = val_ds[["ds"]].copy()
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p_forecast = prophet_model._model.predict(p_df)
                p_preds = np.maximum(0, p_forecast["yhat"].values)
                mae_p = float(np.mean(np.abs(y_true - p_preds)))
            except Exception as e:
                logger.warning("Ensemble: Prophet predict failed: %s", e)

        if np.isinf(mae_x) and np.isinf(mae_p):
            self.w_xgb = 0.5
            self.w_prophet = 0.5
        elif np.isinf(mae_x):
            self.w_xgb = 0.0
            self.w_prophet = 1.0
        elif np.isinf(mae_p):
            self.w_xgb = 1.0
            self.w_prophet = 0.0
        elif mae_x == 0 and mae_p == 0:
            self.w_xgb = 0.5
            self.w_prophet = 0.5
        else:
            inv_x = 1.0 / max(mae_x, 1e-9)
            inv_p = 1.0 / max(mae_p, 1e-9)
            total = inv_x + inv_p
            self.w_xgb = inv_x / total
            self.w_prophet = inv_p / total

        return self

    def predict(self, h: int, future_exog: pd.DataFrame,
                future_ds: pd.DataFrame = None) -> pd.DataFrame:
        x_preds = np.zeros(h)
        p_preds = np.zeros(h)
        x_lower = np.zeros(h)
        x_upper = np.zeros(h)
        p_lower = np.zeros(h)
        p_upper = np.zeros(h)

        if self._xgb is not None and self.w_xgb > 0:
            try:
                x_df = self._xgb.predict_interval(h, future_exog)
                x_preds = x_df["yhat"].values
                x_lower = x_df["yhat_lower"].values
                x_upper = x_df["yhat_upper"].values
            except Exception as e:
                logger.warning("Ensemble XGBoost predict failed: %s", e)

        if self._prophet is not None and self.w_prophet > 0:
            try:
                if future_ds is not None:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p_forecast = self._prophet._model.predict(future_ds[["ds"]])
                    p_preds = np.maximum(0, p_forecast["yhat"].values[:h])
                    p_lower = np.maximum(0, p_forecast["yhat_lower"].values[:h])
                    p_upper = np.maximum(0, p_forecast["yhat_upper"].values[:h])
                else:
                    p_df = self._prophet.predict_interval(h)
                    p_preds = p_df["yhat"].values
                    p_lower = p_df["yhat_lower"].values
                    p_upper = p_df["yhat_upper"].values
            except Exception as e:
                logger.warning("Ensemble Prophet predict failed: %s", e)

        yhat = self.w_xgb * x_preds + self.w_prophet * p_preds
        lower = np.minimum(x_lower, p_lower)
        upper = np.maximum(x_upper, p_upper)

        if self.ceiling is not None:
            yhat = np.minimum(yhat, self.ceiling)
            upper = np.minimum(upper, self.ceiling)
        yhat = np.maximum(0, yhat)
        lower = np.maximum(0, lower)

        return pd.DataFrame({
            "yhat": yhat,
            "yhat_lower": lower,
            "yhat_upper": upper,
            "w_xgb": self.w_xgb,
            "w_prophet": self.w_prophet,
        })

    def predict_interval(self, h: int, future_exog: pd.DataFrame,
                          future_ds: pd.DataFrame = None) -> pd.DataFrame:
        return self.predict(h, future_exog, future_ds)
