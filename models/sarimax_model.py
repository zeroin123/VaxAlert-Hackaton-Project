import logging
import threading
import numpy as np
import pandas as pd
import pmdarima as pm

logger = logging.getLogger(__name__)

SARIMAX_TIMEOUT_SECONDS = 10
FALLBACK_ORDER = (1, 1, 1)
FALLBACK_SEASONAL = (0, 0, 0, 0)

EXOG_COLS_BASE = [
    "is_rainy_season",
    "demand_shock_multiplier",
    "supply_shock_multiplier",
    "birth_seasonality_factor",
    "log_weeks_since_resupply",
]


def _get_exog_cols(antigen: str) -> list:
    cols = EXOG_COLS_BASE.copy()
    if antigen == "MCV":
        cols.append("is_measles_sia")
    if antigen == "OPV":
        cols.append("is_polio_snid")
    return cols


def _run_auto_arima(series, exog, result_holder, exc_holder):
    try:
        model = pm.auto_arima(
            series,
            exogenous=exog,
            start_p=1, max_p=2,
            start_q=0, max_q=1,
            d=None,
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            n_jobs=1,
            error_action="ignore",
            with_oob_score=False,
            suppress_warnings=True,
        )
        result_holder[0] = model
    except Exception as e:
        exc_holder[0] = e


class SARIMAXModel:
    """
    Auto-order SARIMAX using pmdarima.auto_arima with exogenous features.
    """

    def __init__(self, antigen: str = "PENTA", ceiling: float = None):
        self.antigen = antigen
        self.ceiling = ceiling
        self._model = None
        self._exog_cols = _get_exog_cols(antigen)
        self.fallback_used = False
        self._order = None

    def _apply_ceiling(self, arr: np.ndarray) -> np.ndarray:
        arr = np.maximum(0, arr)
        if self.ceiling is not None:
            arr = np.minimum(arr, self.ceiling)
        return arr

    def _fit_with_order(self, series: np.ndarray, exog: np.ndarray,
                         order: tuple, seasonal_order: tuple) -> object:
        model = pm.ARIMA(
            order=order,
            seasonal_order=seasonal_order,
            suppress_warnings=True,
        )
        model.fit(series, exogenous=exog)
        return model

    def fit(self, series: pd.Series, exog: pd.DataFrame) -> "SARIMAXModel":
        y = series.values.astype(float)
        X = exog[self._exog_cols].values.astype(float)

        result = [None]
        exc = [None]

        t = threading.Thread(target=_run_auto_arima, args=(y, X, result, exc), daemon=True)
        t.start()
        t.join(timeout=SARIMAX_TIMEOUT_SECONDS)

        if t.is_alive() or result[0] is None:
            logger.warning("SARIMAX auto_arima timed out or failed for antigen=%s; using fallback order", self.antigen)
            self.fallback_used = True
            try:
                self._model = self._fit_with_order(y, X, FALLBACK_ORDER, FALLBACK_SEASONAL)
            except Exception as e2:
                logger.error("Fallback SARIMAX also failed: %s", e2)
                self._model = None
        else:
            self._model = result[0]
            self.fallback_used = False

        if self._model is not None:
            try:
                o = self._model.order
                so = self._model.seasonal_order
                self._order = (o[0], o[1], o[2], so[0], so[1], so[2])
            except Exception:
                self._order = FALLBACK_ORDER + (0, 0, 0)

        return self

    def fit_with_cached_order(self, series: pd.Series, exog: pd.DataFrame,
                               order: tuple, seasonal_order: tuple) -> "SARIMAXModel":
        """Refit coefficients only using a pre-determined order (no grid search)."""
        y = series.values.astype(float)
        X = exog[self._exog_cols].values.astype(float)
        try:
            self._model = self._fit_with_order(y, X, order, seasonal_order)
            self.fallback_used = False
            self._order = order[:3] + seasonal_order[:3]
        except Exception as e:
            logger.warning("Cached-order SARIMAX fit failed (%s); falling back to ARIMA(1,1,1)", e)
            self.fallback_used = True
            try:
                self._model = self._fit_with_order(y, X, FALLBACK_ORDER, FALLBACK_SEASONAL)
            except Exception:
                self._model = None
        return self

    def predict(self, h: int, future_exog: pd.DataFrame) -> pd.Series:
        if self._model is None:
            return pd.Series([0.0] * h)
        X_future = future_exog[self._exog_cols].values.astype(float)
        try:
            preds = self._model.predict(n_periods=h, exogenous=X_future)
            return pd.Series(self._apply_ceiling(preds))
        except Exception as e:
            logger.warning("SARIMAX predict failed: %s", e)
            return pd.Series([0.0] * h)

    def predict_interval(self, h: int, future_exog: pd.DataFrame,
                          alpha: float = 0.80) -> pd.DataFrame:
        if self._model is None:
            return pd.DataFrame({"yhat": [0.0]*h, "yhat_lower": [0.0]*h, "yhat_upper": [0.0]*h})
        X_future = future_exog[self._exog_cols].values.astype(float)
        try:
            preds, conf = self._model.predict(
                n_periods=h, exogenous=X_future, return_conf_int=True, alpha=1 - alpha
            )
            yhat = self._apply_ceiling(preds)
            lower = self._apply_ceiling(conf[:, 0])
            upper = self._apply_ceiling(conf[:, 1])
            return pd.DataFrame({"yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})
        except Exception as e:
            logger.warning("SARIMAX predict_interval failed: %s", e)
            yhat = self.predict(h, future_exog).values
            return pd.DataFrame({"yhat": yhat, "yhat_lower": yhat, "yhat_upper": yhat})

    def get_aic(self) -> float:
        if self._model is None:
            return float("inf")
        try:
            return self._model.aic()
        except Exception:
            return float("inf")

    def get_order(self) -> tuple:
        return self._order or FALLBACK_ORDER + (0, 0, 0)
