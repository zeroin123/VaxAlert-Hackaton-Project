import logging
import numpy as np
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


class ProphetModel:
    """
    Facebook Prophet configured for weekly EPI stock consumption data.
    """

    def __init__(self, facility_id: str = None, ceiling: float = None):
        self.facility_id = facility_id
        self.ceiling = ceiling
        self._model = None
        self._train_df = None

    def _apply_ceiling(self, arr: np.ndarray) -> np.ndarray:
        arr = np.maximum(0, arr)
        if self.ceiling is not None:
            arr = np.minimum(arr, self.ceiling)
        return arr

    def _build_model(self) -> Prophet:
        from utils.features import build_prophet_events
        events = None
        if self.facility_id:
            try:
                events = build_prophet_events(self.facility_id)
                if events.empty:
                    events = None
            except Exception:
                events = None

        model = Prophet(
            growth="linear",
            seasonality_mode="additive",
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=15,
            interval_width=0.80,
            mcmc_samples=0,
            n_changepoints=20,
            holidays=events,
        )
        model.add_seasonality(name="annual", period=52.18, fourier_order=5)
        return model

    def fit(self, series: pd.DataFrame) -> "ProphetModel":
        """series must have columns ds (datetime) and y (float)."""
        df = series[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)
        self._train_df = df

        self._model = self._build_model()
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(df)
        except Exception as e:
            logger.error("Prophet fit failed: %s", e)
            self._model = None

        return self

    def _make_future(self, h: int) -> pd.DataFrame:
        if self._train_df is None:
            return pd.DataFrame()
        last_ds = self._train_df["ds"].max()
        future_dates = pd.date_range(
            start=last_ds + pd.Timedelta(weeks=1), periods=h, freq="W-MON"
        )
        return pd.DataFrame({"ds": future_dates})

    def predict(self, h: int) -> pd.Series:
        if self._model is None:
            return pd.Series([0.0] * h)
        future = self._make_future(h)
        try:
            forecast = self._model.predict(future)
            return pd.Series(self._apply_ceiling(forecast["yhat"].values))
        except Exception as e:
            logger.warning("Prophet predict failed: %s", e)
            return pd.Series([0.0] * h)

    def predict_interval(self, h: int, alpha: float = 0.80) -> pd.DataFrame:
        if self._model is None:
            return pd.DataFrame({"yhat": [0.0]*h, "yhat_lower": [0.0]*h, "yhat_upper": [0.0]*h})
        future = self._make_future(h)
        try:
            forecast = self._model.predict(future)
            yhat = self._apply_ceiling(forecast["yhat"].values)
            lower = self._apply_ceiling(forecast["yhat_lower"].values)
            upper = self._apply_ceiling(forecast["yhat_upper"].values)
            return pd.DataFrame({"yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})
        except Exception as e:
            logger.warning("Prophet predict_interval failed: %s", e)
            yhat = self.predict(h).values
            return pd.DataFrame({"yhat": yhat, "yhat_lower": yhat, "yhat_upper": yhat})

    def get_components(self) -> pd.DataFrame:
        if self._model is None or self._train_df is None:
            return pd.DataFrame()
        try:
            return self._model.predict(self._train_df)
        except Exception:
            return pd.DataFrame()
