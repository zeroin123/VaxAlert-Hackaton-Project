import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class NaiveModel:
    """
    Two variants: last_value and seasonal_naive.
    """

    def __init__(self, variant: str = "last_value"):
        assert variant in ("last_value", "seasonal_naive"), \
            "variant must be 'last_value' or 'seasonal_naive'"
        self.variant = variant
        self._series = None

    def fit(self, series: pd.Series) -> "NaiveModel":
        self._series = series.reset_index(drop=True)
        return self

    def predict(self, h: int) -> pd.Series:
        if self.variant == "last_value":
            val = float(self._series.iloc[-1])
            return pd.Series([max(0.0, val)] * h)
        else:
            n = len(self._series)
            preds = []
            for i in range(h):
                lag = n - 52 + i
                if lag >= 0:
                    preds.append(max(0.0, float(self._series.iloc[lag % n])))
                else:
                    preds.append(max(0.0, float(self._series.iloc[0])))
            return pd.Series(preds)

    def predict_interval(self, h: int, alpha: float = 0.80) -> pd.DataFrame:
        yhat = self.predict(h)
        residuals = self._series.diff().dropna()
        sigma = residuals.std() if len(residuals) > 1 else 1.0
        z = 1.282 if alpha == 0.80 else 1.645  # 80% or 90% normal z-score
        margin = z * sigma * np.sqrt(np.arange(1, h + 1))
        return pd.DataFrame({
            "yhat": yhat.values,
            "yhat_lower": np.maximum(0, yhat.values - margin),
            "yhat_upper": yhat.values + margin,
        })


class HoltWintersModel:
    """
    Wrapper around statsmodels ExponentialSmoothing.
    Falls back to trend-only ETS when series length < 2 × seasonal_periods.
    """

    SEASONAL_PERIODS = 52

    def __init__(self):
        self._model = None
        self._fit_result = None
        self._use_seasonal = True
        self._series = None

    def fit(self, series: pd.Series) -> "HoltWintersModel":
        self._series = series.reset_index(drop=True)
        n = len(self._series)
        self._use_seasonal = n >= 2 * self.SEASONAL_PERIODS

        try:
            if self._use_seasonal:
                model = ExponentialSmoothing(
                    self._series,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=self.SEASONAL_PERIODS,
                    damped_trend=True,
                    initialization_method="estimated",
                )
            else:
                model = ExponentialSmoothing(
                    self._series,
                    trend="add",
                    seasonal=None,
                    damped_trend=True,
                    initialization_method="estimated",
                )
            self._fit_result = model.fit(optimized=True, remove_bias=False)
        except Exception:
            # Last-resort fallback: fit without estimation
            try:
                model = ExponentialSmoothing(
                    self._series,
                    trend="add",
                    seasonal=None,
                    damped_trend=False,
                )
                self._fit_result = model.fit()
            except Exception:
                self._fit_result = None

        return self

    def predict(self, h: int) -> pd.Series:
        if self._fit_result is None:
            last = float(self._series.iloc[-1])
            return pd.Series([max(0.0, last)] * h)
        preds = self._fit_result.forecast(h)
        return pd.Series(np.maximum(0, preds.values))

    def predict_interval(self, h: int, alpha: float = 0.80) -> pd.DataFrame:
        yhat = self.predict(h)
        if self._fit_result is not None:
            try:
                sim = self._fit_result.simulate(nsimulations=h, repetitions=500, random_errors="bootstrap")
                lower_p = (1 - alpha) / 2 * 100
                upper_p = (1 + alpha) / 2 * 100
                lower = np.maximum(0, np.percentile(sim, lower_p, axis=1))
                upper = np.percentile(sim, upper_p, axis=1)
                return pd.DataFrame({
                    "yhat": yhat.values,
                    "yhat_lower": lower,
                    "yhat_upper": upper,
                })
            except Exception:
                pass

        # Fallback interval based on residuals
        residuals = self._series.values - (self._fit_result.fittedvalues.values if self._fit_result else self._series.values)
        sigma = np.std(residuals) if len(residuals) > 0 else 1.0
        z = 1.282 if alpha == 0.80 else 1.645
        margin = z * sigma * np.sqrt(np.arange(1, h + 1))
        return pd.DataFrame({
            "yhat": yhat.values,
            "yhat_lower": np.maximum(0, yhat.values - margin),
            "yhat_upper": yhat.values + margin,
        })
