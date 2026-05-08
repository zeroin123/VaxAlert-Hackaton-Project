import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in doses."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, min_threshold: float = 5.0) -> float:
    """
    Mean Absolute Percentage Error. Only computed on observations where y_true >= min_threshold.
    Returns NaN if fewer than 5 valid observations.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= min_threshold
    if mask.sum() < 5:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Proportion of true values within the prediction interval. Should be ~0.80 for 80% PI."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def stockout_detection_rate(
    actual_stockouts: pd.Series,
    predicted_dts: pd.Series,
    lead_time_days: float,
    window_weeks: int = 2,
) -> dict:
    """
    Core operational metric.
    A stockout is 'detected' if predicted_dts <= (lead_time_days + safety_buffer)
    within window_weeks before the actual stockout event.

    safety_buffer = lead_time_days * 0.5 (50% buffer on lead time)
    """
    actual_stockouts = pd.Series(actual_stockouts).reset_index(drop=True)
    predicted_dts = pd.Series(predicted_dts).reset_index(drop=True)
    n = len(actual_stockouts)

    safety_buffer = lead_time_days * 0.5
    alert_threshold = lead_time_days + safety_buffer

    # Find actual stockout weeks (transition into stockout)
    stockout_starts = []
    in_stockout = False
    for i in range(n):
        if actual_stockouts.iloc[i] and not in_stockout:
            stockout_starts.append(i)
            in_stockout = True
        elif not actual_stockouts.iloc[i]:
            in_stockout = False

    total_stockouts = len(stockout_starts)
    if total_stockouts == 0:
        return {
            "detection_rate": float("nan"),
            "false_alert_rate": 0.0,
            "mean_warning_lead_days": float("nan"),
            "missed_stockouts": 0,
            "total_stockouts": 0,
        }

    detected = 0
    warning_leads = []

    for so_week in stockout_starts:
        window_start = max(0, so_week - window_weeks)
        found = False
        for w in range(window_start, so_week):
            if predicted_dts.iloc[w] <= alert_threshold:
                found = True
                lead_days = (so_week - w) * 7
                warning_leads.append(lead_days)
                break
        if found:
            detected += 1

    # False alert rate: warnings that did NOT precede a stockout within window_weeks
    alert_weeks = (predicted_dts <= alert_threshold).sum()
    true_positive_alerts = detected
    false_alerts = alert_weeks - true_positive_alerts
    false_alert_rate = float(false_alerts / max(alert_weeks, 1))

    return {
        "detection_rate": float(detected / total_stockouts),
        "false_alert_rate": false_alert_rate,
        "mean_warning_lead_days": float(np.mean(warning_leads)) if warning_leads else float("nan"),
        "missed_stockouts": total_stockouts - detected,
        "total_stockouts": total_stockouts,
    }


def compute_all_metrics(
    y_true, y_pred, lower, upper,
    stockout_actual, predicted_dts, lead_time_days,
) -> dict:
    """Returns dict of all metrics for one facility × antigen series."""
    sdr = stockout_detection_rate(stockout_actual, predicted_dts, lead_time_days)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "interval_coverage": interval_coverage(y_true, lower, upper),
        "stockout_detection_rate": sdr["detection_rate"],
        "false_alert_rate": sdr["false_alert_rate"],
        "mean_warning_lead_days": sdr["mean_warning_lead_days"],
        "missed_stockouts": sdr["missed_stockouts"],
        "total_stockouts": sdr["total_stockouts"],
    }
