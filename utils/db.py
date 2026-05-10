import sqlite3
import os
import pandas as pd

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vaxalert.db")


def get_connection(db_path=None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def get_facilities() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM facilities", conn)


def get_vaccines() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM vaccines", conn)


def get_clusters() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM hc_hp_clusters", conn)


def get_stock_series(facility_id: str, antigen: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            """
            SELECT * FROM stock_ledger
            WHERE facility_id = ? AND antigen = ?
            ORDER BY week
            """,
            conn,
            params=(facility_id, antigen),
        )


def get_all_series() -> dict:
    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM stock_ledger ORDER BY facility_id, antigen, week", conn)
    result = {}
    for (fid, ant), grp in df.groupby(["facility_id", "antigen"]):
        result[(fid, ant)] = grp.reset_index(drop=True)
    return result


def get_shocks_for_facility(facility_id: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT * FROM shock_events WHERE facility_id = ? ORDER BY week",
            conn,
            params=(facility_id,),
        )


def get_target_population(facility_id: str, antigen: str) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM target_population WHERE facility_id = ? AND antigen = ?",
            (facility_id, antigen),
        ).fetchone()
    if row is None:
        return {}
    return dict(row)


def get_delivery_log(facility_id: str, antigen: str = None) -> pd.DataFrame:
    with get_connection() as conn:
        if antigen is None:
            return pd.read_sql(
                "SELECT * FROM delivery_log WHERE facility_id = ? ORDER BY week",
                conn,
                params=(facility_id,),
            )
        return pd.read_sql(
            "SELECT * FROM delivery_log WHERE facility_id = ? AND antigen = ? ORDER BY week",
            conn,
            params=(facility_id, antigen),
        )


def get_session_log(facility_id: str, antigen: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            """
            SELECT * FROM session_log
            WHERE facility_id = ? AND antigen = ?
            ORDER BY week
            """,
            conn,
            params=(facility_id, antigen),
        )


def write_forecasts(forecast_df: pd.DataFrame):
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecast_output (
                facility_id TEXT,
                antigen TEXT,
                forecast_week INT,
                forecast_date TEXT,
                model TEXT,
                yhat REAL,
                yhat_lower REAL,
                yhat_upper REAL,
                predicted_days_to_stockout INT,
                alert_threshold_days INT,
                alert_status TEXT,
                ensemble_w_sarimax REAL,
                ensemble_w_prophet REAL,
                generated_at TEXT,
                PRIMARY KEY (facility_id, antigen, forecast_week, model)
            )
        """)
        forecast_df.to_sql(
            "forecast_output", conn, if_exists="append", index=False,
            chunksize=50,
        )


def write_feature_importance(df: pd.DataFrame):
    """Persist top-N feature importances per facility (mean across antigens).
    df columns: facility_id, feature, importance, rank, generated_at."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance (
                facility_id TEXT,
                feature TEXT,
                importance REAL,
                rank INT,
                generated_at TEXT,
                PRIMARY KEY (facility_id, feature)
            )
        """)
        df.to_sql("feature_importance", conn, if_exists="append",
                  index=False, chunksize=200)


_MODEL_METRICS_COLS = [
    "facility_id", "antigen", "model", "fold",
    "mae", "rmse", "mape", "interval_coverage",
    "stockout_detection_rate", "false_alert_rate", "mean_warning_lead_days",
    "w_sarimax", "w_prophet", "n_stockout_events",
    "zero_inflated", "sarimax_order", "fallback_used",
]


def write_model_metrics(metrics_df: pd.DataFrame):
    cols = [c for c in _MODEL_METRICS_COLS if c in metrics_df.columns]
    df = metrics_df[cols].copy()
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                facility_id TEXT,
                antigen TEXT,
                model TEXT,
                fold TEXT,
                mae REAL,
                rmse REAL,
                mape REAL,
                interval_coverage REAL,
                stockout_detection_rate REAL,
                false_alert_rate REAL,
                mean_warning_lead_days REAL,
                w_sarimax REAL,
                w_prophet REAL,
                n_stockout_events INT,
                zero_inflated INT,
                sarimax_order TEXT,
                fallback_used INT,
                PRIMARY KEY (facility_id, antigen, model, fold)
            )
        """)
        df.to_sql(
            "model_metrics", conn, if_exists="append", index=False,
            chunksize=50,
        )
