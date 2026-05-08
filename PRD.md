# VaxAlert: Product Requirements Document
### Full Implementation Guide

---

## 1. Project Overview

**VaxAlert** is a facility-level vaccine stock forecasting and alert system for Ethiopia's Expanded Programme on Immunization (EPI). It predicts days-to-stockout for each vaccine at each health facility, fires tiered alerts before stockouts occur, and surfaces the HC→HP cascade effect when a health center's supply failure propagates to its satellite health posts.

**Competition track:** Health Trend & Risk Analysis  
**Scope:** 30 synthetic Ethiopian health facilities × 7 EPHI infant schedule antigens × 156 weeks (3 years)  
**Core output:** A Streamlit dashboard with forecasts, alert status, and 5 operational KPIs

---

## 2. Current State — What Already Exists

> These files must not be regenerated or modified during implementation. They are the foundation everything else builds on.

### 2.1 File locations
```
project_root/
├── vaxalert_sdg.py          # Synthetic Data Generator — DO NOT MODIFY
└── data/
    └── vaxalert.db          # SQLite database — already generated, read-only input
```

### 2.2 Database tables (already populated)

The `vaxalert.db` SQLite database contains 8 tables. All model training, evaluation, and dashboard data comes from this file.

#### `facilities` — 30 rows
| Column | Type | Notes |
|--------|------|-------|
| facility_id | TEXT PK | e.g. FAC-001 |
| name | TEXT | Real facility name from Ethiopia facility registry |
| type | TEXT | Health Post / Health Center / Hospital |
| access_tier | TEXT | urban / rural_road / rural_remote / pastoral |
| region | TEXT | Ethiopian administrative region |
| woreda | TEXT | Sub-regional administrative unit |
| latitude | FLOAT | GPS coordinate |
| longitude | FLOAT | GPS coordinate |
| catchment_pop | INT | Standard MoH catchment: HP=5,000, HC=25,000, Hospital=150,000 |
| cbr_per_1000 | FLOAT | Crude birth rate — regional, from 2019 EMDHS |
| imr_per_1000 | FLOAT | Infant mortality rate — regional, from GBD 2019 |
| target_infants_annual | INT | catchment × CBR × (1−IMR) |
| eff_coverage_rate | FLOAT | national WUENIC × regional multiplier × urban/rural adj |
| institutional_delivery_rate | FLOAT | Proportion of births at facility — affects Hospital BCG/OPV0 demand |
| lead_time_days_mean | FLOAT | Mean days from order to delivery |
| lead_time_days_sd | FLOAT | SD on lead time |
| sessions_per_week | FLOAT | Immunization sessions per week (0.5 = biweekly) |
| infants_per_session | FLOAT | Expected target infants per session |
| coverage_multiplier | FLOAT | Regional WUENIC multiplier vs national average |
| urban_rural_adj | FLOAT | Urban +12%, rural −5 to −35% |
| has_fridge | INT | 0=Health Post (no fridge), 1=HC/Hospital |
| supervising_hc_id | TEXT | FK to facilities.facility_id — which HC manages this HP |

#### `vaccines` — 7 rows
| Column | Type | Notes |
|--------|------|-------|
| antigen_code | TEXT PK | BCG / OPV / PENTA / PCV / ROTA / IPV / MCV |
| description | TEXT | Full name |
| schedule_age_weeks | INT | 0=birth dose, 6=6-week schedule, 36=9-month (MCV) |
| doses_in_series | INT | BCG=1, OPV=4, PENTA=3, PCV=3, ROTA=2, IPV=1, MCV=1 |
| vial_size | INT | BCG=20, OPV=20, PENTA=10, PCV=1, ROTA=1, IPV=5, MCV=10 |
| wastage_rate_low_vol | FLOAT | Wastage at low-volume facilities (e.g. BCG=0.50) |
| wastage_rate_high_vol | FLOAT | Wastage at high-volume facilities (e.g. BCG=0.10) |
| national_wuenic_2024 | INT | WHO national coverage % |

#### `hc_hp_clusters` — 35 rows
Maps each HP to its supervising HC. Critical for cascade visualization.
| Column | Type |
|--------|------|
| hc_id | TEXT |
| hp_id | TEXT |
| hc_name | TEXT |
| hp_name | TEXT |
| region | TEXT |

#### `target_population` — 210 rows (30 facilities × 7 antigens)
| Column | Type | Notes |
|--------|------|-------|
| facility_id | TEXT |  |
| antigen | TEXT |  |
| role | TEXT | primary / catch-up / birth-dose |
| target_infants | INT | Infants this facility targets for this antigen |
| doses_in_series | INT |  |
| target_doses_annual | INT | target_infants × doses_in_series |
| expected_consumption_annual | INT | target_doses × eff_coverage_rate |
| wastage_rate | FLOAT | Assigned based on volume tier |
| stock_needed_annual | INT | consumption × (1 + wastage) |
| weekly_consumption_baseline | FLOAT | expected_consumption_annual / 52 |
| vial_size | INT |  |

#### `shock_events` — 1,042 rows
| Column | Type | Notes |
|--------|------|-------|
| shock_id | INT PK |  |
| facility_id | TEXT |  |
| week | INT | 0-indexed week number |
| week_date | TEXT | YYYY-MM-DD of Monday |
| shock_type | TEXT | rainy_season_road_closure / rainy_season_delay / measles_sia_campaign / polio_snid_campaign / cold_chain_failure / epss_hub_stockout / conflict_disruption / pandemic_disruption |
| severity | TEXT | low / medium / high / critical / planned |
| demand_multiplier | FLOAT | Multiplier on consumption (>1 = spike, <1 = drop) |
| supply_multiplier | FLOAT | Multiplier on resupply (0 = no delivery) |
| lead_time_multiplier | FLOAT | Multiplier on lead time |
| duration_weeks | INT |  |
| affected_antigens | TEXT | "ALL" or specific antigen code |
| notes | TEXT | Human-readable description |

#### `stock_ledger` — 32,760 rows — **primary training table**
| Column | Type | Notes |
|--------|------|-------|
| facility_id | TEXT |  |
| antigen | TEXT |  |
| week | INT | 0–155 |
| week_date | TEXT | YYYY-MM-DD |
| opening_stock | INT |  |
| doses_administered | INT | Actual doses given |
| doses_wasted | INT | Open-vial wastage |
| total_consumed | INT | administered + wasted |
| resupply_received | INT | Doses received this week |
| closing_stock | INT |  |
| target_demand | INT | What demand would have been without stockout |
| children_missed | INT | target_demand − doses_administered |
| is_stockout | INT | 1 if closing_stock == 0 |
| days_to_stockout | INT | Estimated days at current burn rate |
| alert_status | TEXT | ok / warning / critical |
| birth_seasonality_factor | FLOAT | Seasonal demand multiplier |
| demand_shock_multiplier | FLOAT | Combined active shock demand multiplier |
| supply_shock_multiplier | FLOAT | Combined active shock supply multiplier |
| cascade_affected | INT | 1 if HP missed supply because supervising HC was stocked out |
| cascade_source_hc | TEXT | HC facility_id causing the cascade, or NULL |

#### `session_log` — 32,419 rows
Weekly aggregation of immunization sessions.
| Column | Type |
|--------|------|
| facility_id | TEXT |
| antigen | TEXT |
| week | INT |
| week_date | TEXT |
| sessions_held | INT |
| sessions_planned | INT |
| doses_administered | INT |
| doses_wasted | INT |
| children_reached | INT |
| children_missed | INT |
| vials_opened | INT |

#### `delivery_log` — 10,341 rows
Resupply delivery records.
| Column | Type | Notes |
|--------|------|-------|
| facility_id | TEXT |  |
| antigen | TEXT |  |
| week | INT |  |
| week_date | TEXT |  |
| quantity_ordered | INT |  |
| quantity_received | INT |  |
| supply_multiplier | FLOAT | <1.0 means partial delivery due to disruption |
| lead_time_actual | INT | Days |
| source | TEXT | EPSS_hub or HC facility_id |
| emergency_order | INT | 1 if triggered by low-stock emergency reorder |

---

## 3. Project Architecture

```
project_root/
├── vaxalert_sdg.py              # Existing — DO NOT MODIFY
├── PRD.md                       # This document
├── requirements.txt
├── data/
│   └── vaxalert.db              # Existing — read-only
├── utils/
│   ├── __init__.py
│   ├── db.py                    # Database connection + query helpers
│   └── features.py              # Feature engineering for models
├── models/
│   ├── __init__.py
│   ├── baseline.py              # Naive + Holt-Winters
│   ├── sarimax_model.py         # SARIMAX with exogenous features
│   ├── prophet_model.py         # Prophet with event calendar
│   └── ensemble.py              # Weighted ensemble + forecast output
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # MAE, RMSE, MAPE, interval coverage, stockout detection
│   └── walk_forward_cv.py       # Cross-validation pipeline
├── forecast/
│   ├── __init__.py
│   └── generate_forecasts.py    # Run all models, write forecast table to DB
├── dashboard/
│   ├── app.py                   # Main Streamlit app
│   └── components/
│       ├── kpi_cards.py
│       ├── facility_map.py
│       ├── stock_chart.py
│       ├── cascade_view.py
│       └── alert_table.py
├── notebooks/
│   └── eda.ipynb                # Exploratory analysis (optional, good for documentation)
└── ethics.md                    # Required by competition
```

---

## 4. Implementation Steps

Follow these steps in order. Each step is a discrete, testable unit.

---

### Step 1: Environment setup

Create `requirements.txt`:

```
pandas>=2.0.0
numpy>=1.24.0
pmdarima>=2.0.3
statsmodels>=0.14.0
prophet>=1.1.4
scikit-learn>=1.3.0
streamlit>=1.32.0
plotly>=5.18.0
folium>=0.15.0
streamlit-folium>=0.18.0
sqlalchemy>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

### Step 2: Database and feature utilities

#### `utils/db.py`

Implement these functions:

```python
def get_connection(db_path="data/vaxalert.db") -> sqlite3.Connection
def get_facilities() -> pd.DataFrame          # all 30 facilities
def get_vaccines() -> pd.DataFrame            # all 7 antigens
def get_clusters() -> pd.DataFrame            # HC-HP relationships
def get_stock_series(facility_id, antigen) -> pd.DataFrame   # full time series for one series
def get_all_series() -> dict                  # {(facility_id, antigen): pd.DataFrame}
def get_shocks_for_facility(facility_id) -> pd.DataFrame
def get_target_population(facility_id, antigen) -> dict
def write_forecasts(forecast_df: pd.DataFrame)  # writes to forecast_output table
def write_model_metrics(metrics_df: pd.DataFrame)  # writes to model_metrics table
```

#### `utils/features.py`

Build the exogenous feature matrix for SARIMAX. For each week in a facility's time series, construct:

```python
def build_exog_features(facility_id: str, n_weeks: int = 156) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - week (int)
    - week_date (datetime)
    - is_rainy_season (binary: 1 if June-September)
    - is_measles_sia (binary: 1 if active SIA campaign for MCV)
    - is_polio_snid (binary: 1 if active SNID campaign for OPV)
    - is_conflict_period (binary)
    - is_pandemic_period (binary: 1 for weeks 8-30)
    - demand_shock_multiplier (float, from shock_events)
    - supply_shock_multiplier (float, from shock_events)
    - lead_time_multiplier (float, from shock_events or 1.0)
    - birth_seasonality_factor (float: 1.0 + 0.15*cos(2π*(month-9.5)/12),
      computed from the actual calendar date — NOT a lookup from stock_ledger.
      For future weeks 156-163, compute from their real calendar dates.)
    - weeks_since_last_resupply (int, computed from delivery_log)
    - log_weeks_since_resupply (float, log transform to reduce skew)
    """
```

Build the Prophet events DataFrame:

```python
def build_prophet_events(facility_id: str) -> pd.DataFrame:
    """
    Returns DataFrame in Prophet holiday format:
    columns: holiday (str), ds (datetime), lower_window (int), upper_window (int)
    
    Events:
    - measles_sia_campaign: Oct-Nov years 1 and 3, lower=-2, upper=2
    - polio_snid_campaign: April each year, lower=-1, upper=1
    - kiremt_rainy_season: June-September, lower=0, upper=0
    - pandemic_disruption: weeks 8-30, lower=0, upper=0
    - conflict_disruption: facility-specific, weeks 60-80 for affected facilities
    - post_disruption_catchup: 4 weeks after each major disruption end
    """
```

---

### Step 3: Baseline models

#### `models/baseline.py`

```python
class NaiveModel:
    """
    Two variants:
    - last_value: forecast = last observed value, repeated h steps
    - seasonal_naive: forecast = value from same week 52 weeks ago
    
    Methods:
    - fit(series: pd.Series) -> self
    - predict(h: int) -> pd.Series  (h = forecast horizon in weeks)
    - predict_interval(h: int, alpha: float = 0.80) -> pd.DataFrame
      columns: yhat, yhat_lower, yhat_upper
    """

class HoltWintersModel:
    """
    Wrapper around statsmodels ExponentialSmoothing.
    
    Configuration:
    - trend='add' (additive trend)
    - seasonal='add' (additive seasonality)
    - seasonal_periods=52 (annual cycle)
    - damped_trend=True (prevents explosive long-run forecasts)
    - initialization_method='estimated'
    
    Methods:
    - fit(series: pd.Series) -> self
    - predict(h: int) -> pd.Series
    - predict_interval(h: int, alpha: float = 0.80) -> pd.DataFrame
      columns: yhat, yhat_lower, yhat_upper
    
    IMPORTANT: If series length < 2 × seasonal_periods (104 weeks), 
    fall back to trend-only ETS (no seasonal component).
    """
```

---

### Step 4: SARIMAX model

#### `models/sarimax_model.py`

```python
class SARIMAXModel:
    """
    Auto-order SARIMAX using pmdarima.auto_arima with exogenous features.
    
    Auto-arima configuration:
    - start_p=1, max_p=3
    - start_q=0, max_q=2
    - d=None (auto-determine via ADF test)
    - seasonal=True, m=52 (annual seasonality)
    - start_P=0, max_P=1
    - start_Q=0, max_Q=1
    - D=None (auto)
    - information_criterion='aic'
    - stepwise=True (faster than exhaustive search)
    - n_jobs=-1
    - error_action='ignore'
    - with_oob_score=False
    
    Exogenous features used (from features.py):
    - is_rainy_season
    - demand_shock_multiplier
    - supply_shock_multiplier
    - birth_seasonality_factor
    - is_measles_sia (for MCV series only)
    - is_polio_snid (for OPV series only)
    - log_weeks_since_resupply
    
    NOTE: Only use antigen-specific campaign features for the relevant antigen.
    
    Methods:
    - fit(series: pd.Series, exog: pd.DataFrame) -> self
    - predict(h: int, future_exog: pd.DataFrame) -> pd.Series
    - predict_interval(h: int, future_exog: pd.DataFrame, alpha: float = 0.80) 
        -> pd.DataFrame  (columns: yhat, yhat_lower, yhat_upper)
    - get_aic() -> float
    - get_order() -> tuple  (p, d, q, P, D, Q)
    
    Error handling:
    - If auto_arima fails to converge, fall back to ARIMA(1,1,1) with no seasonal
    - Log fallback cases for reporting
    - Never raise — always return a prediction
    
    Post-processing:
    - Clip all predictions to >= 0 (stock cannot be negative)
    - Apply logistic ceiling: predictions cannot exceed 
      target_population.stock_needed_annual / 52 × 1.5
    """
```

---

### Step 5: Prophet model

#### `models/prophet_model.py`

```python
class ProphetModel:
    """
    Facebook Prophet configured for weekly EPI stock consumption data.
    
    Configuration:
    - growth='linear'  (not logistic — we handle ceiling separately)
    - seasonality_mode='additive'
    - weekly_seasonality=False  (not relevant for aggregate facility data)
    - daily_seasonality=False
    - yearly_seasonality=True  (captures annual birth cycle)
    - add_seasonality: annual, period=52.18, fourier_order=5
    - changepoint_prior_scale=0.05  (conservative — avoid overfitting on 156 points)
    - seasonality_prior_scale=10
    - holidays_prior_scale=15  (events matter a lot in this system)
    - interval_width=0.80  (80% prediction interval to match evaluation metric)
    - mcmc_samples=0  (use MAP estimation, not MCMC — much faster)
    
    Events (holidays param):
    - Pass the DataFrame from features.build_prophet_events(facility_id)
    - Ensure events are named clearly: measles_sia_campaign, kiremt_rainy_season, etc.
    
    Input format:
    - Prophet expects ds (datetime) and y (float) columns
    - Convert week_date to datetime before fitting
    
    Methods:
    - fit(series: pd.DataFrame) -> self  (must have ds, y columns)
    - predict(h: int) -> pd.Series
    - predict_interval(h: int, alpha: float = 0.80) 
        -> pd.DataFrame  (columns: yhat, yhat_lower, yhat_upper)
    - get_components() -> pd.DataFrame  (trend, seasonality, event contributions)
    
    Post-processing:
    - Same ceiling logic as SARIMAX
    - Clip to >= 0
    """
```

---

### Step 6: Ensemble

#### `models/ensemble.py`

```python
class WeightedEnsemble:
    """
    Inverse-MAE weighted combination of SARIMAX and Prophet.
    Weights are fit per-series on the validation set.
    
    Weight computation:
    - Use MAE from the final CV fold only (test window: weeks 128-139).
      This is the most recent pre-test data and closest to the actual forecast
      horizon, making it the most relevant window for weighting.
    - val_mae_sarimax = MAE of SARIMAX on weeks 128-139 (fold 3 test window)
    - val_mae_prophet = MAE of Prophet on weeks 128-139 (fold 3 test window)
    - w_sarimax = (1/val_mae_sarimax) / (1/val_mae_sarimax + 1/val_mae_prophet)
    - w_prophet  = 1 - w_sarimax
    
    Edge cases:
    - If either model fails on a series, weight = 1.0 for the surviving model
    - If both MAEs are equal, use equal weights (0.5 / 0.5)
    - If a series has zero stockout events in the fold 3 window (very low volume),
      fall back to equal weights rather than optimising on a flat signal
    - Store weights in model_metrics table for transparency
    
    Prediction:
    - ensemble_yhat = w_sarimax × sarimax_yhat + w_prophet × prophet_yhat
    - ensemble_lower = min(sarimax_lower, prophet_lower)  (conservative: widest CI)
    - ensemble_upper = max(sarimax_upper, prophet_upper)
    
    Methods:
    - fit(sarimax_model, prophet_model, val_series, val_exog) -> self
    - predict(h: int, future_exog: pd.DataFrame) -> pd.DataFrame
      columns: yhat, yhat_lower, yhat_upper, w_sarimax, w_prophet
    - predict_interval(h: int, ...) -> pd.DataFrame
    """
```

---

### Step 7: Evaluation framework

#### `evaluation/metrics.py`

Implement all metrics. These are used in both walk-forward CV and final reporting.

```python
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in doses (primary metric)"""

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error — penalises large misses"""

def mape(y_true: np.ndarray, y_pred: np.ndarray, min_threshold: float = 5.0) -> float:
    """
    Mean Absolute Percentage Error.
    CRITICAL: Only compute on observations where y_true >= min_threshold.
    Small denominators produce meaningless MAPE. Return NaN if < 5 valid obs.
    """

def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Proportion of true values falling within the prediction interval.
    For 80% PI, this should be approximately 0.80.
    Returns float in [0, 1].
    """

def stockout_detection_rate(
    actual_stockouts: pd.Series,   # boolean series: True = stockout week
    predicted_dts: pd.Series,      # predicted days_to_stockout
    lead_time_days: float,
    window_weeks: int = 2
) -> dict:
    """
    Core operational metric for VaxAlert.
    
    A stockout is 'detected' if the model predicted days_to_stockout <= 
    (lead_time_days + safety_buffer) within window_weeks before the actual stockout.
    
    Returns:
    {
        'detection_rate': float,     # proportion of stockouts correctly warned
        'false_alert_rate': float,   # proportion of warnings that didn't lead to stockout
        'mean_warning_lead_days': float,  # how many days before stockout alert fired
        'missed_stockouts': int,
        'total_stockouts': int,
    }
    """

def compute_all_metrics(
    y_true, y_pred, lower, upper, 
    stockout_actual, predicted_dts, lead_time_days
) -> dict:
    """Returns dict of all metrics for one facility × antigen series."""
```

#### `evaluation/walk_forward_cv.py`

```python
"""
Walk-forward cross-validation pipeline.

WHY WALK-FORWARD CV (not a single holdout split):
A single train/test split gives one MAE estimate from one 16-week window.
If that window is unusually calm or disrupted, the reported metrics are not
representative. Walk-forward CV averages across multiple evaluation windows,
giving a more stable and honest estimate of model performance. It also
reveals whether performance degrades over time — a fold-by-fold MAE trend
that worsens indicates the model is not generalising.

DATA SPLITS — 3 expanding folds + 1 locked final test:

  Fold 1:  Train weeks 0-79  (80 wks)  | Test weeks 80-91   (12 wks)
  Fold 2:  Train weeks 0-91  (92 wks)  | Test weeks 92-103  (12 wks)
  Fold 3:  Train weeks 0-103 (104 wks) | Test weeks 104-115 (12 wks)

  [Weeks 116-139: buffer + ensemble weight window — not a CV fold]
  [Weeks 140-155: FINAL HELD-OUT TEST — never touched during CV]

  ENSEMBLE WEIGHT WINDOW (weeks 128-139) — 4th fitting step, NOT a CV fold:
  Fit models on weeks 0-127, evaluate on weeks 128-139, use those MAEs to
  compute ensemble weights. Weeks 116-127 are a buffer preventing leakage
  between the last CV fold and the weight window. Never call this "fold 3".

  Reported CV metrics = mean and std across folds 1, 2, 3 per model.
  Reported final metrics = each model evaluated on weeks 140-155 only.

SARIMAX ORDER CACHING to control runtime:
  auto_arima order search is expensive. Running it fresh for every fold of
  every series would make the pipeline prohibitively slow.

  Strategy:
  - Run auto_arima order search ONCE per series on fold 1 training data
  - Cache the best (p,d,q)(P,D,Q) order using joblib.Memory
  - For folds 2 and 3, refit SARIMAX coefficients using the cached order
    (no grid search — just MLE fitting on the expanded window)
  - Cache location: .cache/arima_orders/
  - Cache key: "{facility_id}_{antigen}"
  - If fold 1 auto_arima fails or times out (>90s per series), assign
    fallback order ARIMA(1,1,1) with no seasonal component and log it

  This reduces SARIMAX fitting from O(n_orders × n_folds) to
  O(n_orders + n_folds), cutting total runtime by ~70%.

ZERO-INFLATION HANDLING:
  Some low-volume series (pastoral HPs × PCV, ROTA, IPV) have >60% zero weeks.
  For these series:
  - Skip SARIMAX entirely (auto_arima will produce garbage)
  - Use only Naive + HoltWinters
  - Set ensemble weights to: w_sarimax=0.0, w_prophet=1.0 (Prophet only)
  - Flag in model_metrics: zero_inflated=True
  Threshold: series is zero-inflated if SUM(y==0) / len(y) > 0.60

PIPELINE PER SERIES (30 facilities x 7 antigens = 210 series):

  For each (facility_id, antigen):
    1. Load full series from stock_ledger (closing_stock, weeks 0-155)
    2. Build exog features via features.build_exog_features()
    3. Check zero-inflation flag
    4. For each fold (1, 2, 3):
       a. Slice train and test windows
       b. Fit NaiveModel (both variants: last_value and seasonal_naive)
       c. Fit HoltWintersModel on train
       d. If not zero_inflated:
          - Fold 1: run auto_arima, cache order, fit SARIMAX
          - Folds 2-3: load cached order, refit SARIMAX coefficients only
          - Fit ProphetModel on train (see prophet_model.py — do not modify)
       e. Generate h=12 week predictions for test window from each model
       f. Compute all metrics (mae, rmse, mape, interval_coverage,
          stockout_detection_rate) against actual test values
       g. Collect results with fold label
    5. After all 3 folds:
       - Compute fold 3 test window (weeks 128-139) metrics for ensemble weights
       - Retrain all models on weeks 0-139 (full pre-test data)
       - Compute WeightedEnsemble weights from fold 3 MAEs
       - Evaluate final test set (weeks 140-155) for all 5 models incl. ensemble
    6. Write all metrics to model_metrics table

SARIMAX timeout handling:
  Wrap auto_arima in a signal.alarm timeout (Unix) or threading.Timer (Windows).
  If it exceeds 90 seconds, log the timeout, assign fallback order, continue.

OUTPUT — model_metrics table columns:
  facility_id, antigen, model, fold (1/2/3/final), mae, rmse, mape,
  interval_coverage, stockout_detection_rate, false_alert_rate,
  mean_warning_lead_days, w_sarimax, w_prophet,
  n_stockout_events, zero_inflated, sarimax_order, fallback_used

REPORTING:
  After all 210 series complete, print summary table:
  - Mean CV MAE per model (averaged across all series and folds)
  - Final test MAE per model
  - Stockout detection rate per model on final test
  - Number of series where each model won (lowest MAE)
  - Number of zero-inflated series
  - Number of SARIMAX fallbacks used

CLI flags:
  --sample N     Run only first N series (for quick testing, use N=10)
  --no-cache     Ignore cached SARIMAX orders, re-run auto_arima fresh
  --folds N      Override number of folds (default 3, min 2, max 5)

Progress output:
  Print progress every 10 series: "42/210 | FAC-007 PENTA | fold 3/3"
  Print fold summary after each fold completes.

Runtime estimate:
  - With caching: ~20-40 min for all 210 series
  - Without caching (first run): ~45-90 min
  - --sample 10: ~3-5 min
"""
```

---

### Step 8: Forecast generation

#### `forecast/generate_forecasts.py`

```python
"""
Runs the final production forecast for all 210 series.

Prerequisites: walk_forward_cv.py must have completed successfully.
The model_metrics table must exist and contain fold=final rows.

Steps:
1. Load CV results from model_metrics to get:
   - Cached SARIMAX orders per series (from .cache/arima_orders/)
   - Ensemble weights per series (w_sarimax, w_prophet from fold 3)
   - Zero-inflation flags per series
2. Retrain all models on weeks 0-139
   (full data excluding the locked test set weeks 140-155)
   Note: weeks 140-155 are excluded from production retraining to preserve
   the integrity of the reported final test metrics. In a real deployment
   you would retrain on all 156 weeks — document this distinction.
3. Generate 8-week ahead forecasts (weeks 156-163)
4. Compute predicted_days_to_stockout from forecast:
   current_closing_stock = closing_stock at week 155
   weekly_burn = mean(yhat over forecast horizon)
   predicted_dts = (current_closing_stock / weekly_burn) × 7  if weekly_burn > 0
                   else 999
5. Compute alert_status per facility × antigen:
   safety_buffer_days = max(cv_mae_weeks × 7, 5)
     where cv_mae_weeks = mean fold MAE / weekly_consumption_baseline
   alert_threshold_days = lead_time_days_mean + safety_buffer_days
   alert_status:
     'critical' if predicted_dts <= alert_threshold × 0.5
     'warning'  if predicted_dts <= alert_threshold
     'ok'       otherwise
6. Write results to forecast_output table in DB

forecast_output table schema:
- facility_id TEXT
- antigen TEXT
- forecast_week INT (0-indexed from simulation start, 156-163)
- forecast_date TEXT
- model TEXT (naive / holt_winters / sarimax / prophet / ensemble)
- yhat FLOAT
- yhat_lower FLOAT
- yhat_upper FLOAT
- predicted_days_to_stockout INT
- alert_status TEXT (ok / warning / critical)
- ensemble_w_sarimax FLOAT  (NULL for non-ensemble models)
- ensemble_w_prophet FLOAT  (NULL for non-ensemble models)
- generated_at TEXT (ISO timestamp)

Primary forecast for dashboard display = ensemble model.
All 5 model forecasts stored for the Model Performance comparison view.
"""
```

---

### Step 9: Dashboard

Build with **Streamlit**. The dashboard has 4 views navigable via sidebar.

#### `dashboard/app.py` — Main entry point

```python
"""
streamlit run dashboard/app.py

Sidebar navigation:
1. National Overview (default)
2. Facility Drill-Down
3. Cascade View
4. Model Performance

Global sidebar filters:
- Antigen selector (multiselect, default: all)
- Access tier filter
- Alert status filter (critical / warning / ok)
- Forecast horizon slider (1-8 weeks)
"""
```

---

#### View 1: National Overview

**KPI cards row (all 5 KPIs must appear here):**

**KPI 1 — Stockout alerts**  
Count of facility × antigen combinations currently at `critical` alert status.  
Formula: `COUNT(*) WHERE alert_status = 'critical'` from forecast_output (latest forecast week).  
Display: Large red number + delta vs previous week.

**KPI 2 — DTP dropout rate (national)**  
Formula (corrected — `schedule_position` column does not exist):  
```python
# From stock_ledger, last 52 weeks, PENTA antigen only:
actual_penta_doses = SUM(doses_administered) WHERE antigen='PENTA'
# From target_population, PENTA antigen only:
total_target_infants = SUM(target_infants) WHERE antigen='PENTA'
doses_in_series = 3  # constant for PENTA
# If every child completed all 3 doses: actual = total_target_infants × 3
completion_rate = actual_penta_doses / (total_target_infants * doses_in_series)
dropout_rate = 1 - completion_rate
```  
Threshold: WHO acceptable < 10%. Display with red/amber/green indicator.

**KPI 3 — Children at risk this week**  
`SUM(children_missed)` from stock_ledger for the most recent week.  
Represents children who could not be vaccinated due to stockout.

**KPI 4 — Wastage rate (system-wide)**  
Formula: `SUM(doses_wasted) / (SUM(doses_administered) + SUM(doses_wasted))` from session_log.  
Computed over last 12 weeks. Display as percentage with benchmark line at 10% (WHO acceptable).

**KPI 5 — Resupply urgency score**  
Composite: for each facility × antigen at warning/critical: `score = (1 / days_to_stockout) × (lead_time_days_mean / 7) × (target_infants_annual / 100)`  
Sum across all at-risk combinations and normalise to 0–100.  
Indicates overall system pressure. Higher = more urgent.

**Map:**  
Use `folium` + `streamlit-folium`. Plot all 30 facilities as circle markers.  
Color: red = any critical alert, amber = warning only, green = all ok.  
Radius: proportional to catchment_pop.  
Click popup: facility name, type, tier, worst alert status, days to stockout for worst antigen.

**Alert table:**  
Sortable table of all warning/critical alerts.  
Columns: Facility, Type, Region, Antigen, Days to Stockout, Lead Time, Alert Status, Cascades (Y/N).  
Sort default: days_to_stockout ascending (most urgent first).

---

#### View 2: Facility Drill-Down

Facility selector dropdown (searchable by name or region).  
Antigen selector (radio buttons, one at a time).

**Stock history + forecast chart:**  
Plotly line chart.  
- Solid line: actual closing_stock (weeks 0–155)
- Shaded region: training period (0–119)  
- Dashed line: ensemble forecast (weeks 156–163) in accent color
- Shaded band: 80% prediction interval
- Red horizontal line: reorder point (stock = lead_time_days × weekly_consumption_baseline / 7)
- Annotations: shock events as vertical dashed lines with labels

**Forecast confidence:**  
Show ensemble weights: "SARIMAX {w}% / Prophet {w}%" for this series.  
Show validation MAE: "Typical error: ±X doses/week"

**Delivery timeline:**  
Bar chart of delivery_log for this facility × antigen.  
Actual received vs ordered. Highlight emergency orders.

**Session performance:**  
Line chart: children_reached vs children_missed over time.

---

#### View 3: Cascade View

This view is specific to Health Center → Health Post relationships.

**HC selector:**  
Dropdown showing only Health Centers (7 facilities).

**Cascade network diagram:**  
Use Plotly scatter/line plot (not networkx — keep it simple).  
HC node at top center.  
HP nodes arranged below, connected by lines.  
Node color: red/amber/green by worst alert status.  
Node size: proportional to target_infants_annual.  
Line thickness: proportional to how many weeks the HC has been stocked out (indicating cascade risk).

**Cascade impact table:**  
For the selected HC, show:
- HC stock status per antigen
- Each satellite HP's stock status per antigen
- Cascade-affected records count
- Children missed due to cascade

**Timeline of cascade events:**  
Heatmap: rows = HPs, columns = weeks, color = cascade_affected (binary).  
Shows which HPs were cut off from supply and when.

---

#### View 4: Model Performance

**Per-model summary table:**  
Rows: Naive / Holt-Winters / SARIMAX / Prophet / Ensemble  
Columns: MAE (doses), RMSE, MAPE (%), Interval Coverage (%), Stockout Detection Rate (%), False Alert Rate (%), n_series

**Metric by access tier:**  
Grouped bar chart: MAE by [model × access_tier].  
Shows that remote/pastoral facilities are harder to forecast (should be visible clearly).

**Stockout detection rate breakdown:**  
Stacked bar: for each model, show detected / missed / false alerts.  
This is the most important chart — it answers "which model saves the most children."

**Individual series explorer:**  
Select any facility × antigen.  
Show overlapping forecast lines from all 5 models on the test period.  
Show actual stock in black.  
Show prediction intervals for ensemble.

---

### Step 10: Database additions

Before running models, add these tables to `data/vaxalert.db`:

```sql
-- Created by generate_forecasts.py
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
    alert_status TEXT,
    ensemble_w_sarimax REAL,
    ensemble_w_prophet REAL,
    generated_at TEXT,
    PRIMARY KEY (facility_id, antigen, forecast_week, model)
);

-- Created by walk_forward_cv.py
CREATE TABLE IF NOT EXISTS model_metrics (
    facility_id TEXT,
    antigen TEXT,
    model TEXT,
    fold TEXT,  -- '1', '2', '3', or 'final' (weeks 140-155)
    mae REAL,
    rmse REAL,
    mape REAL,
    interval_coverage REAL,
    stockout_detection_rate REAL,
    false_alert_rate REAL,
    mean_warning_lead_days REAL,
    w_sarimax REAL,         -- NULL except for ensemble model
    w_prophet REAL,         -- NULL except for ensemble model
    n_stockout_events INT,
    zero_inflated INT,      -- 1 if series skipped SARIMAX due to >60% zeros
    sarimax_order TEXT,     -- e.g. "(1,1,1)(1,0,0,52)" — NULL if zero_inflated
    fallback_used INT,      -- 1 if auto_arima timed out and fallback order applied
    PRIMARY KEY (facility_id, antigen, model, fold)
);
```

---

## 5. Running Order

Execute in this order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick test on 10 series (3-5 min)
python evaluation/walk_forward_cv.py --sample 10

# 3. Full CV run — builds SARIMAX order cache on first run (~45-90 min)
#    Subsequent runs with cache: ~20-40 min
python evaluation/walk_forward_cv.py

# 4. Generate forecasts (~5 min, uses cached orders)
python forecast/generate_forecasts.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## 6. Key Design Constraints

**Do not retrain models on test data.** The final test set is weeks 140–155. It is locked and never used during CV or ensemble weight computation. Ensemble weights are determined from fold 3 test results (weeks 128–139) only. The final test set is evaluated exactly once, after all CV and weight decisions are made.

**Never forecast negative stock.** All model outputs must be clipped to >= 0.

**Handle 210 series gracefully.** Some low-volume series (pastoral HPs × ROTA or PCV) will have many zero weeks. Implement zero-inflation handling: if the series has > 60% zeros, skip SARIMAX and use only Holt-Winters + Naive for that series.

**SARIMAX seasonal period.** With weekly data and m=52, SARIMAX can be very slow. If auto_arima takes > 60 seconds per series, catch the timeout and fall back to non-seasonal ARIMA. Use `joblib.Memory` caching so re-runs are fast.

**Prophet speed.** Prophet is slow for 210 series. Use `n_changepoints=20` (not default 25) and `mcmc_samples=0` (MAP only). Still expect 5–10 min for full run.

**Dashboard performance.** Cache all DB queries using `@st.cache_data`. The stock_ledger has 32,760 rows — load it once at startup, not per interaction.

---

## 7. Ethics Documentation

Create `ethics.md` with the following sections:

### Data privacy and confidentiality
The dataset used in VaxAlert is entirely synthetic. It was generated by `vaxalert_sdg.py` using a fixed random seed (42) for full reproducibility. No real patient records, personal health data, or individual identifiers are present. Facility names are drawn from a public Ethiopian facility registry (Ministry of Health). All analyses are at facility aggregate level — no individual is identifiable.

### Bias and fairness
The synthetic data was calibrated using real regional demographic and coverage data from the 2019 Ethiopian Mini Demographic and Health Survey (EMDHS) and WHO/WUENIC estimates. This calibration deliberately preserves the documented inequities in the Ethiopian health system — pastoral and remote facilities have higher stockout rates and fewer children reached. This is not a modelling artefact; it reflects documented reality. VaxAlert surfaces these inequities explicitly through the access-tier-stratified dashboard views, with the intent of directing resources toward the most underserved facilities rather than optimising system-wide averages.

### Model explainability
SARIMAX coefficients are interpretable and logged for each series. Prophet component plots are available in the Facility Drill-Down view, showing the contribution of trend, seasonality, and named events to each forecast. Ensemble weights are stored and displayed per series. No black-box methods are used.

### Limitations and responsible use
VaxAlert is trained on synthetic data. Its real-world deployment would require retraining on actual DHIS2 facility stock data. Forecasts carry uncertainty — the 80% prediction interval and stockout detection rate metrics must be reviewed before operational use. The system should support, not replace, human supply chain decision-making.

---

## 8. Competition Checklist

The competition requires the following deliverables. Implementation should flag any not yet complete.

- [ ] GitHub repository with all code
- [ ] `requirements.txt` present and complete
- [ ] `ethics.md` present
- [ ] `README.md` with setup instructions and project description
- [ ] Executable demo: `streamlit run dashboard/app.py`
- [ ] Dashboard with minimum 3 KPIs (we have 5)
- [ ] All 5 KPIs visible without scrolling on the National Overview page
- [ ] 3-5 min demo video (recorded by team)
- [ ] 5-slide presentation (created by team)

---

## 9. Validated Stockout Rates (Reference)

These are the calibrated stockout rates in the current database. They are consistent with published literature on Ethiopian EPI stockout rates. Do not modify the generator to change these — they are intentional.

| Facility Type | Access Tier | Stockout Rate | Literature Range |
|--------------|-------------|---------------|-----------------|
| Health Post | urban | 4.6% | 2–8% |
| Health Post | rural_road | 10.2% | 8–15% |
| Health Post | rural_remote | 25.5% | 18–35% |
| Health Post | pastoral | 38.1% | 30–50% |
| Health Center | urban | 8.2% | 5–12% |
| Health Center | rural_road | 7.5% | 5–12% |
| Health Center | rural_remote | 10.3% | 7–14% |
| Hospital | all | 0.1–0.2% | <2% |

---

## 10. README Template

Create `README.md` with:

```markdown
# VaxAlert — Vaccine Stockout Alert System for Ethiopia

## Overview
VaxAlert forecasts days-to-stockout for 7 EPHI infant schedule vaccines across 
30 Ethiopian health facilities. It fires tiered alerts before stockouts occur 
and surfaces cascading failures from health center to satellite health posts.

## Setup
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Running
\`\`\`bash
# Run full forecasting pipeline (first time only, ~10-20 min)
python evaluation/walk_forward_cv.py
python forecast/generate_forecasts.py

# Launch dashboard
streamlit run dashboard/app.py
\`\`\`

## Data
The dataset is fully synthetic, generated by vaxalert_sdg.py from a fixed seed 
and calibrated against WHO/WUENIC coverage data and 2019 EMDHS regional parameters.
See ethics.md for full data documentation.

## Models
- Naive baseline (last-value and seasonal naive)
- Holt-Winters Exponential Smoothing
- SARIMAX with exogenous shock features
- Prophet with named event calendar
- Weighted ensemble (inverse-MAE weighted SARIMAX + Prophet)

## Key Findings
[To be filled in after model evaluation]

## Ethics
See ethics.md
```

---

*Document version: 1.0*
