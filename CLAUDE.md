# VaxAlert — Project Memory

## What this project is
Vaccine stockout alert system for Ethiopian EPI facilities.
See PRD.md for full specification.

## Current state
- vaxalert_sdg.py — synthetic data generator. DO NOT MODIFY.
- data/vaxalert.db — generated SQLite database. DO NOT MODIFY or regenerate.
- PRD.md — full implementation spec. Follow it exactly.

## Implementation order
Follow the steps in PRD.md Section 4 in sequence:
Step 1 → Step 2 → Step 3 → Step 4 → Step 6 → Step 5 (Prophet, unchanged) 
→ Step 7 → Step 8 → Step 9 → Step 10

## Critical constraints (do not override these)
- Never retrain on test data (weeks 140–155 are locked)
- Never modify vaxalert.db directly — models write to forecast_output 
  and model_metrics tables only
- Prophet model config must not be changed from PRD spec
- All predictions clipped to >= 0
- SARIMAX order cached after fold 1, reused for folds 2–3

## Database location
data/vaxalert.db — relative path from project root