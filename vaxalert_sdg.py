"""
VaxAlert Synthetic Data Generator (SDG)
========================================
Generates realistic facility-level vaccine stock time series for 30 Ethiopian
health facilities, calibrated against WHO/WUENIC coverage data, 2019 EMDHS
regional estimates, and MoH facility standards.

Output: SQLite database with tables:
  - facilities          (30 rows)
  - vaccines            (7 rows — EPHI infant schedule)
  - hc_hp_clusters      (HC → HP linkages)
  - target_population   (per facility × antigen annual targets)
  - stock_ledger        (weekly stock movements per facility × antigen)
  - session_log         (HP/HC/Hospital session-level delivery records)
  - shock_events        (logged disruption events)
  - delivery_log        (resupply deliveries from EPSS hub / HC)

Fixed seed for full reproducibility.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import date, timedelta

# ============================================================
# 0. CONFIGURATION
# ============================================================
SEED = 42
RNG = np.random.default_rng(SEED)
N_WEEKS = 364  # 7 years
START_DATE = date(2023, 1, 2)  # Monday of week 1
DB_PATH = "data/vaxalert.db"
FACILITY_CSV = "data/sdg_input/ethiopian-health-facilities.csv"
COVERAGE_XLSX = "/mnt/project/coveragedataethiopia.xlsx"

# ============================================================
# 1. REGIONAL DEMOGRAPHIC PARAMETERS
# ============================================================
# Sources: 2019 EMDHS, GBD 2019 subnational, UN WPP 2024
REGIONAL_PARAMS = {
    # region: (TFR, CBR_per_1000, IMR_per_1000_lb, coverage_multiplier, institutional_delivery_rate)
    "Addis Ababa":          (1.8, 13.0, 25.0, 1.35, 0.90),
    "Harari":               (3.5, 25.0, 33.0, 1.25, 0.70),
    "Dire Dawa":            (3.6, 25.0, 35.0, 1.10, 0.65),
    "Tigray":               (4.0, 28.0, 38.0, 1.15, 0.55),
    "Amhara":               (3.7, 26.0, 46.0, 1.05, 0.45),
    "Oromia":               (5.3, 37.0, 54.0, 0.85, 0.35),
    "SNNP":                 (4.3, 30.0, 40.0, 0.90, 0.40),
    "Sidama":               (4.5, 32.0, 40.0, 0.88, 0.38),
    "South West Ethiopia":  (4.3, 30.0, 40.0, 0.88, 0.35),
    "Gambela":              (4.0, 28.0, 42.0, 0.95, 0.40),
    "Benishangul Gumz":     (4.5, 32.0, 48.0, 0.75, 0.30),
    "Afar":                 (5.5, 39.0, 52.0, 0.45, 0.20),
    "Somali":               (6.4, 45.0, 54.0, 0.35, 0.15),
}

# Urban/rural adjustment applied on top of regional multiplier
URBAN_RURAL_COVERAGE_ADJ = {
    "urban": 1.12,
    "rural_road": 0.95,
    "rural_remote": 0.80,
    "pastoral": 0.65,
}

# ============================================================
# 2. VACCINE PARAMETERS (EPHI infant schedule)
# ============================================================
# antigen_code, description, schedule_age_weeks, doses_in_series,
# vial_size, wastage_rate_low_vol, wastage_rate_high_vol,
# national_wuenic_2024
VACCINES = [
    ("BCG",   "BCG",                         0,  1, 20, 0.50, 0.10, 84),
    ("OPV",   "Oral Polio (0,1,2,3)",        0,  4, 20, 0.35, 0.10, 73),
    ("PENTA", "Pentavalent (1,2,3)",          6,  3, 10, 0.15, 0.05, 73),
    ("PCV",   "Pneumococcal (1,2,3)",         6,  3,  1, 0.02, 0.01, 73),
    ("ROTA",  "Rotavirus (1,2)",              6,  2,  1, 0.02, 0.01, 73),
    ("IPV",   "Inactivated Polio (1)",        6,  1,  5, 0.20, 0.05, 73),
    ("MCV",   "Measles (1)",                 36,  1, 10, 0.25, 0.05, 72),
]

# ============================================================
# 3. FACILITY SAMPLING STRATEGY
# ============================================================
# Stratified sample: 20 HPs, 7 HCs, 3 Hospitals across 4 access tiers
SAMPLING_PLAN = [
    # (type, access_tier, region_pool, count, catchment, lead_time_days_mean, lead_time_sd, sessions_per_week)
    # URBAN
    ("Health Post",   "urban",        ["Addis Ababa", "Dire Dawa", "Harari"],                    5, 5000,   2, 0.5, 3),
    ("Health Center", "urban",        ["Addis Ababa", "Dire Dawa"],                              2, 25000,  2, 0.5, 5),
    ("Hospital",      "urban",        ["Addis Ababa", "Oromia"],                                 2, 150000, 1, 0.3, 5),
    # RURAL ROAD
    ("Health Post",   "rural_road",   ["Amhara", "Oromia", "SNNP", "Sidama"],                   8, 5000,   4, 1.0, 2),
    ("Health Center", "rural_road",   ["Amhara", "Oromia", "SNNP"],                             3, 25000,  4, 1.0, 3),
    ("Hospital",      "rural_road",   ["Amhara"],                                                1, 150000, 3, 1.0, 5),
    # RURAL REMOTE
    ("Health Post",   "rural_remote", ["Tigray", "Benishangul Gumz", "Gambela", "South West Ethiopia"], 5, 5000, 8, 2.0, 1),
    ("Health Center", "rural_remote", ["Tigray", "Benishangul Gumz"],                           2, 25000,  8, 2.0, 2),
    # PASTORAL
    ("Health Post",   "pastoral",     ["Somali", "Afar"],                                        2, 5000,  14, 4.0, 0.5),  # 0.5 = biweekly
]


def sample_facilities(csv_path):
    """Sample 30 real facilities from the CSV using stratified approach."""
    df = pd.read_csv(csv_path)
    epi = df[(df["Type"].isin(["Health Post", "Health Center", "Hospital"])) &
             (df["Ownership"] == "Public/Government")].copy()

    facilities = []
    fac_id = 0

    for ftype, tier, regions, count, catchment, lt_mean, lt_sd, sess_wk in SAMPLING_PLAN:
        pool = epi[(epi["Type"] == ftype) & (epi["admin1Name"].isin(regions))]
        if len(pool) < count:
            pool = epi[(epi["Type"] == ftype)]  # fallback to national pool

        sampled = pool.sample(n=count, random_state=SEED + fac_id)

        for _, row in sampled.iterrows():
            fac_id += 1
            region = row["admin1Name"]
            params = REGIONAL_PARAMS.get(region, REGIONAL_PARAMS["Oromia"])
            tfr, cbr, imr, cov_mult, inst_del_rate = params
            urban_adj = URBAN_RURAL_COVERAGE_ADJ[tier]

            # Calculate target infants
            surviving_infants = catchment * (cbr / 1000) * (1 - imr / 1000)

            # Effective coverage for consumption estimation
            eff_coverage = min(1.0, cov_mult * urban_adj)

            # Wastage factor depends on session volume
            infants_per_session = surviving_infants / (sess_wk * 52) if sess_wk > 0 else surviving_infants / 26

            facilities.append({
                "facility_id": f"FAC-{fac_id:03d}",
                "name": row["Name"],
                "type": ftype,
                "access_tier": tier,
                "region": region,
                "woreda": row.get("admin2Name", "Unknown"),
                "latitude": row["Latitude"],
                "longitude": row["Longitude"],
                "catchment_pop": catchment,
                "cbr_per_1000": cbr,
                "imr_per_1000": imr,
                "target_infants_annual": round(surviving_infants),
                "eff_coverage_rate": round(eff_coverage, 3),
                "institutional_delivery_rate": inst_del_rate,
                "lead_time_days_mean": lt_mean,
                "lead_time_days_sd": lt_sd,
                "sessions_per_week": sess_wk,
                "infants_per_session": round(infants_per_session, 1),
                "coverage_multiplier": cov_mult,
                "urban_rural_adj": urban_adj,
                "has_fridge": ftype != "Health Post",
                "supervising_hc_id": None,  # filled later for HPs
            })

    return pd.DataFrame(facilities)


# ============================================================
# 4. HC-HP CLUSTER ASSIGNMENT
# ============================================================
def assign_hc_hp_clusters(facilities_df):
    """Assign each HP to a supervising HC based on proximity within region."""
    hcs = facilities_df[facilities_df["type"] == "Health Center"].copy()
    hps = facilities_df[facilities_df["type"] == "Health Post"].copy()

    clusters = []

    for _, hc in hcs.iterrows():
        # Find HPs in same region or nearest HPs
        region_hps = hps[hps["region"] == hc["region"]]
        if len(region_hps) == 0:
            region_hps = hps  # fallback

        # Sort by distance (simple euclidean on lat/lon — sufficient for assignment)
        dists = np.sqrt((region_hps["latitude"] - hc["latitude"])**2 +
                        (region_hps["longitude"] - hc["longitude"])**2)
        nearest = dists.nsmallest(min(5, len(region_hps))).index

        # Assign up to 5 HPs (or however many are available in region)
        for hp_idx in nearest:
            hp_fac_id = facilities_df.loc[hp_idx, "facility_id"]
            facilities_df.loc[hp_idx, "supervising_hc_id"] = hc["facility_id"]
            clusters.append({
                "hc_id": hc["facility_id"],
                "hp_id": hp_fac_id,
                "hc_name": hc["name"],
                "hp_name": facilities_df.loc[hp_idx, "name"],
                "region": hc["region"],
            })

    # HPs without HC get assigned to nearest HC overall
    unassigned = facilities_df[(facilities_df["type"] == "Health Post") &
                               (facilities_df["supervising_hc_id"].isna())]
    if len(unassigned) > 0 and len(hcs) > 0:
        for idx, hp in unassigned.iterrows():
            dists = np.sqrt((hcs["latitude"] - hp["latitude"])**2 +
                            (hcs["longitude"] - hp["longitude"])**2)
            nearest_hc = hcs.loc[dists.idxmin()]
            facilities_df.loc[idx, "supervising_hc_id"] = nearest_hc["facility_id"]
            clusters.append({
                "hc_id": nearest_hc["facility_id"],
                "hp_id": hp["facility_id"],
                "hc_name": nearest_hc["name"],
                "hp_name": hp["name"],
                "region": nearest_hc["region"],
            })

    return pd.DataFrame(clusters) if clusters else pd.DataFrame()


# ============================================================
# 5. TARGET POPULATION PER FACILITY × ANTIGEN
# ============================================================
def calculate_targets(facilities_df, vaccines):
    """Calculate annual target doses per facility × antigen."""
    targets = []
    for _, fac in facilities_df.iterrows():
        target_infants = fac["target_infants_annual"]
        eff_cov = fac["eff_coverage_rate"]
        inst_del = fac["institutional_delivery_rate"]
        ftype = fac["type"]

        for vax_code, vax_desc, sched_wk, doses_series, vial_sz, wst_lo, wst_hi, nat_cov in vaccines:
            # Determine facility's role for this antigen
            if ftype == "Health Post":
                # Primary delivery point — serves full target
                demand_infants = target_infants
                role = "primary"
            elif ftype == "Health Center":
                # Catch-up only (~10% of satellite HPs' combined target)
                # The HC's "own" target infants represent its catchment,
                # but it only directly vaccinates the ~10% dropout/catch-up fraction
                demand_infants = target_infants * 0.10
                role = "catch-up"
            elif ftype == "Hospital":
                if sched_wk == 0:  # Birth doses (BCG, OPV0)
                    # Hospital delivers birth doses to institutional deliveries
                    demand_infants = target_infants * inst_del
                    role = "birth-dose"
                else:
                    # Other schedule vaccines: minimal catch-up only
                    demand_infants = target_infants * 0.03
                    role = "catch-up"

            # Annual target doses = infants × doses in series
            target_doses_annual = demand_infants * doses_series

            # Expected consumption = target × effective coverage
            expected_consumption_annual = target_doses_annual * eff_cov

            # Wastage depends on volume: low-volume facilities waste more
            weekly_doses = expected_consumption_annual / 52
            is_low_volume = weekly_doses < (vial_sz * 2)
            wastage_rate = wst_lo if is_low_volume else wst_hi

            # Total stock needed = consumption + wastage
            stock_needed_annual = expected_consumption_annual * (1 + wastage_rate)

            targets.append({
                "facility_id": fac["facility_id"],
                "antigen": vax_code,
                "role": role,
                "target_infants": round(demand_infants),
                "doses_in_series": doses_series,
                "target_doses_annual": round(target_doses_annual),
                "expected_consumption_annual": round(expected_consumption_annual),
                "wastage_rate": round(wastage_rate, 3),
                "stock_needed_annual": round(stock_needed_annual),
                "weekly_consumption_baseline": round(expected_consumption_annual / 52, 1),
                "vial_size": vial_sz,
            })

    return pd.DataFrame(targets)


# ============================================================
# 6. SHOCK EVENT CALENDAR
# ============================================================
def generate_shock_calendar(facilities_df, n_weeks):
    """Generate disruption events for the 3-year simulation period."""
    shocks = []
    shock_id = 0

    for week in range(n_weeks):
        week_date = START_DATE + timedelta(weeks=week)
        month = week_date.month

        for _, fac in facilities_df.iterrows():
            fid = fac["facility_id"]
            tier = fac["access_tier"]

            # --- RAINY SEASON (Kiremt: June-September) ---
            if month in [6, 7, 8, 9]:
                if tier in ["rural_remote", "pastoral"]:
                    # Lead time multiplier during rainy season
                    lt_mult = RNG.uniform(1.5, 2.5)
                    # Chance of complete missed delivery
                    if RNG.random() < 0.15:  # 15% chance per week
                        shock_id += 1
                        shocks.append({
                            "shock_id": shock_id,
                            "facility_id": fid,
                            "week": week,
                            "week_date": str(week_date),
                            "shock_type": "rainy_season_road_closure",
                            "severity": "high" if tier == "pastoral" else "medium",
                            "demand_multiplier": 0.7,  # reduced attendance too
                            "supply_multiplier": 0.0,  # no delivery
                            "lead_time_multiplier": lt_mult,
                            "duration_weeks": 1,
                            "affected_antigens": "ALL",
                            "notes": f"Kiremt road closure, {tier} tier",
                        })
                elif tier == "rural_road":
                    if RNG.random() < 0.05:  # lower probability
                        shock_id += 1
                        shocks.append({
                            "shock_id": shock_id,
                            "facility_id": fid,
                            "week": week,
                            "week_date": str(week_date),
                            "shock_type": "rainy_season_delay",
                            "severity": "low",
                            "demand_multiplier": 0.9,
                            "supply_multiplier": 0.5,
                            "lead_time_multiplier": 1.5,
                            "duration_weeks": 1,
                            "affected_antigens": "ALL",
                            "notes": "Kiremt supply delay, rural road",
                        })

        # --- MEASLES SIA (Oct-Nov, biennial: years 1, 3, 5, 7 of simulation) ---
        year_offset = week // 52
        if month in [10, 11] and year_offset in [0, 2, 4, 6]:
            if week % 2 == 0:  # inject once per campaign window
                for _, fac in facilities_df.iterrows():
                    if fac["type"] in ["Health Post", "Health Center"]:
                        shock_id += 1
                        shocks.append({
                            "shock_id": shock_id,
                            "facility_id": fac["facility_id"],
                            "week": week,
                            "week_date": str(week_date),
                            "shock_type": "measles_sia_campaign",
                            "severity": "planned",
                            "demand_multiplier": RNG.uniform(4.0, 8.0),
                            "supply_multiplier": 1.5,  # pre-positioned stock
                            "lead_time_multiplier": 1.0,
                            "duration_weeks": 2,
                            "affected_antigens": "MCV",
                            "notes": "Planned measles SIA campaign",
                        })

        # --- POLIO SNID (April each year) ---
        if month == 4 and week % 52 < 5:
            if week_date.day <= 7:  # first week of April
                for _, fac in facilities_df.iterrows():
                    if fac["type"] in ["Health Post", "Health Center"]:
                        shock_id += 1
                        shocks.append({
                            "shock_id": shock_id,
                            "facility_id": fac["facility_id"],
                            "week": week,
                            "week_date": str(week_date),
                            "shock_type": "polio_snid_campaign",
                            "severity": "planned",
                            "demand_multiplier": RNG.uniform(3.0, 6.0),
                            "supply_multiplier": 1.5,
                            "lead_time_multiplier": 1.0,
                            "duration_weeks": 2,
                            "affected_antigens": "OPV",
                            "notes": "Planned polio SNID campaign",
                        })

    # --- COLD CHAIN FAILURES (random, HC only) ---
    hcs = facilities_df[facilities_df["type"] == "Health Center"]
    for _, hc in hcs.iterrows():
        # 2-3 failures over 3 years for remote, 0-1 for urban
        n_failures = RNG.poisson(2 if hc["access_tier"] in ["rural_remote", "pastoral"] else 0.5)
        for _ in range(min(n_failures, 4)):
            fail_week = RNG.integers(0, n_weeks)
            duration = RNG.integers(1, 5)  # 1-4 weeks
            shock_id += 1
            shocks.append({
                "shock_id": shock_id,
                "facility_id": hc["facility_id"],
                "week": fail_week,
                "week_date": str(START_DATE + timedelta(weeks=int(fail_week))),
                "shock_type": "cold_chain_failure",
                "severity": "high",
                "demand_multiplier": 1.0,
                "supply_multiplier": 0.0,  # can't distribute
                "lead_time_multiplier": 1.0,
                "duration_weeks": int(duration),
                "affected_antigens": "ALL",
                "notes": f"Cold chain failure at HC, {duration}wk duration. Stock at risk of wastage.",
            })

    # --- EPSS HUB STOCKOUT (affects all facilities in a region) ---
    hub_events = [
        (RNG.integers(25, 45), "PENTA", ["Oromia", "SNNP"]),
        (RNG.integers(80, 110), "PCV", ["Amhara", "Tigray"]),
    ]
    for hub_week, antigen, affected_regions in hub_events:
        for _, fac in facilities_df.iterrows():
            if fac["region"] in affected_regions:
                shock_id += 1
                shocks.append({
                    "shock_id": shock_id,
                    "facility_id": fac["facility_id"],
                    "week": int(hub_week),
                    "week_date": str(START_DATE + timedelta(weeks=int(hub_week))),
                    "shock_type": "epss_hub_stockout",
                    "severity": "high",
                    "demand_multiplier": 1.0,
                    "supply_multiplier": 0.0,
                    "lead_time_multiplier": 3.0,
                    "duration_weeks": RNG.integers(3, 7),
                    "affected_antigens": antigen,
                    "notes": f"EPSS hub stockout of {antigen} affecting {', '.join(affected_regions)}",
                })

    # --- CONFLICT DISRUPTION (region-targeted, historically grounded) ---
    # Event 1: Tigray war analogue — weeks 60-80 (~Mar-Aug 2024 in simulation)
    # Event 2: Amhara conflict analogue — weeks 200-220 (~Oct 2026 - Mar 2027)
    # Both based on real recent Ethiopian conflicts.
    conflict_events = [
        {"region": "Tigray",  "week_start": 60,  "week_end": 80,
         "notes": "Tigray-region service disruption (conflict analogue)"},
        {"region": "Amhara",  "week_start": 200, "week_end": 220,
         "notes": "Amhara-region service disruption (conflict analogue)"},
    ]
    for evt_idx, evt in enumerate(conflict_events):
        if evt["week_start"] >= n_weeks:
            continue  # event falls beyond simulation horizon
        region_facs = facilities_df[facilities_df["region"] == evt["region"]]
        if region_facs.empty:
            continue
        affected = region_facs.sample(
            n=min(3, len(region_facs)), random_state=SEED + evt_idx
        )
        for _, fac in affected.iterrows():
            for w in range(evt["week_start"], min(evt["week_end"], n_weeks)):
                shock_id += 1
                shocks.append({
                    "shock_id": shock_id,
                    "facility_id": fac["facility_id"],
                    "week": w,
                    "week_date": str(START_DATE + timedelta(weeks=w)),
                    "shock_type": "conflict_disruption",
                    "severity": "critical",
                    "demand_multiplier": RNG.uniform(0.1, 0.3),
                    "supply_multiplier": 0.0,
                    "lead_time_multiplier": 5.0,
                    "duration_weeks": 1,
                    "affected_antigens": "ALL",
                    "notes": evt["notes"],
                })

    # --- PANDEMIC SHOCK (all facilities, weeks 8-30) ---
    for _, fac in facilities_df.iterrows():
        for w in range(8, 30):
            severity = max(0.4, 1.0 - 0.03 * (w - 8))  # gradual recovery
            shock_id += 1
            shocks.append({
                "shock_id": shock_id,
                "facility_id": fac["facility_id"],
                "week": w,
                "week_date": str(START_DATE + timedelta(weeks=w)),
                "shock_type": "pandemic_disruption",
                "severity": "high" if w < 15 else "medium",
                "demand_multiplier": round(severity, 2),
                "supply_multiplier": round(max(0.5, severity), 2),
                "lead_time_multiplier": 1.5 if w < 15 else 1.2,
                "duration_weeks": 1,
                "affected_antigens": "ALL",
                "notes": f"Pandemic disruption (recovery phase: {round((1-severity)*100)}% reduced)",
            })

    return pd.DataFrame(shocks)


# ============================================================
# 7. BIRTH SEASONALITY
# ============================================================
def birth_seasonality_factor(week_date):
    """Ethiopia birth seasonality: peak Sep-Oct, trough Feb-Mar.
    Returns multiplier centered on 1.0 with ~15-20% amplitude."""
    month = week_date.month
    # Sinusoidal model: peak at month 9.5 (late Sep)
    phase = 2 * np.pi * (month - 9.5) / 12
    return 1.0 + 0.15 * np.cos(phase)


# ============================================================
# 8. MAIN SIMULATION ENGINE
# ============================================================
def simulate_stock_ledger(facilities_df, targets_df, shocks_df, clusters_df, n_weeks):
    """Generate weekly stock ledger and session logs for all facilities."""

    stock_ledger = []
    session_log = []
    delivery_log = []

    # Build lookup structures
    shock_lookup = {}
    if len(shocks_df) > 0:
        for _, shock in shocks_df.iterrows():
            key = (shock["facility_id"], shock["week"])
            if key not in shock_lookup:
                shock_lookup[key] = []
            shock_lookup[key].append(shock)

    # HC → HP cluster lookup
    hc_to_hps = {}
    if len(clusters_df) > 0:
        for _, cl in clusters_df.iterrows():
            hc_id = cl["hc_id"]
            if hc_id not in hc_to_hps:
                hc_to_hps[hc_id] = []
            hc_to_hps[hc_id].append(cl["hp_id"])

    # HP collection parameters by access tier
    # Interval = how many weeks between HC collection trips
    # Reliability = probability HEW successfully makes the collection trip
    HP_COLLECTION_INTERVAL = {
        "urban":        2,
        "rural_road":   2,
        "rural_remote": 3,
        "pastoral":     3,   # bimonthly rather than monthly — more realistic
    }
    HP_COLLECTION_RELIABILITY = {
        "urban":        0.96,   # target 3-5%
        "rural_road":   0.82,   # target 8-12%
        "rural_remote": 0.76,   # target 15-20%
        "pastoral":     0.63,   # target 28-38%
    }
    HP_INITIAL_BUFFER_WEEKS = {
        "urban":        5,
        "rural_road":   7,
        "rural_remote": 11,
        "pastoral":     14,
    }

    # Initialize stock levels per facility x antigen
    stock = {}
    reorder_timer = {}
    for _, tgt in targets_df.iterrows():
        fid = tgt["facility_id"]
        ag = tgt["antigen"]
        fac = facilities_df[facilities_df["facility_id"] == fid].iloc[0]
        tier = fac["access_tier"]

        if fac["type"] == "Health Post":
            buf = HP_INITIAL_BUFFER_WEEKS.get(tier, 4)
        else:
            buf = 4

        initial = max(tgt["vial_size"], tgt["weekly_consumption_baseline"] * buf)
        stock[(fid, ag)] = round(initial)

        if fac["type"] == "Health Post":
            reorder_timer[(fid, ag)] = HP_COLLECTION_INTERVAL.get(tier, 2)
        else:
            reorder_timer[(fid, ag)] = 4

    # Simulate week by week
    for week in range(n_weeks):
        week_date = START_DATE + timedelta(weeks=week)
        birth_mult = birth_seasonality_factor(week_date)

        for _, fac in facilities_df.iterrows():
            fid = fac["facility_id"]
            ftype = fac["type"]
            tier = fac["access_tier"]
            sess_wk = fac["sessions_per_week"]

            # Check for active shocks this week
            active_shocks = shock_lookup.get((fid, week), [])
            demand_mult = 1.0
            supply_mult = 1.0
            lt_mult = 1.0
            for shock in active_shocks:
                ag_filter = shock["affected_antigens"]
                demand_mult *= shock["demand_multiplier"]
                supply_mult *= shock["supply_multiplier"]
                lt_mult *= shock["lead_time_multiplier"]

            for _, tgt in targets_df[targets_df["facility_id"] == fid].iterrows():
                ag = tgt["antigen"]
                baseline = tgt["weekly_consumption_baseline"]
                vial_sz = tgt["vial_size"]
                wastage = tgt["wastage_rate"]

                # Check if this antigen is affected by antigen-specific shocks
                # Rebuild multipliers considering only shocks that apply to this antigen
                ag_demand_mult = 1.0
                ag_supply_mult = 1.0
                ag_lt_mult = 1.0
                for shock in active_shocks:
                    af = shock["affected_antigens"]
                    if af == "ALL" or af == ag:
                        ag_demand_mult *= shock["demand_multiplier"]
                        ag_supply_mult *= shock["supply_multiplier"]
                        ag_lt_mult *= shock["lead_time_multiplier"]

                # --- DEMAND CALCULATION ---
                # Birth seasonality affects birth-dose vaccines differently
                vax_info = [v for v in VACCINES if v[0] == ag][0]
                sched_wk = vax_info[2]

                if sched_wk == 0:
                    # Birth dose: immediate birth seasonality effect
                    seasonal = birth_mult
                elif sched_wk <= 14:
                    # 6-14 week schedule: lagged birth effect (~6 weeks)
                    lagged_date = week_date - timedelta(weeks=6)
                    seasonal = birth_seasonality_factor(lagged_date)
                else:
                    # 9-month schedule (MCV): 9-month lagged effect
                    lagged_date = week_date - timedelta(weeks=36)
                    seasonal = birth_seasonality_factor(lagged_date)

                # Weekly consumption — Poisson demand (count-data noise)
                # Real session attendance varies a lot; Poisson is natural for it.
                expected_demand = max(0.5, baseline * seasonal * ag_demand_mult)
                consumption_raw = float(RNG.poisson(expected_demand))

                # Round to vial increments (can't open half a vial)
                if vial_sz > 1:
                    vials_opened = max(1, int(np.ceil(consumption_raw / vial_sz)))
                    doses_used = min(consumption_raw, vials_opened * vial_sz)
                    doses_wasted = (vials_opened * vial_sz - consumption_raw) * wastage
                    doses_wasted = max(0, round(doses_wasted))
                else:
                    doses_used = round(consumption_raw)
                    doses_wasted = round(consumption_raw * wastage)

                doses_administered = round(consumption_raw)
                total_consumed = doses_administered + doses_wasted

                # --- STOCK DRAWDOWN ---
                opening_stock = stock[(fid, ag)]
                if opening_stock >= total_consumed:
                    actual_administered = doses_administered
                    actual_wasted = doses_wasted
                elif opening_stock > 0:
                    # Partial: administer what we can, reduced wastage
                    actual_administered = min(doses_administered, opening_stock)
                    actual_wasted = max(0, opening_stock - actual_administered)
                    total_consumed = actual_administered + actual_wasted
                else:
                    # STOCKOUT
                    actual_administered = 0
                    actual_wasted = 0
                    total_consumed = 0

                children_missed = max(0, doses_administered - actual_administered)
                closing_stock = max(0, opening_stock - total_consumed)

                # --- RESUPPLY LOGIC ---
                resupply_received = 0
                reorder_timer[(fid, ag)] -= 1

                # HC emergency reorder: if stock < 3.0 weeks, don't wait for 4-week cycle
                hc_emergency = (
                    ftype == "Health Center" and
                    closing_stock < baseline * 3.0 and
                    closing_stock > 0 and
                    reorder_timer[(fid, ag)] > 1
                )
                if hc_emergency:
                    reorder_timer[(fid, ag)] = 1  # trigger next week

                collection_due = reorder_timer[(fid, ag)] <= 0

                # HP collection reliability — HEW may miss the trip
                if collection_due and ftype == "Health Post":
                    reliability = HP_COLLECTION_RELIABILITY.get(tier, 0.70)
                    if RNG.random() > reliability:
                        # Missed collection — reset timer, no resupply this cycle
                        collection_due = False
                        reorder_timer[(fid, ag)] = HP_COLLECTION_INTERVAL.get(tier, 2)

                if collection_due:
                    if ag_supply_mult > 0:
                        # Replenish target: HPs aim for interval+2 weeks, HCs for 7 weeks
                        # Add stochastic variability — real EPSS deliveries are never exact.
                        if ftype == "Health Post":
                            interval = HP_COLLECTION_INTERVAL.get(tier, 2)
                            base_target = baseline * (interval + 2)
                        else:
                            base_target = baseline * 7
                        # Resupply target jitters ±30% (truck capacity, ordering errors, vial availability)
                        target_stock = base_target * RNG.uniform(0.70, 1.30)

                        order_qty = max(0, target_stock - closing_stock)
                        if vial_sz > 1:
                            order_qty = int(np.ceil(order_qty / vial_sz)) * vial_sz

                        # Partial fulfilment noise — even when supply chain is "OK", deliveries
                        # are typically 80-100% of order due to picking errors, vial breakage, etc.
                        fulfilment_noise = RNG.uniform(0.80, 1.00)
                        order_qty = round(order_qty * ag_supply_mult * fulfilment_noise)
                        resupply_received = order_qty
                        closing_stock += resupply_received

                        delivery_log.append({
                            "facility_id": fid,
                            "antigen": ag,
                            "week": week,
                            "week_date": str(week_date),
                            "quantity_ordered": round(max(0, target_stock - (closing_stock - resupply_received))),
                            "quantity_received": resupply_received,
                            "supply_multiplier": round(ag_supply_mult, 2),
                            "lead_time_actual": round(fac["lead_time_days_mean"] * lt_mult),
                            "source": "EPSS_hub" if ftype != "Health Post" else str(fac["supervising_hc_id"]),
                            "emergency_order": hc_emergency,
                        })

                    # Reset timer with ±1-week jitter — collection trips don't happen
                    # like clockwork (vehicle availability, road conditions, HEW workload).
                    if ftype == "Health Post":
                        nominal = HP_COLLECTION_INTERVAL.get(tier, 2)
                        reorder_timer[(fid, ag)] = max(1, nominal + int(RNG.integers(-1, 2)))
                    else:
                        reorder_timer[(fid, ag)] = max(2, 4 + int(RNG.integers(-1, 2)))

                stock[(fid, ag)] = round(closing_stock)

                # --- DAYS TO STOCKOUT ESTIMATE ---
                weekly_burn = max(0.1, baseline * seasonal)
                days_to_stockout = round((closing_stock / weekly_burn) * 7) if weekly_burn > 0 else 999

                # Alert threshold
                effective_lead_time = fac["lead_time_days_mean"] * lt_mult
                safety_buffer = 5  # days
                alert_threshold = effective_lead_time + safety_buffer
                alert_status = "critical" if days_to_stockout <= alert_threshold * 0.5 else \
                               "warning" if days_to_stockout <= alert_threshold else \
                               "ok"

                # --- RECORD ---
                stock_ledger.append({
                    "facility_id": fid,
                    "antigen": ag,
                    "week": week,
                    "week_date": str(week_date),
                    "opening_stock": opening_stock,
                    "doses_administered": actual_administered,
                    "doses_wasted": actual_wasted,
                    "total_consumed": total_consumed,
                    "resupply_received": resupply_received,
                    "closing_stock": round(closing_stock),
                    "target_demand": doses_administered,
                    "children_missed": children_missed,
                    "is_stockout": closing_stock <= 0,
                    "days_to_stockout": days_to_stockout,
                    "alert_status": alert_status,
                    "birth_seasonality_factor": round(seasonal, 3),
                    "demand_shock_multiplier": round(ag_demand_mult, 3),
                    "supply_shock_multiplier": round(ag_supply_mult, 3),
                })

                # Session log (aggregate for the week)
                if actual_administered > 0 or children_missed > 0:
                    sessions_held = max(1, round(sess_wk)) if closing_stock > 0 or opening_stock > 0 else 0
                    session_log.append({
                        "facility_id": fid,
                        "antigen": ag,
                        "week": week,
                        "week_date": str(week_date),
                        "sessions_held": sessions_held,
                        "sessions_planned": max(1, round(sess_wk)),
                        "doses_administered": actual_administered,
                        "doses_wasted": actual_wasted,
                        "children_reached": actual_administered,
                        "children_missed": children_missed,
                        "vials_opened": int(np.ceil(actual_administered / vial_sz)) if vial_sz > 0 else actual_administered,
                    })

    return pd.DataFrame(stock_ledger), pd.DataFrame(session_log), pd.DataFrame(delivery_log)


# ============================================================
# 9. HC CASCADE — POST-PROCESSING
# ============================================================
def apply_hc_cascade(stock_ledger_df, facilities_df, clusters_df):
    """When an HC stocks out, propagate the effect to its satellite HPs.
    Adds cascade flags and adjusts HP stock for weeks where HC was unable to supply."""

    if len(clusters_df) == 0:
        stock_ledger_df["cascade_affected"] = False
        stock_ledger_df["cascade_source_hc"] = None
        return stock_ledger_df

    stock_ledger_df["cascade_affected"] = False
    stock_ledger_df["cascade_source_hc"] = None

    hc_to_hps_map = {}
    for _, cl in clusters_df.iterrows():
        hc_id = cl["hc_id"]
        if hc_id not in hc_to_hps_map:
            hc_to_hps_map[hc_id] = []
        hc_to_hps_map[hc_id].append(cl["hp_id"])

    # Find HC stockout weeks
    hc_ids = facilities_df[facilities_df["type"] == "Health Center"]["facility_id"].values
    hc_stockouts = stock_ledger_df[
        (stock_ledger_df["facility_id"].isin(hc_ids)) &
        (stock_ledger_df["is_stockout"] == True)
    ]

    for _, hc_row in hc_stockouts.iterrows():
        hc_id = hc_row["facility_id"]
        week = hc_row["week"]
        ag = hc_row["antigen"]

        if hc_id not in hc_to_hps_map:
            continue

        for hp_id in hc_to_hps_map[hc_id]:
            mask = (
                (stock_ledger_df["facility_id"] == hp_id) &
                (stock_ledger_df["week"] == week) &
                (stock_ledger_df["antigen"] == ag)
            )
            if mask.any():
                stock_ledger_df.loc[mask, "cascade_affected"] = True
                stock_ledger_df.loc[mask, "cascade_source_hc"] = hc_id
                # Reduce HP resupply to 0 for this week (HC can't provide)
                stock_ledger_df.loc[mask, "resupply_received"] = 0
                # Recalculate closing stock
                idx = stock_ledger_df[mask].index[0]
                opening = stock_ledger_df.loc[idx, "opening_stock"]
                consumed = stock_ledger_df.loc[idx, "total_consumed"]
                new_closing = max(0, opening - consumed)
                stock_ledger_df.loc[idx, "closing_stock"] = new_closing
                stock_ledger_df.loc[idx, "is_stockout"] = new_closing <= 0
                if new_closing <= 0:
                    stock_ledger_df.loc[idx, "alert_status"] = "critical"

    return stock_ledger_df


# ============================================================
# 10. WRITE TO SQLITE
# ============================================================
def write_to_sqlite(db_path, facilities_df, vaccines, targets_df,
                    clusters_df, shocks_df, stock_ledger_df,
                    session_log_df, delivery_log_df):
    """Write all tables to SQLite database."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    # Facilities
    facilities_df.to_sql("facilities", conn, index=False, if_exists="replace")

    # Vaccines
    vax_df = pd.DataFrame(vaccines, columns=[
        "antigen_code", "description", "schedule_age_weeks", "doses_in_series",
        "vial_size", "wastage_rate_low_vol", "wastage_rate_high_vol", "national_wuenic_2024"
    ])
    vax_df.to_sql("vaccines", conn, index=False, if_exists="replace")

    # HC-HP clusters
    if len(clusters_df) > 0:
        clusters_df.to_sql("hc_hp_clusters", conn, index=False, if_exists="replace")

    # Target population
    targets_df.to_sql("target_population", conn, index=False, if_exists="replace")

    # Shock events
    if len(shocks_df) > 0:
        shocks_df.to_sql("shock_events", conn, index=False, if_exists="replace")

    # Stock ledger
    stock_ledger_df.to_sql("stock_ledger", conn, index=False, if_exists="replace")

    # Session log
    session_log_df.to_sql("session_log", conn, index=False, if_exists="replace")

    # Delivery log
    delivery_log_df.to_sql("delivery_log", conn, index=False, if_exists="replace")

    # Create indexes for performance
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX idx_stock_fac_week ON stock_ledger(facility_id, week)")
    cursor.execute("CREATE INDEX idx_stock_fac_ag ON stock_ledger(facility_id, antigen)")
    cursor.execute("CREATE INDEX idx_stock_alert ON stock_ledger(alert_status)")
    cursor.execute("CREATE INDEX idx_session_fac ON session_log(facility_id, week)")
    cursor.execute("CREATE INDEX idx_delivery_fac ON delivery_log(facility_id, week)")
    cursor.execute("CREATE INDEX idx_shock_fac ON shock_events(facility_id, week)")
    conn.commit()
    conn.close()


# ============================================================
# 11. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VaxAlert Synthetic Data Generator")
    print("=" * 60)

    # Step 1: Sample facilities
    print("\n[1/7] Sampling 30 facilities from CSV...")
    facilities_df = sample_facilities(FACILITY_CSV)
    print(f"  Sampled: {len(facilities_df)} facilities")
    print(f"  Types: {facilities_df['type'].value_counts().to_dict()}")
    print(f"  Regions: {facilities_df['region'].nunique()} unique")

    # Step 2: Assign HC-HP clusters
    print("\n[2/7] Assigning HC-HP clusters...")
    clusters_df = assign_hc_hp_clusters(facilities_df)
    print(f"  Created {len(clusters_df)} HC-HP linkages")

    # Step 3: Calculate targets
    print("\n[3/7] Calculating target populations...")
    targets_df = calculate_targets(facilities_df, VACCINES)
    print(f"  Generated {len(targets_df)} facility × antigen targets")
    print(f"  Total target infants (HP primary): {targets_df[targets_df['role']=='primary']['target_infants'].sum():,}")

    # Step 4: Generate shock calendar
    print("\n[4/7] Generating shock events...")
    shocks_df = generate_shock_calendar(facilities_df, N_WEEKS)
    print(f"  Generated {len(shocks_df)} shock events")
    print(f"  Types: {shocks_df['shock_type'].value_counts().to_dict()}")

    # Step 5: Simulate stock ledger
    print("\n[5/7] Simulating weekly stock movements (this takes a moment)...")
    stock_ledger_df, session_log_df, delivery_log_df = simulate_stock_ledger(
        facilities_df, targets_df, shocks_df, clusters_df, N_WEEKS
    )
    print(f"  Stock ledger: {len(stock_ledger_df):,} rows")
    print(f"  Session log: {len(session_log_df):,} rows")
    print(f"  Delivery log: {len(delivery_log_df):,} rows")

    # Step 6: Apply HC cascade
    print("\n[6/7] Applying HC→HP cascade effects...")
    stock_ledger_df = apply_hc_cascade(stock_ledger_df, facilities_df, clusters_df)
    cascade_count = stock_ledger_df["cascade_affected"].sum()
    print(f"  Cascade-affected records: {cascade_count}")

    # Step 7: Write to SQLite
    print(f"\n[7/7] Writing to SQLite: {DB_PATH}")
    write_to_sqlite(DB_PATH, facilities_df, VACCINES, targets_df,
                    clusters_df, shocks_df, stock_ledger_df,
                    session_log_df, delivery_log_df)

    # Summary statistics
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    conn = sqlite3.connect(DB_PATH)
    for table in ["facilities", "vaccines", "hc_hp_clusters", "target_population",
                  "shock_events", "stock_ledger", "session_log", "delivery_log"]:
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn).iloc[0, 0]
            print(f"  {table:25s} {count:>10,} rows")
        except:
            print(f"  {table:25s} (table not created)")

    # Key metrics
    print("\n--- Key Metrics ---")
    stockout_weeks = pd.read_sql(
        "SELECT COUNT(*) as n FROM stock_ledger WHERE is_stockout = 1", conn
    ).iloc[0, 0]
    total_weeks = pd.read_sql("SELECT COUNT(*) as n FROM stock_ledger", conn).iloc[0, 0]
    print(f"  Stockout rate: {stockout_weeks}/{total_weeks} ({stockout_weeks/total_weeks*100:.1f}%)")

    critical_alerts = pd.read_sql(
        "SELECT COUNT(*) as n FROM stock_ledger WHERE alert_status = 'critical'", conn
    ).iloc[0, 0]
    print(f"  Critical alerts: {critical_alerts:,}")

    children_missed = pd.read_sql(
        "SELECT SUM(children_missed) as n FROM stock_ledger", conn
    ).iloc[0, 0]
    print(f"  Total children missed: {int(children_missed):,}")

    cascade_affected = pd.read_sql(
        "SELECT COUNT(*) as n FROM stock_ledger WHERE cascade_affected = 1", conn
    ).iloc[0, 0]
    print(f"  Cascade-affected records: {cascade_affected:,}")

    conn.close()
    print(f"\nDatabase saved to: {DB_PATH}")
    print(f"File size: {os.path.getsize(DB_PATH) / 1024 / 1024:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
