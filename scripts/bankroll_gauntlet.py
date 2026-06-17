#!/usr/bin/env python3
"""
THE 100x GAUNTLET — Bankroll Growth Simulator
======================================================================
v3.7 CHANGES vs v3.6:
  · START_BANKROLL raised $150 → $350.
  · FB8 family ($168/race) commented out — too expensive for Phase 1.
  · Gates widened across all families:
      FB3: purse ceiling raised, sum windows opened slightly
      FB4: fav2 floor lowered 3.0→2.5, sum windows opened
      FB5: fav2 floor lowered 3.0→2.5, sum windows opened
      FB6: chalk_req removed on two gates, sum windows opened
      FB7: purse ceiling raised 20k→25k, sum windows opened
  · All other ruin machinery (cold-day 15%, forced-miss 12%,
    jitter ±5%, NaN→loss) retained from v3.6.
"""

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from numba import njit, prange
from numba.typed import List

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_FILE     = "RaceRecords_Output_2025.csv"
TEST_FILE      = "RaceRecords_Output_2026.csv"

START_BANKROLL = 350.0      # v3.7: raised from $150
RUIN_FLOOR     = 15.0
TARGET         = 77_777.0
MAX_STAKE_CAP  = 200.0
N_PATHS        = 2_000
MAX_TRADE_DAYS = 1_500
RACES_PER_DAY  = 8
RANDOM_SEED    = 42

# ── Ruin pressure (retained from v3.6) ───────────────────────────────────────
P_COLD_DAY           = 0.15
P_RACE_FORCED_MISS   = 0.12
PAYOUT_JITTER_FRAC   = 0.05

# ── Staking table ─────────────────────────────────────────────────────────────
STAKING_TABLE = [
    (0,       2_000,  {"T1": 0.025, "T2": 0.015, "T3": 0.010, "T4": 0.010}),
    (2_000,   8_000,  {"T1": 0.020, "T2": 0.0125,"T3": 0.0075,"T4": 0.0025}),
    (8_000,  20_000,  {"T1": 0.015, "T2": 0.010, "T3": 0.005, "T4": 0.000}),
    (20_000, 80_000,  {"T1": 0.010, "T2": 0.0075,"T3": 0.0025,"T4": 0.000}),
]

PHASES = [
    (0,      2_000,  "Phase 1\nSurvival", "#f43f5e"),
    (2_000,  8_000,  "Phase 2\nPlatform", "#f59e0b"),
    (8_000, 40_001,  "Phase 3\nScaling",  "#10b981"),
]

WOWSUPERWOW_ABS   = 25_000.0
WOWSUPERWOW_RATIO = 200.0

BG, GRID, TEXT, MUT = "#0f172a", "#1e293b", "#f8fafc", "#64748b"
ACCENT, RED, GOLD, CYAN = "#10b981", "#f43f5e", "#f59e0b", "#06b6d4"
TIER_COLORS = {"T1": GOLD, "T2": ACCENT, "T3": CYAN, "T4": MUT}
TIER_LABELS = {"T1": "TIER 1 🏆", "T2": "TIER 2 ✅", "T3": "TIER 3 💎", "T4": "TIER 4 📋"}

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENTAL GATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _chalky(row):
    return str(row.get("ChalkYN", "N")).strip().upper() == "Y"

def _first_race(row):
    return str(row.get("FirstRaceYN", "N")).strip().upper() == "Y"

def _is_sprint(row):
    dist = pd.to_numeric(row.get("Miles", row.get("Distance", 0)), errors="coerce")
    if pd.isna(dist): return False
    if dist > 4.0: dist = dist / 8.0
    return dist < 0.875

def _which_race_ok(row, mn=1, mx=11):
    wr = pd.to_numeric(row.get("WhichRace", 0), errors="coerce")
    return pd.notna(wr) and mn <= int(wr) <= mx

def _field_ok(row, mn=3, mx=12):
    n = pd.to_numeric(row.get("Runners", 0), errors="coerce")
    return pd.notna(n) and mn <= int(n) <= mx

def _purse_ok(row, lo=0, hi=999_999):
    p = pd.to_numeric(row.get("Purse", 0), errors="coerce")
    return pd.notna(p) and lo <= float(p) <= hi

def _sum_ok(row, lo=0.0, hi=999.0):
    s = pd.to_numeric(row.get("SumOf1st2Odds", 0), errors="coerce")
    return pd.notna(s) and lo <= float(s) <= hi

def _fav2_ok(row, lo=0.0, hi=999.0):
    f2_val = row.get("Fav2Exact", row.get("Fav2_odds", 0))
    f2 = pd.to_numeric(f2_val, errors="coerce")
    return pd.notna(f2) and lo <= float(f2) <= hi

def _make_env_vec(df, chalk_req=None, purse_lo=0, purse_hi=999_999,
                  field_min=3, field_max=12, sprint_req=None,
                  first_req=None, race_min=1, race_max=11,
                  sum_min=0.0, sum_max=999.0,
                  fav2_min=0.0, fav2_max=999.0):
    mask = np.ones(len(df), dtype=bool)

    if "WhichRace" in df.columns:
        wr = pd.to_numeric(df["WhichRace"], errors="coerce")
        mask &= wr.notna().values & (wr >= race_min).values & (wr <= race_max).values

    if "Runners" in df.columns:
        rn = pd.to_numeric(df["Runners"], errors="coerce")
        mask &= rn.notna().values & (rn >= field_min).values & (rn <= field_max).values

    if "Purse" in df.columns:
        pu = pd.to_numeric(df["Purse"], errors="coerce")
        mask &= pu.notna().values & (pu >= purse_lo).values & (pu <= purse_hi).values

    if "SumOf1st2Odds" in df.columns:
        su = pd.to_numeric(df["SumOf1st2Odds"], errors="coerce")
        mask &= su.notna().values & (su >= sum_min).values & (su <= sum_max).values

    fav2_col = next((c for c in ["Fav2Exact", "Fav2_odds"] if c in df.columns), None)
    if fav2_col:
        f2 = pd.to_numeric(df[fav2_col], errors="coerce")
        mask &= f2.notna().values & (f2 >= fav2_min).values & (f2 <= fav2_max).values

    if chalk_req is not None and "ChalkYN" in df.columns:
        chalk = (df["ChalkYN"].astype(str).str.strip().str.upper() == "Y").values
        mask &= chalk if chalk_req == "Y" else ~chalk

    if first_req is not None and "FirstRaceYN" in df.columns:
        first = (df["FirstRaceYN"].astype(str).str.strip().str.upper() == "Y").values
        mask &= first if first_req == "Y" else ~first

    if sprint_req is not None:
        dist_col = next((c for c in ["Miles", "Distance"] if c in df.columns), None)
        if dist_col:
            dist      = pd.to_numeric(df[dist_col], errors="coerce")
            furlongs  = dist.where(dist <= 4.0, dist / 8.0)
            is_sprint = (furlongs < 0.875).fillna(False).values
            mask &= is_sprint if sprint_req == "Y" else ~is_sprint

    return mask

def _make_env(chalk_req=None, purse_lo=0, purse_hi=999_999,
              field_min=3, field_max=12, sprint_req=None,
              first_req=None, race_min=1, race_max=11,
              sum_min=0.0, sum_max=999.0,
              fav2_min=0.0, fav2_max=999.0):

    def _f(row):
        if not _which_race_ok(row, race_min, race_max): return False
        if not _field_ok(row, field_min, field_max):    return False
        if not _purse_ok(row, purse_lo, purse_hi):      return False
        if not _sum_ok(row, sum_min, sum_max):           return False
        if not _fav2_ok(row, fav2_min, fav2_max):        return False
        if chalk_req == "N" and _chalky(row):            return False
        if chalk_req == "Y" and not _chalky(row):        return False
        if first_req == "N" and _first_race(row):        return False
        if first_req == "Y" and not _first_race(row):    return False
        if sprint_req == "Y" and not _is_sprint(row):    return False
        if sprint_req == "N" and _is_sprint(row):        return False
        return True

    def _fvec(df):
        return _make_env_vec(
            df,
            chalk_req=chalk_req, purse_lo=purse_lo, purse_hi=purse_hi,
            field_min=field_min, field_max=field_max, sprint_req=sprint_req,
            first_req=first_req, race_min=race_min, race_max=race_max,
            sum_min=sum_min, sum_max=sum_max,
            fav2_min=fav2_min, fav2_max=fav2_max,
        )

    _f.vec = _fvec
    return _f

# ══════════════════════════════════════════════════════════════════════════════
# HIT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_ranked(raw):
    if pd.isna(raw): return None
    tokens = str(raw).strip().split()
    if len(tokens) < 2: return None
    try:    return [int(t) for t in tokens]
    except: return None

def _hit_tri145(pos, n):
    return (
        len(pos) >= 3
        and pos[0] == 1
        and pos[1] in {2, 3, 4}
        and pos[2] in {2, 3, 4, 5}
    )

def _hit_always_true(pos, n):
    return True

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT REGISTRY — v3.7
#
# TICKET COST REMINDER:
#   FB3 : 3! = 6 combos × $1.00 base = $6.00   | payout_mult=0.50 ($1 on $2 base)
#   FB4 : P(4,4)=24  × $0.10       = $2.40     | payout_mult=0.10 (dime on $1 base)
#   FB5 : P(5,4)=120 × $0.10       = $12.00    | payout_mult=0.10
#   FB6 : P(6,4)=360 × $0.10       = $36.00    | payout_mult=0.10
#   FB7 : P(7,4)=840 × $0.10       = $84.00    | payout_mult=0.10
#   FB8 : commented out — $168/race too expensive for Phase 1 survival
#
# GATE WIDENING vs v3.6:
#   FB3 : purse_hi removed on FB3_1/3/4; sum_max 4.99→5.99 on FB3_2/5/6
#   FB4 : fav2_min 3.0→2.5; sum windows ±1 wider; race window 3-9→2-10
#   FB5 : fav2_min 3.0→2.5; sum windows ±1 wider on FB5_2/4
#   FB6 : chalk_req removed on FB6_3/6; sum_max raised on FB6_4/6
#   FB7 : purse_hi 20k→25k on FB7_1/2/3/6; sum_max raised on FB7_1/2/4/5
# ══════════════════════════════════════════════════════════════════════════════

INSTRUMENTS = {

    # ── ANCHOR ────────────────────────────────────────────────────────────────
    # TRI145: 9 combos × $2 = $18.00
    # Trif_paid on $2 base → payout_mult = 1.0
    # MinJK positive, n=491, EV +$10.89 — keeper
    "TRI145": {
        "tier":        "T1",
        "ticket_cost": 18.00,
        "hit_func":    _hit_tri145,
        "payout_col":  "Trif_paid",
        "payout_mult": 1.0,
        "min_runners": 5,
        "ewpd":        2.8,
        "desc":        "Trifecta [1]×[2-4]×[2-5] — 9 combos × $2",
        "env_filter":  _make_env(sum_min=9.0, fav2_min=2.5),
    },

    # ── n=3: Full 3-horse trifecta box ($6.00) ────────────────────────────────
    # 3! = 6 combos × $1.00 base = $6.00
    # Trif_paid on $2 base → payout_mult = 0.50
    #
    # DROPPED vs v3.7:
    #   FB3_1  — MinJK deeply negative (outlier-driven, n=238 borderline)
    #   FB3_5  — MinJK negative, EV +$7.25 but collapses without outliers
    #
    # KEPT: FB3_2/3/4/6 — all show solid MinJK retention and positive raw EV
    "FB3_2": {
        "tier": "T1", "ticket_cost": 6.00,
        "hit_func": _hit_always_true, "payout_col": "Trif_paid",
        "payout_mult": 0.50, "min_runners": 3, "ewpd": 1.5,
        "desc": "FB3 Sprint + Sum:<6",
        "env_filter": _make_env(field_min=3, field_max=3,
                                sprint_req="Y", sum_max=5.99),
    },
    "FB3_3": {
        "tier": "T1", "ticket_cost": 6.00,
        "hit_func": _hit_always_true, "payout_col": "Trif_paid",
        "payout_mult": 0.50, "min_runners": 3, "ewpd": 1.5,
        "desc": "FB3 Sprint — any purse",
        "env_filter": _make_env(field_min=3, field_max=3, sprint_req="Y"),
    },
    "FB3_4": {
        "tier": "T1", "ticket_cost": 6.00,
        "hit_func": _hit_always_true, "payout_col": "Trif_paid",
        "payout_mult": 0.50, "min_runners": 3, "ewpd": 1.5,
        "desc": "FB3 Early(1-5) + Chalk:Y",
        "env_filter": _make_env(field_min=3, field_max=3,
                                race_max=5, chalk_req="Y"),
    },
    "FB3_6": {
        "tier": "T1", "ticket_cost": 6.00,
        "hit_func": _hit_always_true, "payout_col": "Trif_paid",
        "payout_mult": 0.50, "min_runners": 3, "ewpd": 1.5,
        "desc": "FB3 Purse:<12k + Sum:<6",
        "env_filter": _make_env(field_min=3, field_max=3,
                                purse_hi=11_999, sum_max=5.99),
    },

    # ── n=4: Dime superfecta box ($2.40) ──────────────────────────────────────
    # P(4,4) = 24 combos × $0.10 = $2.40
    # Superf_paid on $1 base → payout_mult = 0.10
    #
    # DROPPED vs v3.7:
    #   FB4_2  EV -$0.51, MinJK negative → DROP
    #   FB4_3  EV -$0.35, MinJK negative → DROP
    #   FB4_4  EV -$0.36, MinJK negative → DROP
    #   FB4_5  EV -$0.14, MinJK negative → DROP
    #   FB4_6  EV -$0.68, MinJK negative → DROP (worst in family)
    #
    # KEPT: FB4_1 only — EV +$8.66, n=119, HR 38.7%
    # FB4_1 has real signal: Chalk:N + Fav2:2.5+ selects upset-prone races
    "FB4_1": {
        "tier": "T1", "ticket_cost": 2.40,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 4, "ewpd": 1.5,
        "desc": "FB4 Chalk:N + Fav2:2.5+",
        "env_filter": _make_env(field_min=4, field_max=4,
                                chalk_req="N", fav2_min=2.5),
    },

    # ── n=5: Dime superfecta box ($12.00) ─────────────────────────────────────
    # P(5,4) = 120 combos × $0.10 = $12.00
    # Superf_paid on $1 base → payout_mult = 0.10
    #
    # DROPPED vs v3.7:
    #   FB5_2  EV +$6.68 but MinJK collapses — outlier-sensitive → DROP
    #   FB5_4  EV +$6.10 similar MinJK collapse pattern → DROP
    #   FB5_6  EV +$1.33 on $12 ticket — 0.011% ROI, pure noise → DROP
    #
    # KEPT: FB5_1/3/5 — all show EV >$7, reasonable MinJK, distinct gates
    "FB5_1": {
        "tier": "T1", "ticket_cost": 12.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 5, "ewpd": 1.5,
        "desc": "FB5 Chalk:N + Fav2:2.5+",
        "env_filter": _make_env(field_min=5, field_max=5,
                                chalk_req="N", fav2_min=2.5),
    },
    "FB5_3": {
        "tier": "T1", "ticket_cost": 12.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 5, "ewpd": 1.5,
        "desc": "FB5 Sprint + Chalk:N + Fav2:2.5+",
        "env_filter": _make_env(field_min=5, field_max=5,
                                sprint_req="Y", chalk_req="N",
                                fav2_min=2.5),
    },
    "FB5_5": {
        "tier": "T1", "ticket_cost": 12.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 5, "ewpd": 1.5,
        "desc": "FB5 First:N + Sum:5-13",
        "env_filter": _make_env(field_min=5, field_max=5,
                                first_req="N",
                                sum_min=5.0, sum_max=13.0),
    },

    # ── n=6: Dime superfecta box ($36.00) ─────────────────────────────────────
    # v3.7 widening: chalk_req removed on FB6_3/6; sum_max raised on FB6_4/6
    "FB6_1": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Sprint + Chalk:N + Fav2:2.5+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                sprint_req="Y", chalk_req="N",
                                fav2_min=2.5),
    },
    "FB6_2": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Race:2-10 + Chalk:N + Fav2:2.5+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                race_min=2, race_max=10,
                                chalk_req="N", fav2_min=2.5),
    },
    "FB6_3": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Sum:4-11",          # chalk_req removed vs v3.6
        "env_filter": _make_env(field_min=6, field_max=6,
                                sum_min=4.0, sum_max=11.0),
    },
    "FB6_4": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Sprint + Chalk:N + Sum:3-11",
        "env_filter": _make_env(field_min=6, field_max=6,
                                sprint_req="Y", chalk_req="N",
                                sum_min=3.0, sum_max=11.0),
    },
    "FB6_5": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Chalk:N + Fav2:2.5+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                chalk_req="N", fav2_min=2.5),
    },
    "FB6_6": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Sum:3-13",          # chalk_req removed vs v3.6
        "env_filter": _make_env(field_min=6, field_max=6,
                                sum_min=3.0, sum_max=13.0),
    },

    # ── n=7: Dime superfecta box ($84.00) ─────────────────────────────────────
    # v3.7 widening: purse_hi 20k→25k on FB7_1/2/3/6;
    #                sum_max raised on FB7_1/2/4/5
    "FB7_1": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Purse:5-25k + Chalk:N + Sum:4-10",
        "env_filter": _make_env(field_min=7, field_max=7,
                                purse_lo=5_000, purse_hi=25_000,
                                chalk_req="N",
                                sum_min=4.0, sum_max=10.0),
    },
    "FB7_2": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Purse:5-25k + Chalk:N + Sum:4-11",
        "env_filter": _make_env(field_min=7, field_max=7,
                                purse_lo=5_000, purse_hi=25_000,
                                chalk_req="N",
                                sum_min=4.0, sum_max=11.0),
    },
    "FB7_3": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Purse:5-25k + Sprint + Chalk:N + Fav2:2.5+",
        "env_filter": _make_env(field_min=7, field_max=7,
                                purse_lo=5_000, purse_hi=25_000,
                                sprint_req="Y", chalk_req="N",
                                fav2_min=2.5),
    },
    "FB7_4": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Chalk:N + Sum:4-11 + Fav2:2.5+",
        "env_filter": _make_env(field_min=7, field_max=7,
                                chalk_req="N",
                                sum_min=4.0, sum_max=11.0,
                                fav2_min=2.5),
    },
    "FB7_5": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Sprint + Race:2-10 + Chalk:N + Sum:4-11",
        "env_filter": _make_env(field_min=7, field_max=7,
                                sprint_req="Y",
                                race_min=2, race_max=10,
                                chalk_req="N",
                                sum_min=4.0, sum_max=11.0),
    },
    "FB7_6": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.10, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Purse:5-25k + Chalk:N + Fav2:2.5+",
        "env_filter": _make_env(field_min=7, field_max=7,
                                purse_lo=5_000, purse_hi=25_000,
                                chalk_req="N", fav2_min=2.5),
    },

    # ── n=8: SUP8888 — COMMENTED OUT ($168/race; too expensive for Phase 1) ───
    # "FB8_1": { ... },
    # "FB8_2": { ... },
    # "FB8_3": { ... },
    # "FB8_4": { ... },
    # "FB8_5": { ... },
    # "FB8_6": { ... },

    # ── n=8: COMMENTED OUT — $168/race; too expensive for Phase 1 ─────────────
    # "FB8_1": { ... },
    # "FB8_2": { ... },
    # "FB8_3": { ... },
    # "FB8_4": { ... },
    # "FB8_5": { ... },
    # "FB8_6": { ... },

    # ── M_ENGINE — DROPPED ────────────────────────────────────────────────────
    # M_ENGINE_FvP: EV -$0.15, n=75,020 → systematic negative drag
    # M_ENGINE_FvS: EV -$0.17, n=75,020 → systematic negative drag
    #
    # The "survival floor" hypothesis is rejected by the data.
    # At 75,020 observations the signal is unambiguous: both instruments
    # destroy capital at every bankroll level. The T4 staking fraction
    # (0.010 in Phase 1) means they fire constantly and bleed steadily.
    # With the FB6/FB7 portfolio providing positive EV, the M_ENGINE
    # instruments are net harmful. DROPPED.
    #
    # "M_ENGINE_FvP": { ... },
    # "M_ENGINE_FvS": { ... },
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING + WOWSUPERWOW NULLING
# ══════════════════════════════════════════════════════════════════════════════

def _apply_wowsuperwow(df):
    sup_col   = next((c for c in ["Superf_paid", "Superfecta_paid"]
                      if c in df.columns), None)
    exact_col = next((c for c in ["ExactaTwoPayout","ExactaPays","Exacta_paid"]
                      if c in df.columns), None)
    if sup_col is None: return df

    sup_vals   = pd.to_numeric(df[sup_col], errors="coerce").fillna(0)
    abs_mask   = sup_vals > WOWSUPERWOW_ABS
    ratio_mask = pd.Series(False, index=df.index)

    if exact_col is not None:
        ex_vals    = pd.to_numeric(df[exact_col], errors="coerce").fillna(0)
        denom      = ex_vals.replace(0, np.nan)
        ratio_mask = (ex_vals > 0) & ((sup_vals / denom) > WOWSUPERWOW_RATIO)

    wow_mask = abs_mask | ratio_mask
    n_wow    = int(wow_mask.sum())

    if n_wow > 0:
        payout_cols = [c for c in df.columns
                       if c.endswith("_paid") or c.endswith("Payout")]
        df = df.copy()
        df.loc[wow_mask, payout_cols] = np.nan
        print(f"    ⚡ WowSuperWow: nulled {n_wow:,} outlier row(s) "
              f"across {len(payout_cols)} payout column(s)")
    return df

def _load_and_prep(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  ✔ Loaded {filepath}  ({len(df):,} rows)")
    except FileNotFoundError:
        print(f"  ⚠  {filepath} not found — skipping")
        return pd.DataFrame()

    df = _apply_wowsuperwow(df)

    if "Superfecta_paid" in df.columns and "Superf_paid" not in df.columns:
        df["Superf_paid"] = df["Superfecta_paid"]
    if "Trifecta_paid" in df.columns and "Trif_paid" not in df.columns:
        df["Trif_paid"] = df["Trifecta_paid"]

    for col in ["Runners", "Purse", "WhichRace", "SumOf1st2Odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["Superf_paid", "Trif_paid", "ExactaTwoPayout", "FvP_pd", "FvS_pd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# EMPIRICAL P&L EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_empirical_pnl_fast(df, name, inst):
    if df.empty:
        return np.array([], dtype=np.float64)

    mask = df['Runners'] >= inst['min_runners']
    if 'WhichRace' in df.columns:
        mask &= (df['WhichRace'] >= 1) & (df['WhichRace'] <= 11)

    env_filter = inst['env_filter']
    if hasattr(env_filter, 'vec'):
        env_mask = env_filter.vec(df)
    else:
        env_mask = df.apply(env_filter, axis=1).values
    mask = mask & env_mask

    if not mask.any():
        return np.array([], dtype=np.float64)

    payout_col  = inst['payout_col']
    ticket_cost = inst["ticket_cost"]
    payout_mult = inst.get("payout_mult", 1.0)

    if inst.get("use_pd_direct"):
        if payout_col in df.columns:
            mask = mask & df[payout_col].notna().values
        if not mask.any():
            return np.array([], dtype=np.float64)
        return df.loc[mask, payout_col].values.astype(np.float64)

    qualifying_idx = np.where(mask)[0]

    if payout_col not in df.columns:
        return np.full(len(qualifying_idx), -ticket_cost, dtype=np.float64)

    raw_payouts = df[payout_col].values
    rng         = np.random.default_rng(RANDOM_SEED)

    if 'RANKED_RESULTS' in df.columns:
        raw_results = df['RANKED_RESULTS'].values
        runners_arr = df['Runners'].values
        pnl_list    = []
        for i in qualifying_idx:
            pay = raw_payouts[i]
            if pd.isna(pay) or float(pay) == 0.0:
                pnl_list.append(-ticket_cost)
            else:
                hit = inst['hit_func'](_parse_ranked(raw_results[i]),
                                       int(runners_arr[i]))
                if hit:
                    gross = float(pay) * payout_mult
                    gross *= 1.0 + rng.uniform(-PAYOUT_JITTER_FRAC,
                                                PAYOUT_JITTER_FRAC)
                    pnl_list.append(gross - ticket_cost)
                else:
                    pnl_list.append(-ticket_cost)
        return np.array(pnl_list, dtype=np.float64)
    else:
        pnl_list = []
        for i in qualifying_idx:
            pay = raw_payouts[i]
            if pd.isna(pay) or float(pay) == 0.0:
                pnl_list.append(-ticket_cost)
            else:
                gross = float(pay) * payout_mult
                gross *= 1.0 + rng.uniform(-PAYOUT_JITTER_FRAC,
                                            PAYOUT_JITTER_FRAC)
                pnl_list.append(gross - ticket_cost)
        return np.array(pnl_list, dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA MONTE CARLO KERNEL
# ══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True)
def run_gauntlet_core(
    n_paths, max_days, races_per_day,
    start_br, target, ruin_floor, max_stake_cap,
    p_cold_day, p_race_forced_miss,
    pnl_pools, pool_stats, staking_table, phase_bounds,
):
    all_paths  = np.zeros((n_paths, max_days + 1), dtype=np.float64)
    all_phases = np.zeros((n_paths, max_days + 1), dtype=np.int8)
    outcomes   = np.zeros(n_paths, dtype=np.int8)
    days_arr   = np.zeros(n_paths, dtype=np.int32)

    n_instruments = len(pnl_pools)
    n_phases      = len(phase_bounds)

    for p in prange(n_paths):
        br = start_br
        all_paths[p, 0] = br

        current_phase = n_phases - 1
        for i in range(n_phases):
            if phase_bounds[i, 0] <= br < phase_bounds[i, 1]:
                current_phase = int(phase_bounds[i, 2])
                break
        all_phases[p, 0] = current_phase

        for d in range(1, max_days + 1):
            t1 = t2 = t3 = t4 = 0.0
            for i in range(len(staking_table)):
                if staking_table[i, 0] <= br < staking_table[i, 1]:
                    t1 = staking_table[i, 2]; t2 = staking_table[i, 3]
                    t3 = staking_table[i, 4]; t4 = staking_table[i, 5]
                    break
            fractions = np.array([t1, t2, t3, t4])

            cold_day = np.random.random() < p_cold_day

            for r in range(races_per_day):
                total_ewpd = 0.0
                for i in range(n_instruments):
                    tier_idx = int(pool_stats[i, 4])
                    if fractions[tier_idx] > 0:
                        total_ewpd += pool_stats[i, 3]

                if total_ewpd == 0.0:
                    continue

                pick_val = np.random.random() * total_ewpd
                cum_sum  = 0.0
                idx      = 0
                for i in range(n_instruments):
                    tier_idx = int(pool_stats[i, 4])
                    if fractions[tier_idx] > 0:
                        cum_sum += pool_stats[i, 3]
                        if pick_val <= cum_sum:
                            idx = i
                            break

                tier_idx  = int(pool_stats[idx, 4])
                unit_frac = fractions[tier_idx]
                tc        = pool_stats[idx, 2]
                hr        = pool_stats[idx, 0]
                avg_pay   = pool_stats[idx, 1]

                b = (avg_pay / tc) - 1.0 if tc > 0.0 else 0.0
                f_full       = (b * hr - (1.0 - hr)) / b if b > 1e-9 else 0.0
                kelly_dollar = max(0.0, br) * (0.25 * f_full)
                unit_ceiling = max(0.0, br) * unit_frac
                stake        = min(kelly_dollar, unit_ceiling)

                if stake > max_stake_cap:
                    stake = max_stake_cap
                if stake <= 0.0:
                    continue

                pool    = pnl_pools[idx]
                raw_pnl = pool[np.random.randint(0, len(pool))]

                if cold_day:
                    raw_pnl = -tc
                elif np.random.random() < p_race_forced_miss:
                    raw_pnl = -tc

                br += raw_pnl * (stake / tc)

                if br >= target or br <= ruin_floor:
                    break

            all_paths[p, d] = br

            current_phase = n_phases - 1
            for i in range(n_phases):
                if phase_bounds[i, 0] <= br < phase_bounds[i, 1]:
                    current_phase = int(phase_bounds[i, 2])
                    break
            all_phases[p, d] = current_phase

            if br >= target:
                outcomes[p] = 1; days_arr[p] = d
                all_paths[p, d:]  = br
                all_phases[p, d:] = current_phase
                break
            if br <= ruin_floor:
                outcomes[p] = 2; days_arr[p] = d
                all_paths[p, d:]  = ruin_floor
                all_phases[p, d:] = current_phase
                break

        if outcomes[p] == 0:
            days_arr[p] = max_days

    return all_paths, outcomes, days_arr, all_phases

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA DATA PREP + SIMULATION WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def _get_numba_ready_data(empirical_map):
    typed_pools = List()
    stats_list  = []
    tier_map    = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}

    for name, pool in empirical_map.items():
        inst = INSTRUMENTS[name]
        typed_pools.append(pool.astype(np.float64))
        tc = inst["ticket_cost"]
        if inst.get("use_pd_direct"):
            hit_mask = pool > 0
        else:
            hit_mask = pool > -tc
        hr      = float(np.mean(hit_mask)) if len(pool) > 0 else 0.0
        avg_pay = float((pool[hit_mask] + tc).mean()) if hit_mask.any() else 0.0
        stats_list.append([hr, avg_pay, tc,
                           inst["ewpd"], float(tier_map[inst["tier"]])])

    stk = [[float(lo), float(hi), f["T1"], f["T2"], f["T3"], f["T4"]]
            for lo, hi, f in STAKING_TABLE]
    pb  = [[float(lo), float(hi), float(i)]
            for i, (lo, hi, _, _) in enumerate(PHASES)]

    return typed_pools, np.array(stats_list), np.array(stk), np.array(pb)

def _simulate_paths(empirical_map):
    pools, stats, staking, phase_bounds = _get_numba_ready_data(empirical_map)
    np.random.seed(RANDOM_SEED)

    try:
        from numba import get_num_threads
        n_threads = get_num_threads()
    except Exception:
        n_threads = "?"

    print(f"🔥 Firing Numba JIT kernels (Parallel Threads: {n_threads})...")
    print(f"   Cold-day probability   : {P_COLD_DAY * 100:.0f}%")
    print(f"   Per-race forced miss   : {P_RACE_FORCED_MISS * 100:.0f}%")
    print(f"   Payout jitter          : ±{PAYOUT_JITTER_FRAC * 100:.0f}%")

    paths_m, outcome_codes, days_arr, phases_m = run_gauntlet_core(
        N_PATHS, MAX_TRADE_DAYS, RACES_PER_DAY,
        START_BANKROLL, TARGET, RUIN_FLOOR, MAX_STAKE_CAP,
        P_COLD_DAY, P_RACE_FORCED_MISS,
        pools, stats, staking, phase_bounds,
    )

    code_map = {0: "timeout", 1: "success", 2: "ruin"}
    outcomes = [code_map[c] for c in outcome_codes]

    paths, phase_log, final_brs, days_list = [], [], [], days_arr.tolist()
    for p in range(N_PATHS):
        d = days_arr[p]
        paths.append(paths_m[p, :d + 1])
        phase_log.append(phases_m[p, :d + 1].tolist())
        final_brs.append(float(paths_m[p, d]))

    return paths, outcomes, days_list, final_brs, phase_log

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_stats(paths, outcomes, days_list, final_brs):
    n         = len(paths)
    n_success = outcomes.count("success")
    n_ruin    = outcomes.count("ruin")
    n_timeout = outcomes.count("timeout")

    success_days = [d for d, o in zip(days_list, outcomes) if o == "success"]
    ruin_days    = [d for d, o in zip(days_list, outcomes) if o == "ruin"]

    med_br  = float(np.median(final_brs))
    med_idx = int(np.argmin(np.abs(np.array(final_brs) - med_br)))

    all_dd  = [float(np.max(np.maximum.accumulate(eq) - eq)) for eq in paths]
    pcts    = [10, 25, 50, 75, 90]
    br_pcts = {p: round(float(np.percentile(final_brs, p)), 2) for p in pcts}

    return {
        "n_paths":              n,
        "n_success":            n_success,
        "n_ruin":               n_ruin,
        "n_timeout":            n_timeout,
        "success_rate":         round(n_success / n * 100, 2),
        "ruin_rate":            round(n_ruin    / n * 100, 2),
        "timeout_rate":         round(n_timeout / n * 100, 2),
        "median_days_success":  int(np.median(success_days)) if success_days else None,
        "median_days_ruin":     int(np.median(ruin_days))    if ruin_days    else None,
        "p10_days_success":     int(np.percentile(success_days, 10)) if len(success_days) > 1 else None,
        "p90_days_success":     int(np.percentile(success_days, 90)) if len(success_days) > 1 else None,
        "median_final_br":      round(med_br, 2),
        "mean_final_br":        round(float(np.mean(final_brs)), 2),
        "br_percentiles":       br_pcts,
        "median_max_dd":        round(float(np.median(all_dd)), 2),
        "required_bankroll_3x": round(float(np.median(all_dd)) * 3, 2),
        "median_path_idx":      med_idx,
        "p_cold_day":           P_COLD_DAY,
        "p_race_forced_miss":   P_RACE_FORCED_MISS,
        "payout_jitter_frac":   PAYOUT_JITTER_FRAC,
    }

# ══════════════════════════════════════════════════════════════════════════════
# CHARTING
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(color=GRID, linestyle=":", alpha=0.4, linewidth=0.6)
    if title:  ax.set_title(title,  color=TEXT, fontsize=11, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=MUT, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUT, fontsize=9)

def chart_paths(paths, outcomes, stats):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    _style_ax(ax,
              title=f"THE 100x GAUNTLET — "
                    f"${START_BANKROLL:,.0f} → ${TARGET:,.0f} Simulation Paths",
              xlabel="Trade Days", ylabel="Bankroll ($, symlog scale)")
    ax.set_yscale('symlog', linthresh=100)

    rng_draw = np.random.default_rng(RANDOM_SEED)
    draw_idx = rng_draw.choice(len(paths), size=min(300, len(paths)), replace=False)

    for i in draw_idx:
        eq    = paths[i]
        color = ACCENT if outcomes[i]=="success" else RED if outcomes[i]=="ruin" else MUT
        alpha = 0.10   if outcomes[i]=="success" else 0.16 if outcomes[i]=="ruin" else 0.05
        ax.plot(eq, color=color, alpha=alpha, linewidth=0.5)

    med = paths[stats["median_path_idx"]]
    ax.plot(med, color=GOLD, linewidth=2.8, zorder=6,
            label=f"Median Path  (final: ${med[-1]:,.0f})")

    max_len = max(len(p) for p in paths)
    padded  = np.full((len(paths), max_len), np.nan)
    for i, p in enumerate(paths): padded[i, :len(p)] = p
    n_active = np.sum(~np.isnan(padded), axis=0)
    with np.errstate(all="ignore"):
        band_lo = np.nanpercentile(padded, 25, axis=0)
        band_hi = np.nanpercentile(padded, 75, axis=0)
    band_lo[n_active < 20] = band_hi[n_active < 20] = np.nan
    xs = np.arange(max_len); valid = ~np.isnan(band_lo) & ~np.isnan(band_hi)
    if valid.any():
        ax.fill_between(xs[valid], band_lo[valid], band_hi[valid],
                        color=GOLD, alpha=0.08, label="25th–75th pct band")

    milestones = {
        RUIN_FLOOR:     (f"${RUIN_FLOOR:,.0f} Ruin Floor",  RED,    "-.", 1.5),
        START_BANKROLL: (f"${START_BANKROLL:,.0f} Start",   ACCENT, ":",  1.2),
        2_000:          ("$2k Phase 1→2",                   MUT,    ":",  0.7),
        8_000:          ("$8k Phase 3",                     CYAN,   ":",  1.0),
        TARGET:         (f"${TARGET:,.0f} TARGET",          GOLD,   "--", 1.8),
    }
    for m, (lbl, color, ls, lw) in milestones.items():
        ax.axhline(m, color=color, linewidth=lw, linestyle=ls, alpha=0.65)
        ax.text(max_len * 0.005, m + (abs(m)*0.05 if m != 0 else 100),
                lbl, color=color, fontsize=7.5, va="bottom", alpha=0.85)

    mds   = stats["median_days_success"]
    p10_d = stats.get("p10_days_success")
    p90_d = stats.get("p90_days_success")
    ann = (f"Success : {stats['success_rate']:.1f}%\n"
           f"Ruin    : {stats['ruin_rate']:.1f}%\n"
           f"Median days : {'N/A' if not mds else f'{mds:,}d'}"
           + (f"  [P10–P90: {p10_d:,}–{p90_d:,}d]" if p10_d and p90_d else "") + "\n"
           f"Paths   : {stats['n_paths']:,}  |  Start: ${START_BANKROLL:,.0f}\n"
           f"Cold: {P_COLD_DAY*100:.0f}%  Miss: {P_RACE_FORCED_MISS*100:.0f}%")
    ax.text(0.98, 0.04, ann, transform=ax.transAxes,
            color=TEXT, fontsize=9, ha="right", va="bottom",
            bbox=dict(facecolor=GRID, edgecolor=MUT,
                      alpha=0.85, boxstyle="round,pad=0.5"))

    ax.legend(handles=[
        Line2D([0],[0], color=GOLD,   lw=2.5, label="Median Path"),
        Line2D([0],[0], color=ACCENT, lw=1.2, alpha=0.6, label="Success"),
        Line2D([0],[0], color=RED,    lw=1.2, alpha=0.6, label="Ruin"),
        Line2D([0],[0], color=MUT,    lw=1.2, alpha=0.4, label="Timeout"),
    ], facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    plt.savefig("Gauntlet_Paths.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Paths.png")

def chart_distribution(paths, outcomes, final_brs, stats):
    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1, title="Final Bankroll Distribution",
              xlabel="Final Bankroll ($)", ylabel="Number of Paths")

    shift   = abs(RUIN_FLOOR) + 1
    shifted = [b + shift for b in final_brs]
    bins    = np.logspace(np.log10(max(1, min(shifted))),
                          np.log10(max(shifted) + 1), 50)
    for ok, (color, alpha) in [("success",(ACCENT,0.75)),
                                 ("timeout",(MUT,0.60)),
                                 ("ruin",(RED,0.80))]:
        brs = [b + shift for b, o in zip(final_brs, outcomes) if o == ok]
        if brs:
            ax1.hist(brs, bins=bins, color=color, alpha=alpha,
                     label=f"{ok.capitalize()} ({len(brs):,})")
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x-shift:,.0f}"))
    ax1.axvline(TARGET + shift, color=GOLD, lw=2.0, linestyle="--",
                label=f"Target ${TARGET:,.0f}")
    ax1.axvline(stats["median_final_br"] + shift, color=CYAN, lw=1.5,
                linestyle=":", label=f"Median ${stats['median_final_br']:,.0f}")
    ylim_top = ax1.get_ylim()[1] or 1
    for p, v in stats["br_percentiles"].items():
        ax1.axvline(v + shift, color=MUT, lw=0.5, linestyle=":", alpha=0.4)
        ax1.text((v+shift)*1.04, ylim_top*0.88,
                 f"P{p}", color=MUT, fontsize=7, rotation=90)
    ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2, title="Cumulative Ruin & Success Probability Over Time",
              xlabel="Trade Days", ylabel="Cumulative Probability (%)")
    max_days = max(len(p) for p in paths)
    n = len(paths)

    def _cum(tgt):
        td = np.array([min(len(p)-1, max_days-1)
                        for p, o in zip(paths, outcomes) if o == tgt], dtype=np.intp)
        if not len(td): return np.zeros(max_days)
        return np.cumsum(np.bincount(td, minlength=max_days).astype(float)) / n * 100

    for curve, color, label in [
        (_cum("ruin"),    RED,    f"Cumulative ruin  (final: {stats['ruin_rate']:.1f}%)"),
        (_cum("success"), ACCENT, f"Cumulative success  (final: {stats['success_rate']:.1f}%)"),
    ]:
        ax2.plot(curve, color=color, linewidth=2.2, label=label)
        ax2.fill_between(range(max_days), curve, color=color, alpha=0.10)

    if stats["median_days_success"]:
        ax2.axvline(stats["median_days_success"], color=GOLD, lw=1.4,
                    linestyle="--",
                    label=f"Median: {stats['median_days_success']:,}d")
    if stats.get("p10_days_success"):
        ax2.axvline(stats["p10_days_success"], color=GOLD, lw=0.8,
                    linestyle=":", label=f"P10: {stats['p10_days_success']:,}d")

    ax2.set_xlim(0, max_days); ax2.set_ylim(0, 105)
    ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    fig.suptitle("THE 100x GAUNTLET — Outcome Analysis",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("Gauntlet_Distribution.png", dpi=150,
                facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Distribution.png")

def chart_instruments(empirical_pnl_map):
    names = sorted(
        [n for n in INSTRUMENTS
         if n in empirical_pnl_map and len(empirical_pnl_map[n]) > 0],
        key=lambda n: (INSTRUMENTS[n]["tier"], n)
    )
    if not names: return

    height = max(8, len(names) * 0.4)
    fig    = plt.figure(figsize=(16, height), facecolor=BG)
    fig.suptitle("THE 100x GAUNTLET — Per-Instrument Profile",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40)

    metrics = []
    for name in names:
        inst = INSTRUMENTS[name]; arr = empirical_pnl_map[name]; tc = inst["ticket_cost"]
        hm   = (arr > 0) if inst.get("use_pd_direct") else (arr > -tc)
        n_   = len(arr)
        hr   = hm.sum() / n_ * 100 if n_ > 0 else 0
        ev   = float(arr.mean()) if n_ > 0 else 0
        pay  = arr[hm] + tc
        avg_pay = float(pay.mean()) if len(pay) > 0 else 0
        roi  = (arr.sum() / (n_ * tc) * 100) if n_ > 0 else 0
        metrics.append({"name":name,"hr":hr,"ev":ev,"avg_pay":avg_pay,
                         "n":n_,"roi":roi,"tier":inst["tier"],"cost":tc})

    colors = [TIER_COLORS.get(m["tier"], MUT) for m in metrics]
    lbls   = [m["name"] for m in metrics]

    def _bar(ax, values, title, xlabel, fmt="$"):
        bars = ax.barh(lbls, values, color=colors, height=0.55)
        _style_ax(ax, title=title, xlabel=xlabel)
        ax.axvline(0, color=MUT, lw=0.8)
        mx = max(abs(v) for v in values) if values else 1
        for bar, val in zip(bars, values):
            xpos  = val + mx*0.02 if val >= 0 else val - mx*0.02
            align = "left" if val >= 0 else "right"
            label = (f"${val:,.2f}" if fmt=="$" else
                     f"{val:,.1f}%" if fmt=="%" else f"{val:,.0f}")
            ax.text(xpos, bar.get_y()+bar.get_height()/2,
                    label, color=TEXT, va="center", ha=align, fontsize=8)
        ax.set_yticks(range(len(lbls)))
        ax.set_yticklabels(lbls, color=MUT, fontsize=8)

    _bar(fig.add_subplot(gs[0,0]), [m["ev"]      for m in metrics],
         "EV per Race",        "Net P&L / Race ($)", "$")
    _bar(fig.add_subplot(gs[0,1]), [m["hr"]      for m in metrics],
         "Hit Rate",           "Hit Rate (%)",       "%")
    _bar(fig.add_subplot(gs[0,2]), [m["roi"]     for m in metrics],
         "IS ROI %",           "Return on Cost (%)", "%")
    _bar(fig.add_subplot(gs[1,0]), [m["avg_pay"] for m in metrics],
         "Avg Payout (hits)",  "Avg Payout ($)",     "$")
    _bar(fig.add_subplot(gs[1,1]), [m["cost"]    for m in metrics],
         "Ticket Cost",        "Cost per Race ($)",  "$")
    _bar(fig.add_subplot(gs[1,2]), [m["n"]       for m in metrics],
         "Sample Size",        "Qualifying Races",   "n")

    fig.legend(
        handles=[Line2D([0],[0], color=TIER_COLORS[t], lw=6, label=TIER_LABELS[t])
                 for t in ["T1","T2","T3","T4"]],
        facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
        fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5,-0.04)
    )
    plt.tight_layout()
    plt.savefig("Gauntlet_Instruments.png", dpi=150,
                facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Instruments.png")

def chart_phases(phase_log):
    if not phase_log: return
    max_days = max(len(p) for p in phase_log)
    pm = np.zeros((len(PHASES), max_days))
    for ps in phase_log:
        for d, ph in enumerate(ps):
            if ph < len(PHASES): pm[ph, d] += 1
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    _style_ax(ax, title="THE 100x GAUNTLET — Phase Transitions",
              xlabel="Trade Days", ylabel="% of Paths")
    dt = pm.sum(axis=0); dt[dt==0] = 1; pct = pm / dt * 100
    yb = np.zeros(max_days)
    for i, (_, _, lbl, color) in enumerate(PHASES):
        yt = yb + pct[i]
        ax.fill_between(range(max_days), yb, yt,
                        color=color, alpha=0.85, label=lbl.replace('\n',' '))
        yb = yt
    ax.set_ylim(0,100); ax.set_xlim(0,max_days)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
              loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig("Gauntlet_Phases.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Phases.png")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(empirical_pnl_map, stats):
    SEP = "═" * 80
    print(f"\n{SEP}")
    print("  THE 100x GAUNTLET — SIMULATION RESULTS  [v3.7]")
    print(f"  ${START_BANKROLL:,.0f} → ${TARGET:,.0f}  |  "
          f"{stats['n_paths']:,} paths  |  "
          f"up to {MAX_TRADE_DAYS:,} trade-days each")
    print(SEP)
    print(f"\n  {'OUTCOME':<25} {'COUNT':>6}  {'RATE':>7}")
    print("  " + "─"*40)
    print(f"  {'Success':<25} {stats['n_success']:>6,}  {stats['success_rate']:>6.1f}%")
    print(f"  {'Ruin':<25} {stats['n_ruin']:>6,}  {stats['ruin_rate']:>6.1f}%")
    print(f"  {'Timeout':<25} {stats['n_timeout']:>6,}  {stats['timeout_rate']:>6.1f}%")
    mds = stats["median_days_success"]
    mdr = stats["median_days_ruin"]
    print(f"\n  Median days to target         : {'N/A' if not mds else f'{mds:,}'}")
    print(f"  Median days to ruin           : {'N/A' if not mdr else f'{mdr:,}'}")
    print(f"  Median Max Drawdown           : ${stats['median_max_dd']:,.2f}")
    print(f"  3× MaxDD Capital Requirement  : ${stats['required_bankroll_3x']:,.2f}")
    print(f"  Median Final Bankroll         : ${stats['median_final_br']:,.2f}")
    print(f"  Mean Final Bankroll           : ${stats['mean_final_br']:,.2f}")
    print(f"\n  Stress parameters:")
    print(f"    Cold-day prob   : {stats['p_cold_day']*100:.0f}%")
    print(f"    Forced miss     : {stats['p_race_forced_miss']*100:.0f}%")
    print(f"    Payout jitter   : ±{stats['payout_jitter_frac']*100:.0f}%")
    print(f"\n  Final Bankroll Percentiles:")
    for p, v in stats["br_percentiles"].items():
        print(f"    P{p}: ${v:,.2f}")
    print(f"\n  Instrument pool sizes:")
    print(f"  {'Instrument':<30} {'N':>7}  {'EV':>9}  {'HR':>7}")
    print("  " + "─"*58)
    for name in sorted(empirical_pnl_map):
        arr = empirical_pnl_map[name]
        tc  = INSTRUMENTS[name]["ticket_cost"]
        hm  = (arr > 0) if INSTRUMENTS[name].get("use_pd_direct") else (arr > -tc)
        print(f"  {name:<30} {len(arr):>7,}  {arr.mean():>+9.2f}  {hm.mean()*100:>6.1f}%")
    print(SEP + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print(" 🏁 STARTING THE 100x GAUNTLET STRESS TESTING PIPELINE v3.7")
    print("=" * 80)

    df_2025 = _load_and_prep(TRAIN_FILE)
    df_2026 = _load_and_prep(TEST_FILE)

    if df_2025.empty and df_2026.empty:
        print("❌ CRITICAL ERROR: Source data empty."); exit(1)

    df_all = pd.concat([df_2025, df_2026], ignore_index=True)

    empirical_pnl_map = {}
    print("\nExtracting cross-validated historical P&L pools...")
    for name, inst in INSTRUMENTS.items():
        pnl_arr = _extract_empirical_pnl_fast(df_all, name, inst)
        if len(pnl_arr) > 0:
            tc = inst["ticket_cost"]
            hm = (pnl_arr > 0) if inst.get("use_pd_direct") else (pnl_arr > -tc)
            print(f"  {name:<30} : {len(pnl_arr):>7,} events  "
                  f"EV: {pnl_arr.mean():>+8.2f}  "
                  f"HR: {hm.mean()*100:>5.1f}%")
            empirical_pnl_map[name] = pnl_arr
        else:
            print(f"  {name:<30} : 0 events — skipped")

    if not empirical_pnl_map:
        print("❌ CRITICAL ERROR: No events passed filters."); exit(1)

    paths, outcomes, days_list, final_brs, phase_log = _simulate_paths(
        empirical_pnl_map)

    print("📊 Computing telemetry...")
    stats = _compute_stats(paths, outcomes, days_list, final_brs)

    print("🎨 Rendering dashboards...")
    chart_paths(paths, outcomes, stats)
    chart_distribution(paths, outcomes, final_brs, stats)
    chart_instruments(empirical_pnl_map)
    chart_phases(phase_log)

    print_report(empirical_pnl_map, stats)

    print("💾 Archiving...")
    with open("Gauntlet_Results.json", "w") as f:
        json.dump(stats, f, indent=4)
    flat = {k: v for k, v in stats.items() if k != "br_percentiles"}
    for p, v in stats["br_percentiles"].items():
        flat[f"br_percentile_P{p}"] = v
    pd.DataFrame([flat]).to_csv("Gauntlet_Results.csv", index=False)
    print("  ✓ Gauntlet_Results.json + Gauntlet_Results.csv")
    print("\n✅ THE 100x GAUNTLET v3.7 SIMULATION COMPLETE.")
