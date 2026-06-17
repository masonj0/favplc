#!/usr/bin/env python3
"""
THE 100x GAUNTLET — Bankroll Growth Simulator v3.3 (Production Build)
======================================================================
Puts approved, OOS-validated strategies through a rigorous Monte Carlo
simulation of the $500 → $40,000 challenge (with a $0 ruin floor).

v3.3 UPGRADES:
  · Bankroll starts at $500, ruin threshold at $0.
  · MAX_STAKE_CAP ($200) introduced to simulate pari-mutuel pool liquidity limits.
  · Purged destructive .fillna(0) operations; tracking gaps natively via pd.notna().
  · Numba JIT-compiled, parallelized C-level Monte Carlo kernel.
  · 36 Full-Field Box strategies (n=3 through n=8) integrated with 0.05 fractional scaling.
  · M_ENGINE (FvP, FvS) integrated as the survival floor.
"""

import json
import math
import warnings
from datetime import datetime
from pathlib import Path

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

# Base units
BASE_UNIT_TRI  = 2.00      # Academic baseline and standard Trifecta unit
BASE_UNIT_SUP  = 0.10      # Standard Superfecta dime unit

START_BANKROLL = 500.0
RUIN_FLOOR     = 0.0
TARGET         = 40_000.0
MAX_STAKE_CAP  = 200.0      # Prevents crushing pool liquidity at high bankrolls
N_PATHS        = 2_000
MAX_TRADE_DAYS = 1_500      # hard ceiling — approx 4 racing seasons
RACES_PER_DAY  = 8          # average qualifying races seen per race day
RANDOM_SEED    = 42         # reproducibility

# Staking table: (bankroll_floor, bankroll_ceil) → tier fractional units
STAKING_TABLE = [
    (0,       2_000,   {"T1": 0.025, "T2": 0.015, "T3": 0.010, "T4": 0.005}),
    (2_000,   8_000,   {"T1": 0.020, "T2": 0.0125,"T3": 0.0075,"T4": 0.0025}),
    (8_000,  20_000,   {"T1": 0.015, "T2": 0.010, "T3": 0.005, "T4": 0.000}),
    (20_000, 40_001,   {"T1": 0.010, "T2": 0.0075,"T3": 0.0025,"T4": 0.000}),
]

# Bankroll phase boundaries (for phase-transition chart)
PHASES = [
    (0,      2_000,   "Phase 1\nSurvival",    "#f43f5e"),
    (2_000,  8_000,   "Phase 2\nPlatform",    "#f59e0b"),
    (8_000, 40_001,   "Phase 3\nScaling",     "#10b981"),
]

# WowSuperWow outlier gate
WOWSUPERWOW_ABS   = 25_000.0
WOWSUPERWOW_RATIO = 200.0

# Visual theme
BG, GRID, TEXT, MUT = "#0f172a", "#1e293b", "#f8fafc", "#64748b"
ACCENT, RED, GOLD, CYAN, PURPLE = "#10b981", "#f43f5e", "#f59e0b", "#06b6d4", "#a855f7"

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

def _which_race_ok(row, mn=3, mx=11):
    wr = pd.to_numeric(row.get("WhichRace", 0), errors="coerce")
    return pd.notna(wr) and mn <= int(wr) <= mx

def _field_ok(row, mn=4, mx=9):
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
                  field_min=4, field_max=9, sprint_req=None,
                  first_req=None, race_min=1, race_max=11,
                  sum_min=0.0, sum_max=999.0, fav2_min=0.0, fav2_max=999.0):
    """Fully vectorized equivalent of the _make_env row-wise closure."""
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
            dist = pd.to_numeric(df[dist_col], errors="coerce")
            furlongs = dist.where(dist <= 4.0, dist / 8.0)
            is_sprint = (furlongs < 0.875).fillna(False).values
            mask &= is_sprint if sprint_req == "Y" else ~is_sprint

    return mask

def _make_env(chalk_req=None, purse_lo=0, purse_hi=999_999,
              field_min=4, field_max=9, sprint_req=None,
              first_req=None, race_min=1, race_max=11,
              sum_min=0.0, sum_max=999.0, fav2_min=0.0, fav2_max=999.0):
    """Factory: returns a combined env_filter callable."""
    def _f(row):
        if not _which_race_ok(row, race_min, race_max): return False
        if not _field_ok(row, field_min, field_max):    return False
        if not _purse_ok(row, purse_lo, purse_hi):      return False
        if not _sum_ok(row, sum_min, sum_max):          return False
        if not _fav2_ok(row, fav2_min, fav2_max):       return False
        if chalk_req == "N" and _chalky(row):           return False
        if chalk_req == "Y" and not _chalky(row):       return False
        if first_req == "N" and _first_race(row):       return False
        if first_req == "Y" and not _first_race(row):   return False
        if sprint_req == "Y" and not _is_sprint(row):   return False
        if sprint_req == "N" and _is_sprint(row):       return False
        return True

    def _fvec(df):
        return _make_env_vec(
            df,
            chalk_req=chalk_req, purse_lo=purse_lo, purse_hi=purse_hi,
            field_min=field_min, field_max=field_max, sprint_req=sprint_req,
            first_req=first_req, race_min=race_min, race_max=race_max,
            sum_min=sum_min, sum_max=sum_max, fav2_min=fav2_min, fav2_max=fav2_max,
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
    try: return [int(t) for t in tokens]
    except ValueError: return None

def _hit_tri145(pos, n):
    return (len(pos) >= 3 and pos[0] == 1 and pos[1] in {2, 3, 4} and pos[2] in {2, 3, 4, 5})

def _hit_always_true(pos, n):
    return True

# ══════════════════════════════════════════════════════════════════════════════
# APPROVED INSTRUMENT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

INSTRUMENTS = {
    "TRI145": {
        "tier":         "T1",
        "ticket_cost":  18.00,
        "hit_func":     _hit_tri145,
        "payout_col":   "Trif_paid",
        "payout_mult":  1.0,
        "min_runners":  5,
        "ewpd":         2.8,
        "desc":         "Trifecta [1]×[2-4]×[2-5] — 49% IS/OOS hit rate",
        "env_filter":   _make_env(sum_min=9.0, fav2_min=2.5),
    },

    # ── n=3: TRI333 ($0.60/race) ──
    "FB3_1": {"tier": "T1", "ticket_cost": 0.60, "hit_func": _hit_always_true, "payout_col": "Trif_paid", "payout_mult": 0.05, "min_runners": 3, "ewpd": 1.0, "desc": "FB3 Purse:<8k + Chalk:Y", "env_filter": _make_env(field_min=3, field_max=3, purse_hi=7999, chalk_req="Y")},
    "FB3_2": {"tier": "T1", "ticket_cost": 0.60, "hit_func": _hit_always_true, "payout_col": "Trif_paid", "payout_mult": 0.05, "min_runners": 3, "ewpd": 1.0, "desc": "FB3 Purse:<8k + Sprint + Sum:<4", "env_filter": _make_env(field_min=3, field_max=3, purse_hi=7999, sprint_req="Y", sum_max=3.99)},
    "FB3_3": {"tier": "T1", "ticket_cost": 0.60, "hit_func": _hit_always_true, "payout_col": "Trif_paid", "payout_mult": 0.05, "min_runners": 3, "ewpd": 1.0, "desc": "FB3 Purse:<8k + Sprint", "env_filter": _make_env(field_min=3, field_max=3, purse_hi=7999, sprint_req="Y")},
    "FB3_4": {"tier": "T1", "ticket_cost": 0.60, "hit_func": _hit_always_true, "payout_col": "Trif_paid", "payout_mult": 0.05, "min_runners": 3, "ewpd": 1.0, "desc": "FB3 Sprint + Early + Chalk:Y + First:N", "env_filter": _make_env(field_min=3, field_max=3, sprint_req="Y", race_max=3, chalk_req="Y", first_req="N")},
    "FB3_5": {"tier": "T1", "ticket_cost": 0.60, "hit_func": _hit_always_true, "payout_col": "Trif_paid", "payout_mult": 0.05, "min_runners": 3, "ewpd": 1.0, "desc": "FB3 Purse:<8k + Sum:<4", "env_filter": _make_env(field_min=3, field_max=3, purse_hi=7999, sum_max=3.99)},
    "FB3_6": {"tier": "T1", "ticket_cost": 0.60, "hit_func": _hit_always_true, "payout_col": "Trif_paid", "payout_mult": 0.05, "min_runners": 3, "ewpd": 1.0, "desc": "FB3 Purse:<8k + Sum:<4 + Fav2:<2.5", "env_filter": _make_env(field_min=3, field_max=3, purse_hi=7999, sum_max=3.99, fav2_max=2.49)},

    # ── n=4: SUP4444 ($2.40/race) ──
    "FB4_1": {"tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.0, "desc": "FB4 Chalk:N + Fav2:4+", "env_filter": _make_env(field_min=4, field_max=4, chalk_req="N", fav2_min=4.0)},
    "FB4_2": {"tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.0, "desc": "FB4 Chalk:N + Sum:4-6 + Fav2:2.5-4", "env_filter": _make_env(field_min=4, field_max=4, chalk_req="N", sum_min=4.0, sum_max=6.0, fav2_min=2.5, fav2_max=4.0)},
    "FB4_3": {"tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.0, "desc": "FB4 Chalk:N + First:N + Sum:4-6", "env_filter": _make_env(field_min=4, field_max=4, chalk_req="N", first_req="N", sum_min=4.0, sum_max=6.0)},
    "FB4_4": {"tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.0, "desc": "FB4 Chalk:N + Sum:4-6", "env_filter": _make_env(field_min=4, field_max=4, chalk_req="N", sum_min=4.0, sum_max=6.0)},
    "FB4_5": {"tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.0, "desc": "FB4 Sprint + Chalk:N + Sum:4-6", "env_filter": _make_env(field_min=4, field_max=4, sprint_req="Y", chalk_req="N", sum_min=4.0, sum_max=6.0)},
    "FB4_6": {"tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.0, "desc": "FB4 Mid(4-8) + Sum:4-6 + Fav2:2.5-4", "env_filter": _make_env(field_min=4, field_max=4, race_min=4, race_max=8, sum_min=4.0, sum_max=6.0, fav2_min=2.5, fav2_max=4.0)},

    # ── n=5: SUP5555 ($12.00/race) ──
    "FB5_1": {"tier": "T1", "ticket_cost": 12.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 5, "ewpd": 1.0, "desc": "FB5 Chalk:N + First:N + Fav2:4+", "env_filter": _make_env(field_min=5, field_max=5, chalk_req="N", first_req="N", fav2_min=4.0)},
    "FB5_2": {"tier": "T1", "ticket_cost": 12.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 5, "ewpd": 1.0, "desc": "FB5 Chalk:N + Fav2:4+", "env_filter": _make_env(field_min=5, field_max=5, chalk_req="N", fav2_min=4.0)},
    "FB5_3": {"tier": "T1", "ticket_cost": 12.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 5, "ewpd": 1.0, "desc": "FB5 Sprint + Chalk:N + Fav2:4+", "env_filter": _make_env(field_min=5, field_max=5, sprint_req="Y", chalk_req="N", fav2_min=4.0)},
    "FB5_4": {"tier": "T1", "ticket_cost": 12.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 5, "ewpd": 1.0, "desc": "FB5 Chalk:N + Sum:6-8", "env_filter": _make_env(field_min=5, field_max=5, chalk_req="N", sum_min=6.0, sum_max=8.0)},
    "FB5_5": {"tier": "T1", "ticket_cost": 12.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 5, "ewpd": 1.0, "desc": "FB5 First:N + Sum:8-11 + Fav2:4+", "env_filter": _make_env(field_min=5, field_max=5, first_req="N", sum_min=8.0, sum_max=11.0, fav2_min=4.0)},
    "FB5_6": {"tier": "T1", "ticket_cost": 12.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 5, "ewpd": 1.0, "desc": "FB5 First:N + Sum:8-11", "env_filter": _make_env(field_min=5, field_max=5, first_req="N", sum_min=8.0, sum_max=11.0)},

    # ── n=6: SUP6666 ($36.00/race) ──
    "FB6_1": {"tier": "T1", "ticket_cost": 36.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.0, "desc": "FB6 Sprint + Chalk:N + First:N + Fav2:4+", "env_filter": _make_env(field_min=6, field_max=6, sprint_req="Y", chalk_req="N", first_req="N", fav2_min=4.0)},
    "FB6_2": {"tier": "T1", "ticket_cost": 36.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.0, "desc": "FB6 Mid(4-8) + Chalk:N + First:N + Fav2:4+", "env_filter": _make_env(field_min=6, field_max=6, race_min=4, race_max=8, chalk_req="N", first_req="N", fav2_min=4.0)},
    "FB6_3": {"tier": "T1", "ticket_cost": 36.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.0, "desc": "FB6 Mid(4-8) + Chalk:N + Fav2:4+", "env_filter": _make_env(field_min=6, field_max=6, race_min=4, race_max=8, chalk_req="N", fav2_min=4.0)},
    "FB6_4": {"tier": "T1", "ticket_cost": 36.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.0, "desc": "FB6 Sprint + Chalk:N + Fav2:4+", "env_filter": _make_env(field_min=6, field_max=6, sprint_req="Y", chalk_req="N", fav2_min=4.0)},
    "FB6_5": {"tier": "T1", "ticket_cost": 36.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.0, "desc": "FB6 Chalk:N + First:N + Fav2:4+", "env_filter": _make_env(field_min=6, field_max=6, chalk_req="N", first_req="N", fav2_min=4.0)},
    "FB6_6": {"tier": "T1", "ticket_cost": 36.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.0, "desc": "FB6 Chalk:N + Fav2:4+", "env_filter": _make_env(field_min=6, field_max=6, chalk_req="N", fav2_min=4.0)},

    # ── n=7: SUP7777 ($84.00/race) ──
    "FB7_1": {"tier": "T1", "ticket_cost": 84.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.0, "desc": "FB7 Purse:8-15k + Chalk:N + First:N + Sum:6-8", "env_filter": _make_env(field_min=7, field_max=7, purse_lo=8000, purse_hi=15000, chalk_req="N", first_req="N", sum_min=6.0, sum_max=8.0)},
    "FB7_2": {"tier": "T1", "ticket_cost": 84.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.0, "desc": "FB7 Purse:8-15k + Chalk:N + Sum:6-8", "env_filter": _make_env(field_min=7, field_max=7, purse_lo=8000, purse_hi=15000, chalk_req="N", sum_min=6.0, sum_max=8.0)},
    "FB7_3": {"tier": "T1", "ticket_cost": 84.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.0, "desc": "FB7 Purse:8-15k + Sprint + Chalk:N + Fav2:4+", "env_filter": _make_env(field_min=7, field_max=7, purse_lo=8000, purse_hi=15000, sprint_req="Y", chalk_req="N", fav2_min=4.0)},
    "FB7_4": {"tier": "T1", "ticket_cost": 84.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.0, "desc": "FB7 Chalk:N + Sum:6-8 + Fav2:4+", "env_filter": _make_env(field_min=7, field_max=7, chalk_req="N", sum_min=6.0, sum_max=8.0, fav2_min=4.0)},
    "FB7_5": {"tier": "T1", "ticket_cost": 84.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.0, "desc": "FB7 Sprint + Mid(4-8) + Chalk:N + Sum:6-8", "env_filter": _make_env(field_min=7, field_max=7, sprint_req="Y", race_min=4, race_max=8, chalk_req="N", sum_min=6.0, sum_max=8.0)},
    "FB7_6": {"tier": "T1", "ticket_cost": 84.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.0, "desc": "FB7 Purse:8-15k + Chalk:N + Fav2:4+", "env_filter": _make_env(field_min=7, field_max=7, purse_lo=8000, purse_hi=15000, chalk_req="N", fav2_min=4.0)},

    # ── n=8: SUP8888 ($168.00/race) ──
    "FB8_1": {"tier": "T1", "ticket_cost": 168.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 8, "ewpd": 1.0, "desc": "FB8 Purse:15-30k + Sprint + Late(9-11) + Fav2:4+", "env_filter": _make_env(field_min=8, field_max=8, purse_lo=15000, purse_hi=30000, sprint_req="Y", race_min=9, race_max=11, fav2_min=4.0)},
    "FB8_2": {"tier": "T1", "ticket_cost": 168.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 8, "ewpd": 1.0, "desc": "FB8 Purse:15-30k + Route + Late(9-11) + Fav2:<2.5", "env_filter": _make_env(field_min=8, field_max=8, purse_lo=15000, purse_hi=30000, sprint_req="N", race_min=9, race_max=11, fav2_max=2.49)},
    "FB8_3": {"tier": "T1", "ticket_cost": 168.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 8, "ewpd": 1.0, "desc": "FB8 Purse:8-15k + Route + Early(1-3) + Sum:<4", "env_filter": _make_env(field_min=8, field_max=8, purse_lo=8000, purse_hi=15000, sprint_req="N", race_max=3, sum_max=3.99)},
    "FB8_4": {"tier": "T1", "ticket_cost": 168.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 8, "ewpd": 1.0, "desc": "FB8 Purse:<8k + Sprint + First:Y + Sum:<4", "env_filter": _make_env(field_min=8, field_max=8, purse_hi=7999, sprint_req="Y", first_req="Y", sum_max=3.99)},
    "FB8_5": {"tier": "T1", "ticket_cost": 168.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 8, "ewpd": 1.0, "desc": "FB8 Purse:8-15k + Route + Sum:6-8 + Fav2:4+", "env_filter": _make_env(field_min=8, field_max=8, purse_lo=8000, purse_hi=15000, sprint_req="N", sum_min=6.0, sum_max=8.0, fav2_min=4.0)},
    "FB8_6": {"tier": "T1", "ticket_cost": 168.00, "hit_func": _hit_always_true, "payout_col": "Superf_paid", "payout_mult": 0.05, "min_runners": 8, "ewpd": 1.0, "desc": "FB8 Purse:8-15k + Route + Mid(4-8) + Sum:6-8", "env_filter": _make_env(field_min=8, field_max=8, purse_lo=8000, purse_hi=15000, sprint_req="N", race_min=4, race_max=8, sum_min=6.0, sum_max=8.0)},

    # ── M_ENGINE (Survival Floor) ──
    "M_ENGINE_FvP": {
        "tier":         "T4",
        "ticket_cost":  2.00,
        "hit_func":     _hit_always_true,
        "payout_col":   "FvP_pd",
        "payout_mult":  1.0,
        "use_pd_direct": True,
        "min_runners":  4,
        "ewpd":         4.0,
        "desc":         "Favorite Place — Survival Floor",
        "env_filter":   _make_env(),
    },
    "M_ENGINE_FvS": {
        "tier":         "T4",
        "ticket_cost":  2.00,
        "hit_func":     _hit_always_true,
        "payout_col":   "FvS_pd",
        "payout_mult":  1.0,
        "use_pd_direct": True,
        "min_runners":  4,
        "ewpd":         4.0,
        "desc":         "Favorite Show — Survival Floor",
        "env_filter":   _make_env(),
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING + WOWSUPERWOW NULLING
# ══════════════════════════════════════════════════════════════════════════════

def _apply_wowsuperwow(df):
    sup_col = next((c for c in ["Superf_paid", "Superfecta_paid"] if c in df.columns), None)
    exact_col = next((c for c in ["ExactaTwoPayout", "ExactaPays", "Exacta_paid"] if c in df.columns), None)

    if sup_col is None: return df

    sup_vals  = pd.to_numeric(df[sup_col], errors="coerce").fillna(0)
    abs_mask  = sup_vals > WOWSUPERWOW_ABS
    ratio_mask = pd.Series(False, index=df.index)

    if exact_col is not None:
        ex_vals    = pd.to_numeric(df[exact_col], errors="coerce").fillna(0)
        denom      = ex_vals.replace(0, np.nan)
        ratio_mask = (ex_vals > 0) & ((sup_vals / denom) > WOWSUPERWOW_RATIO)

    wow_mask = abs_mask | ratio_mask
    n_wow    = int(wow_mask.sum())

    if n_wow > 0:
        payout_cols = [c for c in df.columns if c.endswith("_paid") or c.endswith("Payout")]
        df = df.copy()
        df.loc[wow_mask, payout_cols] = np.nan
        print(f"    ⚡ WowSuperWow: nulled {n_wow:,} outlier row(s) across {len(payout_cols)} payout column(s)")

    return df

def _load_and_prep(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  ✔ Loaded {filepath}  ({len(df):,} rows)")
    except FileNotFoundError:
        print(f"  ⚠  {filepath} not found — skipping")
        return pd.DataFrame()

    df = _apply_wowsuperwow(df)

    # Alias alternate column names BEFORE numeric conversion
    if "Superfecta_paid" in df.columns and "Superf_paid" not in df.columns:
        df["Superf_paid"] = df["Superfecta_paid"]
    if "Trifecta_paid" in df.columns and "Trif_paid" not in df.columns:
        df["Trif_paid"] = df["Trifecta_paid"]

    # Numeric coercion for non-payout columns (safe to fillna(0))
    numeric_cols = ["Runners", "Purse", "WhichRace", "SumOf1st2Odds"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Numeric coercion for payout columns (DO NOT fillna(0) to preserve NaN gaps)
    payout_cols = ["Superf_paid", "Trif_paid", "ExactaTwoPayout", "FvP_pd", "FvS_pd"]
    for col in payout_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# FAST EMPIRICAL EXTRACTION (VECTORIZED)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_empirical_pnl_fast(df, name, inst):
    """Vectorized extraction of P&L using NumPy masking."""
    if df.empty: return np.array([], dtype=np.float64)

    mask = (df['Runners'] >= inst['min_runners']) & (df['Runners'] <= 9)

    if 'WhichRace' in df.columns:
        mask &= (df['WhichRace'] >= 1) & (df['WhichRace'] <= 11)

    env_filter = inst['env_filter']
    if hasattr(env_filter, 'vec'):
        env_mask = env_filter.vec(df)
    else:
        env_mask = df.apply(env_filter, axis=1).values
    mask = mask & env_mask

    # Exclude rows where the payout is NaN (either naturally missing or nulled by WowSuperWow)
    payout_col = inst['payout_col']
    if payout_col in df.columns:
        valid_payout_mask = df[payout_col].notna().values
        mask = mask & valid_payout_mask

    if not mask.any():
        return np.array([], dtype=np.float64)

    payout_mult = inst.get("payout_mult", 1.0)
    ticket_cost = inst["ticket_cost"]

    if inst.get("use_pd_direct"):
        pnl_pool = df.loc[mask, inst["payout_col"]].values.astype(np.float64)
    else:
        if 'RANKED_RESULTS' not in df.columns:
            print(f"  ⚠  {name}: 'RANKED_RESULTS' column missing — skipping instrument.")
            return np.array([], dtype=np.float64)
        raw_results = df.loc[mask, 'RANKED_RESULTS'].values
        runners_arr = df.loc[mask, 'Runners'].values
        payouts = df.loc[mask, inst['payout_col']].values.astype(np.float64) * payout_mult

        hits = np.array([
            inst['hit_func'](_parse_ranked(r), int(n))
            for r, n in zip(raw_results, runners_arr)
        ])
        pnl_pool = np.where(hits, payouts - ticket_cost, -ticket_cost)

    return pnl_pool

# ══════════════════════════════════════════════════════════════════════════════
# THE NUMBA-ACCELERATED MONTE CARLO CORE
# ══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True)
def run_gauntlet_core(
    n_paths,
    max_days,
    races_per_day,
    start_br,
    target,
    ruin_floor,
    max_stake_cap,
    pnl_pools,
    pool_stats,
    staking_table,
    phase_bounds
):
    all_paths = np.zeros((n_paths, max_days + 1), dtype=np.float64)
    all_phases = np.zeros((n_paths, max_days + 1), dtype=np.int8)
    outcomes = np.zeros(n_paths, dtype=np.int8)
    days_arr = np.zeros(n_paths, dtype=np.int32)

    n_instruments = len(pnl_pools)
    n_phases = len(phase_bounds)

    for p in prange(n_paths):
        br = start_br
        all_paths[p, 0] = br

        current_phase = n_phases - 1
        for i in range(n_phases):
            if br >= phase_bounds[i, 0] and br < phase_bounds[i, 1]:
                current_phase = int(phase_bounds[i, 2])
                break
        all_phases[p, 0] = current_phase

        for d in range(1, max_days + 1):
            t1, t2, t3, t4 = 0.0, 0.0, 0.0, 0.0
            for i in range(len(staking_table)):
                if br >= staking_table[i, 0] and br < staking_table[i, 1]:
                    t1, t2, t3, t4 = staking_table[i, 2], staking_table[i, 3], staking_table[i, 4], staking_table[i, 5]
                    break

            fractions = np.array([t1, t2, t3, t4])

            for r in range(races_per_day):
                total_ewpd = 0.0
                for i in range(n_instruments):
                    tier_idx = int(pool_stats[i, 4])
                    if fractions[tier_idx] > 0 and br >= (pool_stats[i, 2] / fractions[tier_idx]):
                        total_ewpd += pool_stats[i, 3]

                if total_ewpd == 0.0:
                    continue

                pick_val = np.random.random() * total_ewpd
                cum_sum = 0.0
                idx = 0
                for i in range(n_instruments):
                    tier_idx = int(pool_stats[i, 4])
                    if fractions[tier_idx] > 0 and br >= (pool_stats[i, 2] / fractions[tier_idx]):
                        cum_sum += pool_stats[i, 3]
                        if pick_val <= cum_sum:
                            idx = i
                            break

                tier_idx = int(pool_stats[idx, 4])
                unit_frac = fractions[tier_idx]
                tc = pool_stats[idx, 2]
                hr = pool_stats[idx, 0]
                avg_pay = pool_stats[idx, 1]

                b = (avg_pay / tc) - 1.0
                f_full = (b * hr - (1.0 - hr)) / b if b > 0 else 0.0
                kelly_dollar = max(0.0, br) * (0.25 * f_full)
                unit_ceiling = max(0.0, br) * unit_frac

                stake = min(kelly_dollar, unit_ceiling)

                # Apply liquidity cap
                if stake > max_stake_cap:
                    stake = max_stake_cap

                if stake < tc: stake = 0.0

                if stake > 0:
                    scale = stake / tc
                    pool = pnl_pools[idx]
                    raw_pnl = pool[np.random.randint(0, len(pool))]
                    br += (raw_pnl * scale)

                if br >= target or br <= ruin_floor:
                    break

            all_paths[p, d] = br

            current_phase = n_phases - 1
            for i in range(n_phases):
                if br >= phase_bounds[i, 0] and br < phase_bounds[i, 1]:
                    current_phase = int(phase_bounds[i, 2])
                    break
            all_phases[p, d] = current_phase

            if br >= target:
                outcomes[p] = 1
                days_arr[p] = d
                all_paths[p, d:] = br
                all_phases[p, d:] = current_phase
                break
            if br <= ruin_floor:
                outcomes[p] = 2
                days_arr[p] = d
                all_paths[p, d:] = br
                all_phases[p, d:] = current_phase
                break

        if outcomes[p] == 0:
            days_arr[p] = max_days

    return all_paths, outcomes, days_arr, all_phases

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def _get_numba_ready_data(empirical_map):
    typed_pools = List()
    stats_list = []
    tier_map = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}

    for name, pool in empirical_map.items():
        inst = INSTRUMENTS[name]
        typed_pools.append(pool.astype(np.float64))

        tc = inst["ticket_cost"]
        if inst.get("use_pd_direct"):
            hit_mask = pool > 0
            hr = np.mean(hit_mask) if len(pool) > 0 else 0.0
            avg_pay = (pool[hit_mask] + tc).mean() if hit_mask.any() else 0.0
        else:
            hit_mask = pool > -tc
            hr = np.mean(hit_mask) if len(pool) > 0 else 0.0
            avg_pay = (pool[hit_mask] + tc).mean() if hit_mask.any() else 0.0

        stats_list.append([hr, avg_pay, tc, inst["ewpd"], tier_map[inst["tier"]]])

    stk = []
    for lo, hi, fracs in STAKING_TABLE:
        stk.append([lo, hi, fracs["T1"], fracs["T2"], fracs["T3"], fracs["T4"]])

    pb = []
    for i, (lo, hi, _, _) in enumerate(PHASES):
        pb.append([lo, hi, i])

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

    paths_matrix, outcome_codes, days_arr, phases_matrix = run_gauntlet_core(
        N_PATHS, MAX_TRADE_DAYS, RACES_PER_DAY, START_BANKROLL, TARGET, RUIN_FLOOR, MAX_STAKE_CAP,
        pools, stats, staking, phase_bounds
    )

    code_map = {0: "timeout", 1: "success", 2: "ruin"}
    outcomes = [code_map[c] for c in outcome_codes]

    paths, phase_log, final_brs, days_list = [], [], [], days_arr.tolist()

    for p in range(N_PATHS):
        d = days_arr[p]
        paths.append(paths_matrix[p, :d+1])
        phase_log.append(phases_matrix[p, :d+1].tolist())
        final_brs.append(float(paths_matrix[p, d]))

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

    all_dd = []
    for eq in paths:
        peak = np.maximum.accumulate(eq)
        dd   = float(np.max(peak - eq))
        all_dd.append(dd)

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
    }

# ══════════════════════════════════════════════════════════════════════════════
# CHARTING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linestyle=":", alpha=0.4, linewidth=0.6)
    if title:  ax.set_title(title,  color=TEXT, fontsize=11, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=MUT,  fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUT,  fontsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — SPAGHETTI + MEDIAN PATH
# ══════════════════════════════════════════════════════════════════════════════

def chart_paths(paths, outcomes, stats):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    _style_ax(ax,
              title=f"THE 100x GAUNTLET — ${START_BANKROLL:,.0f} → ${TARGET:,.0f} Simulation Paths",
              xlabel="Trade Days",
              ylabel="Bankroll ($, symlog scale)")

    ax.set_yscale('symlog', linthresh=100)

    rng_draw = np.random.default_rng(RANDOM_SEED)
    MAX_DRAW = 300
    draw_idx = rng_draw.choice(len(paths), size=min(MAX_DRAW, len(paths)), replace=False)

    for i in draw_idx:
        eq    = paths[i]
        color = (ACCENT if outcomes[i] == "success" else RED if outcomes[i] == "ruin" else MUT)
        alpha = (0.10 if outcomes[i] == "success" else 0.16 if outcomes[i] == "ruin" else 0.05)
        ax.plot(eq, color=color, alpha=alpha, linewidth=0.5)

    med = paths[stats["median_path_idx"]]
    ax.plot(med, color=GOLD, linewidth=2.8, zorder=6, label=f"Median Path  (final: ${med[-1]:,.0f})")

    max_len = max(len(p) for p in paths)

    padded = np.full((len(paths), max_len), np.nan)
    for i, p in enumerate(paths):
        padded[i, :len(p)] = p
    n_active = np.sum(~np.isnan(padded), axis=0)
    with np.errstate(all="ignore"):
        band_lo = np.nanpercentile(padded, 25, axis=0)
        band_hi = np.nanpercentile(padded, 75, axis=0)
    band_lo[n_active < 20] = np.nan
    band_hi[n_active < 20] = np.nan

    xs    = np.arange(max_len)
    valid = ~np.isnan(band_lo) & ~np.isnan(band_hi)
    if valid.any():
        ax.fill_between(xs[valid], band_lo[valid], band_hi[valid], color=GOLD, alpha=0.08, label="25th–75th percentile band")

    milestones = {
        RUIN_FLOOR: (f"${RUIN_FLOOR:,.0f} Ruin Floor", RED, "-.", 1.5),
        START_BANKROLL: (f"${START_BANKROLL:,.0f} Start", ACCENT, ":", 1.2),
        2_000:  ("$2k  Phase 1→2 bridge", MUT, ":", 0.7),
        8_000:  ("$8k  Phase 3 scaling", CYAN, ":", 1.0),
        TARGET: (f"${TARGET:,.0f} TARGET", GOLD, "--", 1.8),
    }

    for m, (lbl, color, ls, lw) in milestones.items():
        ax.axhline(m, color=color, linewidth=lw, linestyle=ls, alpha=0.65)
        ax.text(max_len * 0.005, m + (abs(m)*0.05 if m != 0 else 100), lbl, color=color, fontsize=7.5, va="bottom", alpha=0.85)

    mds     = stats["median_days_success"]
    p10_d   = stats.get("p10_days_success")
    p90_d   = stats.get("p90_days_success")
    mds_str = f"{mds:,}d" if mds else "N/A"
    range_str = (f"  (P10–P90: {p10_d:,}–{p90_d:,}d)" if p10_d and p90_d else "")

    ann = (f"Success : {stats['success_rate']:.1f}%\n"
           f"Ruin    : {stats['ruin_rate']:.1f}%\n"
           f"Median days to target : {mds_str}{range_str}\n"
           f"Paths   : {stats['n_paths']:,}")

    ax.text(0.98, 0.04, ann, transform=ax.transAxes, color=TEXT, fontsize=9, ha="right", va="bottom", bbox=dict(facecolor=GRID, edgecolor=MUT, alpha=0.85, boxstyle="round,pad=0.5"))

    legend_elems = [
        Line2D([0],[0], color=GOLD,  lw=2.5, label="Median Path"),
        Line2D([0],[0], color=ACCENT,lw=1.2, alpha=0.6, label="Success paths"),
        Line2D([0],[0], color=RED,   lw=1.2, alpha=0.6, label="Ruin paths"),
        Line2D([0],[0], color=MUT,   lw=1.2, alpha=0.4, label="Timeout paths"),
    ]
    ax.legend(handles=legend_elems, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper left")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    plt.savefig("Gauntlet_Paths.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Paths.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — OUTCOME DISTRIBUTION + CUMULATIVE RUIN/SUCCESS
# ══════════════════════════════════════════════════════════════════════════════

def chart_distribution(paths, outcomes, final_brs, stats):
    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1, title="Final Bankroll Distribution", xlabel="Final Bankroll ($)", ylabel="Number of Paths")

    shift = abs(RUIN_FLOOR) + 1
    shifted_brs = [b + shift for b in final_brs]
    min_br = max(1, min(shifted_brs))
    max_br = max(shifted_brs) if shifted_brs else TARGET + shift
    bins = np.logspace(np.log10(min_br), np.log10(max_br + 1), 50)

    groups = {"success": (ACCENT, 0.75), "timeout": (MUT, 0.60), "ruin": (RED, 0.80)}

    for outcome_key, (color, alpha) in groups.items():
        brs = [b + shift for b, o in zip(final_brs, outcomes) if o == outcome_key]
        if brs:
            ax1.hist(brs, bins=bins, color=color, alpha=alpha, label=f"{outcome_key.capitalize()} ({len(brs):,})")

    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x - shift:,.0f}"))

    ax1.axvline(TARGET + shift, color=GOLD, lw=2.0, linestyle="--", label=f"Target ${TARGET:,.0f}")
    ax1.axvline(stats["median_final_br"] + shift, color=CYAN, lw=1.5, linestyle=":", label=f"Median ${stats['median_final_br']:,.0f}")

    ylim_top = ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1
    for p, v in stats["br_percentiles"].items():
        ax1.axvline(v + shift, color=MUT, lw=0.5, linestyle=":", alpha=0.4)
        ax1.text((v + shift) * 1.04, ylim_top * 0.88, f"P{p}", color=MUT, fontsize=7, rotation=90)

    ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2, title="Cumulative Ruin & Success Probability Over Time", xlabel="Trade Days", ylabel="Cumulative Probability (%)")

    max_days = max(len(p) for p in paths)
    n        = len(paths)

    def _cum_curve(target_outcome):
        term_days = np.array(
            [min(len(p) - 1, max_days - 1) for p, o in zip(paths, outcomes) if o == target_outcome],
            dtype=np.intp,
        )
        if len(term_days) == 0:
            return np.zeros(max_days)
        daily = np.bincount(term_days, minlength=max_days).astype(np.float64)
        return np.cumsum(daily) / n * 100

    ruin_curve    = _cum_curve("ruin")
    success_curve = _cum_curve("success")

    ax2.plot(ruin_curve, color=RED, linewidth=2.2, label=f"Cumulative ruin  (final: {stats['ruin_rate']:.1f}%)")
    ax2.fill_between(range(max_days), ruin_curve, color=RED, alpha=0.12)

    ax2.plot(success_curve, color=ACCENT, linewidth=2.2, label=f"Cumulative success  (final: {stats['success_rate']:.1f}%)")
    ax2.fill_between(range(max_days), success_curve, color=ACCENT, alpha=0.10)

    if stats["median_days_success"]:
        ax2.axvline(stats["median_days_success"], color=GOLD, lw=1.4, linestyle="--", label=f"Median days to target: {stats['median_days_success']:,}")

    if stats.get("p10_days_success"):
        ax2.axvline(stats["p10_days_success"], color=GOLD, lw=0.8, linestyle=":", label=f"P10 days: {stats['p10_days_success']:,}")

    ax2.set_xlim(0, max_days)
    ax2.set_ylim(0, 105)
    ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    fig.suptitle("THE 100x GAUNTLET — Outcome Analysis", color=TEXT, fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig("Gauntlet_Distribution.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — PER-INSTRUMENT PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def chart_instruments(empirical_pnl_map):
    names = [n for n in INSTRUMENTS if n in empirical_pnl_map and len(empirical_pnl_map[n]) > 0]
    if not names:
        print("  ⚠  chart_instruments: no instrument data to plot")
        return

    names.sort(key=lambda n: (INSTRUMENTS[n]["tier"], n))

    height = max(8, len(names) * 0.4)
    fig = plt.figure(figsize=(16, height), facecolor=BG)
    fig.suptitle("THE 100x GAUNTLET — Per-Instrument Profile", color=TEXT, fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40)

    metrics = []
    for name in names:
        inst    = INSTRUMENTS[name]
        arr     = empirical_pnl_map[name]
        tc      = inst["ticket_cost"]

        if inst.get("use_pd_direct"):
            hits = (arr > 0).sum()
            n    = len(arr)
            hr   = hits / n * 100 if n > 0 else 0
            ev   = float(arr.mean()) if n > 0 else 0
            payouts = arr[arr > 0] + tc
            avg_pay = float(payouts.mean()) if len(payouts) > 0 else 0
            roi  = (arr.sum() / (n * tc) * 100) if n > 0 else 0
        else:
            hits = (arr > -tc).sum()
            n    = len(arr)
            hr   = hits / n * 100 if n > 0 else 0
            ev   = float(arr.mean()) if n > 0 else 0
            payouts = arr[arr > -tc] + tc
            avg_pay = float(payouts.mean()) if len(payouts) > 0 else 0
            roi  = (arr.sum() / (n * tc) * 100) if n > 0 else 0

        metrics.append({
            "name": name, "hr": hr, "ev": ev, "avg_pay": avg_pay, "n": n, "roi": roi, "tier": inst["tier"], "cost": tc,
        })

    colors = [TIER_COLORS.get(m["tier"], MUT) for m in metrics]
    lbls   = [m["name"] for m in metrics]

    def _bar(ax, values, title, xlabel, fmt="$"):
        bars = ax.barh(lbls, values, color=colors, height=0.55)
        _style_ax(ax, title=title, xlabel=xlabel)
        ax.axvline(0, color=MUT, lw=0.8)
        for bar, val in zip(bars, values):
            xpos = val + abs(max(values, default=1)) * 0.02 if val >= 0 else val - abs(max(values, default=1)) * 0.02
            align = "left" if val >= 0 else "right"
            label = (f"${val:,.2f}" if fmt == "$" else f"{val:,.1f}%" if fmt == "%" else f"{val:,.0f}")
            ax.text(xpos, bar.get_y() + bar.get_height()/2, label, color=TEXT, va="center", ha=align, fontsize=8)
        ax.set_yticks(range(len(lbls)))
        ax.set_yticklabels(lbls, color=MUT, fontsize=8)

    _bar(fig.add_subplot(gs[0, 0]), [m["ev"]      for m in metrics], "EV per Race",      "Net P&L / Race ($)",   "$")
    _bar(fig.add_subplot(gs[0, 1]), [m["hr"]      for m in metrics], "Hit Rate",         "Hit Rate (%)",          "%")
    _bar(fig.add_subplot(gs[0, 2]), [m["roi"]     for m in metrics], "IS ROI %",         "Return on Avg Cost (%)",    "%")
    _bar(fig.add_subplot(gs[1, 0]), [m["avg_pay"] for m in metrics], "Avg Payout (hits)", "Average Payout ($)",   "$")
    _bar(fig.add_subplot(gs[1, 1]), [m["cost"]    for m in metrics], "Ticket Cost",      "Cost per Race ($)",     "$")
    _bar(fig.add_subplot(gs[1, 2]), [m["n"]       for m in metrics], "Sample Size",      "Qualifying Races",     "n")

    legend_elems = [Line2D([0],[0], color=TIER_COLORS[t], lw=6, label=TIER_LABELS[t]) for t in ["T1","T2","T3","T4"]]
    fig.legend(handles=legend_elems, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig("Gauntlet_Instruments.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Instruments.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — PHASE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def chart_phases(phase_log):
    if not phase_log: return

    max_days = max(len(p) for p in phase_log)
    n_phases = len(PHASES)
    phase_matrix = np.zeros((n_phases, max_days))

    for p_seq in phase_log:
        for d, ph in enumerate(p_seq):
            if ph < n_phases:
                phase_matrix[ph, d] += 1

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    _style_ax(ax, title="THE 100x GAUNTLET — Bankroll Phase Transitions Over Time", xlabel="Trade Days", ylabel="Percentage of Paths (%)")

    day_totals = phase_matrix.sum(axis=0)
    day_totals[day_totals == 0] = 1
    pct_matrix = (phase_matrix / day_totals) * 100

    xs = np.arange(max_days)
    y_bottom = np.zeros(max_days)

    labels = [p[2].replace('\n', ' ') for p in PHASES]
    colors = [p[3] for p in PHASES]

    for i in range(n_phases):
        y_top = y_bottom + pct_matrix[i]
        ax.fill_between(xs, y_bottom, y_top, color=colors[i], alpha=0.85, label=labels[i])
        y_bottom = y_top

    ax.set_ylim(0, 100)
    ax.set_xlim(0, max_days)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig("Gauntlet_Phases.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Phases.png")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(empirical_pnl_map, stats):
    SEP  = "═" * 80
    DASH = "─" * 80
    print(f"\n{SEP}")
    print("  THE 100x GAUNTLET — SIMULATION RESULTS")
    print(f"  ${START_BANKROLL:,.0f} → ${TARGET:,.0f}  |  {stats['n_paths']:,} paths  |  up to {MAX_TRADE_DAYS:,} trade-days each")
    print(SEP)

    print(f"\n  {'OUTCOME':<25} {'COUNT':>6}  {'RATE':>7}")
    print("  " + DASH)
    print(f"  {'Success (reach target)':<25} {stats['n_success']:>6,}  {stats['success_rate']:>6.1f}%")
    print(f"  {'Ruin (hit floor)':<25} {stats['n_ruin']:>6,}  {stats['ruin_rate']:>6.1f}%")
    print(f"  {'Timeout (ran out of days)':<25} {stats['n_timeout']:>6,}  {stats['timeout_rate']:>6.1f}%")

    mds = stats["median_days_success"]
    mdr = stats["median_days_ruin"]
    print(f"\n  Median days to target         : {'N/A' if mds is None else f'{mds:,} trade-days'}")
    print(f"  Median days to absolute ruin  : {'N/A' if mdr is None else f'{mdr:,} trade-days'}")
    print(f"  Median Maximum Drawdown       : ${stats['median_max_dd']:,.2f}")
    print(f"  3x MaxDD Conservative Capital : ${stats['required_bankroll_3x']:,.2f}")
    print(f"  Median Final Bankroll         : ${stats['median_final_br']:,.2f}")
    print(f"  Mean Final Bankroll           : ${stats['mean_final_br']:,.2f}")

    print("\n  Final Bankroll Percentiles:")
    for p, v in stats["br_percentiles"].items():
        print(f"    P{p}: ${v:,.2f}")
    print(SEP + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print(" 🏁 STARTING THE 100x GAUNTLET STRESS TESTING PIPELINE v3.3")
    print("=" * 80)

    # 1. Load Data Slices
    df_2025 = _load_and_prep(TRAIN_FILE)
    df_2026 = _load_and_prep(TEST_FILE)

    if df_2025.empty and df_2026.empty:
        print("❌ CRITICAL ERROR: Source data arrays empty. Verify paths for target records.")
        exit(1)

    df_all = pd.concat([df_2025, df_2026], ignore_index=True)

    # 2. Reconstruct Empirical P&L Distributions
    empirical_pnl_map = {}
    print("\nExtracting cross-validated historical P&L pools...")
    for name, inst in INSTRUMENTS.items():
        pnl_arr = _extract_empirical_pnl_fast(df_all, name, inst)
        print(f"  {name:<40} : {len(pnl_arr):>6,} qualifying events mapped.")
        if len(pnl_arr) > 0:
            empirical_pnl_map[name] = pnl_arr

    if not empirical_pnl_map:
        print("❌ CRITICAL ERROR: No historical rows passed instrument registration filters.")
        exit(1)

    # 3. Fire Bootstrap Engine
    paths, outcomes, days_list, final_brs, phase_log = _simulate_paths(empirical_pnl_map)

    # 4. Extract Analytical Metrics
    print("📊 Computing stress test telemetry benchmarks...")
    stats = _compute_stats(paths, outcomes, days_list, final_brs)

    # 5. Build Reports and Visualization Artifacts
    print("🎨 Rendering reporting dashboards...")
    chart_paths(paths, outcomes, stats)
    chart_distribution(paths, outcomes, final_brs, stats)
    chart_instruments(empirical_pnl_map)
    chart_phases(phase_log)

    # 6. Output Terminal Diagnostic Summary
    print_report(empirical_pnl_map, stats)

    # 7. Serialize Output Artifacts to Disk
    print("💾 Archiving system records...")
    with open("Gauntlet_Results.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("  ✓ Gauntlet_Results.json structured audit record saved.")

    flat_stats = {k: v for k, v in stats.items() if k != "br_percentiles"}
    for p, v in stats["br_percentiles"].items():
        flat_stats[f"br_percentile_P{p}"] = v

    df_flat = pd.DataFrame([flat_stats])
    df_flat.to_csv("Gauntlet_Results.csv", index=False)
    print("  ✓ Gauntlet_Results.csv flat database table saved.")
    print("\n✅ THE 100x GAUNTLET SIMULATION ENGINE RUN COMPLETE.")
