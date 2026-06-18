#!/usr/bin/env python3
"""
THE 100x GAUNTLET — Bankroll Growth Simulator
======================================================================
v5.4.1 CHANGES:

  FIX 1 — max_runners field added to instrument registry:
    SB* instruments require EXACT field size matching.
    _extract_empirical_pnl_fast now respects inst["max_runners"].
    Previously all instruments used Runners >= min_runners which
    caused SB7 instruments to include 8,9,10+ horse fields,
    corrupting the P&L pool. FB* instruments unaffected (no max).

  FIX 2 — Staking table extended with terminal sprint bracket:
    New bracket: (60_000, 999_999, T1:0.020, T2:0.010, T3:0, T4:0)
    At $80k bankroll, T1 fraction rises from 1.0% to 2.0%.
    Kelly still governs actual bet size — ruin protection intact.
    Addresses the 96.6% timeout problem from v5.3.

  FIX 3 — MAX_STAKE_CAP replaced with dynamic BR-relative cap:
    Old: flat $200 hard cap regardless of bankroll
    New: dynamic_cap = bankroll * 0.05 (5% of current BR)
    At $150: cap=$7.50  At $5k: cap=$250  At $81k: cap=$4,050
    Implemented in Numba kernel via br * dynamic_cap_frac param.

  FIX 4 — RUIN_FLOOR raised to $50:
    $15 floor was unreachable with Kelly sizing.
    $50 (33% of start) gives honest ruin statistics.
    Kelly still prevents most ruin; floor catches edge cases.

  NEW INSTRUMENTS — v5.1 PRUNED + SixBox scout discoveries:
    Added NM_Sup4445_N6_C, NM_Sup4445_N7_D, NM_Sup2266_N10_D,
    NM_Sup2266_N11_C, NM_Sup2266_N11_D from v5.1 Pruned list.
    Retained SB6_S5556_A/B/C/D, SB11_S6666_A, SB12_S6667_A, SB12_S6678_A.

  CHART FIX — chart_comfort_score implemented:
    Visualizes the stability vs. performance trade-off for all instruments.

  Registry: 33 instruments (v5.1 Pruned + v5.4 Scouts)
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

TRAIN_FILE  = "RaceRecords_Output_2025.csv"
TEST_FILE   = "RaceRecords_Output_2026.csv"

START_BANKROLL      = 150.0
RUIN_FLOOR          = 50.0          # v5.4: raised from $15
TARGET              = 88_888.0
DYNAMIC_CAP_FRAC    = 0.05          # v5.4: replaces flat MAX_STAKE_CAP
MAX_STAKE_FLOOR     = 200.0         # minimum cap floor (protects early game)
N_PATHS             = 2_000
RANDOM_SEED         = 42

MAX_TRADE_RACES     = 30_000
RACES_PER_DAY       = 25

P_RACE_COLD         = 0.05
P_RACE_FORCED_MISS  = 0.12
PAYOUT_JITTER_FRAC  = 0.05
HR_MIN_JBM          = 2

# v5.4: Terminal sprint bracket added
STAKING_TABLE = [
    (0,       2_000,  {"T1": 0.025, "T2": 0.015, "T3": 0.010, "T4": 0.010}),
    (2_000,   8_000,  {"T1": 0.020, "T2": 0.0125,"T3": 0.0075,"T4": 0.0025}),
    (8_000,  20_000,  {"T1": 0.015, "T2": 0.010, "T3": 0.005, "T4": 0.000}),
    (20_000, 60_000,  {"T1": 0.010, "T2": 0.0075,"T3": 0.0025,"T4": 0.000}),
    (60_000, 999_999, {"T1": 0.020, "T2": 0.010, "T3": 0.000, "T4": 0.000}),
]

PHASES = [
    (0,       2_000,  "Phase 1\nSurvival", "#f43f5e"),
    (2_000,   8_000,  "Phase 2\nPlatform", "#f59e0b"),
    (8_000,  40_001,  "Phase 3\nScaling",  "#10b981"),
]

WOWSUPERWOW_ABS   = 25_000.0
WOWSUPERWOW_RATIO = 200.0

BG, GRID, TEXT, MUT = "#0f172a", "#1e293b", "#f8fafc", "#64748b"
ACCENT, RED, GOLD, CYAN = "#10b981", "#f43f5e", "#f59e0b", "#06b6d4"
TIER_COLORS = {"T1": GOLD, "T2": ACCENT, "T3": CYAN, "T4": MUT}
TIER_LABELS = {
    "T1": "TIER 1 🏆 (Core)",
    "T2": "TIER 2 ✅ (Active)",
    "T3": "TIER 3 💎 (Monitor)",
    "T4": "TIER 4 📋 (Bench)",
}

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENTAL GATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _chalky(row):
    return str(row.get("ChalkYN", "N")).strip().upper() == "Y"

def _first_race(row):
    return str(row.get("FirstRaceYN", "N")).strip().upper() == "Y"

def _is_sprint(row):
    dist = pd.to_numeric(
        row.get("Miles", row.get("Distance", 0)), errors="coerce")
    if pd.isna(dist): return False
    if dist > 4.0: dist = dist / 8.0
    return dist < 0.875

def _which_race_ok(row, mn=1, mx=11):
    wr = pd.to_numeric(row.get("WhichRace", 0), errors="coerce")
    return pd.notna(wr) and mn <= int(wr) <= mx

def _field_ok(row, mn=3, mx=13):
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
                  field_min=3, field_max=13, sprint_req=None,
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
              field_min=3, field_max=13, sprint_req=None,
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
            df, chalk_req=chalk_req, purse_lo=purse_lo, purse_hi=purse_hi,
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
    if len(tokens) < 4: return None
    try:
        return [int(t) for t in tokens[:4]]
    except (ValueError, TypeError):
        return None

def _hit_tri145(pos, n):
    return (len(pos) >= 3
            and pos[0] == 1
            and pos[1] in {2, 3, 4}
            and pos[2] in {2, 3, 4, 5})

def _hit_always_true(pos, n):
    return True

def _hit_supr3666(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 3 and pos[1] <= 6 and pos[2] <= 6 and pos[3] <= 6

def _hit_sup4455(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 4 and pos[1] <= 4 and pos[2] <= 5 and pos[3] <= 5

def _hit_sup4456(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 4 and pos[1] <= 4 and pos[2] <= 5 and pos[3] <= 6

def _hit_sup4445(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 4 and pos[1] <= 4 and pos[2] <= 4 and pos[3] <= 5

def _hit_sup2266(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 2 and pos[1] <= 2 and pos[2] <= 6 and pos[3] <= 6

# v5.4 NEW
def _hit_sup5556(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 5 and pos[1] <= 5 and pos[2] <= 5 and pos[3] <= 6

def _hit_sup6666(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 6 and pos[1] <= 6 and pos[2] <= 6 and pos[3] <= 6

def _hit_sup6667(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 6 and pos[1] <= 6 and pos[2] <= 6 and pos[3] <= 7

def _hit_sup6678(pos, n):
    if pos is None or len(pos) < 4: return False
    return pos[0] <= 6 and pos[1] <= 6 and pos[2] <= 7 and pos[3] <= 8

_VEC_HIT = {
    _hit_supr3666: lambda a,b,c,d: (a<=3)&(b<=6)&(c<=6)&(d<=6),
    _hit_sup4455:  lambda a,b,c,d: (a<=4)&(b<=4)&(c<=5)&(d<=5),
    _hit_sup4456:  lambda a,b,c,d: (a<=4)&(b<=4)&(c<=5)&(d<=6),
    _hit_sup4445:  lambda a,b,c,d: (a<=4)&(b<=4)&(c<=4)&(d<=5),
    _hit_sup2266:  lambda a,b,c,d: (a<=2)&(b<=2)&(c<=6)&(d<=6),
    _hit_sup5556:  lambda a,b,c,d: (a<=5)&(b<=5)&(c<=5)&(d<=6),
    _hit_sup6666:  lambda a,b,c,d: (a<=6)&(b<=6)&(c<=6)&(d<=6),
    _hit_sup6667:  lambda a,b,c,d: (a<=6)&(b<=6)&(c<=6)&(d<=7),
    _hit_sup6678:  lambda a,b,c,d: (a<=6)&(b<=6)&(c<=7)&(d<=8),
}

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT REGISTRY — v5.4.1 (Integrated)
# ══════════════════════════════════════════════════════════════════════════════

INSTRUMENTS = {

    # ── ANCHOR ────────────────────────────────────────────────────────────────
    "TRI145": {
        "tier": "T1", "ticket_cost": 18.00,
        "hit_func": _hit_tri145, "payout_col": "Trif_paid",
        "payout_mult": 1.0, "min_runners": 5, "ewpd": 2.8,
        "desc": "Trifecta [1]×[2-4]×[2-5] — 9 combos × $2",
        "env_filter": _make_env(sum_min=9.0, fav2_min=2.5),
    },

    # ── FB4 family ────────────────────────────────────────────────────────────
    "FB4_2": {
        "tier": "T1", "ticket_cost": 2.40,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.5,
        "desc": "FB4 Chalk:N + Sum:4-6 + Fav2:2.5-4",
        "env_filter": _make_env(field_min=4, field_max=4, chalk_req="N",
                                sum_min=4.0, sum_max=6.0,
                                fav2_min=2.5, fav2_max=4.0),
    },
    "FB4_3": {
        "tier": "T1", "ticket_cost": 2.40,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.5,
        "desc": "FB4 Chalk:N + First:N + Sum:4-6",
        "env_filter": _make_env(field_min=4, field_max=4,
                                chalk_req="N", first_req="N",
                                sum_min=4.0, sum_max=6.0),
    },
    "FB4_4": {
        "tier": "T1", "ticket_cost": 2.40,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.5,
        "desc": "FB4 Chalk:N + Sum:4-6",
        "env_filter": _make_env(field_min=4, field_max=4,
                                chalk_req="N", sum_min=4.0, sum_max=6.0),
    },
    "FB4_5": {
        "tier": "T1", "ticket_cost": 2.40,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.5,
        "desc": "FB4 Sprint + Chalk:N + Sum:4-6",
        "env_filter": _make_env(field_min=4, field_max=4,
                                sprint_req="Y", chalk_req="N",
                                sum_min=4.0, sum_max=6.0),
    },
    "FB4_6": {
        "tier": "T1", "ticket_cost": 2.40,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 4, "ewpd": 1.5,
        "desc": "FB4 Race:4-8 + Sum:4-6 + Fav2:2.5-4",
        "env_filter": _make_env(field_min=4, field_max=4,
                                race_min=4, race_max=8,
                                sum_min=4.0, sum_max=6.0,
                                fav2_min=2.5, fav2_max=4.0),
    },

    # ── FB5 family ────────────────────────────────────────────────────────────
    "FB5_2": {
        "tier": "T1", "ticket_cost": 12.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 5, "ewpd": 1.5,
        "desc": "FB5 Chalk:N + Fav2:4.0+",
        "env_filter": _make_env(field_min=5, field_max=5,
                                chalk_req="N", fav2_min=4.0),
    },

    # ── FB6 family ────────────────────────────────────────────────────────────
    "FB6_1": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Sprint + Chalk:N + First:N + Fav2:4.0+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                sprint_req="Y", chalk_req="N",
                                first_req="N", fav2_min=4.0),
    },
    "FB6_2": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Race:4-8 + Chalk:N + First:N + Fav2:4.0+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                race_min=4, race_max=8,
                                chalk_req="N", first_req="N",
                                fav2_min=4.0),
    },
    "FB6_3": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Race:4-8 + Chalk:N + Fav2:4.0+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                race_min=4, race_max=8,
                                chalk_req="N", fav2_min=4.0),
    },
    "FB6_4": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Sprint + Chalk:N + Fav2:4.0+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                sprint_req="Y", chalk_req="N",
                                fav2_min=4.0),
    },
    "FB6_5": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Chalk:N + First:N + Fav2:4.0+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                chalk_req="N", first_req="N",
                                fav2_min=4.0),
    },
    "FB6_6": {
        "tier": "T1", "ticket_cost": 36.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "FB6 Chalk:N + Fav2:4.0+",
        "env_filter": _make_env(field_min=6, field_max=6,
                                chalk_req="N", fav2_min=4.0),
    },

    # ── FB7 family ────────────────────────────────────────────────────────────
    "FB7_1": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Purse:8-15k + Chalk:N + First:N + Sum:6-8",
        "env_filter": _make_env(field_min=7, field_max=7,
                                purse_lo=8_000, purse_hi=15_000,
                                chalk_req="N", first_req="N",
                                sum_min=6.0, sum_max=8.0),
    },
    "FB7_2": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Purse:8-15k + Chalk:N + Sum:6-8",
        "env_filter": _make_env(field_min=7, field_max=7,
                                purse_lo=8_000, purse_hi=15_000,
                                chalk_req="N",
                                sum_min=6.0, sum_max=8.0),
    },
    "FB7_3": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Purse:8-15k + Sprint + Chalk:N + Fav2:4.0+",
        "env_filter": _make_env(field_min=7, field_max=7,
                                purse_lo=8_000, purse_hi=15_000,
                                sprint_req="Y", chalk_req="N",
                                fav2_min=4.0),
    },
    "FB7_4": {
        "tier": "T1", "ticket_cost": 84.00,
        "hit_func": _hit_always_true, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 7, "ewpd": 1.5,
        "desc": "FB7 Chalk:N + Sum:6-8 + Fav2:4.0+",
        "env_filter": _make_env(field_min=7, field_max=7,
                                chalk_req="N",
                                sum_min=6.0, sum_max=8.0,
                                fav2_min=4.0),
    },

    # ── NM: Non-Monotonic Structured Exotics ──────────────────────────────────
    "NM_Supr3666_N6_D": {
        "tier": "T2", "ticket_cost": 18.00,
        "hit_func": _hit_supr3666, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "Supr3666 n=6 Chalk:N Fav2>=4.0 Sum:6-8.5",
        "env_filter": _make_env(field_min=6, field_max=6,
                                chalk_req="N", fav2_min=4.0,
                                sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup4455_N6_C": {
        "tier": "T1", "ticket_cost": 7.20,
        "hit_func": _hit_sup4455, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "Sup4455 n=6 Chalk:N Fav2>=4.0",
        "env_filter": _make_env(field_min=6, field_max=6,
                                chalk_req="N", fav2_min=4.0),
    },
    "NM_Sup4455_N6_D": {
        "tier": "T2", "ticket_cost": 7.20,
        "hit_func": _hit_sup4455, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "Sup4455 n=6 Chalk:N Fav2>=4.0 Sum:6-8.5",
        "env_filter": _make_env(field_min=6, field_max=6,
                                chalk_req="N", fav2_min=4.0,
                                sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup4456_N6_D": {
        "tier": "T2", "ticket_cost": 10.80,
        "hit_func": _hit_sup4456, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "ewpd": 1.5,
        "desc": "Sup4456 n=6 Chalk:N Fav2>=4.0 Sum:6-8.5",
        "env_filter": _make_env(field_min=6, field_max=6,
                                chalk_req="N", fav2_min=4.0,
                                sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup4445_N6_C": {
        "tier": "T1", "ticket_cost": 4.80, "hit_func": _hit_sup4445, "payout_col": "Superf_paid", "payout_mult": 0.05,
        "min_runners": 6, "ewpd": 1.5, "desc": "Sup4445 n=6 Chalk:N Fav2>=4.0",
        "env_filter": _make_env(field_min=6, field_max=6, chalk_req="N", fav2_min=4.0),
    },
    "NM_Sup4445_N7_D": {
        "tier": "T1", "ticket_cost": 4.80, "hit_func": _hit_sup4445, "payout_col": "Superf_paid", "payout_mult": 0.05,
        "min_runners": 7, "ewpd": 1.5, "desc": "Sup4445 n=7 Chalk:N Fav2>=4.0 Sum:6-8.5",
        "env_filter": _make_env(field_min=7, field_max=7, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup2266_N10_D": {
        "tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_sup2266, "payout_col": "Superf_paid", "payout_mult": 0.05,
        "min_runners": 10, "ewpd": 1.5, "desc": "Sup2266 n=10 Chalk:N Fav2>=4.0 Sum:6-8.5",
        "env_filter": _make_env(field_min=10, field_max=10, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup2266_N11_C": {
        "tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_sup2266, "payout_col": "Superf_paid", "payout_mult": 0.05,
        "min_runners": 11, "ewpd": 1.5, "desc": "Sup2266 n=11 Chalk:N Fav2>=4.0",
        "env_filter": _make_env(field_min=11, field_max=11, chalk_req="N", fav2_min=4.0),
    },
    "NM_Sup2266_N11_D": {
        "tier": "T1", "ticket_cost": 2.40, "hit_func": _hit_sup2266, "payout_col": "Superf_paid", "payout_mult": 0.05,
        "min_runners": 11, "ewpd": 1.5, "desc": "Sup2266 n=11 Chalk:N Fav2>=4.0 Sum:6-8.5",
        "env_filter": _make_env(field_min=11, field_max=11, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },

    # ── SB6: Scout discoveries — sup5556 at n=6 ──────────────────────────────
    "SB6_S5556_A": {
        "tier": "T1", "ticket_cost": 18.00,
        "hit_func": _hit_sup5556, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "max_runners": 6,
        "ewpd": 2.0,
        "desc": "SB6 sup5556 Chalk:N+Sum:6-8.5+Purse:8-25k [n=61 EV+$16]",
        "env_filter": _make_env(
            field_min=6, field_max=6,
            chalk_req="N",
            sum_min=6.0, sum_max=8.5,
            purse_lo=8_000, purse_hi=25_000,
        ),
    },
    "SB6_S5556_B": {
        "tier": "T1", "ticket_cost": 18.00,
        "hit_func": _hit_sup5556, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "max_runners": 6,
        "ewpd": 2.0,
        "desc": "SB6 sup5556 Chalk:N+First:N+Sum:6-8.5+Purse:8-25k [n=55 EV+$18]",
        "env_filter": _make_env(
            field_min=6, field_max=6,
            chalk_req="N", first_req="N",
            sum_min=6.0, sum_max=8.5,
            purse_lo=8_000, purse_hi=25_000,
        ),
    },
    "SB6_S5556_C": {
        "tier": "T2", "ticket_cost": 18.00,
        "hit_func": _hit_sup5556, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "max_runners": 6,
        "ewpd": 1.8,
        "desc": "SB6 sup5556 Chalk:N+Sprint:Y+Sum:6-8.5+Purse:8-25k [n=39 EV+$21]",
        "env_filter": _make_env(
            field_min=6, field_max=6,
            chalk_req="N", sprint_req="Y",
            sum_min=6.0, sum_max=8.5,
            purse_lo=8_000, purse_hi=25_000,
        ),
    },
    "SB6_S5556_D": {
        "tier": "T2", "ticket_cost": 18.00,
        "hit_func": _hit_sup5556, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 6, "max_runners": 6,
        "ewpd": 1.5,
        "desc": "SB6 sup5556 Chalk:N+First:N+Sprint:Y+Sum:6-8.5+Purse:8-25k [n=34 EV+$25]",
        "env_filter": _make_env(
            field_min=6, field_max=6,
            chalk_req="N", first_req="N", sprint_req="Y",
            sum_min=6.0, sum_max=8.5,
            purse_lo=8_000, purse_hi=25_000,
        ),
    },

    # ── SB11: Scout discovery — sup6666 at n=11 ──────────────────────────────
    "SB11_S6666_A": {
        "tier": "T2", "ticket_cost": 36.00,
        "hit_func": _hit_sup6666, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 11, "max_runners": 11,
        "ewpd": 1.5,
        "desc": "SB11 sup6666 Sprint:N+Race:4-8+Sum:6-8.5+Fav2:2.5-4+Purse:8-25k [n=32 EV+$12]",
        "env_filter": _make_env(
            field_min=11, field_max=11,
            sprint_req="N",
            race_min=4, race_max=8,
            sum_min=6.0, sum_max=8.5,
            fav2_min=2.5, fav2_max=4.0,
            purse_lo=8_000, purse_hi=25_000,
        ),
    },

    # ── SB12: Scout discoveries ───────────────────────────────────────────────
    "SB12_S6667_A": {
        "tier": "T2", "ticket_cost": 48.00,
        "hit_func": _hit_sup6667, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 12, "max_runners": 12,
        "ewpd": 1.5,
        "desc": "SB12 sup6667 Chalk:N+Sprint:N+Race:4-8+Sum:4-8.5+Fav2:4++Purse:8-25k [n=38 EV+$23]",
        "env_filter": _make_env(
            field_min=12, field_max=12,
            chalk_req="N", sprint_req="N",
            race_min=4, race_max=8,
            sum_min=4.0, sum_max=8.5,
            fav2_min=4.0, fav2_max=999.0,
            purse_lo=8_000, purse_hi=25_000,
        ),
    },
    "SB12_S6678_A": {
        "tier": "T3", "ticket_cost": 75.00,
        "hit_func": _hit_sup6678, "payout_col": "Superf_paid",
        "payout_mult": 0.05, "min_runners": 12, "max_runners": 12,
        "ewpd": 1.0,
        "desc": "SB12 sup6678 Chalk:N+Race:4-8+Sum:8.5++Purse:15k+ [n=30 EV+$38 T3]",
        "env_filter": _make_env(
            field_min=12, field_max=12,
            chalk_req="N",
            race_min=4, race_max=8,
            sum_min=8.5, sum_max=999.0,
            purse_lo=15_000, purse_hi=999_999,
        ),
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING + WOWSUPERWOW NULLING
# ══════════════════════════════════════════════════════════════════════════════

def _apply_wowsuperwow(df):
    sup_col   = next((c for c in ["Superf_paid", "Superfecta_paid"]
                      if c in df.columns), None)
    trif_col  = next((c for c in ["Trif_paid", "Trifecta_paid"]
                      if c in df.columns), None)
    exact_col = next((c for c in ["ExactaTwoPayout", "ExactaPays", "Exacta_paid"]
                      if c in df.columns), None)
    wow_mask  = pd.Series(False, index=df.index)
    if sup_col is not None:
        sup_vals = pd.to_numeric(df[sup_col], errors="coerce").fillna(0)
        wow_mask |= sup_vals > WOWSUPERWOW_ABS
        if exact_col is not None:
            ex_vals   = pd.to_numeric(df[exact_col], errors="coerce").fillna(0)
            denom     = ex_vals.replace(0, np.nan)
            wow_mask |= (ex_vals > 0) & ((sup_vals / denom) > WOWSUPERWOW_RATIO)
    if trif_col is not None:
        trif_vals = pd.to_numeric(df[trif_col], errors="coerce").fillna(0)
        wow_mask |= trif_vals > WOWSUPERWOW_ABS
    n_wow = int(wow_mask.sum())
    if n_wow > 0:
        payout_cols = [c for c in df.columns
                       if c.endswith("_paid") or c.endswith("Payout")
                       or c.endswith("_pd")]
        df = df.copy()
        df.loc[wow_mask, payout_cols] = np.nan
        print(f"  ⚡ WowSuperWow: nulled {n_wow:,} outlier row(s) "
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
    all_payout_cols = set(inst["payout_col"] for inst in INSTRUMENTS.values())
    for col in all_payout_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# EMPIRICAL P&L EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_empirical_pnl_fast(df, name, inst):
    if df.empty:
        return np.array([], dtype=np.float64)

    max_runners = inst.get("max_runners", 999)
    mask = ((df["Runners"] >= inst["min_runners"]) &
            (df["Runners"] <= max_runners))

    if "WhichRace" in df.columns:
        mask &= (df["WhichRace"] >= 1) & (df["WhichRace"] <= 11)

    env_filter = inst["env_filter"]
    env_mask   = (env_filter.vec(df) if hasattr(env_filter, "vec")
                  else df.apply(env_filter, axis=1).values)
    mask = mask & env_mask

    if not mask.any():
        return np.array([], dtype=np.float64)

    payout_col  = inst["payout_col"]
    ticket_cost = inst["ticket_cost"]
    payout_mult = inst.get("payout_mult", 1.0)
    hit_func    = inst["hit_func"]

    qualifying_idx = np.where(mask)[0]
    n_q = len(qualifying_idx)

    if payout_col not in df.columns:
        return np.full(n_q, -ticket_cost, dtype=np.float64)

    raw_payouts = df[payout_col].values
    rng = np.random.default_rng(RANDOM_SEED + (abs(hash(name)) % 1_000_000))

    pays   = raw_payouts[qualifying_idx]
    pays_f = pays.astype(float)
    valid  = ~(np.isnan(pays_f) | (pays_f == 0.0))
    jitter = rng.uniform(1.0 - PAYOUT_JITTER_FRAC,
                         1.0 + PAYOUT_JITTER_FRAC, size=n_q)

    if hit_func is _hit_always_true:
        return np.where(
            valid,
            pays_f * payout_mult * jitter - ticket_cost,
            -ticket_cost,
        ).astype(np.float64)

    vec_fn = _VEC_HIT.get(hit_func)
    if vec_fn is not None and "RANKED_RESULTS" in df.columns:
        rr_sub = df["RANKED_RESULTS"].iloc[qualifying_idx]
        try:
            split = rr_sub.str.split(expand=True).iloc[:, :4]
            while split.shape[1] < 4:
                split[split.shape[1]] = np.nan
            p1 = pd.to_numeric(split.iloc[:, 0], errors="coerce").values
            p2 = pd.to_numeric(split.iloc[:, 1], errors="coerce").values
            p3 = pd.to_numeric(split.iloc[:, 2], errors="coerce").values
            p4 = pd.to_numeric(split.iloc[:, 3], errors="coerce").values
            parseable = ~(np.isnan(p1) | np.isnan(p2)
                          | np.isnan(p3) | np.isnan(p4))
            hit = vec_fn(p1, p2, p3, p4) & parseable
            return np.where(
                valid & hit,
                pays_f * payout_mult * jitter - ticket_cost,
                -ticket_cost,
            ).astype(np.float64)
        except Exception:
            pass

    raw_results = df["RANKED_RESULTS"].values
    runners_arr = df["Runners"].values
    pnl_list    = []
    for i in qualifying_idx:
        pay = raw_payouts[i]
        if pd.isna(pay) or float(pay) == 0.0:
            pnl_list.append(-ticket_cost)
        else:
            pos = _parse_ranked(raw_results[i])
            if hit_func(pos, int(runners_arr[i])):
                gross  = float(pay) * payout_mult
                gross *= 1.0 + rng.uniform(-PAYOUT_JITTER_FRAC,
                                            PAYOUT_JITTER_FRAC)
                pnl_list.append(gross - ticket_cost)
            else:
                pnl_list.append(-ticket_cost)
    return np.array(pnl_list, dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA MONTE CARLO KERNEL
# ══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True)
def run_gauntlet_core(
    n_paths, max_races,
    start_br, target, ruin_floor,
    dynamic_cap_frac, min_cap_floor,
    p_race_cold, p_race_forced_miss,
    pnl_pools, pool_stats, staking_table, phase_bounds,
):
    all_paths  = np.zeros((n_paths, max_races + 1), dtype=np.float64)
    all_phases = np.zeros((n_paths, max_races + 1), dtype=np.int8)
    outcomes   = np.zeros(n_paths, dtype=np.int8)
    races_arr  = np.zeros(n_paths, dtype=np.int32)

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

        for race in range(1, max_races + 1):
            t1 = t2 = t3 = t4 = 0.0
            for i in range(len(staking_table)):
                if staking_table[i, 0] <= br < staking_table[i, 1]:
                    t1 = staking_table[i, 2]; t2 = staking_table[i, 3]
                    t3 = staking_table[i, 4]; t4 = staking_table[i, 5]
                    break
            fractions = np.array([t1, t2, t3, t4])
            dynamic_cap = max(max(br, 0.0) * dynamic_cap_frac, min_cap_floor)

            is_cold_race   = np.random.random() < p_race_cold
            is_forced_miss = np.random.random() < p_race_forced_miss

            total_ewpd = 0.0
            for i in range(n_instruments):
                tier_idx = int(pool_stats[i, 4])
                if fractions[tier_idx] > 0:
                    total_ewpd += pool_stats[i, 3]

            if total_ewpd == 0.0:
                all_paths[p, race]  = br
                all_phases[p, race] = current_phase
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

            if stake > dynamic_cap:
                stake = dynamic_cap
            if stake <= 0.0:
                all_paths[p, race]  = br
                all_phases[p, race] = current_phase
                continue

            pool    = pnl_pools[idx]
            raw_pnl = pool[np.random.randint(0, len(pool))]

            if is_cold_race or is_forced_miss:
                raw_pnl = -tc

            br += raw_pnl * (stake / tc)
            all_paths[p, race] = br

            current_phase = n_phases - 1
            for i in range(n_phases):
                if phase_bounds[i, 0] <= br < phase_bounds[i, 1]:
                    current_phase = int(phase_bounds[i, 2])
                    break
            all_phases[p, race] = current_phase

            if br >= target:
                outcomes[p]  = 1; races_arr[p] = race
                for r in range(race + 1, max_races + 1):
                    all_paths[p, r]  = br
                    all_phases[p, r] = current_phase
                break
            if br <= ruin_floor:
                outcomes[p]  = 2; races_arr[p] = race
                for r in range(race + 1, max_races + 1):
                    all_paths[p, r]  = ruin_floor
                    all_phases[p, r] = current_phase
                break
        else:
            races_arr[p] = max_races

    return all_paths, all_phases, outcomes, races_arr

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
        hit_mask = pool > HR_MIN_JBM
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

    all_paths_m, all_phases_m, outcome_codes, races_arr = run_gauntlet_core(
        N_PATHS, MAX_TRADE_RACES,
        START_BANKROLL, TARGET, RUIN_FLOOR,
        DYNAMIC_CAP_FRAC, MAX_STAKE_FLOOR,
        P_RACE_COLD, P_RACE_FORCED_MISS,
        pools, stats, staking, phase_bounds,
    )

    code_map   = {0: "timeout", 1: "success", 2: "ruin"}
    outcomes   = [code_map[int(c)] for c in outcome_codes]
    races_list = races_arr.tolist()

    paths, phase_log, final_brs = [], [], []
    for p in range(N_PATHS):
        r = races_arr[p]
        paths.append(all_paths_m[p, :r + 1])
        phase_log.append(all_phases_m[p, :r + 1].tolist())
        final_brs.append(float(all_paths_m[p, r]))

    return paths, outcomes, races_list, final_brs, phase_log

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_stats(paths, outcomes, races_list, final_brs):
    n         = len(paths)
    n_success = outcomes.count("success")
    n_ruin    = outcomes.count("ruin")
    n_timeout = outcomes.count("timeout")

    success_races = [r for r, o in zip(races_list, outcomes) if o == "success"]
    ruin_races    = [r for r, o in zip(races_list, outcomes) if o == "ruin"]

    med_br  = float(np.median(final_brs))
    med_idx = int(np.argmin(np.abs(np.array(final_brs) - med_br)))
    all_dd  = [float(np.max(np.maximum.accumulate(eq) - eq)) for eq in paths]
    pcts    = [10, 25, 50, 75, 90]
    br_pcts = {p: round(float(np.percentile(final_brs, p)), 2) for p in pcts}

    def _safe_median(lst):
        return int(np.median(lst)) if lst else None
    def _safe_pct(lst, q):
        return int(np.percentile(lst, q)) if len(lst) > 1 else None
    def _to_days(r):
        return r // RACES_PER_DAY if r is not None else None

    mrs = _safe_median(success_races)
    mrr = _safe_median(ruin_races)
    p10 = _safe_pct(success_races, 10)
    p90 = _safe_pct(success_races, 90)

    return {
        "n_paths": n, "n_success": n_success, "n_ruin": n_ruin,
        "n_timeout": n_timeout,
        "success_rate":  round(n_success / n * 100, 2),
        "ruin_rate":     round(n_ruin    / n * 100, 2),
        "timeout_rate":  round(n_timeout / n * 100, 2),
        "median_races_success": mrs, "median_races_ruin": mrr,
        "p10_races_success": p10,   "p90_races_success": p90,
        "median_days_success": _to_days(mrs),
        "median_days_ruin":    _to_days(mrr),
        "p10_days_success":    _to_days(p10),
        "p90_days_success":    _to_days(p90),
        "races_per_day_display": RACES_PER_DAY,
        "median_final_br":      round(med_br, 2),
        "mean_final_br":        round(float(np.mean(final_brs)), 2),
        "br_percentiles":       br_pcts,
        "median_max_dd":        round(float(np.median(all_dd)), 2),
        "required_bankroll_3x": round(float(np.median(all_dd)) * 3, 2),
        "median_path_idx":      med_idx,
        "p_race_cold":          P_RACE_COLD,
        "p_race_forced_miss":   P_RACE_FORCED_MISS,
        "payout_jitter_frac":   PAYOUT_JITTER_FRAC,
        "hr_min_jbm":           HR_MIN_JBM,
        "dynamic_cap_frac":     DYNAMIC_CAP_FRAC,
        "ruin_floor":           RUIN_FLOOR,
        "hit_rate_definition":  f"pnl > {HR_MIN_JBM} (unified)",
        "version":              "v5.4.1",
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
                    f"${START_BANKROLL:,.0f} → ${TARGET:,.0f} Simulation Paths [v5.4.1]",
              xlabel="Race Events", ylabel="Bankroll ($, symlog scale)")
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
           f"Cold: {stats['p_race_cold']*100:.0f}%  Miss: {stats['p_race_forced_miss']*100:.0f}%")
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
    _style_ax(ax1, title="Final Bankroll Distribution [v5.4.1]",
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
    _style_ax(ax2, title="Cumulative Ruin & Success Probability Over Time [v5.4.1]",
              xlabel="Race Events", ylabel="Cumulative Probability (%)")
    max_races = max(len(p) for p in paths)
    n = len(paths)

    def _cum(tgt):
        td = np.array([min(len(p)-1, max_races-1)
                        for p, o in zip(paths, outcomes) if o == tgt], dtype=np.intp)
        if not len(td): return np.zeros(max_races)
        return np.cumsum(np.bincount(td, minlength=max_races).astype(float)) / n * 100

    for curve, color, label in [
        (_cum("ruin"),    RED,    f"Cumulative ruin  (final: {stats['ruin_rate']:.1f}%)"),
        (_cum("success"), ACCENT, f"Cumulative success  (final: {stats['success_rate']:.1f}%)"),
    ]:
        ax2.plot(curve, color=color, linewidth=2.2, label=label)
        ax2.fill_between(range(max_races), curve, color=color, alpha=0.10)

    if stats["median_races_success"]:
        ax2.axvline(stats["median_races_success"], color=GOLD, lw=1.4,
                    linestyle="--",
                    label=f"Median: {stats['median_races_success']:,} races")
    if stats.get("p10_races_success"):
        ax2.axvline(stats["p10_races_success"], color=GOLD, lw=0.8,
                    linestyle=":", label=f"P10: {stats['p10_races_success']:,} races")

    ax2.set_xlim(0, max_races); ax2.set_ylim(0, 105)
    ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    fig.suptitle("THE 100x GAUNTLET — Outcome Analysis [v5.4.1]",
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
    fig.suptitle("THE 100x GAUNTLET — Per-Instrument Profile [v5.4.1]",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40)

    metrics = []
    for name in names:
        inst = INSTRUMENTS[name]; arr = empirical_pnl_map[name]; tc = inst["ticket_cost"]
        hm   = (arr > HR_MIN_JBM)
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
         f"Hit Rate (pnl>{HR_MIN_JBM})",           "Hit Rate (%)",       "%")
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
    max_races = max(len(p) for p in phase_log)
    pm = np.zeros((len(PHASES), max_races))
    for ps in phase_log:
        for d, ph in enumerate(ps):
            if ph < len(PHASES): pm[ph, d] += 1
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    _style_ax(ax, title="THE 100x GAUNTLET — Phase Transitions [v5.4.1]",
              xlabel="Race Events", ylabel="% of Paths")
    dt = pm.sum(axis=0); dt[dt==0] = 1; pct = pm / dt * 100
    yb = np.zeros(max_races)
    for i, (_, _, lbl, color) in enumerate(PHASES):
        yt = yb + pct[i]
        ax.fill_between(range(max_races), yb, yt,
                        color=color, alpha=0.85, label=lbl.replace('\n',' '))
        yb = yt
    ax.set_ylim(0,100); ax.set_xlim(0,max_races)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
              loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig("Gauntlet_Phases.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Phases.png")

def chart_comfort_score(empirical_pnl_map):
    """Visualizes instrument stability (Sharpe-like) vs performance (EV)."""
    names = sorted(
        [n for n in INSTRUMENTS
         if n in empirical_pnl_map and len(empirical_pnl_map[n]) > 0],
        key=lambda n: (INSTRUMENTS[n]["tier"], n)
    )
    if not names: return

    evs = []
    stds = []
    colors = []

    for name in names:
        arr = empirical_pnl_map[name]
        ev = float(arr.mean())
        std = float(arr.std())
        evs.append(ev)
        stds.append(std)
        colors.append(TIER_COLORS.get(INSTRUMENTS[name]["tier"], MUT))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    _style_ax(ax, title="Instrument Comfort Score: Stability vs Performance",
              xlabel="Volatility (Std Dev of P&L)", ylabel="Performance (Mean EV)")

    ax.scatter(stds, evs, c=colors, s=100, alpha=0.7, edgecolors=GRID)

    for i, name in enumerate(names):
        ax.annotate(name, (stds[i], evs[i]), color=TEXT, fontsize=8, xytext=(5, 5), textcoords='offset points')

    ax.axhline(0, color=MUT, lw=1, linestyle='--')

    plt.tight_layout()
    plt.savefig("Gauntlet_Comfort.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("  📊 Gauntlet_Comfort.png")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(empirical_pnl_map, stats):
    SEP = "═" * 80
    print(f"\n{SEP}")
    print("  THE 100x GAUNTLET — SIMULATION RESULTS  [v5.4.1]")
    print(f"  ${START_BANKROLL:,.0f} → ${TARGET:,.0f}  |  "
          f"{stats['n_paths']:,} paths  |  "
          f"up to {MAX_TRADE_RACES:,} race events each")
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
    print(f"    Per-race cold   : {stats['p_race_cold']*100:.0f}%")
    print(f"    Forced miss     : {stats['p_race_forced_miss']*100:.0f}%")
    print(f"    Payout jitter   : ±{stats['payout_jitter_frac']*100:.0f}%")
    print(f"    Dynamic stake cap: {DYNAMIC_CAP_FRAC*100:.0f}% (floor ${MAX_STAKE_FLOOR:.0f})")
    print(f"\n  Final Bankroll Percentiles:")
    for p, v in stats["br_percentiles"].items():
        print(f"    P{p}: ${v:,.2f}")
    print(f"\n  Instrument pool sizes:")
    print(f"  {'Instrument':<32} {'N':>7}  {'EV':>9}  {'HR':>7}")
    print("  " + "─"*60)
    for name in sorted(empirical_pnl_map):
        arr = empirical_pnl_map[name]
        tc  = INSTRUMENTS[name]["ticket_cost"]
        hm  = (arr > HR_MIN_JBM)
        print(f"  {name:<32} {len(arr):>7,}  {arr.mean():>+9.2f}  {hm.mean()*100:>6.1f}%")
    print(SEP + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print(" 🏁 STARTING THE 100x GAUNTLET STRESS TESTING PIPELINE v5.4.1")
    print(f"    PRIMARY UNIT: Race Events  "
          f"(max {MAX_TRADE_RACES:,} races ≈ "
          f"{MAX_TRADE_RACES//RACES_PER_DAY:,} days)")
    print(f"    Dynamic stake cap: {DYNAMIC_CAP_FRAC*100:.0f}% of BR "
          f"(floor ${MAX_STAKE_FLOOR:.0f})")
    print(f"    Ruin floor: ${RUIN_FLOOR:.0f}  |  "
          f"Terminal sprint bracket: $60k+")
    print(f"    v5.4.1: max_runners fix | staking fixes | v5.1 Pruned + v5.4 Scouts")
    print("=" * 80)

    df_2025 = _load_and_prep(TRAIN_FILE)
    df_2026 = _load_and_prep(TEST_FILE)

    if df_2025.empty and df_2026.empty:
        print("❌ CRITICAL ERROR: Source data empty."); exit(1)

    df_all = pd.concat([df_2025, df_2026], ignore_index=True)

    empirical_pnl_map = {}
    print(f"\nExtracting cross-validated historical P&L pools...")
    print(f"  Registry: {len(INSTRUMENTS)} instruments")

    for name, inst in INSTRUMENTS.items():
        pnl_arr = _extract_empirical_pnl_fast(df_all, name, inst)
        if len(pnl_arr) > 0:
            hm  = pnl_arr > HR_MIN_JBM
            tag = " ★NM" if name.startswith("NM_") else \
                  " ★SB" if name.startswith("SB") else ""
            print(f"  {name:<32} : {len(pnl_arr):>6,} events  "
                  f"EV: {pnl_arr.mean():>+8.2f}  "
                  f"HR(pnl>{HR_MIN_JBM}): {hm.mean()*100:>5.1f}%{tag}")
            empirical_pnl_map[name] = pnl_arr
        else:
            print(f"  {name:<32} : 0 events — skipped")

    if not empirical_pnl_map:
        print("❌ CRITICAL ERROR: No events passed filters."); exit(1)

    paths, outcomes, races_list, final_brs, phase_log = _simulate_paths(
        empirical_pnl_map)

    print("📊 Computing telemetry...")
    stats = _compute_stats(paths, outcomes, races_list, final_brs)

    print("🎨 Rendering dashboards...")
    chart_paths(paths, outcomes, stats)
    chart_distribution(paths, outcomes, final_brs, stats)
    chart_instruments(empirical_pnl_map)
    chart_phases(phase_log)
    chart_comfort_score(empirical_pnl_map)

    print_report(empirical_pnl_map, stats)

    print("💾 Archiving...")
    with open("Gauntlet_Results.json", "w") as f:
        json.dump(stats, f, indent=4)
    flat = {k: v for k, v in stats.items() if k != "br_percentiles"}
    for p, v in stats["br_percentiles"].items():
        flat[f"br_percentile_P{p}"] = v
    pd.DataFrame([flat]).to_csv("Gauntlet_Results.csv", index=False)
    print("  ✓ Gauntlet_Results.json + Gauntlet_Results.csv")

    try:
        from importable_GC_chartmaker__BankrollGrowthV5 import run_comparison_charts
        run_comparison_charts(empirical_pnl_map, INSTRUMENTS)
    except ImportError:
        print("\n  ℹ  importable_GC_chartmaker__BankrollGrowthV5.py not found.")

    print("\n✅ THE 100x GAUNTLET v5.4.1 SIMULATION COMPLETE.")
