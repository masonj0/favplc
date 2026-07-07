#!/usr/bin/env python3
"""
THE 100x GAUNTLET — v1.1.0 GOLD MASTER
"If the math don't work, we don't bet."
"""

import os
import sys
import datetime
import math
import warnings
import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
BG, MUT, GRD, TXT = "#0f172a", "#64748b", "#1e293b", "#f1f5f9"
ACC, RED, GLD = "#10b981", "#f43f5e", "#f59e0b"

# Global Sim Constants
S_BR    = 2000.0   # Starting Bankroll
R_FLR   = 200.0    # Ruin Floor
TGT     = 200000.0 # Target/Retirement
MAX_BET = 5000.0   # Max wager per race
N_PTH   = 1000     # Number of paths to simulate
M_RCS   = 10000    # Max races per path
RPD     = 25       # Average races per day
SEED    = 42
JITTER  = 0.05     # Randomized payout noise
UNIT_SUPERF = 2.00
UNIT_TRI    = 2.00
BASE_BET_FACTOR = 2.00

TRAIN_FILE, TEST_FILE = "RaceRecords_Output_2025.csv", "RaceRecords_Output_2026.csv"

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT FILTERS
# ══════════════════════════════════════════════════════════════════════════════
def _mk_env(fmin=0, fmax=99, c=None, fr=None, spr=None, rmin=1, rmax=99,
            smin=0.0, smax=99.0, f2min=0.0, f2max=99.0, plo=0, phi=1000000):
    def env_filter(df):
        m = (df["Runners"].between(fmin, fmax)) & \
            (df["WhichRace"].between(rmin, rmax)) & \
            (df["SumOf1st2Odds"].between(smin, smax)) & \
            (df["Purse"].between(plo, phi))
        if c is not None:
            c_col = "ChalkStatus" if "ChalkStatus" in df else "Chalk"
            if c_col in df: m &= (df[c_col] == c)
        if fr is not None:
            f_col = "FirstStatus" if "FirstStatus" in df else "First"
            if f_col in df: m &= (df[f_col] == fr)
        if spr is not None and "Sprint" in df:
            m &= (df["Sprint"] == spr)
        if "Fav2Odds" in df:
            m &= df["Fav2Odds"].between(f2min, f2max)
        elif f2min > 0.0 or f2max < 99.0:
            f2 = df["SumOf1st2Odds"] - df["1stOdds"] if "1stOdds" in df else pd.Series(0, index=df.index)
            m &= f2.between(f2min, f2max)
        return m
    return env_filter

# ══════════════════════════════════════════════════════════════════════════════
# HIT FUNCTIONS (VECTORIZED)
# ══════════════════════════════════════════════════════════════════════════════
def _h_al(p1, p2, p3, p4):    return np.ones_like(p1, dtype=bool)
def _h_t145(p1, p2, p3, p4):  return (p1 == 1) & (p2 <= 4) & (p3 <= 5)
def _h_s4455(p1, p2, p3, p4): return (p1 <= 4) & (p2 <= 4) & (p3 <= 5) & (p4 <= 5)
def _h_s5556(p1, p2, p3, p4): return (p1 <= 5) & (p2 <= 5) & (p3 <= 5) & (p4 <= 6)
def _h_s4456(p1, p2, p3, p4): return (p1 <= 4) & (p2 <= 4) & (p3 <= 5) & (p4 <= 6)
def _h_s3666(p1, p2, p3, p4): return (p1 <= 3) & (p2 <= 6) & (p3 <= 6) & (p4 <= 6)
def _h_s6667(p1, p2, p3, p4): return (p1 <= 6) & (p2 <= 6) & (p3 <= 6) & (p4 <= 7)
def _h_s4466(p1, p2, p3, p4): return (p1 <= 4) & (p2 <= 4) & (p3 <= 6) & (p4 <= 6)
def _h_s4444(p1, p2, p3, p4): return (p1 <= 4) & (p2 <= 4) & (p3 <= 4) & (p4 <= 4)
def _h_s5567(p1, p2, p3, p4): return (p1 <= 5) & (p2 <= 5) & (p3 <= 6) & (p4 <= 7)
def _h_s2444(p1, p2, p3, p4): return (p1 <= 2) & (p2 <= 4) & (p3 <= 4) & (p4 <= 4)
def _h_s2555(p1, p2, p3, p4): return (p1 <= 2) & (p2 <= 5) & (p3 <= 5) & (p4 <= 5)
def _h_s1234(p1, p2, p3, p4): return (p1 == 1) & (p2 == 2) & (p3 == 3) & (p4 == 4)
def _h_s2355(p1, p2, p3, p4): return (p1 <= 2) & (p2 <= 3) & (p3 <= 5) & (p4 <= 5)
def _h_tri2l1(p1, p2, p3, p4): return (p1 <= 2) & (p2 <= 6) & (p3 <= 2)
def _h_tria22(p1, p2, p3, p4): return (p1 >= 3) & (p2 <= 2) & (p3 <= 2) & (p2 != p3)
def _h_s1445(p1, p2, p3, p4): return (p1 == 1) & (p2 <= 4) & (p3 <= 4) & (p4 <= 5)
def _h_s3444(p1, p2, p3, p4): return (p1 <= 3) & (p2 <= 4) & (p3 <= 4) & (p4 <= 4)
def _h_s2266(p1, p2, p3, p4): return (p1 <= 2) & (p2 <= 2) & (p3 <= 6) & (p4 <= 6)
def _h_s1555(p1, p2, p3, p4): return (p1 == 1) & (p2 <= 5) & (p3 <= 5) & (p4 <= 5)
def _h_s4144(p1, p2, p3, p4): return (p1 <= 4) & (p2 == 1) & (p3 <= 4) & (p4 <= 4)
def _h_t123(p1, p2, p3, p4): return (p1 == 1) & (p2 == 2) & (p3 == 3)
def _h_s6678(p1, p2, p3, p4): return (p1 <= 6) & (p2 <= 6) & (p3 <= 7) & (p4 <= 8)

_VH = {
    "_h_al": _h_al, "_h_t145": _h_t145, "_h_s4455": _h_s4455,
    "_h_s5556": _h_s5556, "_h_s4456": _h_s4456, "_h_s3666": _h_s3666,
    "_h_s6667": _h_s6667, "_h_s4466": _h_s4466, "_h_s4444": _h_s4444,
    "_h_s5567": _h_s5567, "_h_s2444": _h_s2444, "_h_s2555": _h_s2555,
    "_h_s1234": _h_s1234, "_h_s2355": _h_s2355, "_h_tri2l1": _h_tri2l1,
    "_h_tria22": _h_tria22, "_h_s1445": _h_s1445, "_h_s3444": _h_s3444,
    "_h_s2266": _h_s2266, "_h_s1555": _h_s1555, "_h_s4144": _h_s4144,
    "_h_t123": _h_t123, "_h_s6678": _h_s6678
}

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT REGISTRY — v7.5.2
# ══════════════════════════════════════════════════════════════════════════════
INS = {
    # ── TIER 1A — active from $0 ──────────────────────────────────────────────
    "FB4_2": {
        "t":"T1A","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":48,"hf":"_h_al","pc":"Superf_paid","mn":4,"mx":4,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=4,fmax=4,c="N",smin=4,smax=7.5,f2min=2.5,f2max=4)},
    "FB4_3": {
        "t":"T1A","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":48,"hf":"_h_al","pc":"Superf_paid","mn":4,"mx":4,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=4,fmax=4,c="N",fr="N",smin=4,smax=6)},
    "FB4_4": {
        "t":"T1A","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":48,"hf":"_h_al","pc":"Superf_paid","mn":4,"mx":4,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=4,fmax=4,c="N",smin=4,smax=6)},
    "FB4_5": {
        "t":"T1A","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":48,"hf":"_h_al","pc":"Superf_paid","mn":4,"mx":4,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=4,fmax=4,c="N",spr="Y",smin=4,smax=6)},

    # ── TIER 1B — active from $3,000 ─────────────────────────────────────────
    "SB6_S5556_A": {
        "t":"T1B","tc_mode":"pattern_superf","pat":[5,5,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s5556","pc":"Superf_paid","mn":6,"mx":6,"ew":3.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",smin=6,smax=8.5,plo=8000,phi=30000)}, # W8
    "SB6_S5556_B": {
        "t":"T1B","tc_mode":"pattern_superf","pat":[5,5,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s5556","pc":"Superf_paid","mn":6,"mx":6,"ew":3.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",smin=6,smax=8.5,plo=8000,phi=30000)}, # W9
    "NM_Sup4455_N6_C": {
        "t":"T1B","tc_mode":"pattern_superf","pat":[4,4,5,5],
        "unit_price":UNIT_SUPERF,"_old_tc":144,
        "hf":"_h_s4455","pc":"Superf_paid","mn":6,"mx":6,"ew":2.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4)},
    "FB5_2": {
        "t":"T1B","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":240,"hf":"_h_al","pc":"Superf_paid","mn":5,"mx":5,"ew":2.5,"v":1,
        "ef":_mk_env(fmin=5,fmax=5,c="N",f2min=3)}, # W5

    # ── TIER 2 — active from $5,000 ──────────────────────────────────────────
    "TRI145": {
        "t":"T2","tc_mode":"pattern_tri","pat":[1,4,5],
        "unit_price":UNIT_TRI,"_old_tc":18,
        "hf":"_h_t145","pc":"Trif_paid","mn":5,"mx":13,"ew":2.8,"v":1,
        "ef":_mk_env(smin=8,f2min=2.5)}, # W1
    "SB6_S5556_C": {
        "t":"T2","tc_mode":"pattern_superf","pat":[5,5,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s5556","pc":"Superf_paid","mn":6,"mx":6,"ew":2.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",spr="Y",smin=6,smax=8.5,plo=8000,phi=35000)},
    "SB6_S5556_SumWide": {
        "t":"T2","tc_mode":"pattern_superf","pat":[5,5,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s5556","pc":"Superf_paid","mn":6,"mx":6,"ew":3.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",smin=6,smax=10.5,plo=8000,phi=25000)}, # W4
    "SB6_S5556_A_P35": {
        "t":"T2","tc_mode":"pattern_superf","pat":[5,5,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s5556","pc":"Superf_paid","mn":6,"mx":6,"ew":3.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",smin=6,smax=8.5,plo=8000,phi=40000)}, # W7
    "NM_Sup4455_N6_D": {
        "t":"T2","tc_mode":"pattern_superf","pat":[4,4,5,5],
        "unit_price":UNIT_SUPERF,"_old_tc":144,
        "hf":"_h_s4455","pc":"Superf_paid","mn":6,"mx":6,"ew":2.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4,smin=6,smax=9.5)},
    "NM_Sup4456_N6_D": {
        "t":"T2","tc_mode":"pattern_superf","pat":[4,4,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":216,
        "hf":"_h_s4456","pc":"Superf_paid","mn":6,"mx":6,"ew":2.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4,smin=6,smax=9.5)},
    "NM_Supr3666_N6_D": {
        "t":"T2","tc_mode":"pattern_superf","pat":[3,6,6,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s3666","pc":"Superf_paid","mn":6,"mx":6,"ew":2.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4,smin=6,smax=10.0)}, # W6
    "FB6_HighPurse": {
        "t":"T2","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=3,plo=25000)}, # W3

    # ── TIER 2 WATCH — v=0, not in active strategies ─────────────────────────
    "SB6_S5556_E": {
        "t":"T2","tc_mode":"pattern_superf","pat":[5,5,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s5556","pc":"Superf_paid","mn":6,"mx":6,"ew":3.0,"v":0,
        "ef":_mk_env(fmin=6,fmax=6,c="N",smin=6,smax=9.5,plo=8000,phi=35000)},

    # ── TIER 3 — active from $10,000 ─────────────────────────────────────────
    "FB4_6": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":48,"hf":"_h_al","pc":"Superf_paid","mn":4,"mx":4,"ew":1.0,"v":1,
        "ef":_mk_env(fmin=4,fmax=4,rmin=4,rmax=8,smin=4,smax=6,f2min=2.5,f2max=4)},
    "FB7_1": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":1680,"hf":"_h_al","pc":"Superf_paid","mn":7,"mx":7,"ew":0.5,"v":1,
        "ef":_mk_env(fmin=7,fmax=7,c="N",fr="N",spr="Y",plo=8000,phi=15000,smin=6,smax=8)},
    "FB7_2": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":1680,"hf":"_h_al","pc":"Superf_paid","mn":7,"mx":7,"ew":0.5,"v":1,
        "ef":_mk_env(fmin=7,fmax=7,c="N",plo=8000,phi=15000,smin=6,smax=8)},
    "FB7_3": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":1680,"hf":"_h_al","pc":"Superf_paid","mn":7,"mx":7,"ew":0.75,"v":1,
        "ef":_mk_env(fmin=7,fmax=7,c="N",spr="Y",plo=8000,phi=20000,f2min=4,smin=6)},
    "FB7_4": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":1680,"hf":"_h_al","pc":"Superf_paid","mn":7,"mx":7,"ew":0.5,"v":1,
        "ef":_mk_env(fmin=7,fmax=7,c="N",smin=6,smax=8,f2min=4)},
    "SB12_S6667_A": {
        "t":"T3","tc_mode":"pattern_superf","pat":[6,6,6,7],
        "unit_price":UNIT_SUPERF,"_old_tc":960,
        "hf":"_h_s6667","pc":"Superf_paid","mn":12,"mx":12,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=12,fmax=12,c="N",spr="N",rmin=4,rmax=9,
                     smin=4,smax=8.5,f2min=4,plo=8000,phi=25000)},
    "FB6_First": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":0,
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="Y",f2min=4)},
    "FB6_1": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",spr="Y",f2min=4)},
    "FB6_2": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",rmin=4,rmax=8,f2min=4)},
    "FB6_3": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",rmin=4,rmax=8,f2min=4)},
    "FB6_4": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",spr="Y",f2min=4)},
    "FB6_5": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",f2min=4)},
    "FB6_6": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4)},
    "NM_Sup4466": {
        "t":"T3","tc_mode":"pattern_superf","pat":[4,4,6,6],
        "unit_price":UNIT_SUPERF,"_old_tc":288,
        "hf":"_h_s4466","pc":"Superf_paid","mn":6,"mx":6,"ew":2.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4)},
    "NM_Sup5567": {
        "t":"T3","tc_mode":"pattern_superf","pat":[5,5,6,7],
        "unit_price":UNIT_SUPERF,"_old_tc":480,
        "hf":"_h_s5567","pc":"Superf_paid","mn":6,"mx":6,"ew":2.0,"v":1,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4)},
    "TRI_First": {
        "t":"T3","tc_mode":"pattern_tri","pat":[1,4,5],
        "unit_price":UNIT_TRI,"_old_tc":18,
        "hf":"_h_t145","pc":"Trif_paid","mn":5,"mx":13,"ew":2.0,"v":1,
        "ef":_mk_env(fr="Y",smin=9,f2min=2.0)}, # W2

    # ── BONUS — discovery, v=0, excluded from active strategies ──────────────
    "BNS_SB6_RouteOnly": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[5,5,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s5556","pc":"Superf_paid","mn":6,"mx":6,"ew":2.5,"v":0,
        "ef":_mk_env(fmin=6,fmax=6,c="N",spr="N",smin=6,smax=8.5,plo=8000,phi=25000)},
    "BNS_SB12_R9": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[6,6,6,7],
        "unit_price":UNIT_SUPERF,"_old_tc":960,
        "hf":"_h_s6667","pc":"Superf_paid","mn":12,"mx":12,"ew":1.5,"v":0,
        "ef":_mk_env(fmin=12,fmax=12,c="N",spr="N",rmin=4,rmax=9,
                     smin=4,smax=8.5,f2min=4,plo=8000,phi=25000)},
    "BNS_TRI_HighPurse": {
        "t":"BONUS","tc_mode":"pattern_tri","pat":[1,4,5],
        "unit_price":UNIT_TRI,"_old_tc":18,
        "hf":"_h_t145","pc":"Trif_paid","mn":5,"mx":13,"ew":2.0,"v":0,
        "ef":_mk_env(smin=9,f2min=2.5,plo=25000)},
    "BNS_TRI_Sum8": {
        "t":"BONUS","tc_mode":"pattern_tri","pat":[1,4,5],
        "unit_price":UNIT_TRI,"_old_tc":18,
        "hf":"_h_t145","pc":"Trif_paid","mn":5,"mx":13,"ew":3.0,"v":0,
        "ef":_mk_env(smin=8,f2min=2.5)},
    "BNS_FB5_Fav3": {
        "t":"BONUS","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":240,"hf":"_h_al","pc":"Superf_paid","mn":5,"mx":5,"ew":2.0,"v":0,
        "ef":_mk_env(fmin=5,fmax=5,c="N",f2min=3)},
    "BNS_Sup2355": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[2,3,5,5],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s2355","pc":"Superf_paid","mn":6,"mx":6,"ew":2.0,"v":0,
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=3,smin=6,smax=9,plo=10000)},
    "BNS_Tri2L1": {
        "t":"BONUS","tc_mode":"pattern_tri","pat":[2,6,2],
        "unit_price":UNIT_TRI,"_old_tc":18,
        "hf":"_h_tri2l1","pc":"Trif_paid","mn":6,"mx":13,"ew":3.0,"v":0,
        "ef":_mk_env(c="N",smin=9,f2min=4)},
    "BNS_TriA22": {
        "t":"BONUS","tc_mode":"all_over_top2_top2_tri",
        "unit_price":UNIT_TRI,"_old_tc":None,
        "hf":"_h_tria22","pc":"Trif_paid","mn":5,"mx":13,"ew":2.5,"v":0,
        "ef":_mk_env(c="N",smin=7,f2min=3)},

    # ── SNIPER DISCOVERIES ────────────────────────────────────────────────────
    "BNS_Supr2555_Sniper": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[2,5,5,5],
        "unit_price":UNIT_SUPERF,"_old_tc":96,
        "hf":"_h_s2555","pc":"Superf_paid","mn":5,"mx":7,"ew":1.5,"v":0,
        "ef":_mk_env(fmin=5,fmax=7,c="Y",smin=4.0,smax=7.5,f2max=2.0)},
    "BNS_Sup2444_High": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[2,4,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":24,
        "hf":"_h_s2444","pc":"Superf_paid","mn":4,"mx":6,"ew":1.0,"v":0,
        "ef":_mk_env(fmin=4,fmax=6,smin=7.5)},
    "BNS_Supr4444_High": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[4,4,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s4444","pc":"Superf_paid","mn":4,"mx":6,"ew":1.0,"v":0,
        "ef":_mk_env(fmin=4,fmax=6,smin=7.5)},
    "BNS_Supr1445_R7_Mid": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[1,4,4,5],
        "unit_price":UNIT_SUPERF,"_old_tc":24,
        "hf":"_h_s1445","pc":"Superf_paid","mn":7,"mx":7,"ew":1.5,"v":0,
        "ef":_mk_env(fmin=7,fmax=7,c="Y",smin=4.0,smax=7.5,f2max=2.0)},
    "BNS_Sup1234_Sniper": {
        "t":"BONUS","tc_mode":"straight_superf",
        "unit_price":UNIT_SUPERF,"_old_tc":2,"tc":2.0,
        "hf":"_h_s1234","pc":"Superf_paid","mn":7,"mx":7,"ew":1.0,"v":0,
        "ef":_mk_env(fmin=7,fmax=7,c="Y",smin=4.0,smax=7.5,f2max=2.0)},

    # ── HISTORICAL CONSENSUS ROUTES & BEST BETS ───────────────────────────────
    "BNS_Sup3444_R4": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[3,4,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":36,
        "hf":"_h_s3444","pc":"Superf_paid","mn":4,"mx":4,"ew":1.5,"v":0,
        "ef":_mk_env(fmin=4,fmax=4,c="N",smin=3.0,smax=7.0)},
    "BNS_Supr4444_R4": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[4,4,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s4444","pc":"Superf_paid","mn":4,"mx":4,"ew":1.5,"v":0,
        "ef":_mk_env(fmin=4,fmax=4,c="N",smin=3.0,smax=7.0)},
    "BNS_Sup2266_Large": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[2,2,6,6],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s2266","pc":"Superf_paid","mn":10,"mx":12,"ew":2.0,"v":0,
        "ef":_mk_env(fmin=10,fmax=12,c="N",smin=5.0,smax=7.0)},
    "BNS_Supr1555_Large": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[1,5,5,5],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s1555","pc":"Superf_paid","mn":10,"mx":12,"ew":2.0,"v":0,
        "ef":_mk_env(fmin=10,fmax=12,c="N",smin=5.0,smax=7.0)},
    "BNS_Supr4144_R8": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[4,1,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":16,
        "hf":"_h_s4144","pc":"Superf_paid","mn":8,"mx":8,"ew":1.5,"v":0,
        "ef":_mk_env(fmin=8,fmax=8,c="N",smin=2.0,smax=7.0,phi=18000)},
    "BNS_Tri123_Large": {
        "t":"BONUS","tc_mode":"straight_superf",
        "unit_price":UNIT_TRI,"_old_tc":2,"tc":2.0,
        "hf":"_h_t123","pc":"Trif_paid","mn":10,"mx":12,"ew":1.0,"v":0,
        "ef":_mk_env(fmin=10,fmax=12,c="N",smin=2.0,smax=7.0)},
    "BNS_Sup1234_Large": {
        "t":"BONUS","tc_mode":"straight_superf",
        "unit_price":UNIT_SUPERF,"_old_tc":2,"tc":2.0,
        "hf":"_h_s1234","pc":"Superf_paid","mn":10,"mx":12,"ew":1.0,"v":0,
        "ef":_mk_env(fmin=10,fmax=12,c="N",smin=2.0,smax=7.0)},

    # ── APEX — inactive, audit visibility only ────────────────────────────────
    "APX_Supr3666_Wide": {
        "t":"APEX","tc_mode":"pattern_superf","pat":[3,6,6,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s3666","pc":"Superf_paid","mn":5,"mx":12,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=5,fmax=12,plo=1000,phi=32500)},
    "APX_Supr3666": {
        "t":"APEX","tc_mode":"pattern_superf","pat":[3,6,6,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s3666","pc":"Superf_paid","mn":6,"mx":12,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=6,fmax=12,plo=500,phi=34500)},
    "SB12_S6678_A": {
        "t":"APEX","tc_mode":"pattern_superf","pat":[6,6,7,8],
        "unit_price":UNIT_SUPERF,"_old_tc":1500,
        "hf":"_h_s6678","pc":"Superf_paid","mn":12,"mx":12,"ew":1.5,"v":1,
        "ef":_mk_env(fmin=12,fmax=12,c="N")},
    "APX_Sup4456": {
        "t":"APEX","tc_mode":"pattern_superf","pat":[4,4,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":216,
        "hf":"_h_s4456","pc":"Superf_paid","mn":8,"mx":12,"ew":1.5,"q":1,
        "ef":_mk_env(fmin=8,fmax=12,plo=500,phi=34500)},
    "APX_Sup1234": {
        "t":"APEX","tc_mode":"straight_superf",
        "unit_price":UNIT_SUPERF,"_old_tc":2,"tc":2.0,
        "hf":"_h_s1234","pc":"Superf_paid","mn":4,"mx":13,"ew":1.5,"v":0,
        "ef":_mk_env(c="N",smin=4,smax=9)},
}

# ── Strategy sets ─────────────────────────────────────────────────────────────
_T1A = {"FB4_2","FB4_3","FB4_4","FB4_5"}
_T1B = {"SB6_S5556_A","SB6_S5556_B","NM_Sup4455_N6_C","FB5_2"}
_T1  = _T1A | _T1B
_T2  = {
    "TRI145",
    "SB6_S5556_C","SB6_S5556_SumWide","SB6_S5556_A_P35",
    "NM_Sup4455_N6_D","NM_Sup4456_N6_D","NM_Supr3666_N6_D",
    "FB6_HighPurse",
}
_T3_VALIDATED = {
    "FB4_6",
    "FB7_1","FB7_2","FB7_3","FB7_4",
    "SB12_S6667_A",
    "NM_Sup4466","NM_Sup5567",
    "TRI_First",
    "FB6_1","FB6_2","FB6_3","FB6_4","FB6_5","FB6_6",
}
_T3_ALL = {k for k,v in INS.items() if v["t"]=="T3"}
BONUS = {k for k,v in INS.items() if v["t"]=="BONUS"}
APX   = {k for k,v in INS.items() if v["t"]=="APEX"}

STRATEGY_SETS = {
    "SAFEST":   _T1,
    "STANDARD": _T1 | _T2,
    "FULL":     _T1 | _T2 | _T3_VALIDATED,
    "TURBO":    _T1 | _T2 | _T3_ALL,
}

# ══════════════════════════════════════════════════════════════════════════════
# STRESS PRESETS
# ══════════════════════════════════════════════════════════════════════════════
STRESS_PRESETS = {
    "VANILLA": {"lbl":"VANILLA", "c":0.03, "fm":0.08, "s":0.00, "mt":2000, "ms":0.00004, "td":0.3, "tdu":20},
    "HALF":    {"lbl":"HALF",    "c":0.05, "fm":0.12, "s":0.05, "mt":2000, "ms":0.00004, "td":0.3, "tdu":20},
    "FULL":    {"lbl":"FULL",    "c":0.05, "fm":0.12, "s":0.07, "mt":2000, "ms":0.00004, "td":0.3, "tdu":20},
    "NIGHTMARE":{"lbl":"NIGHTMARE", "c":0.08, "fm":0.15, "s":0.12, "mt":1000, "ms":0.0008, "td":0.2, "tdu":30},
}

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA CORE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True)
def run_gauntlet_core(npth, mxr, sbr, tgt, rfl, mb, p_pls, p_sts, t_tbl, c, fm, s, mt, ms, td, tdu):
    ap  = np.zeros((npth, mxr+1), dtype=np.float64)
    oc  = np.zeros(npth, dtype=np.int8)
    ra  = np.zeros(npth, dtype=np.int32)
    n_in = len(p_pls)

    for p in prange(npth):
        br, pk, tr = sbr, sbr, 0; ap[p,0] = br
        for r in range(1, mxr+1):
            tx0, tx1, tx2 = 0.0, 0.0, 0.0
            found = False
            for i in range(len(t_tbl)):
                if t_tbl[i,0] <= br < t_tbl[i,1]:
                    tx0, tx1, tx2 = t_tbl[i,2], t_tbl[i,3], t_tbl[i,4]
                    found = True; break
            if not found: tx0, tx1, tx2 = 10.0, 8.0, 6.0

            tot_w = 0.0
            for i in range(n_in):
                t_idx = int(p_sts[i,4]) # 1=T1, 2=T2, 3=T3
                if (t_idx == 1 and tx0 > 0) or (t_idx == 2 and tx1 > 0) or (t_idx == 3 and tx2 > 0):
                    tot_w += p_sts[i,3]
            if tot_w == 0.0: ap[p,r]=br; continue

            pv = np.random.random() * tot_w
            cs_sum, idx = 0.0, -1
            for i in range(n_in):
                t_idx = int(p_sts[i,4])
                if (t_idx == 1 and tx0 > 0) or (t_idx == 2 and tx1 > 0) or (t_idx == 3 and tx2 > 0):
                    cs_sum += p_sts[i,3]
                    if pv <= cs_sum: idx=i; break
            if idx == -1: idx = 0

            tc = p_sts[idx,2]
            t_idx = int(p_sts[idx,4])
            nt = tx0 if t_idx==1 else tx1 if t_idx==2 else tx2
            nt = min(nt, mb/tc)
            if nt < 1.0: nt = 1.0

            if np.random.random() < 0.02: # Simulated cold cycle
                ap[p,r]=br; continue

            pool = p_pls[idx]
            pnl = pool[np.random.randint(0, len(pool))]

            if np.random.random()<c or np.random.random()<fm or np.random.random() < 0.12:
                pnl = -tc
            elif pnl>0 and (nt*tc)>mt: # Market Impact
                pnl = (pnl+tc)*(1.0-min(0.5, (nt*tc-mt)*ms))-tc

            br += pnl * nt
            ap[p,r] = br

            if br >= tgt or br <= rfl:
                oc[p] = 1 if br >= tgt else 2
                ra[p] = r
                for x in range(r+1, mxr+1): ap[p,x] = br
                break
        else: ra[p] = mxr
    return ap, oc, ra

# ══════════════════════════════════════════════════════════════════════════════
# TICKET COST ENGINE (v7.5.2)
# ══════════════════════════════════════════════════════════════════════════════
def get_ticket_cost(inst, n_runners):
    mode = inst.get("tc_mode")
    unit = inst.get("unit_price", 1.0)

    if mode == "full_box_superf":
        # P(n, 4) = n * (n-1) * (n-2) * (n-3)
        if n_runners < 4: return 0.0
        perms = n_runners * (n_runners - 1) * (n_runners - 2) * (n_runners - 3)
        return perms * unit
    elif mode == "pattern_superf":
        # Product of wheel sizes
        pat = inst.get("pat", [1,1,1,1])
        return math.prod(pat) * unit
    elif mode == "pattern_tri":
        pat = inst.get("pat", [1,1,1])
        return math.prod(pat) * unit
    elif mode == "all_over_top2_top2_tri":
        # (n-2) * 2
        if n_runners < 3: return 0.0
        return (n_runners - 2) * 2 * unit
    elif mode == "straight_superf":
        return inst.get("tc", unit)

    return inst.get("_old_tc", 1.0)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PNL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_pnl(df, k, i):
    if df.empty: return np.array([], dtype=np.float64)
    m = df["Runners"].between(i["mn"], i.get("mx", 99))
    ef = i["ef"]; m &= ef(df)
    if not m.any(): return np.array([], dtype=np.float64)

    q = np.where(m)[0]; pc = i["pc"]
    if pc not in df: return np.array([], dtype=np.float64)

    # Dynamic TC extraction per race based on runner count
    runner_counts = df["Runners"].values[q].astype(int)
    tcs = np.array([get_ticket_cost(i, n) for n in runner_counts])

    payouts = df[pc].values[q].astype(float) * (BASE_BET_FACTOR / 2.0)
    valid_payouts = (payouts > 0) & (~np.isnan(payouts))

    rng = np.random.default_rng(SEED + abs(hash(k)) % 1000000)
    jitter = rng.uniform(1 - JITTER, 1 + JITTER, len(q))

    hf_key = i["hf"]
    if hf_key == "_h_al":
        return np.where(valid_payouts, payouts * jitter - tcs, -tcs).astype(np.float64)
    elif (fn := _VH.get(hf_key)) and "RANKED_RESULTS" in df:
        try:
            s = df["RANKED_RESULTS"].iloc[q].str.split(expand=True)
            req_ranks = 3 if "Trif" in pc else 4
            p = []
            mask = np.ones(len(q), dtype=bool)
            for x in range(4):
                arr = pd.to_numeric(s.iloc[:,x] if x < s.shape[1] else np.nan, errors="coerce").values
                p.append(arr)
                if x < req_ranks: mask &= ~np.isnan(arr)

            hits = mask & fn(*p)
            return np.where(valid_payouts & hits, payouts * jitter - tcs, -tcs).astype(np.float64)
        except Exception: return np.full(len(q), -tcs.mean(), dtype=np.float64)
    return np.full(len(q), -tcs.mean(), dtype=np.float64)

def load_data(f1, f2):
    dfs = []
    for f in [f1, f2]:
        if os.path.exists(f):
            raw = pd.read_csv(f)
            for c in ["Superf_paid", "Trif_paid", "Purse", "Exacta_paid", "Superfecta_paid", "Trifecta_paid"]:
                if c in raw:
                    raw[c] = raw[c].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
                    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)
            if "Superfecta_paid" in raw and "Superf_paid" not in raw: raw["Superf_paid"] = raw["Superfecta_paid"]
            if "Trifecta_paid" in raw and "Trif_paid" not in raw: raw["Trif_paid"] = raw["Trifecta_paid"]
            dfs.append(raw)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def main():
    print("="*72); print("  THE 100x GAUNTLET — v1.1.0 GOLD MASTER"); print("="*72)
    df = load_data(TRAIN_FILE, TEST_FILE)
    if df.empty: print("❌ Error: No race records found."); return
    print(f"✅ Loaded {len(df):,} historical races.")

    ql = (input("\n Quick Launch? [1] Yes (FULL HALF Gauntlet) [2] Customize\n Choice [1]: ").strip() or "1")
    if ql=="1": slbl, stress = "FULL", "HALF"
    else:
        slbl = {"1":"SAFEST","2":"STANDARD","3":"FULL","4":"TURBO"}.get(input("\n Strat: [1] SAFEST [2] STANDARD [3] FULL [4] TURBO [3]: ").strip() or "3", "FULL")
        stress = {"1":"VANILLA","2":"HALF","3":"FULL","4":"NIGHTMARE"}.get(input("\n Stress: [1] VANILLA [2] HALF [3] FULL [4] NIGHTMARE [2]: ").strip() or "2", "HALF")

    so = STRESS_PRESETS[stress]
    tkt_table = np.array([[0,2000,1,0,0],[2000,3000,1,1,0],[3000,5000,2,1,0],[5000,7500,2,2,0],[7500,10000,3,2,1],[10000,15000,3,3,1],[15000,25000,4,3,2],[25000,40000,5,4,2],[40000,70000,6,5,3],[70000,120000,8,6,4],[120000,999999,10,8,6]], dtype=np.float64)

    pnl_pools, stats_matrix = List(), []
    tmap = {"T1A":1,"T1B":1,"T1":1,"T2":2,"T3":3,"BONUS":3,"APEX":3}
    for k in sorted(STRATEGY_SETS[slbl]):
        inst = INS[k]
        pnl = extract_pnl(df, k, inst)
        if len(pnl)==0: continue
        pnl_pools.append(pnl)
        # Using inst["ew"] (Edge Weight) as replacement for comfort score
        avg_tc = np.mean([get_ticket_cost(inst, n) for n in [inst["mn"], inst.get("mx", inst["mn"])]])
        stats_matrix.append([(pnl>0).mean(), pnl[pnl>0].mean() if (pnl>0).any() else 0, avg_tc, inst["ew"], tmap.get(inst["t"],3)])

    if not stats_matrix: print("❌ No valid data for strategy."); return
    stats_matrix = np.array(stats_matrix, dtype=np.float64)
    print(f"🚀 Running {N_PTH} paths with {slbl} strategy...");
    ap, oc, ra = run_gauntlet_core(N_PTH, M_RCS, S_BR, TGT, R_FLR, MAX_BET, pnl_pools, stats_matrix, tkt_table, so["c"], so["fm"], so["s"], so["mt"], so["ms"], so["td"], so["tdu"])

    success, ruin = (oc==1).sum(), (oc==2).sum()
    print("-" * 30); print(f" SUCCESS: {success/N_PTH*100:>.1f}% ({success})"); print(f" RUIN:    {ruin/N_PTH*100:>.1f}% ({ruin})"); print("-" * 30)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG); ax.set_facecolor(BG); ax.set_yscale("log")
    for i in range(min(300, N_PTH)):
        c = ACC if oc[i]==1 else RED if oc[i]==2 else MUT
        ax.plot(ap[i, :ra[i]], color=c, alpha=0.1, lw=0.5)

    final_brs = np.array([ap[i, ra[i]] for i in range(N_PTH)])
    med_idx = np.argsort(final_brs)[len(final_brs)//2]
    ax.plot(ap[med_idx, :ra[med_idx]], color=GLD, lw=2, label="Median Path")
    ax.set_title(f"GAUNTLET v1.1.0 — {slbl} | {stress}", color=TXT, fontsize=14)
    plt.savefig(f"Gauntlet_Results_{slbl}_{stress}.png", facecolor=BG)
    print(f"📈 Chart saved to 'Gauntlet_Results_{slbl}_{stress}.png'")

if __name__ == "__main__": main()
