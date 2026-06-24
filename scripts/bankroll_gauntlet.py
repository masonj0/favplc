#!/usr/bin/env python3
"""
THE 100x GAUNTLET — v7.5.2 "OMNI RESTORE"
warnings.filterwarnings("ignore")
"""

import os
import sys
import datetime
import math
import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
BG, MUT, GRD, TXT = "#0f172a", "#64748b", "#1e293b", "#f1f5f9"
ACC, RED, GLD = "#10b981", "#f43f5e", "#f59e0b"

def _safe_print(m):
    try: print(m)
    except: pass

def _inst_snapshot(pmap):
    return {k: {"n": len(v), "ev": float(v.mean()) if len(v) else 0.0}
            for k, v in pmap.items()}

def _mk_env(field_min=0, field_max=99, chalk_req=None, first_req=None, sprint_req=None,
            race_min=1, race_max=99, sum_min=0.0, sum_max=99.0,
            fav2_min=0.0, fav2_max=99.0, purse_lo=0, purse_hi=1_000_000,
            fmin=None, fmax=None, c=None, fr=None, spr=None, rmin=None, rmax=None,
            smin=None, smax=None, f2min=None, f2max=None, plo=None, phi=None):
    # Map short names to long names if provided
    field_min = fmin if fmin is not None else field_min
    field_max = fmax if fmax is not None else field_max
    chalk_req = c if c is not None else chalk_req
    first_req = fr if fr is not None else first_req
    sprint_req = spr if spr is not None else sprint_req
    race_min = rmin if rmin is not None else race_min
    race_max = rmax if rmax is not None else race_max
    sum_min = smin if smin is not None else sum_min
    sum_max = smax if smax is not None else sum_max
    fav2_min = f2min if f2min is not None else fav2_min
    fav2_max = f2max if f2max is not None else fav2_max
    purse_lo = plo if plo is not None else purse_lo
    purse_hi = phi if phi is not None else purse_hi

    def env_filter(df):
        m = (df["Runners"].between(field_min, field_max)) & \
            (df["WhichRace"].between(race_min, race_max)) & \
            (df["SumOf1st2Odds"].between(sum_min, sum_max)) & \
            (df["Purse"].between(purse_lo, purse_hi))

        if chalk_req is not None:
            c_col = "Chalk" if "Chalk" in df else "ChalkStatus" if "ChalkStatus" in df else None
            if c_col: m &= (df[c_col] == chalk_req)

        if first_req is not None:
            f_col = "First" if "First" in df else "FirstStatus" if "FirstStatus" in df else None
            if f_col: m &= (df[f_col] == first_req)

        if sprint_req is not None and "Sprint" in df:
            m &= (df["Sprint"] == sprint_req)

        if fav2_min > 0.0 or fav2_max < 99.0:
            f2 = df["Fav2Odds"] if "Fav2Odds" in df else (
                df["SumOf1st2Odds"] - df["1stOdds"] if "1stOdds" in df else pd.Series(0, index=df.index))
            m &= f2.between(fav2_min, fav2_max)
        return m
    return env_filter

# ══════════════════════════════════════════════════════════════════════════════
# HIT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
_h_al = lambda a,b,c,d: True
_h_t145 = lambda a,b,c,d: (a==1)&(b>=2)&(b<=4)&(c>=2)&(c<=5)
_h_s5556 = lambda a,b,c,d: (a<=5)&(b<=5)&(c<=5)&(d<=6)
_h_s4455 = lambda a,b,c,d: (a<=4)&(b<=4)&(c<=5)&(d<=5)
_h_s4456 = lambda a,b,c,d: (a<=4)&(b<=4)&(c<=5)&(d<=6)
_h_s3666 = lambda a,b,c,d: (a<=3)&(b<=6)&(c<=6)&(d<=6)
_h_s6667 = lambda a,b,c,d: (a<=6)&(b<=6)&(c<=6)&(d<=7)
_h_s4466 = lambda a,b,c,d: (a<=4)&(b<=4)&(c<=6)&(d<=6)
_h_s5567 = lambda a,b,c,d: (a<=5)&(b<=5)&(c<=6)&(d<=7)
_h_s2355 = lambda a,b,c,d: (a<=2)&(b<=3)&(c<=5)&(d<=5)
_h_tri2l1 = lambda a,b,c,d: (a<=2)&(b<=6)&(c<=2)
_h_tria22 = lambda a,b,c,d: (b<=2)&(c<=2)
_h_s2555 = lambda a,b,c,d: (a<=2)&(b<=5)&(c<=5)&(d<=5)
_h_s2444 = lambda a,b,c,d: (a<=2)&(b<=4)&(c<=4)&(d<=4)
_h_s4444 = lambda a,b,c,d: (a<=4)&(b<=4)&(c<=4)&(d<=4)
_h_s1445 = lambda a,b,c,d: (a==1)&(b<=4)&(c<=4)&(d<=5)
_h_s1234 = lambda a,b,c,d: (a==1)&(b==2)&(c==3)&(d==4)
_h_s3444 = lambda a,b,c,d: (a<=3)&(b<=4)&(c<=4)&(d<=4)
_h_s2266 = lambda a,b,c,d: (a<=2)&(b<=2)&(c<=6)&(d<=6)
_h_s1555 = lambda a,b,c,d: (a==1)&(b<=5)&(c<=5)&(d<=5)
_h_s4144 = lambda a,b,c,d: (a<=4)&(b==1)&(c<=4)&(d<=4)
_h_t123 = lambda a,b,c,d: (a==1)&(b==2)&(c==3)
_h_s6678 = lambda a,b,c,d: (a<=6)&(b<=6)&(c<=7)&(d<=8)

_VH = {
    "_h_al": _h_al, "_h_t145": _h_t145, "_h_s5556": _h_s5556,
    "_h_s4455": _h_s4455, "_h_s4456": _h_s4456, "_h_s3666": _h_s3666,
    "_h_s6667": _h_s6667, "_h_s4466": _h_s4466, "_h_s5567": _h_s5567,
    "_h_s2355": _h_s2355, "_h_tri2l1": _h_tri2l1, "_h_tria22": _h_tria22,
    "_h_s2555": _h_s2555, "_h_s2444": _h_s2444, "_h_s4444": _h_s4444,
    "_h_s1445": _h_s1445, "_h_s1234": _h_s1234, "_h_s3444": _h_s3444,
    "_h_s2266": _h_s2266, "_h_s1555": _h_s1555, "_h_s4144": _h_s4144,
    "_h_t123": _h_t123, "_h_s6678": _h_s6678
}

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_FILE, TEST_FILE = "RaceRecords_Output_2025.csv", "RaceRecords_Output_2026.csv"

S_BR    = 2_000.0
R_FLR   = 200.0
TGT     = 200_000.0
MAX_BET = 5_000.0

N_PTH, M_RCS, RPD, SEED = 1000, 10000, 25, 42
JITTER, HR_MIN, WOW_ABS, WOW_RAT = 0.05, 2, 25000.0, 200.0

BASE_BET_FACTOR = 2.00
UNIT_SUPERF = 2.00
UNIT_TRI    = 2.00

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
        "watch_list":True,
        "watch_reason":"EV/tc=+0.335 confirmed. plo=25000 is differentiator.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=3,plo=25000)}, # W3

    # ── TIER 2 WATCH — v=0, not in active strategies ─────────────────────────
    "SB6_S5556_E": {
        "t":"T2","tc_mode":"pattern_superf","pat":[5,5,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s5556","pc":"Superf_paid","mn":6,"mx":6,"ew":3.0,"v":0,
        "watch_list":True,
        "watch_reason":"EV=+300 N=90. v=0 policy. Needs OOS confirm before v=1.",
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
        "watch_list":True,"watch_reason":"EV=+574 EV/tc=+0.342 — highest absolute EV in T3.",
        "ef":_mk_env(fmin=7,fmax=7,c="N",spr="Y",plo=8000,phi=20000,f2min=4,smin=6)},
    "FB7_4": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":1680,"hf":"_h_al","pc":"Superf_paid","mn":7,"mx":7,"ew":0.5,"v":1,
        "ef":_mk_env(fmin=7,fmax=7,c="N",smin=6,smax=8,f2min=4)},
    "SB12_S6667_A": {
        "t":"T3","tc_mode":"pattern_superf","pat":[6,6,6,7],
        "unit_price":UNIT_SUPERF,"_old_tc":960,
        "hf":"_h_s6667","pc":"Superf_paid","mn":12,"mx":12,"ew":1.5,"v":1,
        "watch_list":True,
        "watch_reason":"36mo EV=+292 N=44. Negative 18mo. Hold T3, monitor.",
        "ef":_mk_env(fmin=12,fmax=12,c="N",spr="N",rmin=4,rmax=9,
                     smin=4,smax=8.5,f2min=4,plo=8000,phi=25000)},
    "FB6_First": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":0,
        "watch_list":True,"watch_reason":"QUARANTINED: EV/tc=-0.183 all windows.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="Y",f2min=4)},
    "FB6_1": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "watch_list":True,"watch_reason":"DEMOTED T2→T3 v7.1.8. EV/tc=+0.239 overall.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",spr="Y",f2min=4)},
    "FB6_2": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "watch_list":True,"watch_reason":"DEMOTED T2→T3 v7.1.8. EV/tc=+0.328 overall.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",rmin=4,rmax=8,f2min=4)},
    "FB6_3": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "watch_list":True,"watch_reason":"DEMOTED T2→T3 v7.1.8. EV/tc=+0.323 overall.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",rmin=4,rmax=8,f2min=4)},
    "FB6_4": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "watch_list":True,"watch_reason":"DEMOTED T2→T3 v7.1.8. EV/tc=+0.201 overall.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",spr="Y",f2min=4)},
    "FB6_5": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "watch_list":True,"watch_reason":"DEMOTED T2→T3 v7.1.8. EV/tc=+0.202 overall.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",f2min=4)},
    "FB6_6": {
        "t":"T3","tc_mode":"full_box_superf","unit_price":UNIT_SUPERF,
        "_old_tc":720,"hf":"_h_al","pc":"Superf_paid","mn":6,"mx":6,"ew":1.5,"v":1,
        "watch_list":True,"watch_reason":"DEMOTED T2→T3 v7.1.8. EV/tc=+0.149 overall.",
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
        "watch_list":True,"watch_reason":"EV=-46 N=11. Thin and negative. Monitor.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",spr="N",smin=6,smax=8.5,plo=8000,phi=25000)},
    "BNS_SB12_R9": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[6,6,6,7],
        "unit_price":UNIT_SUPERF,"_old_tc":960,
        "hf":"_h_s6667","pc":"Superf_paid","mn":12,"mx":12,"ew":1.5,"v":0,
        "watch_list":True,"watch_reason":"EV=-28 18mo N=23. Inconsistent. Monitor.",
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
        "watch_list":True,"watch_reason":"Gates tightened plo=10000 smin=6. Monitor HR.",
        "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=3,smin=6,smax=9,plo=10000)},
    "BNS_Tri2L1": {
        "t":"BONUS","tc_mode":"pattern_tri","pat":[2,6,2],
        "unit_price":UNIT_TRI,"_old_tc":18,
        "hf":"_h_tri2l1","pc":"Trif_paid","mn":6,"mx":13,"ew":3.0,"v":0,
        "watch_list":True,"watch_reason":"QUARANTINE: TC drift -11.1%. HR=4% N=3139.",
        "ef":_mk_env(c="N",smin=9,f2min=4)},
    "BNS_TriA22": {
        "t":"BONUS","tc_mode":"all_over_top2_top2_tri",
        "unit_price":UNIT_TRI,"_old_tc":None,
        "hf":"_h_tria22","pc":"Trif_paid","mn":5,"mx":13,"ew":2.5,"v":0,
        "watch_list":True,
        "watch_reason":"CORRECTED v7.1.10: ANY winner/Top2/Top2. TC=(n-2)*2. "
                       "_old_tc=None — revalidate from scratch.",
        "ef":_mk_env(c="N",smin=7,f2min=3)},

    # ── SNIPER DISCOVERIES ────────────────────────────────────────────────────
    "BNS_Supr2555_Sniper": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[2,5,5,5],
        "unit_price":UNIT_SUPERF,"_old_tc":96,
        "hf":"_h_s2555","pc":"Superf_paid","mn":5,"mx":7,"ew":1.5,"v":0,
        "watch_list":True,
        "watch_reason":"SNIPER: N=141 AvgP=30.06 Smth=0.212. Validate N>=200.",
        "ef":_mk_env(fmin=5,fmax=7,c="Y",smin=4.0,smax=7.5,f2max=2.0)},
    "BNS_Sup2444_High": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[2,4,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":24,
        "hf":"_h_s2444","pc":"Superf_paid","mn":4,"mx":6,"ew":1.0,"v":0,
        "watch_list":True,
        "watch_reason":"SNIPER: N=73 AvgP=29.68 Smth=0.174. SI_HIGH gate.",
        "ef":_mk_env(fmin=4,fmax=6,smin=7.5)},
    "BNS_Supr4444_High": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[4,4,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s4444","pc":"Superf_paid","mn":4,"mx":6,"ew":1.0,"v":0,
        "watch_list":True,
        "watch_reason":"SNIPER: N=73 AvgP=26.95 Smth=0.143. SI_HIGH gate.",
        "ef":_mk_env(fmin=4,fmax=6,smin=7.5)},
    "BNS_Supr1445_R7_Mid": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[1,4,4,5],
        "unit_price":UNIT_SUPERF,"_old_tc":24,
        "hf":"_h_s1445","pc":"Superf_paid","mn":7,"mx":7,"ew":1.5,"v":0,
        "watch_list":True,
        "watch_reason":"SNIPER: N=94 AvgP=20.07 Smth=0.217 HIGHEST. SI_MID R7.",
        "ef":_mk_env(fmin=7,fmax=7,c="Y",smin=4.0,smax=7.5,f2max=2.0)},
    "BNS_Sup1234_Sniper": {
        "t":"BONUS","tc_mode":"straight_superf",
        "unit_price":UNIT_SUPERF,"_old_tc":2,"tc":2.0,
        "hf":"_h_s1234","pc":"Superf_paid","mn":7,"mx":7,"ew":1.0,"v":0,
        "watch_list":True,
        "watch_reason":"SNIPER: N=94 AvgP=11.01 Smth=0.184. Straight 1-2-3-4. "
                       "tc=$2 at $2 base. SI_MID R7 chalk gate.",
        "ef":_mk_env(fmin=7,fmax=7,c="Y",smin=4.0,smax=7.5,f2max=2.0)},

    # ── HISTORICAL CONSENSUS ROUTES & BEST BETS ───────────────────────────────
    "BNS_Sup3444_R4": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[3,4,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":36,
        "hf":"_h_s3444","pc":"Superf_paid","mn":4,"mx":4,"ew":1.5,"v":0,
        "watch_list":True,"watch_reason":"ROUTE-01 Primary. 45.1% hit rate in R4.",
        "ef":_mk_env(fmin=4,fmax=4,c="N",smin=3.0,smax=7.0)},
    "BNS_Supr4444_R4": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[4,4,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s4444","pc":"Superf_paid","mn":4,"mx":4,"ew":1.5,"v":0,
        "watch_list":True,"watch_reason":"ROUTE-01 Secondary. 41.2% hit rate in R4.",
        "ef":_mk_env(fmin=4,fmax=4,c="N",smin=3.0,smax=7.0)},
    "BNS_Sup2266_Large": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[2,2,6,6],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s2266","pc":"Superf_paid","mn":10,"mx":12,"ew":2.0,"v":0,
        "watch_list":True,"watch_reason":"ROUTE-04. Highest absolute payout pocket.",
        "ef":_mk_env(fmin=10,fmax=12,c="N",smin=5.0,smax=7.0)},
    "BNS_Supr1555_Large": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[1,5,5,5],
        "unit_price":UNIT_SUPERF,"_old_tc":48,
        "hf":"_h_s1555","pc":"Superf_paid","mn":10,"mx":12,"ew":2.0,"v":0,
        "watch_list":True,"watch_reason":"ROUTE-04. Strong large-field player.",
        "ef":_mk_env(fmin=10,fmax=12,c="N",smin=5.0,smax=7.0)},
    "BNS_Supr4144_R8": {
        "t":"BONUS","tc_mode":"pattern_superf","pat":[4,1,4,4],
        "unit_price":UNIT_SUPERF,"_old_tc":16,
        "hf":"_h_s4144","pc":"Superf_paid","mn":8,"mx":8,"ew":1.5,"v":0,
        "watch_list":True,"watch_reason":"ROUTE-05. GUM Tier 1 validated anchor.",
        "ef":_mk_env(fmin=8,fmax=8,c="N",smin=2.0,smax=7.0,phi=18000)},
    "BNS_Tri123_Large": {
        "t":"BONUS","tc_mode":"straight_superf",
        "unit_price":UNIT_TRI,"_old_tc":2,"tc":2.0,
        "hf":"_h_t123","pc":"Trif_paid","mn":10,"mx":12,"ew":1.0,"v":0,
        "watch_list":True,"watch_reason":"BestBets: NoChalk x 10-12 runners.",
        "ef":_mk_env(fmin=10,fmax=12,c="N",smin=2.0,smax=7.0)},
    "BNS_Sup1234_Large": {
        "t":"BONUS","tc_mode":"straight_superf",
        "unit_price":UNIT_SUPERF,"_old_tc":2,"tc":2.0,
        "hf":"_h_s1234","pc":"Superf_paid","mn":10,"mx":12,"ew":1.0,"v":0,
        "watch_list":True,"watch_reason":"BestBets: Top JK2 priority for large fields.",
        "ef":_mk_env(fmin=10,fmax=12,c="N",smin=2.0,smax=7.0)},

    # ── APEX — inactive, audit visibility only ────────────────────────────────
    "APX_Supr3666_Wide": {
        "t":"APEX","tc_mode":"pattern_superf","pat":[3,6,6,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s3666","pc":"Superf_paid","mn":5,"mx":12,"ew":1.5,"v":1,
        "watch_list":True,
        "watch_reason":"INACTIVE: EV=-0.319/tc N=50228. Gates too wide. "
                       "Sniper version: BNS_Supr2555_Sniper.",
        "ef":_mk_env(fmin=5,fmax=12,plo=1000,phi=32500)},
    "APX_Supr3666": {
        "t":"APEX","tc_mode":"pattern_superf","pat":[3,6,6,6],
        "unit_price":UNIT_SUPERF,"_old_tc":360,
        "hf":"_h_s3666","pc":"Superf_paid","mn":6,"mx":12,"ew":1.5,"v":1,
        "watch_list":True,
        "watch_reason":"INACTIVE: EV=-0.313/tc N=46238. Gates too wide.",
        "ef":_mk_env(fmin=6,fmax=12,plo=500,phi=34500)},
    "SB12_S6678_A": {
        "t":"APEX","tc_mode":"pattern_superf","pat":[6,6,7,8],
        "unit_price":UNIT_SUPERF,"_old_tc":1500,
        "hf":"_h_s6678","pc":"Superf_paid","mn":12,"mx":12,"ew":1.5,"v":1,
        "watch_list":True,
        "watch_reason":"INACTIVE: Both windows negative EV. tc=$1500 ruin machine.",
        "ef":_mk_env(fmin=12,fmax=12,c="N")},
    "APX_Sup4456": {
        "t":"APEX","tc_mode":"pattern_superf","pat":[4,4,5,6],
        "unit_price":UNIT_SUPERF,"_old_tc":216,
        "hf":"_h_s4456","pc":"Superf_paid","mn":8,"mx":12,"ew":1.5,"q":1,
        "watch_list":True,
        "watch_reason":"score=0.410. Promote T3 if score>=0.45 OOS confirmed.",
        "ef":_mk_env(fmin=8,fmax=12,plo=500,phi=34500)},
    "APX_Sup1234": {
        "t":"APEX","tc_mode":"straight_superf",
        "unit_price":UNIT_SUPERF,"_old_tc":2,"tc":2.0,
        "hf":"_h_s1234","pc":"Superf_paid","mn":4,"mx":13,"ew":1.5,"v":0,
        "watch_list":True,
        "watch_reason":"RETIRED: HR=1% EV=0 N=16627. Wide gate noise. "
                       "Sniper version: BNS_Sup1234_Sniper.",
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

# Add test compatibility keys
def get_tc(inst):
    mode = inst.get("tc_mode", "fixed")
    if mode == "fixed": return inst.get("tc", inst.get("_old_tc", 0))
    if mode == "full_box_superf":
        n = inst["mn"]
        return math.perm(n, 4) * (inst["unit_price"] / 10.0) # $2 unit / 10 = $0.20 base
    if mode == "pattern_superf":
        pat = inst["pat"]
        return math.prod(pat) * (inst["unit_price"] / 10.0)
    if mode == "pattern_tri":
        pat = inst["pat"]
        return math.prod(pat) * (inst["unit_price"] / 2.0) # $2 unit / 2 = $1 base
    if mode == "straight_superf":
        return inst.get("tc", 2.0)
    if mode == "all_over_top2_top2_tri":
        n = inst["mn"]
        return (n - 2) * 2 * (inst["unit_price"] / 2.0)
    return inst.get("_old_tc", 0)

for k in list(INS.keys()):
    INS[k]["tier"] = INS[k]["t"]
    INS[k]["tc"] = get_tc(INS[k])
    INS[k]["ticket_cost"] = INS[k]["tc"]
    INS[k]["hit_func"] = INS[k]["hf"]
    INS[k]["payout_col"] = INS[k]["pc"]

    # Default pm to 0.05 (dime) for superf, 1.0 for tri unless specified
    if "pm" not in INS[k]:
        if "Superf" in INS[k]["pc"]: INS[k]["pm"] = 0.05
        else: INS[k]["pm"] = 1.0

    INS[k]["payout_mult"] = INS[k]["pm"]
    INS[k]["min_runners"] = INS[k]["mn"]
    INS[k]["ewpd"] = INS[k]["ew"]
    INS[k]["env_filter"] = INS[k]["ef"]

# ══════════════════════════════════════════════════════════════════════════════
# STRESS PRESETS
# ══════════════════════════════════════════════════════════════════════════════
STRESS_PRESETS = {
    "VANILLA":   {"lbl":"VANILLA",   "emo":"🌤",
                  "f":0.00,"s":0.00,"mi":False,"mt":2000,"ms":0.00004,
                  "ti":False,"td":0.3,"tdu":20,"c":0.03,"fm":0.08},
    "HALF":      {"lbl":"HALF",      "emo":"🌦",
                  "f":0.01,"s":0.05,"mi":False,"mt":2000,"ms":0.00004,
                  "ti":False,"td":0.3,"tdu":20,"c":0.05,"fm":0.12},
    "FULL":      {"lbl":"FULL",      "emo":"⛈",
                  "f":0.01,"s":0.07,"mi":True, "mt":2000,"ms":0.00004,
                  "ti":True, "td":0.3,"tdu":20,"c":0.05,"fm":0.12},
    "NIGHTMARE": {"lbl":"NIGHTMARE", "emo":"💀",
                  "f":0.02,"s":0.12,"mi":True, "mt":1000,"ms":0.0008,
                  "ti":True, "td":0.2,"tdu":30,"c":0.08,"fm":0.15},
}

STRESS_PAIRS = {
    "VANILLA":   ("VANILLA",   "HALF"),
    "HALF":      ("HALF",      "FULL"),
    "FULL":      ("FULL",      "NIGHTMARE"),
    "NIGHTMARE": ("NIGHTMARE", "NIGHTMARE"),
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _get_df(f1, f2, cut):
    def _read(f):
        try: return pd.read_csv(f, low_memory=False)
        except Exception: return pd.DataFrame()
    frames = [fr for fr in [_read(f1), _read(f2)] if not fr.empty]
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    dc = next((c for c in df if "Date" in c or "date" in c), None)
    if dc:
        df = df[pd.to_datetime(df[dc], errors="coerce").dt.date >= pd.Timestamp(cut).date()]
    for c in ["Runners","Purse","WhichRace","SumOf1st2Odds","Superf_paid","Trif_paid","Exacta_paid","Superfecta_paid","Trifecta_paid"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "Superfecta_paid" in df and "Superf_paid" not in df: df["Superf_paid"] = df["Superfecta_paid"]
    if "Trifecta_paid" in df and "Trif_paid" not in df: df["Trif_paid"] = df["Trifecta_paid"]
    return df

# ══════════════════════════════════════════════════════════════════════════════
# PnL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def _x_pnl(df, k, i, ft):
    if df.empty: return np.array([], dtype=np.float64)
    m = df["Runners"].between(i["mn"], i.get("mx",999))
    if "WhichRace" in df: m &= df["WhichRace"].between(1, 11)
    ef = i["ef"]; m &= ef(df)
    if not m.any(): return np.array([], dtype=np.float64)
    q = np.where(m)[0]; tc = i["tc"]; pc = i["pc"]
    if pc not in df: return np.full(len(q), -tc, dtype=np.float64)
    pf = df[pc].values[q].astype(float) * (BASE_BET_FACTOR / 2.0)
    v = ~(np.isnan(pf) | (pf == 0))
    rng = np.random.default_rng(SEED + abs(hash(k)) % 1_000_000)
    j = rng.uniform(1 - JITTER, 1 + JITTER, len(q))

    hf_key = i["hf"]
    if hf_key == "_hit_always_true":
        pnl = np.where(v, pf * j - tc, -tc).astype(np.float64)
    elif (fn := _VH.get(hf_key)) and "RANKED_RESULTS" in df:
        try:
            s = df["RANKED_RESULTS"].iloc[q].str.split(expand=True).iloc[:, :4]
            while s.shape[1] < 4: s[s.shape[1]] = np.nan
            p = [pd.to_numeric(s.iloc[:,x], errors="coerce").values for x in range(4)]
            h = fn(*p) & ~(np.isnan(p[0])|np.isnan(p[1])|np.isnan(p[2])|np.isnan(p[3]))
            pnl = np.where(v & h, pf * j - tc, -tc).astype(np.float64)
        except Exception: pnl = np.full(len(q), -tc, dtype=np.float64)
    else: pnl = np.full(len(q), -tc, dtype=np.float64)
    return pnl

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA KERNEL
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True)
def run_gauntlet_core(npth, mxr, sbr, tgt, rfl, mb, c, fm, ls, m_on, mt, ms, t_on, t_dd, t_du, p_pls, p_sts, t_tbl, pb):
    ap  = np.zeros((npth, mxr+1), dtype=np.float64)
    aph = np.zeros((npth, mxr+1), dtype=np.int8)
    oc  = np.zeros(npth, dtype=np.int8)
    ra  = np.zeros(npth, dtype=np.int32)
    n_in, n_ph = len(p_pls), len(pb)
    for p in prange(npth):
        br, pk, tr = sbr, sbr, 0; ap[p,0] = br
        cph = n_ph - 1
        for i in range(n_ph):
            if pb[i,0] <= br < pb[i,1]: cph = int(pb[i,2]); break
        aph[p,0] = cph
        tx = np.zeros(6, dtype=np.float64)
        for r in range(1, mxr+1):
            for t in range(6): tx[t] = 0.0
            for i in range(len(t_tbl)):
                if t_tbl[i,0] <= br < t_tbl[i,1]:
                    for t in range(6): tx[t] = t_tbl[i,2+t]
                    break
            tot = 0.0
            for i in range(n_in):
                if tx[int(p_sts[i,4])] > 0: tot += p_sts[i,3]
            if tot == 0.0: ap[p,r], aph[p,r] = br, cph; continue
            pv = np.random.random() * tot; cs = 0.0; idx = 0; le = 0
            for i in range(n_in):
                if tx[int(p_sts[i,4])] > 0:
                    le = i; cs += p_sts[i,3]
                    if pv <= cs: idx = i; break
            else: idx = le
            tc = p_sts[idx,2]; nt = min(int(tx[int(p_sts[idx,4])]), int(mb/tc))
            if nt <= 0: ap[p,r], aph[p,r] = br, cph; continue
            st = min(float(nt)*tc, br)
            if br > pk: pk, tr = br, 0
            if t_on > 0.5 and pk > 0.0 and (pk-br)/pk >= t_dd:
                if tr <= 0: tr = int(t_du)
            if tr > 0: st = min(float(min(nt+1, int(mb/tc)))*tc, br); tr -= 1
            pool = p_pls[idx]; pnl = pool[np.random.randint(0, len(pool))]
            if (np.random.random()<c or np.random.random()<fm or np.random.random()<ls): pnl = -tc
            elif pnl>0 and m_on>0.5 and st>mt: pnl = (pnl+tc)*(1.0-min(0.5,(st-mt)*ms))-tc
            br += pnl*(st/tc); ap[p,r] = br
            cph = n_ph-1
            for i in range(n_ph):
                if pb[i,0] <= br < pb[i,1]: cph = int(pb[i,2]); break
            aph[p,r] = cph
            if br >= tgt or br <= rfl:
                oc[p] = 1 if br>=tgt else 2; ra[p] = r
                for x in range(r+1, mxr + 1): ap[p,x], aph[p,x] = br, cph
                break
        else: ra[p] = mxr
    return ap, aph, oc, ra

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT SNAPSHOT  (stored in mega-JSON per run)
# ══════════════════════════════════════════════════════════════════════════════
def _sax(a, t=None, x=None, y=None):
    a.set_facecolor(BG); a.tick_params(colors=MUT, labelsize=8)
    for sp in a.spines.values(): sp.set_color(GRD)
    a.grid(color=GRD, ls=":", alpha=0.4)
    if t: a.set_title(t, color=TXT, fontsize=10, fontweight="bold", pad=10)
    if x: a.set_xlabel(x, color=MUT, fontsize=8)
    if y: a.set_ylabel(y, color=MUT, fontsize=8)

def _save_charts(paths, outcomes, fbr, st_o, key):
    sub = f"{st_o['strategy']} | {st_o['stress']} | {st_o['window']}"
    fig, ax = plt.subplots(figsize=(13,6), facecolor=BG)
    _sax(ax, f"GAUNTLET v7.5.2 | ${S_BR:,.0f}->${TGT:,.0f} | {sub}", "Races", "Bankroll ($)")
    ax.set_yscale("symlog", linthresh=1000)
    rng = np.random.default_rng(SEED)
    for i in rng.choice(len(paths), min(300,len(paths)), replace=False):
        c = ACC if outcomes[i]=="success" else RED if outcomes[i]=="ruin" else MUT
        ax.plot(paths[i], color=c, alpha=0.1, lw=0.5)
    med = paths[st_o["midx"]]
    ax.plot(med, color=GLD, lw=2, label=f"Median (${med[-1]:,.0f})")
    plt.savefig(f"Gauntlet_Paths_{key}.png", dpi=120, facecolor=BG); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# RUNNERS
# ══════════════════════════════════════════════════════════════════════════════
def _prep(pmap):
    tp, sl = List(), []; tm = {"T1A":0,"T1B":0,"T1":0,"T2":1,"T3":2,"T4":3,"BONUS":4,"APEX":5}
    for k, v in pmap.items():
        inst = INS[k]; tc = inst["tc"]; tp.append(v.astype(np.float64))
        ew = max(float(inst.get("ew", 0.1)), 0.01)
        sl.append([float((v>-tc).mean()), float((v[v>-tc]+tc).mean()) if (v>-tc).any() else 0.0, tc, ew, float(tm.get(inst.get("t","T1"),0))])
    tkt_tbl = np.array([[0,2000,1,0,0,0,0,0],[2000,5000,1,1,0,0,1,0],[5000,10000,2,2,1,0,1,0],[10000,20000,3,3,1,0,1,0],[20000,40000,3,3,2,0,2,1],[40000,80000,5,5,3,0,3,1],[80000,120000,8,8,5,0,5,2],[120000,999999,10,10,0,0,0,0]], dtype=np.float64)
    pb = np.array([[0,5000,0],[5000,20000,1],[20000,999999,2]], dtype=np.float64)
    return tp, np.array(sl, dtype=np.float64), tkt_tbl, pb

def _stat(paths, outcomes, races, fbr, slbl, so, window, mode):
    n = len(paths); ns = outcomes.count("success"); nr = outcomes.count("ruin")
    mb = float(np.median(fbr)); mi = int(np.argmin(np.abs(np.array(fbr)-mb)))
    return {"window":window,"strategy":slbl,"stress":so["lbl"],"mode":mode,"n":n,"ns":ns,"nr":nr,"sr":round(ns/n*100,2),"rr":round(nr/n*100,2),"mbr":round(mb,2),"midx":mi}

def main():
    _safe_print("="*72); _safe_print("  THE 100x GAUNTLET v7.5.2 OMNI RESTORE"); _safe_print("="*72)
    tdy = datetime.date.today(); cuts = {"18mo":(tdy-datetime.timedelta(days=548)).strftime("%Y-%m-%d"), "36mo":(tdy-datetime.timedelta(days=1096)).strftime("%Y-%m-%d")}

    ql = (input("\n Quick Launch? [1] Yes (36mo FULL HALF Gauntlet $2)  [2] Customize\n Choice [1]: ").strip() or "1")
    if ql == "1":
        window, slbl, stress_choice = "36mo", "FULL", "HALF"
    else:
        window = "18mo" if (input("\n Window: [1] 18mo  [2] 36mo [2]: ").strip() or "2")=="1" else "36mo"
        slbl = {"1":"SAFEST","2":"STANDARD","3":"FULL"}.get(input("\n Strategy: [1] SAFEST [2] STANDARD [3] FULL [3]: ").strip() or "3", "FULL")
        stress_choice = {"1":"VANILLA","2":"HALF","3":"FULL","4":"NIGHTMARE"}.get(input("\n Stress: [1] VANILLA [2] HALF [3] FULL [4] NIGHTMARE [2]: ").strip() or "2", "HALF")

    df = _get_df(TRAIN_FILE, TEST_FILE, cuts[window])
    s1, s2 = STRESS_PAIRS[stress_choice]
    for s in [s1, s2]:
        so = STRESS_PRESETS[s]
        pmap = {k:pnl for k in sorted(STRATEGY_SETS[slbl]) if len(pnl := _x_pnl(df, k, INS[k], so["f"])) > 0}
        if not pmap: print(f" ❌ No instruments passed for {s}"); continue
        tp, sl, tt, pb = _prep(pmap)
        ap, aph, oc, ra = run_gauntlet_core(N_PTH, M_RCS, S_BR, TGT, R_FLR, MAX_BET, float(so["c"]),float(so["fm"]),float(so["s"]), 0.0, float(so["mt"]), float(so["ms"]), 0.0, float(so["td"]), float(so["tdu"]), tp, sl, tt, pb)
        oc_ = ["timeout" if c==0 else "success" if c==1 else "ruin" for c in oc]
        fbr_ = [float(ap[i,ra[i]]) for i in range(N_PTH)]
        paths_ = [ap[i,:ra[i]+1] for i in range(N_PTH)]
        st = _stat(paths_, oc_, ra.tolist(), fbr_, slbl, so, window, "gauntlet")
        _save_charts(paths_, oc_, fbr_, st, f"{window}_{slbl}_{s}")
        print(f" ✅ {s}: SR={st['sr']}% MedianBR=${st['mbr']:,.0f}")

if __name__ == "__main__": main()
