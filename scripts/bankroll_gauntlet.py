#!/usr/bin/env python3
"""
THE 100x GAUNTLET — v6.6.0 DIAMOND APEX REFINED
BATCH RUNNER EDITION
=================================================
New in Batch Runner:
  - One-time data load + PnL extraction for all windows
  - Automated loop over all Window × Strategy × Stress × Mode combos
  - Mega-JSON output: Gauntlet_MegaResults.json
  - Cross-run leaderboard CSV: Gauntlet_Leaderboard.csv
  - Ladder results embedded in mega-JSON
  - Zero manual prompts during batch run
  - Progress bar across all runs
  - Summary leaderboard printed to console at end
  - Per-run charts saved as: Gauntlet_Paths_{key}.png etc.
  - APEX confirmation gate auto-skipped in batch (safety)

v6.5.1 → v6.6.0 changes also retained:
  1.  APX_Supr3666 promoted APEX→T3, v=1
  2.  APX_Supr2555 promoted APEX→T3, v=1
  3.  FB7_3 promoted to v=1 validated
  4.  HARD DELETE: BNS_Sup3335/BNS_Sup2266/BNS_NM_4455_Fav3/BNS_SB10_6667
  5.  BNS_SB12_R9 demoted→BONUS
  6.  BNS_FB5_Route gate tightened
  7.  SB12_S6667_A watch_list=True
  8.  Early-ruin annotation in charts + stats
  9.  Tier-aware prune thresholds
  10. All v6.5.1 fixes retained
"""
import warnings, datetime, json, os, sys, time, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from numba import njit, prange
from numba.typed import List
from tqdm import tqdm

if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_FILE, TEST_FILE = "RaceRecords_Output_2025.csv", "RaceRecords_Output_2026.csv"
S_BR, R_FLR, TGT, MAX_BET = 250.0, 50.0, 88888.0, 100.0
N_PTH, M_RCS, RPD, SEED   = 2000, 30000, 25, 42
JITTER, HR_MIN, WOW_ABS, WOW_RAT = 0.05, 2, 25000.0, 200.0

# Ladder config
LADDER_N_PTH = 500
LADDER_MXR   = 50000
LDR_RNGS     = [200, 2000, 20000]

SSN_SWP, SSN_W, SSN_N = False, 365, 500
SSN_STS = ["01-01","02-01","03-01","04-01","05-01","06-01",
           "07-01","08-01","09-01","10-01","11-01","12-01"]

PRUNE_EV_MIN     = 0.0
PRUNE_N_MIN      = 20
PRUNE_HR_MIN_PCT = 5.0

# ══════════════════════════════════════════════════════════════════════════════
# BATCH CONFIGURATION — edit these to control what runs
# ══════════════════════════════════════════════════════════════════════════════
BATCH_WINDOWS    = ["18mo", "36mo"]
BATCH_STRATEGIES = ["SAFEST", "STANDARD", "FULL", "TURBO"]
# APEX excluded from batch by default (quarantine)
BATCH_STRESSES   = ["VANILLA", "HALF", "FULL", "NIGHTMARE"]
BATCH_MODES      = ["gauntlet", "ladder"]

# Charts: True = save PNG for every run (slow, lots of files)
#         False = only save charts for runs matching CHART_FILTER
BATCH_SAVE_CHARTS = True
CHART_FILTER = {
    # Only save charts for these combos when BATCH_SAVE_CHARTS=False
    # Format: ("36mo", "FULL", "HALF", "gauntlet")
    ("36mo", "FULL",   "HALF",      "gauntlet"),
    ("18mo", "TURBO",  "HALF",      "gauntlet"),
    ("36mo", "TURBO",  "VANILLA",   "ladder"),
}

# Output filenames
MEGA_JSON_FILE  = "Gauntlet_MegaResults.json"
LEADERBOARD_CSV = "Gauntlet_Leaderboard.csv"

def _prune_score_min(tier):
    return {"T1":0.25,"T2":0.35,"T3":0.35,"BONUS":0.40,"APEX":0.45}.get(tier,0.35)

# ══════════════════════════════════════════════════════════════════════════════
# STRESS PRESETS
# ══════════════════════════════════════════════════════════════════════════════
STRESS_PRESETS = {
    "VANILLA":   {"lbl":"VANILLA",   "emo":"🌤",
                  "f":0.00,"s":0.00,"mi":False,"mt":200,"ms":0.0004,
                  "ti":False,"td":0.3,"tdu":20,"c":0.03,"fm":0.08},
    "HALF":      {"lbl":"HALF",      "emo":"🌦",
                  "f":0.01,"s":0.05,"mi":False,"mt":200,"ms":0.0004,
                  "ti":False,"td":0.3,"tdu":20,"c":0.05,"fm":0.12},
    "FULL":      {"lbl":"FULL",      "emo":"⛈",
                  "f":0.01,"s":0.07,"mi":True, "mt":200,"ms":0.0004,
                  "ti":True, "td":0.3,"tdu":20,"c":0.05,"fm":0.12},
    "NIGHTMARE": {"lbl":"NIGHTMARE", "emo":"💀",
                  "f":0.02,"s":0.12,"mi":True, "mt":100,"ms":0.0008,
                  "ti":True, "td":0.2,"tdu":30,"c":0.08,"fm":0.15},
}

# ══════════════════════════════════════════════════════════════════════════════
# TICKET TABLE
# ══════════════════════════════════════════════════════════════════════════════
TKT_TBL = [
    (0,      500,    1,  0,  0, 0, 0, 0),
    (500,    2000,   1,  1,  0, 0, 1, 0),
    (2000,   4000,   2,  2,  1, 0, 1, 0),
    (4000,   8000,   3,  3,  1, 0, 1, 0),
    (8000,   15000,  3,  3,  2, 0, 2, 1),
    (15000,  25000,  5,  5,  3, 0, 3, 1),
    (25000,  40000,  8,  8,  5, 0, 5, 2),
    (40000,  60000, 10, 10,  0, 0, 0, 0),
    (60000, 999999, 10, 10,  0, 0, 0, 0),
]

def _tt_arr():
    return np.array(
        [[float(lo),float(hi),float(t1),float(t2),float(t3),
          float(t4),float(tb),float(ta)]
         for lo,hi,t1,t2,t3,t4,tb,ta in TKT_TBL],
        dtype=np.float64)

PHS = [
    (0,      2000,  "Phase 1\nEconomy",        "#f43f5e"),
    (2000,   8000,  "Phase 2\nBusiness Class", "#f59e0b"),
    (8000,  40001,  "Phase 3\nFirst Class",    "#10b981"),
]

def _pb_arr():
    return np.array(
        [[float(lo),float(hi),float(idx)]
         for idx,(lo,hi,_,_) in enumerate(PHS)],
        dtype=np.float64)

BG,GRD,TXT,MUT,ACC,RED,GLD,CYN = (
    "#0f172a","#1e293b","#f8fafc","#64748b",
    "#10b981","#f43f5e","#f59e0b","#06b6d4")
T_COL = {"T1":GLD,"T2":ACC,"T3":CYN,"T4":MUT,"BONUS":"#f97316","APEX":"#ec4899"}
_TM   = {"T1":0,"T2":1,"T3":2,"T4":3,"BONUS":4,"APEX":5}

def _safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        for src, dst in [
            ("⚔️","[GAUNTLET]"),("🌤","[VANILLA]"),("🌦","[HALF]"),
            ("⛈","[FULL]"),("💀","[NIGHTMARE]"),("🚀","[RUNNING]"),
            ("💎","[TIER3]"),("✅","[OK]"),("🏆","[TIER1]"),("🔥","[BONUS]"),
            ("⚠️","[WARNING]"),("💾","[SAVE]"),("🎨","[RENDER]"),
            ("🔪","[PRUNE]"),("❌","[NO]"),("🔍","[WATCH]"),("📊","[CHART]"),
        ]:
            text = text.replace(src, dst)
        print(text)

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT GATES
# ══════════════════════════════════════════════════════════════════════════════
def _env_vec(df, c=None, plo=0, phi=999999, fmin=3, fmax=13,
             spr=None, fr=None, rmin=1, rmax=11,
             smin=0.0, smax=999.0, f2min=0.0, f2max=999.0):
    m = np.ones(len(df), dtype=bool)
    if "WhichRace"     in df: m &= df["WhichRace"].between(rmin, rmax)
    if "Runners"       in df: m &= df["Runners"].between(fmin, fmax)
    if "Purse"         in df: m &= df["Purse"].between(plo, phi)
    if "SumOf1st2Odds" in df: m &= df["SumOf1st2Odds"].between(smin, smax)
    f2c = None
    for _c in ["Fav2Exact", "Fav2_odds"]:
        if _c in df: f2c = _c; break
    if f2c is not None:
        m &= df[f2c].between(f2min, f2max)
    if c and "ChalkYN" in df:
        ck = df["ChalkYN"].str.strip().str.upper() == "Y"
        m &= ck if c == "Y" else ~ck
    if fr and "FirstRaceYN" in df:
        fk = df["FirstRaceYN"].str.strip().str.upper() == "Y"
        m &= fk if fr == "Y" else ~fk
    if spr:
        dc = None
        for _c in ["Miles", "Distance"]:
            if _c in df: dc = _c; break
        if dc is not None:
            fl = df[dc].where(df[dc] <= 4.0, df[dc] / 8.0)
            sp = (fl < 0.875).fillna(False)
            m &= sp if spr == "Y" else ~sp
    return m

def _mk_env(**kwargs):
    def _apply(df):
        return _env_vec(df, **kwargs)
    _apply.vec = _apply
    return _apply

# ══════════════════════════════════════════════════════════════════════════════
# HIT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
_h_al   = lambda p,n: True
_h_t145 = lambda p,n: p and len(p)>2 and p[0]==1 and p[1] in {2,3,4} and p[2] in {2,3,4,5}
_h_t133 = lambda p,n: p and len(p)>2 and p[0]==1 and max(p[1:3])<=3
_h_t223 = lambda p,n: p and len(p)>2 and max(p[:2])<=2 and p[2]<=3
_h_s3666= lambda p,n: p and len(p)>3 and p[0]<=3 and max(p[1:4])<=6
_h_s4455= lambda p,n: p and len(p)>3 and p[0]<=4 and p[1]<=4 and max(p[2:4])<=5
_h_s4456= lambda p,n: p and len(p)>3 and p[0]<=4 and p[1]<=4 and p[2]<=5 and p[3]<=6
_h_s5556= lambda p,n: p and len(p)>3 and max(p[:3])<=5 and p[3]<=6
_h_s6667= lambda p,n: p and len(p)>3 and max(p[:3])<=6 and p[3]<=7
_h_s6678= lambda p,n: p and len(p)>3 and max(p[:2])<=6 and p[2]<=7 and p[3]<=8
_h_s5567= lambda p,n: p and len(p)>3 and max(p[:2])<=5 and p[2]<=6 and p[3]<=7
_h_s4466= lambda p,n: p and len(p)>3 and max(p[:2])<=4 and max(p[2:4])<=6
_h_s5566= lambda p,n: p and len(p)>3 and max(p[:2])<=5 and max(p[2:4])<=6
_h_s2244= lambda p,n: p and len(p)>3 and max(p[:2])<=2 and max(p[2:4])<=4
_h_s3335= lambda p,n: p and len(p)>3 and max(p[:3])<=3 and p[3]<=5
_h_r4145= lambda p,n: p and len(p)>3 and p[0]<=4 and p[1]==1 and p[2]<=4 and p[3]<=5
_h_s1345= lambda p,n: p and len(p)>3 and p[0]==1 and p[1]<=3 and p[2]<=4 and p[3]<=5
_h_s3444= lambda p,n: p and len(p)>3 and p[0]<=3 and max(p[1:4])<=4
_h_s2255= lambda p,n: p and len(p)>3 and max(p[:2])<=2 and max(p[2:4])<=5
_h_s2345= lambda p,n: p and len(p)>3 and p[0]<=2 and p[1]<=3 and p[2]<=4 and p[3]<=5
_h_s2266= lambda p,n: p and len(p)>3 and max(p[:2])<=2 and max(p[2:4])<=6

_VH = {
    _h_t145: lambda a,b,c,d:(a==1)&(b>=2)&(b<=4)&(c>=2)&(c<=5),
    _h_t133: lambda a,b,c,d:(a==1)&(b<=3)&(c<=3),
    _h_t223: lambda a,b,c,d:(a<=2)&(b<=2)&(c<=3),
    _h_s3666:lambda a,b,c,d:(a<=3)&(b<=6)&(c<=6)&(d<=6),
    _h_s4455:lambda a,b,c,d:(a<=4)&(b<=4)&(c<=5)&(d<=5),
    _h_s4456:lambda a,b,c,d:(a<=4)&(b<=4)&(c<=5)&(d<=6),
    _h_s5556:lambda a,b,c,d:(a<=5)&(b<=5)&(c<=5)&(d<=6),
    _h_s6667:lambda a,b,c,d:(a<=6)&(b<=6)&(c<=6)&(d<=7),
    _h_s6678:lambda a,b,c,d:(a<=6)&(b<=6)&(c<=7)&(d<=8),
    _h_s5567:lambda a,b,c,d:(a<=5)&(b<=5)&(c<=6)&(d<=7),
    _h_s4466:lambda a,b,c,d:(a<=4)&(b<=4)&(c<=6)&(d<=6),
    _h_s5566:lambda a,b,c,d:(a<=5)&(b<=5)&(c<=6)&(d<=6),
    _h_s2244:lambda a,b,c,d:(a<=2)&(b<=2)&(c<=4)&(d<=4),
    _h_s3335:lambda a,b,c,d:(a<=3)&(b<=3)&(c<=3)&(d<=5),
    _h_r4145:lambda a,b,c,d:(a<=4)&(b==1)&(c<=4)&(d<=5),
    _h_s1345:lambda a,b,c,d:(a==1)&(b<=3)&(c<=4)&(d<=5),
    _h_s3444:lambda a,b,c,d:(a<=3)&(b<=4)&(c<=4)&(d<=4),
    _h_s2255:lambda a,b,c,d:(a<=2)&(b<=2)&(c<=5)&(d<=5),
    _h_s2345:lambda a,b,c,d:(a<=2)&(b<=3)&(c<=4)&(d<=5),
    _h_s2266:lambda a,b,c,d:(a<=2)&(b<=2)&(c<=6)&(d<=6),
}

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT REGISTRY  — v6.6.0
# ══════════════════════════════════════════════════════════════════════════════
INS = {
    # ── TIER 1 ───────────────────────────────────────────────────────────────
    "SB6_S5556_A":      {"t":"T1","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":3.0,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",smin=6,smax=8.5,plo=8000,phi=25000)},
    "SB6_S5556_B":      {"t":"T1","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":3.0,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",smin=6,smax=8.5,plo=8000,phi=25000)},
    "FB4_2":            {"t":"T1","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=4,fmax=4,c="N",smin=4,smax=6,f2min=2.5,f2max=4)},
    "FB4_3":            {"t":"T1","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=4,fmax=4,c="N",fr="N",smin=4,smax=6)},
    "FB4_4":            {"t":"T1","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=4,fmax=4,c="N",smin=4,smax=6)},
    "FB4_5":            {"t":"T1","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=4,fmax=4,c="N",spr="Y",smin=4,smax=6)},
    "NM_Sup4455_N6_C":  {"t":"T1","tc":7.2, "hf":_h_s4455,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4)},
    "FB5_2":            {"t":"T1","tc":12,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":5, "mx":5, "ew":2.5,"v":1,
                         "ef":_mk_env(fmin=5,fmax=5,c="N",f2min=4)},
    # ── TIER 2 ───────────────────────────────────────────────────────────────
    "TRI145":           {"t":"T2","tc":18,  "hf":_h_t145, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":13,"ew":2.8,"v":1,
                         "ef":_mk_env(smin=9,f2min=2.5)},
    "SB6_S5556_C":      {"t":"T2","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.5,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",spr="Y",smin=6,smax=8.5,plo=8000,phi=35000)},
    "NM_Sup4455_N6_D":  {"t":"T2","tc":7.2, "hf":_h_s4455,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4,smin=6,smax=9.5)},
    "NM_Sup4456_N6_D":  {"t":"T2","tc":10.8,"hf":_h_s4456,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4,smin=6,smax=9.5)},
    "NM_Supr3666_N6_D": {"t":"T2","tc":18,  "hf":_h_s3666,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4,smin=6,smax=9.5)},
    "FB6_1":            {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",spr="Y",f2min=4)},
    "FB6_2":            {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",rmin=4,rmax=8,f2min=4)},
    "FB6_3":            {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",rmin=4,rmax=8,f2min=4)},
    "FB6_4":            {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",spr="Y",f2min=4)},
    "FB6_5":            {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",fr="N",f2min=4)},
    "FB6_6":            {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4)},
    "SB12_S6678_A":     {"t":"T2","tc":48,  "hf":_h_s6678,"pc":"Superf_paid","pm":0.05,"mn":12,"mx":12,"ew":1.5,"v":1,
                         "ef":_mk_env(fmin=12,fmax=12,c="N")},
    # ── TIER 3 ───────────────────────────────────────────────────────────────
    "FB4_6":            {"t":"T3","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.0,"v":1,
                         "ef":_mk_env(fmin=4,fmax=4,rmin=4,rmax=8,smin=4,smax=6,f2min=2.5,f2max=4)},
    "FB7_1":            {"t":"T3","tc":84,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.0,"v":1,
                         "ef":_mk_env(fmin=7,fmax=7,c="N",fr="N",spr="Y",plo=8000,phi=15000,smin=6,smax=8)},
    "FB7_2":            {"t":"T3","tc":84,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.0,"v":1,
                         "ef":_mk_env(fmin=7,fmax=7,c="N",plo=8000,phi=15000,smin=6,smax=8)},
    "FB7_3":            {"t":"T3","tc":84,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.5,"v":1,
                         "ef":_mk_env(fmin=7,fmax=7,c="N",spr="Y",plo=8000,phi=20000,f2min=4,smin=6)},
    "FB7_4":            {"t":"T3","tc":84,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.0,"v":1,
                         "ef":_mk_env(fmin=7,fmax=7,c="N",smin=6,smax=8,f2min=4)},
    "SB12_S6667_A":     {"t":"T3","tc":48,  "hf":_h_s6667,"pc":"Superf_paid","pm":0.05,"mn":12,"mx":12,"ew":1.5,"v":1,
                         "watch_list":True,
                         "watch_reason":"EV=-0.74 in 18mo window. Monitor for regression. "
                                        "Do not prune on 18mo alone. Re-evaluate at N>=50 OOS.",
                         "ef":_mk_env(fmin=12,fmax=12,c="N",spr="N",rmin=4,rmax=9,
                                      smin=4,smax=8.5,f2min=4,plo=8000,phi=25000)},
    "SB6_S5556_D":      {"t":"T3","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.5,"v":1,
                         "watch_list":True,
                         "watch_reason":"Gate review: phi=25000 vs SB6_S5556_C phi=35000. "
                                        "Promote T3→T2 if N>=60 OOS confirmed.",
                         "ef":_mk_env(fmin=6,fmax=6,c="N",spr="Y",smin=6,smax=8.5,plo=8000,phi=25000)},
    # Promoted APEX→T3 v6.6.0
    "APX_Supr3666":     {"t":"T3","tc":4,   "hf":_h_s3666,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":12,"ew":1.5,"v":1,
                         "ef":_mk_env(fmin=6,fmax=12,plo=500,phi=34500)},
    "APX_Supr2555":     {"t":"T3","tc":4,   "hf":_h_s3666,"pc":"Superf_paid","pm":0.05,"mn":5, "mx":12,"ew":1.5,"v":1,
                         "ef":_mk_env(fmin=5,fmax=12,plo=1000,phi=32500)},
    # Promoted BONUS→T3
    "BNS_NM_5567":      {"t":"T3","tc":12,  "hf":_h_s5567,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":0,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4)},
    "BNS_NM_4466":      {"t":"T3","tc":9.6, "hf":_h_s4466,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":0,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4)},
    "BNS_TRI_First":    {"t":"T3","tc":18,  "hf":_h_t145, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":13,"ew":2.0,"v":0,
                         "ef":_mk_env(fr="Y",smin=9,f2min=2.5)},
    "BNS_TRI_HighPurse":{"t":"T3","tc":18,  "hf":_h_t145, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":13,"ew":2.0,"v":0,
                         "ef":_mk_env(smin=9,f2min=2.5,plo=25000)},
    "BNS_SB6_SumWide":  {"t":"T3","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":3.0,"v":0,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",smin=6,smax=9.5,plo=8000,phi=25000)},
    "BNS_SB6_A_P35":    {"t":"T3","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":3.0,"v":0,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",smin=6,smax=8.5,plo=8000,phi=35000)},
    "BNS_TRI_Sum8":     {"t":"T3","tc":18,  "hf":_h_t145, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":13,"ew":3.0,"v":0,
                         "ef":_mk_env(smin=8,f2min=2.5)},
    "BNS_FB5_Fav3":     {"t":"T3","tc":12,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":5, "mx":5, "ew":2.0,"v":0,
                         "ef":_mk_env(fmin=5,fmax=5,c="N",f2min=3)},
    "BNS_FB6_HighPurse":{"t":"T3","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":0,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",f2min=4,plo=25000)},
    # ── BONUS ────────────────────────────────────────────────────────────────
    "BNS_SB6_C_P35":    {"t":"BONUS","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.5,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",spr="Y",smin=6,smax=8.5,plo=8000,phi=35000)},
    "BNS_SB6_RouteOnly":{"t":"BONUS","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.5,
                         "watch_list":True,
                         "watch_reason":"EV=-1.26 in 18mo (N=10 very thin). Monitor or tighten smin.",
                         "ef":_mk_env(fmin=6,fmax=6,c="N",spr="N",smin=6,smax=8.5,plo=8000,phi=25000)},
    "BNS_SB12_R9":      {"t":"BONUS","tc":48,  "hf":_h_s6667,"pc":"Superf_paid","pm":0.05,"mn":12,"mx":12,"ew":1.5,
                         "watch_list":True,
                         "watch_reason":"Inconsistent EV: -1.33 18mo vs +15 36mo. Demoted T3→BONUS.",
                         "ef":_mk_env(fmin=12,fmax=12,c="N",spr="N",rmin=4,rmax=9,
                                      smin=4,smax=8.5,f2min=4,plo=8000,phi=25000)},
    "BNS_FB5_Route":    {"t":"BONUS","tc":12,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":5, "mx":5, "ew":1.5,
                         "watch_list":True,
                         "watch_reason":"N=10 thin. rmin=3,rmax=9 gate added. Promote if N>=30 OOS EV>0.",
                         "ef":_mk_env(fmin=5,fmax=5,c="N",spr="N",rmin=3,rmax=9,f2min=4)},
    "BNS_FB6_First":    {"t":"BONUS","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,
                         "ef":_mk_env(fmin=6,fmax=6,c="N",fr="Y",f2min=4)},
    # ── APEX (one survivor) ───────────────────────────────────────────────────
    "APX_Sup4456":      {"t":"APEX","tc":4,   "hf":_h_s4456,"pc":"Superf_paid","pm":0.05,"mn":8, "mx":12,"ew":1.5,"q":1,
                         "watch_list":True,
                         "watch_reason":"score=0.410 WATCH. EV=+3.39 HR=22.9% N=10,453. "
                                        "Promote T3 if score>=0.45 confirmed OOS.",
                         "ef":_mk_env(fmin=8,fmax=12,plo=500,phi=34500)},
}

# ── Strategy sets ─────────────────────────────────────────────────────────────
_T1  = {"SB6_S5556_A","SB6_S5556_B","FB4_2","FB4_3","FB4_4","FB4_5",
        "NM_Sup4455_N6_C","FB5_2"}
_T2  = {"TRI145","SB6_S5556_C","NM_Sup4455_N6_D","NM_Sup4456_N6_D",
        "NM_Supr3666_N6_D","FB6_1","FB6_2","FB6_3","FB6_4","FB6_5","FB6_6",
        "SB12_S6678_A"}
_T3  = {"FB4_6","FB7_1","FB7_2","FB7_3","FB7_4","SB12_S6667_A","SB6_S5556_D",
        "BNS_NM_5567","BNS_NM_4466","BNS_TRI_First","BNS_TRI_HighPurse",
        "BNS_SB6_SumWide","BNS_SB6_A_P35","BNS_TRI_Sum8","BNS_FB5_Fav3",
        "BNS_FB6_HighPurse","APX_Supr3666","APX_Supr2555"}
M_A  = _T1
M_B  = _T1 | _T2
M_C  = _T1 | _T2 | _T3
BONUS= {k for k,v in INS.items() if v["t"]=="BONUS"}
APX  = {k for k,v in INS.items() if v["t"]=="APEX"}

STRATEGY_SETS = {
    "SAFEST":   M_A,
    "STANDARD": M_B,
    "FULL":     M_C,
    "TURBO":    M_C | BONUS,
    # APEX excluded from batch
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
        df = df[pd.to_datetime(df[dc], errors="coerce").dt.date
                >= pd.Timestamp(cut).date()]
    for c in ["Runners","Purse","WhichRace","SumOf1st2Odds",
              "Superf_paid","Trif_paid","Exacta_paid",
              "Superfecta_paid","Trifecta_paid"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "Superfecta_paid" in df and "Superf_paid" not in df:
        df["Superf_paid"] = df["Superfecta_paid"]
    if "Trifecta_paid" in df and "Trif_paid" not in df:
        df["Trif_paid"] = df["Trifecta_paid"]
    wm = pd.Series(False, index=df.index)
    if "Superf_paid" in df and "Exacta_paid" in df:
        exacta_safe = df["Exacta_paid"].clip(lower=1).astype(float)
        wm |= (df["Superf_paid"] > WOW_ABS) | (
               (df["Exacta_paid"] > 0) &
               (df["Superf_paid"].astype(float) / exacta_safe > WOW_RAT))
    if "Trif_paid" in df:
        wm |= df["Trif_paid"] > WOW_ABS
    if (nw := int(wm.sum())) > 0:
        pc = [c for c in df.columns if c.endswith("_paid") or
              c.endswith("Payout") or c.endswith("_pd")]
        df = df.copy()
        df.loc[wm, pc] = np.nan
        print(f"  ⚡ WowSuperWow: nulled {nw:,} outlier rows")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# PnL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def _x_pnl(df, k, i, ft):
    if df.empty: return np.array([], dtype=np.float64)
    m = df["Runners"].between(i["mn"], i.get("mx", 999))
    if "WhichRace" in df: m &= df["WhichRace"].between(1, 11)
    ef = i["ef"]
    m &= ef(df) if callable(ef) else df.apply(ef, axis=1).values
    if not m.any(): return np.array([], dtype=np.float64)
    q  = np.where(m)[0]
    tc, pm, hf, pc = i["tc"], i.get("pm", 1.0), i["hf"], i["pc"]
    if pc not in df: return np.full(len(q), -tc, dtype=np.float64)
    pf = df[pc].values[q].astype(float)
    v  = ~(np.isnan(pf) | (pf == 0))
    rng = np.random.default_rng(SEED + abs(hash(k)) % 1_000_000)
    j   = rng.uniform(1 - JITTER, 1 + JITTER, len(q))
    if hf is _h_al:
        pnl = np.where(v, pf * pm * j - tc, -tc).astype(np.float64)
    elif (fn := _VH.get(hf)) and "RANKED_RESULTS" in df:
        try:
            s = (df["RANKED_RESULTS"].iloc[q]
                   .str.split(expand=True).iloc[:, :4])
            while s.shape[1] < 4: s[s.shape[1]] = np.nan
            p = [pd.to_numeric(s.iloc[:, x], errors="coerce").values
                 for x in range(4)]
            nan_mask = (np.isnan(p[0])|np.isnan(p[1])|
                        np.isnan(p[2])|np.isnan(p[3]))
            h   = fn(*p) & ~nan_mask
            pnl = np.where(v & h, pf * pm * j - tc, -tc).astype(np.float64)
        except Exception:
            pnl = np.full(len(q), -tc, dtype=np.float64)
    else:
        pnl = np.full(len(q), -tc, dtype=np.float64)
    if ft > 0:
        pos_mask = pnl > 0
        n_pos    = pos_mask.sum()
        min_need = max(10, int(1 / ft))
        if n_pos >= min_need:
            threshold = float(np.percentile(pnl[pos_mask], (1 - ft) * 100))
            pnl = np.minimum(pnl, threshold)
    return pnl

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA KERNEL
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True)
def run_gauntlet_core(npth, mxr, sbr, tgt, rfl, mb,
                      c, fm, ls, m_on, mt, ms,
                      t_on, t_dd, t_du,
                      p_pls, p_sts, t_tbl, pb):
    ap  = np.zeros((npth, mxr + 1), dtype=np.float64)
    aph = np.zeros((npth, mxr + 1), dtype=np.int8)
    oc  = np.zeros(npth, dtype=np.int8)
    ra  = np.zeros(npth, dtype=np.int32)
    n_in, n_ph = len(p_pls), len(pb)
    for p in prange(npth):
        br, pk, tr = sbr, sbr, 0
        ap[p, 0] = br
        cph = n_ph - 1
        for i in range(n_ph):
            if pb[i, 0] <= br < pb[i, 1]:
                cph = int(pb[i, 2]); break
        aph[p, 0] = cph
        for r in range(1, mxr + 1):
            tx = np.zeros(6, dtype=np.float64)
            for i in range(len(t_tbl)):
                if t_tbl[i, 0] <= br < t_tbl[i, 1]:
                    for t in range(6): tx[t] = t_tbl[i, 2 + t]
                    break
            tot = 0.0
            for i in range(n_in):
                if tx[int(p_sts[i, 4])] > 0: tot += p_sts[i, 3]
            if tot == 0.0:
                ap[p, r], aph[p, r] = br, cph; continue
            pv = np.random.random() * tot
            cs = 0.0; idx = 0; le = 0
            for i in range(n_in):
                if tx[int(p_sts[i, 4])] > 0:
                    le = i; cs += p_sts[i, 3]
                    if pv <= cs: idx = i; break
            else:
                idx = le
            tc  = p_sts[idx, 2]
            nt  = min(int(tx[int(p_sts[idx, 4])]), int(mb / tc))
            if nt <= 0:
                ap[p, r], aph[p, r] = br, cph; continue
            st  = min(float(nt) * tc, br)
            if br > pk: pk, tr = br, 0
            if t_on > 0.5 and pk > 0.0 and (pk - br) / pk >= t_dd:
                if tr <= 0: tr = int(t_du)
            if tr > 0:
                st  = min(float(min(nt + 1, int(mb / tc))) * tc, br)
                tr -= 1
            pool = p_pls[idx]
            pnl  = pool[np.random.randint(0, len(pool))]
            if (np.random.random() < c or
                    np.random.random() < fm or
                    np.random.random() < ls):
                pnl = -tc
            elif pnl > 0 and m_on > 0.5 and st > mt:
                pnl = (pnl + tc) * (1.0 - min(0.5, (st - mt) * ms)) - tc
            br += pnl * (st / tc)
            ap[p, r] = br
            cph = n_ph - 1
            for i in range(n_ph):
                if pb[i, 0] <= br < pb[i, 1]:
                    cph = int(pb[i, 2]); break
            aph[p, r] = cph
            if br >= tgt or br <= rfl:
                oc[p] = 1 if br >= tgt else 2
                ra[p] = r
                for x in range(r + 1, mxr + 1):
                    ap[p, x], aph[p, x] = br, cph
                break
        else:
            ra[p] = mxr
    return ap, aph, oc, ra

# ══════════════════════════════════════════════════════════════════════════════
# PREP
# ══════════════════════════════════════════════════════════════════════════════
def _prep(pmap):
    tp, sl = List(), []
    for k, v in pmap.items():
        inst = INS[k]; tc = inst["tc"]; hm = v > HR_MIN
        tp.append(v.astype(np.float64))
        ew = max(float(inst["ew"]), 0.01)
        sl.append([
            float(hm.mean()) if len(v) else 0.0,
            float((v[hm] + tc).mean()) if hm.any() else 0.0,
            tc, ew, float(_TM[inst["t"]])
        ])
    return tp, np.array(sl, dtype=np.float64), _tt_arr(), _pb_arr()

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def _stat(paths, outcomes, races, fbr, slbl, so, window, mode):
    n  = len(paths)
    ns = outcomes.count("success")
    nr = outcomes.count("ruin")
    nt = outcomes.count("timeout")
    sr_list = [x for x, y in zip(races, outcomes) if y == "success"]
    rr_list = [x for x, y in zip(races, outcomes) if y == "ruin"]
    mb  = float(np.median(fbr))
    mi  = int(np.argmin(np.abs(np.array(fbr) - mb)))
    dd  = [float(np.max(np.maximum.accumulate(p) - p)) for p in paths]
    fm  = lambda l: int(np.median(l)) if l else None
    fp  = lambda l, q: int(np.percentile(l, q)) if len(l) > 1 else None
    adv = (1 - (1 - so["c"]) * (1 - so["fm"]) * (1 - so["s"])) * 100
    ruin_races  = [x for x, y in zip(races, outcomes) if y == "ruin"]
    early_ruin  = sum(1 for r in ruin_races if r < 200)
    mrs = fm(sr_list)
    mrr = fm(rr_list)
    return {
        # identification
        "window":   window,
        "strategy": slbl,
        "stress":   so["lbl"],
        "mode":     mode,
        "version":  "v6.6.0",
        "edition":  "DIAMOND APEX REFINED",
        "run_ts":   datetime.datetime.now().isoformat(),
        # core outcomes
        "n":        n,
        "ns":       ns,
        "nr":       nr,
        "nt":       nt,
        "sr":       round(ns / n * 100, 4),
        "rr":       round(nr / n * 100, 4),
        "tr":       round(nt / n * 100, 4),
        # timing
        "mrs":      mrs,
        "mrr":      mrr,
        "p1s":      fp(sr_list, 10),
        "p9s":      fp(sr_list, 90),
        "p25s":     fp(sr_list, 25),
        "p75s":     fp(sr_list, 75),
        "mds":      mrs // RPD if mrs else None,
        "mdr":      mrr // RPD if mrr else None,
        # bankroll
        "mbr":      round(mb, 4),
        "midx":     mi,
        "mdd":      round(float(np.median(dd)), 4),
        "max_dd":   round(float(np.max(dd)), 4),
        "pct_br": {
            str(p): round(float(np.percentile(fbr, p)), 2)
            for p in [5, 10, 25, 50, 75, 90, 95]
        },
        # risk
        "adv":                  round(adv, 6),
        "effective_adverse_rate": round(adv, 6),
        "early_ruin_count":     early_ruin,
        "early_ruin_pct":       round(early_ruin / n * 100, 4),
        # config echo
        "max_bet_dollars":      MAX_BET,
        "start_bankroll":       S_BR,
        "ruin_floor":           R_FLR,
        "target":               TGT,
        "payout_jitter_frac":   JITTER,
        "n_paths":              N_PTH,
        "max_races":            M_RCS,
    }

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT SNAPSHOT  (stored in mega-JSON per run)
# ══════════════════════════════════════════════════════════════════════════════
def _inst_snapshot(pmap):
    """Compact per-instrument stats for embedding in mega-JSON."""
    snap = {}
    for k, v in pmap.items():
        i   = INS[k]
        tc  = i["tc"]
        hm  = v > HR_MIN
        n   = len(v)
        ev  = float(v.mean()) if n else 0.0
        hr  = float(hm.mean() * 100) if n else 0.0
        sd  = float(v.std()) if n else 0.0
        snap[k] = {
            "tier":     i["t"],
            "n":        n,
            "ev":       round(ev, 4),
            "hr_pct":   round(hr, 2),
            "sd":       round(sd, 4),
            "tc":       tc,
            "ev_per_tc":round(ev / tc if tc else 0, 4),
            "watch":    i.get("watch_list", False),
            "validated":bool(i.get("v", 0)),
        }
    return snap

# ══════════════════════════════════════════════════════════════════════════════
# LADDER RUNNER  (returns result dict, no prints during batch)
# ══════════════════════════════════════════════════════════════════════════════
def _run_ladder(tp, sl, tt, pb, so, slbl, window, silent=False):
    results = []
    for i, b in enumerate(LDR_RNGS):
        tgt_l = float(b * 10)
        rfl_l = max(2.0, float(b) * 0.10)
        np.random.seed(SEED)
        ap, aph, oc, ra = run_gauntlet_core(
            LADDER_N_PTH, LADDER_MXR, float(b), tgt_l, rfl_l, MAX_BET,
            float(so["c"]), float(so["fm"]), float(so["s"]),
            1.0 if so["mi"] else 0.0, float(so["mt"]), float(so["ms"]),
            1.0 if so["ti"] else 0.0, float(so["td"]), float(so["tdu"]),
            tp, sl, tt, pb)
        ns_  = int(np.sum(oc == 1))
        nr_  = int(np.sum(oc == 2))
        nt_  = int(np.sum(oc == 0))
        sr_  = ra[oc == 1]
        m    = int(np.median(sr_)) if len(sr_) else 0
        p10  = int(np.percentile(sr_, 10)) if len(sr_) > 1 else None
        p90  = int(np.percentile(sr_, 90)) if len(sr_) > 1 else None
        d    = m // RPD if m else None
        sr_p = ns_ / LADDER_N_PTH * 100
        rr_p = nr_ / LADDER_N_PTH * 100
        tr_p = nt_ / LADDER_N_PTH * 100
        results.append({
            "rung":        i + 1,
            "start":       b,
            "target":      b * 10,
            "ruin_floor":  rfl_l,
            "sr":          round(sr_p, 2),
            "rr":          round(rr_p, 2),
            "tr":          round(tr_p, 2),
            "median_races":m,
            "median_days": d,
            "p10_races":   p10,
            "p90_races":   p90,
            "ns":          ns_,
            "nr":          nr_,
            "nt":          nt_,
        })
        if not silent:
            print(f"    Rung {i+1}: ${b:,}→${b*10:,}  "
                  f"SR:{sr_p:.1f}%  RR:{rr_p:.1f}%  Med:{m:,}r (~{d or '?'}d)")
    # compound SR
    cum = 1.0
    for r in results:
        if r["sr"] > 0: cum *= r["sr"] / 100.0
    return {
        "window":       window,
        "strategy":     slbl,
        "stress":       so["lbl"],
        "mode":         "ladder",
        "version":      "v6.6.0",
        "edition":      "DIAMOND APEX REFINED",
        "run_ts":       datetime.datetime.now().isoformat(),
        "paths_per_rung": LADDER_N_PTH,
        "max_races_per_rung": LADDER_MXR,
        "compound_sr":  round(cum * 100, 4),
        "rungs":        results,
    }

# ══════════════════════════════════════════════════════════════════════════════
# CHARTING  (slim version — only called when chart saving is on)
# ══════════════════════════════════════════════════════════════════════════════
def _sax(a, t=None, x=None, y=None):
    a.set_facecolor(BG)
    a.tick_params(colors=MUT, labelsize=8)
    for sp in a.spines.values(): sp.set_color(GRD)
    a.grid(color=GRD, ls=":", alpha=0.4)
    if t: a.set_title(t, color=TXT, fontsize=10, fontweight="bold", pad=10)
    if x: a.set_xlabel(x, color=MUT, fontsize=8)
    if y: a.set_ylabel(y, color=MUT, fontsize=8)

def _save_charts(paths, outcomes, fbr, st_o, pmap, plg, key):
    """Save all 4 charts with key-suffixed filenames."""
    sub = f"{st_o['strategy']} | {st_o['stress']} | {st_o['window']}"

    # ── Chart 1: Paths ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    _sax(ax, f"GAUNTLET v6.6.0 | ${S_BR:,.0f}→${TGT:,.0f} | {sub}",
         f"Races (~days@{RPD}/d)", "Bankroll ($)")
    ax.set_yscale("symlog", linthresh=100)
    rng = np.random.default_rng(SEED)
    for i in rng.choice(len(paths), min(300, len(paths)), replace=False):
        c = (ACC if outcomes[i]=="success" else
             RED if outcomes[i]=="ruin" else MUT)
        alpha = 0.10 if outcomes[i]=="success" else 0.18 if outcomes[i]=="ruin" else 0.05
        ax.plot(paths[i], color=c, alpha=alpha, lw=0.5)
    med = paths[st_o["midx"]]
    ax.plot(med, color=GLD, lw=2.2, zorder=6, label=f"Median (${med[-1]:,.0f})")
    ml  = max(len(p) for p in paths)
    pad = np.full((len(paths), ml), np.nan)
    for i, p in enumerate(paths): pad[i, :len(p)] = p
    act = np.sum(~np.isnan(pad), axis=0)
    with np.errstate(all="ignore"):
        blo = np.nanpercentile(pad, 25, axis=0)
        bhi = np.nanpercentile(pad, 75, axis=0)
    blo[act < 5] = bhi[act < 5] = np.nan
    xs = np.arange(ml); ok = ~np.isnan(blo) & ~np.isnan(bhi)
    if ok.any():
        ax.fill_between(xs[ok], blo[ok], bhi[ok], color=GLD, alpha=0.07, label="P25-P75")
    for mv, (lb, col, ls, lw) in {
        R_FLR:("Ruin",RED,"-.",1.2), TGT:("Target",GLD,"--",1.7)
    }.items():
        ax.axhline(mv, color=col, ls=ls, lw=lw, alpha=0.65)
        ax.text(ml*0.004, mv*1.05 if mv>0 else mv+30, lb, color=col, fontsize=7.5)
    ann = (f"SR:{st_o['sr']:.1f}%  RR:{st_o['rr']:.1f}%\n"
           f"Med:{st_o['mrs'] or 'N/A'}r"
           + (f"(~{st_o['mds']}d)" if st_o['mds'] else "") + "\n"
           f"Adv:{st_o['adv']:.1f}%  ER:{st_o['early_ruin_count']} paths")
    ax.text(0.98, 0.04, ann, transform=ax.transAxes, color=TXT,
            fontsize=8.5, ha="right", va="bottom",
            bbox=dict(facecolor=GRD, edgecolor=MUT, alpha=0.85,
                      boxstyle="round,pad=0.4"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(facecolor=BG, edgecolor=GRD, labelcolor=TXT, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"Gauntlet_Paths_{key}.png", dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()

    # ── Chart 2: Distribution ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _sax(a1, "Final Bankroll Distribution", "Bankroll ($)", "Paths")
    sh  = abs(R_FLR) + 1
    shf = [b + sh for b in fbr]
    bns = np.logspace(np.log10(max(1,min(shf))), np.log10(max(shf)+1), 50)
    for lb, col in [("success",ACC),("timeout",MUT),("ruin",RED)]:
        vs = [b+sh for b,o in zip(fbr,outcomes) if o==lb]
        if vs: a1.hist(vs, bins=bns, color=col, alpha=0.72, label=f"{lb} ({len(vs):,})")
    a1.set_xscale("log")
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v-sh:,.0f}"))
    a1.axvline(TGT+sh, color=GLD, lw=1.8, ls="--", label=f"Target ${TGT:,.0f}")
    a1.axvline(st_o["mbr"]+sh, color=CYN, lw=1.2, ls=":", label=f"Median ${st_o['mbr']:,.0f}")
    if st_o["early_ruin_count"] > 0:
        a1.text(0.04, 0.92,
                f"Early ruin (<200r):\n{st_o['early_ruin_count']} paths "
                f"({st_o['early_ruin_pct']:.2f}%)",
                transform=a1.transAxes, color=RED, fontsize=8, va="top",
                bbox=dict(facecolor=GRD, edgecolor=RED, alpha=0.8,
                          boxstyle="round,pad=0.3"))
    a1.legend(facecolor=BG, labelcolor=TXT, fontsize=8)
    _sax(a2, "Cumulative Ruin & Success", "Race Events", "Prob (%)")
    mxr  = max(len(x) for x in paths)
    npth = len(paths)
    for lb, col in [("ruin",RED),("success",ACC)]:
        td = np.array([min(len(x)-1,mxr-1) for x,y in zip(paths,outcomes) if y==lb],
                      dtype=np.intp)
        if len(td):
            cv = (np.cumsum(np.bincount(td, minlength=mxr).astype(float)) / npth * 100)
            a2.plot(cv, color=col, lw=2, label=f"{lb.capitalize()} {cv[-1]:.1f}%")
            a2.fill_between(range(mxr), cv, color=col, alpha=0.09)
    if st_o["mrs"]:
        a2.axvline(st_o["mrs"], color=GLD, lw=1.4, ls="--",
                   label=f"Median {st_o['mrs']:,} races")
    a2.set_xlim(0, mxr); a2.set_ylim(0, 105)
    a2.legend(facecolor=BG, labelcolor=TXT, fontsize=8)
    fig.suptitle(f"v6.6.0 Outcome | {sub}", color=TXT, fontsize=11,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"Gauntlet_Dist_{key}.png", dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()

    # ── Chart 3: Comfort Score ────────────────────────────────────────────────
    def _cscr_local(pm):
        rows = []
        ah = [float((v > HR_MIN).mean()) for v in pm.values()]
        an = [len(v) for v in pm.values()]
        mh = max(ah) if ah else 1
        mn = max(an) if an else 1
        for k, v in pm.items():
            i   = INS[k]; tc = i["tc"] if i["tc"] != 0 else 1e-6
            n   = len(v); hr = float((v > HR_MIN).mean()) if n else 0
            ev  = float(v.mean()) if n else 0
            sd  = float(v.std())  if n else 0
            hs  = hr / mh; vr = sd / abs(tc); st2 = 1 / (1 + vr)
            ss  = np.log10(n + 1) / np.log10(mn + 1) if mn else 0
            es  = float(np.clip(ev / abs(tc), 0, 3) / 3)
            rows.append({
                "n":k,"t":i["t"],"sz":n,"h":hr*100,"e":ev,"tc":tc,"sd":sd,
                "c":hs*.35+st2*.30+ss*.20+es*.15,
                "mb":min(int(MAX_BET/tc),10)*tc,
                "wl":i.get("watch_list",False),"b":i["t"]=="BONUS",
                "a":i["t"]=="APEX","val":i.get("v",0),
            })
        return sorted(rows, key=lambda x: x["c"], reverse=True)
    rc  = _cscr_local(pmap)
    nms = [x["n"]+(" 🔍" if x["wl"] else " 🔥" if x["b"] else
                    " ⚠️"  if x["a"]  else "") for x in rc]
    ec  = [T_COL.get(x["t"], MUT) for x in rc]
    ht  = ["" if x["val"] else "//" for x in rc]
    fig, ax = plt.subplots(figsize=(13, max(6, len(rc)*0.42)), facecolor=BG)
    _sax(ax, f"Comfort Score | {sub}  [Hatched=Discovery]", "Score", "")
    y   = np.arange(len(rc))
    mh_ = max(x["h"] for x in rc) or 1
    mn_ = max(x["sz"] for x in rc) or 1
    hw  = [x["h"]/mh_*.35 for x in rc]
    sw  = [1/(1+(x["sd"]/x["tc"] if x["tc"] else 0))*.30 for x in rc]
    ls_ = [a+b for a,b in zip(hw,sw)]
    aw  = [np.log10(x["sz"]+1)/np.log10(mn_+1)*.20 for x in rc]
    le_ = [a+b for a,b in zip(ls_,aw)]
    ew  = [np.clip(x["e"]/x["tc"] if x["tc"] else 0,0,3)/3*.15 for x in rc]
    ax.barh(y, hw,  color="#06b6d4", label="HR×0.35")
    ax.barh(y, sw,  left=hw,  color="#10b981", label="Stability×0.30")
    ax.barh(y, aw,  left=ls_, color="#8b5cf6", label="Sample×0.20")
    ax.barh(y, ew,  left=le_, color="#f59e0b", label="EV/Cost×0.15")
    for i, x in enumerate(rc):
        ax.barh(i, x["c"], fill=False, edgecolor=ec[i], hatch=ht[i], lw=1.5, zorder=5)
        ax.text(x["c"]+0.006, i,
                f" {x['c']:.3f}  HR:{x['h']:.0f}%  N:{x['sz']:,}"
                f"  EV:{'+' if x['e']>=0 else ''}{x['e']:.2f}",
                color=TXT, va="center", fontsize=6.5, fontfamily="monospace")
    ax.set_yticks(y); ax.set_yticklabels(nms, color=MUT, fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color=RED, lw=1, ls="--", alpha=0.6)
    ax.legend(facecolor=BG, edgecolor=GRD, labelcolor=TXT, fontsize=7.5, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"Gauntlet_Comfort_{key}.png", dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()

    # ── Chart 4: Phase Transitions ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    _sax(ax, f"Phase Transitions | {sub}", "Race Events", "% of Paths")
    mxr = max(len(x) for x in plg)
    pm_ = np.zeros((len(PHS), mxr))
    for pl in plg:
        for r, ph in enumerate(pl):
            if ph < len(PHS): pm_[ph, r] += 1
    dt = pm_.sum(axis=0); dt[dt==0] = 1
    pct = pm_/dt*100; yb = np.zeros(mxr)
    for i, (_,_,lbl,col) in enumerate(PHS):
        yt = yb + pct[i]
        ax.fill_between(range(mxr), yb, yt, color=col, alpha=0.85,
                        label=lbl.replace("\n"," "))
        yb = yt
    ax.set_xlim(0, mxr); ax.set_ylim(0, 100)
    ax.legend(facecolor=BG, edgecolor=GRD, labelcolor=TXT, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"Gauntlet_Phases_{key}.png", dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD  (printed + saved after all runs)
# ══════════════════════════════════════════════════════════════════════════════
def _print_leaderboard(all_results):
    """Print ranked table of all gauntlet runs by SR descending."""
    gauntlet = [r for r in all_results if r.get("mode") == "gauntlet"]
    if not gauntlet: return
    gauntlet.sort(key=lambda x: (-x["sr"], x["rr"], x.get("mrs", 99999) or 99999))
    SEP = "═"*100
    print(f"\n{SEP}")
    print("  MEGA-BATCH LEADERBOARD — GAUNTLET RUNS RANKED BY SUCCESS RATE")
    print(SEP)
    print(f"  {'Rank':<5} {'Window':<7} {'Strategy':<12} {'Stress':<11} "
          f"{'SR%':>6} {'RR%':>6} {'TR%':>6} {'MedRaces':>9} {'MedDays':>8} "
          f"{'EarlyRuin':>10} {'MedBR':>12}")
    print("  "+"─"*95)
    for i, r in enumerate(gauntlet, 1):
        tier_emoji = ("🏆" if r["sr"] >= 98 else
                      "✅" if r["sr"] >= 95 else
                      "⚠️"  if r["sr"] >= 90 else "❌")
        print(f"  {i:<5} {r['window']:<7} {r['strategy']:<12} {r['stress']:<11} "
              f"{r['sr']:>6.2f}% {r['rr']:>5.2f}% {r['tr']:>5.2f}% "
              f"{r.get('mrs') or 'N/A':>9} {r.get('mds') or 'N/A':>8} "
              f"{r['early_ruin_count']:>10} "
              f"${r['mbr']:>10,.0f}  {tier_emoji}")
    print(SEP)

    # Ladder compound SR ranking
    ladders = [r for r in all_results if r.get("mode") == "ladder"]
    if ladders:
        ladders.sort(key=lambda x: -x.get("compound_sr", 0))
        print(f"\n  LADDER RUNS — RANKED BY COMPOUND SR")
        print("  "+"─"*60)
        print(f"  {'Rank':<5} {'Window':<7} {'Strategy':<12} {'Stress':<11} {'CompSR%':>8}")
        print("  "+"─"*60)
        for i, r in enumerate(ladders, 1):
            print(f"  {i:<5} {r['window']:<7} {r['strategy']:<12} "
                  f"{r['stress']:<11} {r.get('compound_sr', 0):>8.2f}%")
    print(SEP+"\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*72)
    print("  THE 100x GAUNTLET v6.6.0 DIAMOND APEX REFINED")
    print("  *** MEGA BATCH RUNNER ***")
    print("="*72)

    # ── Step 1: Compute date cutoffs ──────────────────────────────────────────
    tdy   = datetime.date.today()
    cuts  = {
        "18mo": (tdy - datetime.timedelta(days=548)).strftime("%Y-%m-%d"),
        "36mo": (tdy - datetime.timedelta(days=1096)).strftime("%Y-%m-%d"),
    }
    print(f"\n  Date windows:")
    for w, c in cuts.items():
        print(f"    {w}: cutoff {c}")

    # ── Step 2: Load raw data ONCE per window ─────────────────────────────────
    print("\n  Loading data (once per window)...")
    raw_dfs = {}
    for w, cut in cuts.items():
        if w not in BATCH_WINDOWS:
            continue
        df = _get_df(TRAIN_FILE, TEST_FILE, cut)
        raw_dfs[w] = df
        print(f"    {w}: {len(df):,} rows loaded")

    # ── Step 3: Extract PnL pools ONCE per (window, strategy) ────────────────
    # We need all instruments across all strategies to extract once per window
    print("\n  Extracting PnL pools (once per window × all instruments)...")
    all_instruments = set()
    for sname in BATCH_STRATEGIES:
        all_instruments |= STRATEGY_SETS.get(sname, set())
    all_instruments = {k for k in all_instruments if k in INS}

    # pmap_cache[window][stress_label] = {instrument: pnl_array}
    # Note: stress affects ft (fat-tail trim) via so["f"]
    # We cache per (window, ft) since ft is the only pmap dependency on stress
    pmap_cache = {}   # key: (window, ft_str) -> {k: pnl}
    ft_vals    = {so["f"] for so in STRESS_PRESETS.values()}

    for w in BATCH_WINDOWS:
        if w not in raw_dfs: continue
        df = raw_dfs[w]
        for ft in ft_vals:
            cache_key = (w, str(ft))
            print(f"    Extracting: window={w}  ft={ft} ...")
            pmap = {}
            for k in tqdm(sorted(all_instruments),
                           desc=f"  {w}/ft={ft}", leave=False, ncols=70):
                pnl = _x_pnl(df, k, INS[k], ft)
                if len(pnl) > 0:
                    pmap[k] = pnl
            pmap_cache[cache_key] = pmap
            print(f"      → {len(pmap)} instruments extracted")

    # ── Step 4: Build combo list ──────────────────────────────────────────────
    combos = list(itertools.product(
        BATCH_WINDOWS, BATCH_STRATEGIES, BATCH_STRESSES, BATCH_MODES))
    total  = len(combos)
    print(f"\n  Total combinations to run: {total}")
    print(f"  Windows:    {BATCH_WINDOWS}")
    print(f"  Strategies: {BATCH_STRATEGIES}")
    print(f"  Stresses:   {BATCH_STRESSES}")
    print(f"  Modes:      {BATCH_MODES}")
    print(f"\n  Starting batch...\n")

    # ── Step 5: BATCH LOOP ────────────────────────────────────────────────────
    mega   = {
        "meta": {
            "version":      "v6.6.0",
            "edition":      "DIAMOND APEX REFINED",
            "generated":    datetime.datetime.now().isoformat(),
            "n_paths":      N_PTH,
            "max_races":    M_RCS,
            "ladder_paths": LADDER_N_PTH,
            "ladder_mxr":   LADDER_MXR,
            "start_br":     S_BR,
            "target":       TGT,
            "ruin_floor":   R_FLR,
            "max_bet":      MAX_BET,
            "jitter":       JITTER,
            "rpd":          RPD,
            "windows":      BATCH_WINDOWS,
            "strategies":   BATCH_STRATEGIES,
            "stresses":     BATCH_STRESSES,
            "modes":        BATCH_MODES,
            "total_runs":   total,
        },
        "runs":    {},   # key -> full result dict
        "ladders": {},   # key -> ladder result dict
        "instrument_pools": {},  # window -> {k: {ev, hr, n, ...}}
    }

    # Store instrument snapshots (per window, per ft)
    for (w, ft_s), pmap in pmap_cache.items():
        snap_key = f"{w}_ft{ft_s}"
        mega["instrument_pools"][snap_key] = _inst_snapshot(pmap)

    all_results_flat = []   # for leaderboard
    leaderboard_rows = []   # for CSV

    batch_start = time.time()

    with tqdm(total=total, desc="Batch", unit="run", ncols=80) as pbar:
        for window, strategy, stress_lbl, mode in combos:
            so       = STRESS_PRESETS[stress_lbl]
            ft       = so["f"]
            cache_key= (window, str(ft))
            all_inst = pmap_cache.get(cache_key, {})

            # Filter to this strategy's instruments
            strat_keys = {k for k in STRATEGY_SETS.get(strategy, set())
                          if k in all_inst}
            pmap = {k: all_inst[k] for k in strat_keys}

            run_key = f"{window}__{strategy}__{stress_lbl}__{mode}"
            pbar.set_postfix_str(run_key[:50])

            if not pmap:
                pbar.update(1)
                continue

            tp, sl, tt, pb = _prep(pmap)

            # ── GAUNTLET ──────────────────────────────────────────────────────
            if mode == "gauntlet":
                np.random.seed(SEED)
                ap, aph, oc, ra = run_gauntlet_core(
                    N_PTH, M_RCS, S_BR, TGT, R_FLR, MAX_BET,
                    float(so["c"]), float(so["fm"]), float(so["s"]),
                    1.0 if so["mi"] else 0.0, float(so["mt"]), float(so["ms"]),
                    1.0 if so["ti"] else 0.0, float(so["td"]), float(so["tdu"]),
                    tp, sl, tt, pb)
                outcomes_ = ["timeout" if c==0 else
                             "success" if c==1 else "ruin" for c in oc]
                fbr_   = [float(ap[i, ra[i]]) for i in range(N_PTH)]
                paths_ = [ap[i, :ra[i]+1]     for i in range(N_PTH)]
                plg_   = [aph[i, :ra[i]+1].tolist() for i in range(N_PTH)]

                st_o = _stat(paths_, outcomes_, ra.tolist(), fbr_,
                             strategy, so, window, mode)

                # Charts?
                do_chart = (BATCH_SAVE_CHARTS or
                            (window, strategy, stress_lbl, mode) in CHART_FILTER)
                if do_chart:
                    chart_key = f"{window}_{strategy}_{stress_lbl}"
                    _save_charts(paths_, outcomes_, fbr_, st_o,
                                 pmap, plg_, chart_key)

                # Embed instrument snapshot in result
                st_o["instruments"] = _inst_snapshot(pmap)
                st_o["n_instruments"] = len(pmap)

                mega["runs"][run_key] = st_o
                all_results_flat.append(st_o)
                leaderboard_rows.append({
                    "key":          run_key,
                    "window":       window,
                    "strategy":     strategy,
                    "stress":       stress_lbl,
                    "mode":         mode,
                    "sr":           st_o["sr"],
                    "rr":           st_o["rr"],
                    "tr":           st_o["tr"],
                    "mrs":          st_o["mrs"],
                    "mds":          st_o["mds"],
                    "mrr":          st_o["mrr"],
                    "mbr":          st_o["mbr"],
                    "mdd":          st_o["mdd"],
                    "early_ruin":   st_o["early_ruin_count"],
                    "early_ruin_pct": st_o["early_ruin_pct"],
                    "adv":          st_o["adv"],
                    "n_instruments":st_o["n_instruments"],
                    "p10_br":       st_o["pct_br"].get("10"),
                    "p90_br":       st_o["pct_br"].get("90"),
                    "p50_br":       st_o["pct_br"].get("50"),
                })

            # ── LADDER ────────────────────────────────────────────────────────
            elif mode == "ladder":
                ldr = _run_ladder(tp, sl, tt, pb, so, strategy, window, silent=True)
                ldr["instruments"] = _inst_snapshot(pmap)
                ldr["n_instruments"] = len(pmap)
                mega["ladders"][run_key] = ldr
                all_results_flat.append(ldr)
                leaderboard_rows.append({
                    "key":          run_key,
                    "window":       window,
                    "strategy":     strategy,
                    "stress":       stress_lbl,
                    "mode":         mode,
                    "sr":           ldr.get("compound_sr"),
                    "rr":           None,
                    "tr":           None,
                    "mrs":          None,
                    "mds":          None,
                    "mrr":          None,
                    "mbr":          None,
                    "mdd":          None,
                    "early_ruin":   None,
                    "early_ruin_pct": None,
                    "adv":          None,
                    "n_instruments":ldr["n_instruments"],
                    "p10_br":       None,
                    "p90_br":       None,
                    "p50_br":       None,
                })

            pbar.update(1)

    batch_elapsed = time.time() - batch_start
    print(f"\n  Batch complete in {batch_elapsed/60:.1f} minutes.")

    # ── Step 6: Cross-run analytics ───────────────────────────────────────────
    gauntlet_results = [r for r in all_results_flat if r.get("mode") == "gauntlet"]

    # Best runs summary
    if gauntlet_results:
        best_sr  = max(gauntlet_results, key=lambda x: x["sr"])
        best_mdr = min((r for r in gauntlet_results if r.get("mrs")),
                       key=lambda x: x.get("mrs", 99999), default=None)
        worst    = min(gauntlet_results, key=lambda x: x["sr"])

        mega["cross_run_analytics"] = {
            "best_sr_run": {
                "key":      f"{best_sr['window']}__{best_sr['strategy']}__{best_sr['stress']}__gauntlet",
                "sr":       best_sr["sr"],
                "rr":       best_sr["rr"],
                "mrs":      best_sr["mrs"],
                "stress":   best_sr["stress"],
                "strategy": best_sr["strategy"],
                "window":   best_sr["window"],
            },
            "fastest_median_run": {
                "key":      f"{best_mdr['window']}__{best_mdr['strategy']}__{best_mdr['stress']}__gauntlet" if best_mdr else None,
                "mrs":      best_mdr["mrs"] if best_mdr else None,
                "mds":      best_mdr["mds"] if best_mdr else None,
                "sr":       best_mdr["sr"]  if best_mdr else None,
            },
            "worst_sr_run": {
                "key":      f"{worst['window']}__{worst['strategy']}__{worst['stress']}__gauntlet",
                "sr":       worst["sr"],
                "rr":       worst["rr"],
                "stress":   worst["stress"],
                "strategy": worst["strategy"],
                "window":   worst["window"],
            },
            "sr_by_stress": {
                s: round(float(np.mean([r["sr"] for r in gauntlet_results
                                        if r["stress"] == s])), 3)
                for s in BATCH_STRESSES
            },
            "sr_by_strategy": {
                s: round(float(np.mean([r["sr"] for r in gauntlet_results
                                        if r["strategy"] == s])), 3)
                for s in BATCH_STRATEGIES
            },
            "sr_by_window": {
                w: round(float(np.mean([r["sr"] for r in gauntlet_results
                                        if r["window"] == w])), 3)
                for w in BATCH_WINDOWS
            },
            "ruin_free_runs": sum(1 for r in gauntlet_results if r["rr"] == 0.0),
            "sub_2pct_ruin":  sum(1 for r in gauntlet_results if r["rr"] < 2.0),
            "sub_5pct_ruin":  sum(1 for r in gauntlet_results if r["rr"] < 5.0),
            "total_gauntlet_runs": len(gauntlet_results),
        }
        mega["meta"]["batch_elapsed_seconds"] = round(batch_elapsed, 1)
        mega["meta"]["batch_elapsed_minutes"] = round(batch_elapsed / 60, 2)

    # ── Step 7: Save outputs ──────────────────────────────────────────────────
    print(f"\n  Saving {MEGA_JSON_FILE}...")
    with open(MEGA_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(mega, f, indent=2, default=str)
    print(f"  Saving {LEADERBOARD_CSV}...")
    pd.DataFrame(leaderboard_rows).to_csv(LEADERBOARD_CSV, index=False)

    # Also save a slim gauntlet-only CSV sorted by SR
    gdf = pd.DataFrame([r for r in leaderboard_rows if r["mode"]=="gauntlet"])
    if not gdf.empty:
        gdf = gdf.sort_values("sr", ascending=False)
        gdf.to_csv("Gauntlet_Leaderboard_Gauntlet.csv", index=False)
        print("  Saving Gauntlet_Leaderboard_Gauntlet.csv...")

    # ── Step 8: Console leaderboard ───────────────────────────────────────────
    _print_leaderboard(all_results_flat)

    # ── Step 9: Print cross-run summary ───────────────────────────────────────
    if "cross_run_analytics" in mega:
        cra = mega["cross_run_analytics"]
        SEP = "═"*72
        print(f"{SEP}\n  CROSS-RUN ANALYTICS\n{SEP}")
        print(f"  Total gauntlet runs  : {cra['total_gauntlet_runs']}")
        print(f"  Zero-ruin runs       : {cra['ruin_free_runs']}")
        print(f"  Sub-2% ruin runs     : {cra['sub_2pct_ruin']}")
        print(f"  Sub-5% ruin runs     : {cra['sub_5pct_ruin']}")
        print(f"\n  Best SR   : {cra['best_sr_run']['sr']:.2f}%  "
              f"({cra['best_sr_run']['window']} {cra['best_sr_run']['strategy']} "
              f"{cra['best_sr_run']['stress']})")
        print(f"  Worst SR  : {cra['worst_sr_run']['sr']:.2f}%  "
              f"({cra['worst_sr_run']['window']} {cra['worst_sr_run']['strategy']} "
              f"{cra['worst_sr_run']['stress']})")
        if cra["fastest_median_run"]["mrs"]:
            print(f"  Fastest   : {cra['fastest_median_run']['mrs']:,} races "
                  f"(~{cra['fastest_median_run']['mds']}d)  "
                  f"SR={cra['fastest_median_run']['sr']:.1f}%")
        print(f"\n  Avg SR by stress:")
        for s, v in cra["sr_by_stress"].items():
            bar = "█" * int(v / 2)
            print(f"    {s:<12}: {v:.2f}%  {bar}")
        print(f"\n  Avg SR by strategy:")
        for s, v in cra["sr_by_strategy"].items():
            bar = "█" * int(v / 2)
            print(f"    {s:<12}: {v:.2f}%  {bar}")
        print(f"\n  Avg SR by window:")
        for s, v in cra["sr_by_window"].items():
            print(f"    {s:<12}: {v:.2f}%")
        print(f"\n  Elapsed: {batch_elapsed/60:.1f} minutes\n{SEP}\n")

    print("✅ MEGA BATCH COMPLETE.")
    print(f"   {MEGA_JSON_FILE}          — full nested results")
    print(f"   {LEADERBOARD_CSV}   — all runs flat")
    print(f"   Gauntlet_Leaderboard_Gauntlet.csv — gauntlet only, sorted")
    if BATCH_SAVE_CHARTS:
        n_charts = len([r for r in leaderboard_rows if r["mode"]=="gauntlet"]) * 4
        print(f"   ~{n_charts} chart PNGs saved")

if __name__ == "__main__":
    main()
