#!/usr/bin/env python3
"""
THE 100x GAUNTLET — v6.3.4 CRYSTAL OMNI EDITION (PATCHED)
===========================================================
Fixes applied:
  1. STS_PRE → STRESS_PRESETS in main()
  2. TKT_TBL unpack in seasonal sweep (8-col explicit unpack)
  3. TKT_TBL unpack in main() tt= line (same fix)
  4. Stress dict key "e"/"l" unified — stress accesses "emo"/"lbl"
  5. _stat stress_label reads so["lbl"] not so.get("l","")
  6. Ladder TXT export upgraded to full per-rung detail
  7. "emo" and "lbl" keys used consistently throughout
"""
import warnings, datetime, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from numba import njit, prange; from numba.typed import List
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_FILE, TEST_FILE = "RaceRecords_Output_2025.csv", "RaceRecords_Output_2026.csv"
S_BR, R_FLR, TGT, MAX_BET = 250.0, 50.0, 88888.0, 100.0
N_PTH, M_RCS, RPD, SEED = 2000, 30000, 25, 42
JITTER, HR_MIN, WOW_ABS, WOW_RAT = 0.05, 2, 25000.0, 200.0
LDR_RNGS = [20 * (2**i) for i in range(14)]  # 20, 40, 80, ..., 163,840 (powers of 2)
SSN_SWP, SSN_W, SSN_N = False, 365, 500
SSN_STS = ["01-01","02-01","03-01","04-01","05-01","06-01",
           "07-01","08-01","09-01","10-01","11-01","12-01"]

# ══════════════════════════════════════════════════════════════════════════════
# STRESS PRESETS  — keys: lbl, emo, f, s, mi, mt, ms, ti, td, tdu, c, fm
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
# TICKET TABLE  — br_lo, br_hi, T1, T2, T3, T4, BNS, APX
# ══════════════════════════════════════════════════════════════════════════════
TKT_TBL = [
    (0,      500,    1,  0,  0, 0, 0, 0),
    (500,    2000,   1,  1,  0, 0, 1, 0),
    (2000,   4000,   2,  2,  1, 0, 1, 0),
    (4000,   8000,   2,  2,  1, 0, 1, 0),
    (8000,   15000,  3,  3,  2, 0, 2, 1),
    (15000,  25000,  5,  5,  3, 0, 3, 1),
    (25000,  40000,  8,  8,  5, 0, 5, 2),
    (40000,  60000, 10, 10,  0, 0, 0, 0),
    (60000, 999999, 10, 10,  0, 0, 0, 0),
]

def _tt_arr():
    """Build float64 array from TKT_TBL — always 8-column explicit unpack."""
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
    """Phase bounds array — [lo, hi, phase_idx]."""
    return np.array(
        [[float(lo),float(hi),float(idx)]
         for idx,(lo,hi,_,_) in enumerate(PHS)],
        dtype=np.float64)

BG,GRD,TXT,MUT,ACC,RED,GLD,CYN = "#0f172a","#1e293b","#f8fafc","#64748b","#10b981","#f43f5e","#f59e0b","#06b6d4"
T_COL = {"T1":GLD,"T2":ACC,"T3":CYN,"T4":MUT,"BNS":"#f97316","APX":"#ec4899"}

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT GATES
# ══════════════════════════════════════════════════════════════════════════════
def _env(cr=None,plo=0,phi=999999,fmi=3,fma=13,spr=None,fr=None,
         rmi=1,rma=11,smi=0.0,sma=999.0,f2mi=0.0,f2ma=999.0):
    def _v(df):
        m = np.ones(len(df), dtype=bool)
        if "WhichRace" in df: m &= df["WhichRace"].between(rmi,rma)
        if "Runners"   in df: m &= df["Runners"].between(fmi,fma)
        if "Purse"     in df: m &= df["Purse"].between(plo,phi)
        if "SumOf1st2Odds" in df: m &= df["SumOf1st2Odds"].between(smi,sma)
        if f2c := next((c for c in ["Fav2Exact","Fav2_odds"] if c in df),None):
            m &= df[f2c].between(f2mi,f2ma)
        if cr and "ChalkYN" in df:
            ck = df["ChalkYN"].str.strip().str.upper()=="Y"
            m &= ck if cr=="Y" else ~ck
        if fr and "FirstRaceYN" in df:
            fk = df["FirstRaceYN"].str.strip().str.upper()=="Y"
            m &= fk if fr=="Y" else ~fk
        if spr and (dc := next((c for c in ["Miles","Distance"] if c in df),None)):
            fl = df[dc].where(df[dc]<=4.0,df[dc]/8.0)
            sp = (fl<0.875).fillna(False)
            m &= sp if spr=="Y" else ~sp
        return m
    f = lambda r: False; f.vec = _v; return f

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
# INSTRUMENT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
INS = {
    # ── TIER 1 ───────────────────────────────────────────────────────────────
    "SB6_S5556_A":     {"t":"T1","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":3.0,"v":1,"ef":_env(fmi=6,fma=6,cr="N",smi=6,sma=8.5,plo=8000,phi=25000)},
    "SB6_S5556_B":     {"t":"T1","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":3.0,"v":1,"ef":_env(fmi=6,fma=6,cr="N",fr="N",smi=6,sma=8.5,plo=8000,phi=25000)},
    "FB4_2":           {"t":"T1","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"v":1,"ef":_env(fmi=4,fma=4,cr="N",smi=4,sma=6,f2mi=2.5,f2ma=4)},
    "FB4_3":           {"t":"T1","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"v":1,"ef":_env(fmi=4,fma=4,cr="N",fr="N",smi=4,sma=6)},
    "FB4_4":           {"t":"T1","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"v":1,"ef":_env(fmi=4,fma=4,cr="N",smi=4,sma=6)},
    "FB4_5":           {"t":"T1","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"v":1,"ef":_env(fmi=4,fma=4,cr="N",spr="Y",smi=4,sma=6)},
    "NM_Sup4455_N6_C": {"t":"T1","tc":7.2, "hf":_h_s4455,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":1,"ef":_env(fmi=6,fma=6,cr="N",f2mi=4)},
    # ── TIER 2 ───────────────────────────────────────────────────────────────
    "TRI145":          {"t":"T2","tc":18,  "hf":_h_t145, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":13,"ew":2.8,"v":1,"ef":_env(smi=9,f2mi=2.5)},
    "SB6_S5556_C":     {"t":"T2","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.5,"v":1,"ef":_env(fmi=6,fma=6,cr="N",spr="Y",smi=6,sma=8.5,plo=8000,phi=35000)},
    "FB5_2":           {"t":"T2","tc":12,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":5, "mx":5, "ew":2.5,"v":1,"ef":_env(fmi=5,fma=5,cr="N",f2mi=4)},
    "NM_Sup4455_N6_D": {"t":"T2","tc":7.2, "hf":_h_s4455,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":1,"ef":_env(fmi=6,fma=6,cr="N",f2mi=4,smi=6,sma=9.5)},
    "NM_Sup4456_N6_D": {"t":"T2","tc":10.8,"hf":_h_s4456,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":1,"ef":_env(fmi=6,fma=6,cr="N",f2mi=4,smi=6,sma=9.5)},
    "NM_Supr3666_N6_D":{"t":"T2","tc":18,  "hf":_h_s3666,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"v":1,"ef":_env(fmi=6,fma=6,cr="N",f2mi=4,smi=6,sma=9.5)},
    "FB6_1":           {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,"ef":_env(fmi=6,fma=6,cr="N",fr="N",spr="Y",f2mi=4)},
    "FB6_2":           {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,"ef":_env(fmi=6,fma=6,cr="N",fr="N",rmi=4,rma=8,f2mi=4)},
    "FB6_3":           {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,"ef":_env(fmi=6,fma=6,cr="N",rmi=4,rma=8,f2mi=4)},
    "FB6_4":           {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,"ef":_env(fmi=6,fma=6,cr="N",spr="Y",f2mi=4)},
    "FB6_5":           {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,"ef":_env(fmi=6,fma=6,cr="N",fr="N",f2mi=4)},
    "FB6_6":           {"t":"T2","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"v":1,"ef":_env(fmi=6,fma=6,cr="N",f2mi=4)},
    # ── TIER 3 ───────────────────────────────────────────────────────────────
    "FB4_6":           {"t":"T3","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.0,"v":1,"ef":_env(fmi=4,fma=4,rmi=4,rma=8,smi=4,sma=6,f2mi=2.5,f2ma=4)},
    "FB7_1":           {"t":"T3","tc":84,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.0,"v":1,"ef":_env(fmi=7,fma=7,cr="N",fr="N",spr="Y",plo=8000,phi=15000,smi=6,sma=8)},
    "FB7_2":           {"t":"T3","tc":84,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.0,"v":1,"ef":_env(fmi=7,fma=7,cr="N",plo=8000,phi=15000,smi=6,sma=8)},
    "FB7_3":           {"t":"T3","tc":84,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.5,"v":1,"ef":_env(fmi=7,fma=7,cr="N",spr="Y",plo=8000,phi=20000,f2mi=4)},
    "FB7_4":           {"t":"T3","tc":84,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.0,"v":1,"ef":_env(fmi=7,fma=7,cr="N",smi=6,sma=8,f2mi=4)},
    "SB12_S6667_A":    {"t":"T3","tc":48,  "hf":_h_s6667,"pc":"Superf_paid","pm":0.05,"mn":12,"mx":12,"ew":1.5,"v":1,"ef":_env(fmi=12,fma=12,cr="N",spr="N",rmi=4,rma=9,smi=4,sma=8.5,f2mi=4,plo=8000,phi=25000)},
    "SB6_S5556_D":     {"t":"T3","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.5,"v":1,"w":1,"wr":"ASV v2.0 Duplicate Alert — gates identical to SB6_S5556_C","ef":_env(fmi=6,fma=6,cr="N",spr="Y",smi=6,sma=8.5,plo=8000,phi=25000)},
    "SB12_S6678_A":    {"t":"T3","tc":48,  "hf":_h_s6678,"pc":"Superf_paid","pm":0.05,"mn":12,"mx":12,"ew":1.5,"v":1,"w":1,"wr":"Needs n>=200 OOS verification","ef":_env(fmi=12,fma=12,cr="N")},
    # ── BONUS ────────────────────────────────────────────────────────────────
    "BNS_SB6_A_P35":   {"t":"BNS","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":3.0,"ef":_env(fmi=6,fma=6,cr="N",smi=6,sma=8.5,plo=8000,phi=35000)},
    "BNS_SB6_C_P35":   {"t":"BNS","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.5,"ef":_env(fmi=6,fma=6,cr="N",spr="Y",smi=6,sma=8.5,plo=8000,phi=35000)},
    "BNS_SB6_SumWide": {"t":"BNS","tc":18,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":3.0,"ef":_env(fmi=6,fma=6,cr="N",smi=6,sma=9.5,plo=8000,phi=25000)},
    "BNS_SB12_R9":     {"t":"BNS","tc":48,  "hf":_h_s6667,"pc":"Superf_paid","pm":0.05,"mn":12,"mx":12,"ew":1.5,"ef":_env(fmi=12,fma=12,cr="N",spr="N",rmi=4,rma=9,smi=4,sma=8.5,f2mi=4,plo=8000,phi=25000)},
    "BNS_FB5_Route":   {"t":"BNS","tc":12,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":5, "mx":5, "ew":1.5,"ef":_env(fmi=5,fma=5,cr="N",spr="N",f2mi=4)},
    "BNS_FB5_Fav3":    {"t":"BNS","tc":12,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":5, "mx":5, "ew":2.0,"ef":_env(fmi=5,fma=5,cr="N",f2mi=3)},
    "BNS_FB6_Early":   {"t":"BNS","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"ef":_env(fmi=6,fma=6,cr="N",rmi=1,rma=3,f2mi=4)},
    "BNS_FB6_Late":    {"t":"BNS","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"ef":_env(fmi=6,fma=6,cr="N",rmi=9,rma=11,f2mi=4)},
    "BNS_FB6_First":   {"t":"BNS","tc":36,  "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"ef":_env(fmi=6,fma=6,cr="N",fr="Y",f2mi=4)},
    "BNS_FB6_HighPurse":{"t":"BNS","tc":36, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":1.5,"ef":_env(fmi=6,fma=6,cr="N",f2mi=4,plo=25000)},
    "BNS_FB7_HighPurse":{"t":"BNS","tc":84, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":7, "mx":7, "ew":1.0,"ef":_env(fmi=7,fma=7,cr="N",plo=15000,f2mi=4)},
    "BNS_SB8_5556":    {"t":"BNS","tc":24,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":8, "mx":8, "ew":2.0,"ef":_env(fmi=8,fma=8,cr="N",f2mi=4)},
    "BNS_SB9_5556":    {"t":"BNS","tc":30,  "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":9, "mx":9, "ew":2.0,"ef":_env(fmi=9,fma=9,cr="N",f2mi=4)},
    "BNS_SB10_6667":   {"t":"BNS","tc":36,  "hf":_h_s6667,"pc":"Superf_paid","pm":0.05,"mn":10,"mx":10,"ew":2.0,"ef":_env(fmi=10,fma=10,cr="N",f2mi=4)},
    "BNS_NM_4455_Fav3":{"t":"BNS","tc":7.2, "hf":_h_s4455,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"ef":_env(fmi=6,fma=6,cr="N",f2mi=3)},
    "BNS_NM_5567":     {"t":"BNS","tc":12,  "hf":_h_s5567,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"ef":_env(fmi=6,fma=6,cr="N",f2mi=4)},
    "BNS_NM_4466":     {"t":"BNS","tc":9.6, "hf":_h_s4466,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.0,"ef":_env(fmi=6,fma=6,cr="N",f2mi=4)},
    "BNS_TRI_Sum8":    {"t":"BNS","tc":18,  "hf":_h_t145, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":13,"ew":3.0,"ef":_env(smi=8,f2mi=2.5)},
    "BNS_TRI_First":   {"t":"BNS","tc":18,  "hf":_h_t145, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":13,"ew":2.0,"ef":_env(fr="Y",smi=9,f2mi=2.5)},
    "BNS_TRI_HighPurse":{"t":"BNS","tc":18, "hf":_h_t145, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":13,"ew":2.0,"ef":_env(smi=9,f2mi=2.5,plo=25000)},
    "BNS_FB4_Route":   {"t":"BNS","tc":2.4, "hf":_h_al,   "pc":"Superf_paid","pm":0.05,"mn":4, "mx":4, "ew":1.5,"ef":_env(fmi=4,fma=4,cr="N",spr="N",smi=4,sma=6)},
    "BNS_SB6_RouteOnly":{"t":"BNS","tc":18, "hf":_h_s5556,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":6, "ew":2.5,"ef":_env(fmi=6,fma=6,cr="N",spr="N",smi=6,sma=8.5,plo=8000,phi=25000)},
    "BNS_Tri133":      {"t":"BNS","tc":4,   "hf":_h_t133, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":10,"ew":3.0,"ef":_env(fmi=5,fma=10,cr="N")},
    "BNS_Tri223":      {"t":"BNS","tc":4,   "hf":_h_t223, "pc":"Trif_paid",  "pm":1.0, "mn":5, "mx":10,"ew":3.0,"ef":_env(fmi=5,fma=10,cr="N")},
    "BNS_Sup2244":     {"t":"BNS","tc":8,   "hf":_h_s2244,"pc":"Superf_paid","pm":0.05,"mn":4, "mx":8, "ew":2.5,"ef":_env(fmi=4,fma=8,cr="N",plo=3000,phi=33500)},
    "BNS_Sup3335":     {"t":"BNS","tc":8,   "hf":_h_s3335,"pc":"Superf_paid","pm":0.05,"mn":5, "mx":11,"ew":2.5,"ef":_env(fmi=5,fma=11,cr="N",plo=3500,phi=26500)},
    "BNS_Supr4145":    {"t":"BNS","tc":24,  "hf":_h_r4145,"pc":"Superf_paid","pm":0.05,"mn":5, "mx":11,"ew":2.0,"ef":_env(fmi=5,fma=11,cr="N",plo=2500,phi=30500)},
    "BNS_Sup1345":     {"t":"BNS","tc":4,   "hf":_h_s1345,"pc":"Superf_paid","pm":0.05,"mn":5, "mx":13,"ew":3.0,"ef":_env(fmi=5,fma=13,cr="N",plo=12000,phi=35000)},
    "BNS_Sup3444":     {"t":"BNS","tc":16,  "hf":_h_s3444,"pc":"Superf_paid","pm":0.05,"mn":4, "mx":11,"ew":2.5,"ef":_env(fmi=4,fma=11,cr="N",plo=2000,phi=32000)},
    "BNS_Sup2255":     {"t":"BNS","tc":8,   "hf":_h_s2255,"pc":"Superf_paid","pm":0.05,"mn":7, "mx":12,"ew":2.0,"ef":_env(fmi=7,fma=12,cr="N",plo=15000,phi=34000)},
    "BNS_Sup2345":     {"t":"BNS","tc":8,   "hf":_h_s2345,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":12,"ew":2.0,"ef":_env(fmi=6,fma=12,cr="N")},
    "BNS_Sup2266":     {"t":"BNS","tc":8,   "hf":_h_s2266,"pc":"Superf_paid","pm":0.05,"mn":7, "mx":12,"ew":2.0,"ef":_env(fmi=7,fma=12,cr="N",plo=3500,phi=35000)},
    # ── APEX ─────────────────────────────────────────────────────────────────
    "APX_Supr3666":    {"t":"APX","tc":4,   "hf":_h_s3666,"pc":"Superf_paid","pm":0.05,"mn":6, "mx":12,"ew":1.5,"q":1,"ef":_env(fmi=6,fma=12,plo=500,phi=34500)},
    "APX_Sup4456":     {"t":"APX","tc":4,   "hf":_h_s4456,"pc":"Superf_paid","pm":0.05,"mn":8, "mx":12,"ew":1.5,"q":1,"ef":_env(fmi=8,fma=12,plo=500,phi=34500)},
    "APX_Sup4455":     {"t":"APX","tc":100, "hf":_h_s4455,"pc":"Superf_paid","pm":0.05,"mn":5, "mx":12,"ew":1.5,"q":1,"ef":_env(fmi=5,fma=12,plo=0,phi=32500)},
    "APX_Supr2555":    {"t":"APX","tc":4,   "hf":_h_s3666,"pc":"Superf_paid","pm":0.05,"mn":5, "mx":12,"ew":1.5,"q":1,"ef":_env(fmi=5,fma=12,plo=1000,phi=32500)},
    "APX_Supr4444":    {"t":"APX","tc":24,  "hf":_h_s4455,"pc":"Superf_paid","pm":0.05,"mn":4, "mx":12,"ew":1.5,"q":1,"ef":_env(fmi=4,fma=12,plo=2500,phi=35000)},
}

INST_EN = {k: True for k in INS}
for _r in ["SB11_S6666_A","Supr2555_Wide","Sup4456_Wide","Supr3666_Wide","SB_Sup5556_R7to9"]:
    INST_EN[_r] = False

_T1 = {"SB6_S5556_A","SB6_S5556_B","FB4_2","FB4_3","FB4_4","FB4_5","NM_Sup4455_N6_C"}
_T2 = {"TRI145","SB6_S5556_C","FB5_2","NM_Sup4455_N6_D","NM_Sup4456_N6_D",
       "NM_Supr3666_N6_D","FB6_1","FB6_2","FB6_3","FB6_4","FB6_5","FB6_6"}
_T3 = {"FB4_6","FB7_1","FB7_2","FB7_3","FB7_4","SB12_S6667_A","SB6_S5556_D","SB12_S6678_A"}
M_A = _T1; M_B = _T1|_T2; M_C = _T1|_T2|_T3
BNS = {k for k,v in INS.items() if v["t"]=="BNS"}
APX = {k for k,v in INS.items() if v["t"]=="APX"}
_TM  = {"T1":0,"T2":1,"T3":2,"T4":3,"BNS":4,"APX":5}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _get_df(f1, f2, cut):
    frames = [pd.read_csv(f, low_memory=False)
              for f in [f1,f2] if os.path.exists(f)]
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if dc := next((c for c in df if "Date" in c or "date" in c), None):
        df = df[pd.to_datetime(df[dc], errors="coerce").dt.date
                >= pd.Timestamp(cut).date()]
    for c in ["Runners","Purse","WhichRace","SumOf1st2Odds",
              "Superf_paid","Trif_paid","Exacta_paid",
              "Superfecta_paid","Trifecta_paid"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "Superfecta_paid" in df and "Superf_paid" not in df:
        df["Superf_paid"] = df["Superfecta_paid"]
    if "Trifecta_paid" in df and "Trif_paid" not in df:
        df["Trif_paid"] = df["Trifecta_paid"]
    wm = pd.Series(False, index=df.index)
    if "Superf_paid" in df and "Exacta_paid" in df:
        wm |= (df["Superf_paid"]>WOW_ABS) | (
            (df["Exacta_paid"]>0) &
            (df["Superf_paid"]/df["Exacta_paid"].replace(0,np.nan)>WOW_RAT))
    if "Trif_paid" in df: wm |= df["Trif_paid"] > WOW_ABS
    if (nw:=int(wm.sum()))>0:
        pc=[c for c in df.columns if c.endswith("_paid") or
            c.endswith("Payout") or c.endswith("_pd")]
        df=df.copy(); df.loc[wm,pc]=np.nan
        print(f"  ⚡ WowSuperWow: nulled {nw:,} outlier rows")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# PnL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def _x_pnl(df, k, i, ft):
    if df.empty: return np.array([], dtype=np.float64)
    m = df["Runners"].between(i["mn"], i.get("mx",999))
    if "WhichRace" in df: m &= df["WhichRace"].between(1,11)
    m &= i["ef"].vec(df) if hasattr(i["ef"],"vec") else df.apply(i["ef"],axis=1).values
    if not m.any(): return np.array([], dtype=np.float64)
    q=np.where(m)[0]; tc,pm,hf,pc=i["tc"],i.get("pm",1.0),i["hf"],i["pc"]
    if pc not in df: return np.full(len(q),-tc,dtype=np.float64)
    pf=df[pc].values[q].astype(float); v=~(np.isnan(pf)|(pf==0))
    j=np.random.default_rng(SEED+abs(hash(k))%1000000).uniform(1-JITTER,1+JITTER,len(q))
    if hf is _h_al:
        pnl=np.where(v,pf*pm*j-tc,-tc).astype(np.float64)
    elif (fn:=_VH.get(hf)) and "RANKED_RESULTS" in df:
        try:
            s=df["RANKED_RESULTS"].iloc[q].str.split(expand=True).iloc[:,:4]
            while s.shape[1]<4: s[s.shape[1]]=np.nan
            p=[pd.to_numeric(s.iloc[:,x],errors="coerce").values for x in range(4)]
            h=fn(*p)&~(np.isnan(p[0])|np.isnan(p[1])|np.isnan(p[2])|np.isnan(p[3]))
            pnl=np.where(v&h,pf*pm*j-tc,-tc).astype(np.float64)
        except Exception: pnl=np.full(len(q),-tc,dtype=np.float64)
    else: pnl=np.full(len(q),-tc,dtype=np.float64)
    if ft>0 and (wm:=pnl>0).sum()>=max(10,int(1/ft)):
        pnl=np.minimum(pnl,float(np.percentile(pnl[wm],(1-ft)*100)))
    return pnl

def _ssn_filt(df, md, wd=365):
    if not md: return df
    dc=next((c for c in df if "Date" in c or "date" in c),None)
    if not dc: return df
    dts=pd.to_datetime(df[dc],errors="coerce")
    try: s_dt=pd.Timestamp(f"2001-{md}")
    except Exception: return df
    sd,ed=s_dt.day_of_year,s_dt.day_of_year+wd-1
    rdoy=dts.dt.dayofyear.where(~(dts.dt.is_leap_year&(dts.dt.month>2)),dts.dt.dayofyear-1)
    m=(rdoy>=sd)&(rdoy<ed) if ed<=365 else (rdoy>=sd)|(rdoy<ed-365)
    return df[m&dts.notna()].copy()

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA KERNEL
# ══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True)
def run_gauntlet_core(npth, mxr, sbr, tgt, rfl, mb,
                      c, fm, ls, m_on, mt, ms,
                      t_on, t_dd, t_du,
                      p_pls, p_sts, t_tbl, pb):
    ap  = np.zeros((npth, mxr+1), dtype=np.float64)
    aph = np.zeros((npth, mxr+1), dtype=np.int8)
    oc  = np.zeros(npth, dtype=np.int8)
    ra  = np.zeros(npth, dtype=np.int32)
    n_in, n_ph = len(p_pls), len(pb)

    for p in prange(npth):
        br, pk, tr = sbr, sbr, 0
        ap[p,0] = br
        cph = n_ph-1
        for i in range(n_ph):
            if pb[i,0] <= br < pb[i,1]: cph=int(pb[i,2]); break
        aph[p,0] = cph

        for r in range(1, mxr+1):
            # Ticket lookup
            tx = np.zeros(6, dtype=np.float64)
            for i in range(len(t_tbl)):
                if t_tbl[i,0] <= br < t_tbl[i,1]:
                    for t in range(6): tx[t]=t_tbl[i,2+t]
                    break
            # ewpd-weighted selection with fp-undershoot guard
            tot=0.0
            for i in range(n_in):
                if tx[int(p_sts[i,4])]>0: tot+=p_sts[i,3]
            if tot==0.0: ap[p,r],aph[p,r]=br,cph; continue
            pv,cs,idx,le=np.random.random()*tot,0.0,0,0
            for i in range(n_in):
                if tx[int(p_sts[i,4])]>0:
                    le=i; cs+=p_sts[i,3]
                    if pv<=cs: idx=i; break
            else: idx=le
            tc=p_sts[idx,2]; nt=min(int(tx[int(p_sts[idx,4])]),int(mb/tc))
            if nt<=0: ap[p,r],aph[p,r]=br,cph; continue
            st=min(float(nt)*tc, br)
            # Tilt
            if br>pk: pk,tr=br,0
            if t_on>0.5 and pk>0.0 and (pk-br)/pk>=t_dd:
                if tr<=0: tr=int(t_du)
            if tr>0: st=min(float(min(nt+1,int(mb/tc)))*tc,br); tr-=1
            # Draw PnL
            pool=p_pls[idx]; pnl=pool[np.random.randint(0,len(pool))]
            if np.random.random()<c or np.random.random()<fm or np.random.random()<ls:
                pnl=-tc
            elif pnl>0 and m_on>0.5 and st>mt:
                pnl=(pnl+tc)*(1.0-min(0.5,(st-mt)*ms))-tc
            br+=pnl*(st/tc); ap[p,r]=br
            cph=n_ph-1
            for i in range(n_ph):
                if pb[i,0]<=br<pb[i,1]: cph=int(pb[i,2]); break
            aph[p,r]=cph
            if br>=tgt or br<=rfl:
                oc[p],ra[p]=1 if br>=tgt else 2, r
                for x in range(r+1,mxr+1): ap[p,x],aph[p,x]=br,cph
                break
        else: ra[p]=mxr
    return ap, aph, oc, ra

# ══════════════════════════════════════════════════════════════════════════════
# POOL → NUMBA PREP
# ══════════════════════════════════════════════════════════════════════════════
def _prep(pmap):
    tp, sl = List(), []
    for k, v in pmap.items():
        inst=INS[k]; tc=inst["tc"]; hm=v>HR_MIN
        tp.append(v.astype(np.float64))
        sl.append([float(hm.mean()) if len(v) else 0.0,
                   float((v[hm]+tc).mean()) if hm.any() else 0.0,
                   tc, inst["ew"], float(_TM[inst["t"]])])
    return tp, np.array(sl, dtype=np.float64), _tt_arr(), _pb_arr()

def _run_core(pmap, so, n=None, sbr=None, tgt=None, rfl=None, mxr=None):
    tp,sl,tt,pb=_prep(pmap); np.random.seed(SEED)
    n=n or N_PTH; sbr=sbr or S_BR; tgt=tgt or TGT; rfl=rfl or R_FLR; mxr=mxr or M_RCS
    return run_gauntlet_core(
        n, mxr, float(sbr), float(tgt), float(rfl), MAX_BET,
        float(so["c"]),float(so["fm"]),float(so["s"]),
        1.0 if so["mi"] else 0.0,float(so["mt"]),float(so["ms"]),
        1.0 if so["ti"] else 0.0,float(so["td"]),float(so["tdu"]),
        tp, sl, tt, pb)

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def _stat(paths, outcomes, races, fbr, slbl, so):
    n=len(paths); ns=outcomes.count("success"); nr=outcomes.count("ruin"); nt=outcomes.count("timeout")
    sr=[x for x,y in zip(races,outcomes) if y=="success"]
    rr=[x for x,y in zip(races,outcomes) if y=="ruin"]
    mb=float(np.median(fbr)); mi=int(np.argmin(np.abs(np.array(fbr)-mb)))
    dd=[float(np.max(np.maximum.accumulate(p)-p)) for p in paths]
    fm=lambda l: int(np.median(l)) if l else None
    fp=lambda l,q: int(np.percentile(l,q)) if len(l)>1 else None
    adv=(1-(1-so["c"])*(1-so["fm"])*(1-so["s"]))*100
    return {
        "n":n,"ns":ns,"nr":nr,"nt":nt,
        "sr":ns/n*100,"rr":nr/n*100,"tr":nt/n*100,
        "mrs":fm(sr),"mrr":fm(rr),"p1s":fp(sr,10),"p9s":fp(sr,90),
        "mds":fm(sr)//RPD if fm(sr) else None,
        "mdr":fm(rr)//RPD if fm(rr) else None,
        "mbr":mb,"midx":mi,"mdd":float(np.median(dd)),
        "pct":{p:round(float(np.percentile(fbr,p)),2) for p in [10,25,50,75,90]},
        "adv":adv,"version":"v6.3.4","edition":"CRYSTAL OMNI",
        "stress_label":so["lbl"],"effective_adverse_rate":adv,
        "max_bet_dollars":MAX_BET,"start_bankroll":S_BR,
        "payout_jitter_frac":JITTER,"ruin_floor":R_FLR,
    }

# ══════════════════════════════════════════════════════════════════════════════
# COMFORT SCORES
# ══════════════════════════════════════════════════════════════════════════════
def _cscr(pmap):
    rows=[]; ah=[float((v>HR_MIN).mean()) for v in pmap.values()]; an=[len(v) for v in pmap.values()]
    mh,mn=max(ah) if ah else 1,max(an) if an else 1
    for k,v in pmap.items():
        i=INS[k]; tc=i["tc"]; n=len(v)
        hr=float((v>HR_MIN).mean()) if n else 0
        ev=float(v.mean()) if n else 0; sd=float(v.std()) if n else 0
        hs=hr/mh; vr=sd/abs(tc) if tc else 0; st=1/(1+vr)
        ss=np.log10(n+1)/np.log10(mn+1) if mn else 0
        es=float(np.clip(ev/abs(tc) if tc else 0,0,3)/3)
        rows.append({"n":k,"t":i["t"],"sz":n,"h":hr*100,"e":ev,"tc":tc,"sd":sd,
                     "c":hs*.35+st*.3+ss*.2+es*.15,
                     "mb":min(int(MAX_BET/tc),10)*tc,
                     "w":i.get("w",0),"b":i["t"]=="BNS","a":i["t"]=="APX","val":i.get("v",0)})
    return sorted(rows,key=lambda x:x["c"],reverse=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHARTING
# ══════════════════════════════════════════════════════════════════════════════
def _sax(a,t=None,x=None,y=None):
    a.set_facecolor(BG); a.tick_params(colors=MUT,labelsize=8)
    for sp in a.spines.values(): sp.set_color(GRD)
    a.grid(color=GRD,ls=":",alpha=0.4)
    if t: a.set_title(t,color=TXT,fontsize=10,fontweight="bold",pad=10)
    if x: a.set_xlabel(x,color=MUT,fontsize=8)
    if y: a.set_ylabel(y,color=MUT,fontsize=8)

def charts(paths, outcomes, fbr, st, pmap, plg, slbl, so):
    sub=f"{slbl} | {so['emo']} {so['lbl']}"

    # Chart 1: Paths
    fig,ax=plt.subplots(figsize=(13,6),facecolor=BG)
    _sax(ax,f"GAUNTLET v6.3.4 CRYSTAL OMNI | ${S_BR:,.0f} → ${TGT:,.0f} | {sub}",
         f"Races (~days@{RPD}/d)","Bankroll ($)")
    ax.set_yscale("symlog",linthresh=100)
    for i in np.random.default_rng(SEED).choice(len(paths),min(300,len(paths)),False):
        c=ACC if outcomes[i]=="success" else RED if outcomes[i]=="ruin" else MUT
        ax.plot(paths[i],color=c,alpha=0.10 if outcomes[i]=="success" else 0.18 if outcomes[i]=="ruin" else 0.05,lw=0.5)
    med=paths[st["midx"]]; ax.plot(med,color=GLD,lw=2.2,zorder=6,label=f"Median (${med[-1]:,.0f})")
    # IQR band
    ml=max(len(p) for p in paths); pad=np.full((len(paths),ml),np.nan)
    for i,p in enumerate(paths): pad[i,:len(p)]=p
    act=np.sum(~np.isnan(pad),axis=0)
    with np.errstate(all="ignore"):
        blo=np.nanpercentile(pad,25,axis=0); bhi=np.nanpercentile(pad,75,axis=0)
    blo[act<20]=bhi[act<20]=np.nan; xs=np.arange(ml); ok=~np.isnan(blo)&~np.isnan(bhi)
    if ok.any(): ax.fill_between(xs[ok],blo[ok],bhi[ok],color=GLD,alpha=0.07,label="P25-P75")
    for m,(lb,col,ls,lw) in {R_FLR:("Ruin",RED,"-.",1.2),TGT:("Target",GLD,"--",1.7)}.items():
        ax.axhline(m,color=col,ls=ls,lw=lw,alpha=0.65)
        ax.text(ml*0.004,m*1.05 if m>0 else m+30,lb,color=col,fontsize=7.5)
    ann=(f"Success: {st['sr']:.1f}%  Ruin: {st['rr']:.1f}%\n"
         f"Median: {st['mrs'] or 'N/A'} races"+(f" (~{st['mds']}d)" if st['mds'] else "")+"\n"
         f"Adverse: {st['adv']:.1f}%  MaxBet: ${MAX_BET:.0f}")
    ax.text(0.98,0.04,ann,transform=ax.transAxes,color=TXT,fontsize=8.5,ha="right",va="bottom",
            bbox=dict(facecolor=GRD,edgecolor=MUT,alpha=0.85,boxstyle="round,pad=0.4"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"${v:,.0f}"))
    ax.legend(facecolor=BG,edgecolor=GRD,labelcolor=TXT,fontsize=8,loc="upper left")
    plt.tight_layout(); plt.savefig("Gauntlet_Paths.png",dpi=150,facecolor=BG,bbox_inches="tight"); plt.close()

    # Chart 2: Distribution + Cumulative
    fig=plt.figure(figsize=(14,6),facecolor=BG); gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.3)
    a1,a2=fig.add_subplot(gs[0]),fig.add_subplot(gs[1])
    _sax(a1,"Final Bankroll Distribution","Bankroll ($)","Paths")
    sh=abs(R_FLR)+1; shf=[b+sh for b in fbr]
    bns=np.logspace(np.log10(max(1,min(shf))),np.log10(max(shf)+1),50)
    for lb,col in [("success",ACC),("timeout",MUT),("ruin",RED)]:
        vs=[b+sh for b,o in zip(fbr,outcomes) if o==lb]
        if vs: a1.hist(vs,bins=bns,color=col,alpha=0.72,label=f"{lb} ({len(vs):,})")
    a1.set_xscale("log"); a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"${v-sh:,.0f}"))
    a1.axvline(TGT+sh,color=GLD,lw=1.8,ls="--",label=f"Target ${TGT:,.0f}")
    a1.axvline(st["mbr"]+sh,color=CYN,lw=1.2,ls=":",label=f"Median ${st['mbr']:,.0f}")
    a1.legend(facecolor=BG,labelcolor=TXT,fontsize=8)
    _sax(a2,"Cumulative Ruin & Success","Race Events","Prob (%)")
    mxr=max(len(x) for x in paths); npth=len(paths)
    for lb,col in [("ruin",RED),("success",ACC)]:
        td=np.array([min(len(x)-1,mxr-1) for x,y in zip(paths,outcomes) if y==lb],dtype=np.intp)
        if len(td):
            cv=np.cumsum(np.bincount(td,minlength=mxr).astype(float))/npth*100
            a2.plot(cv,color=col,lw=2,label=f"{lb.capitalize()} {cv[-1]:.1f}%")
            a2.fill_between(range(mxr),cv,color=col,alpha=0.09)
    if st["mrs"]: a2.axvline(st["mrs"],color=GLD,lw=1.4,ls="--",label=f"Median {st['mrs']:,} races")
    a2.set_xlim(0,mxr); a2.set_ylim(0,105); a2.legend(facecolor=BG,labelcolor=TXT,fontsize=8)
    fig.suptitle(f"GAUNTLET v6.3.4 Outcome Analysis | {sub}",color=TXT,fontsize=11,fontweight="bold",y=1.01)
    plt.tight_layout(); plt.savefig("Gauntlet_Dist.png",dpi=150,facecolor=BG,bbox_inches="tight"); plt.close()

    # Chart 3: Comfort Score
    rc=_cscr(pmap); nms=[x["n"]+(" 🔍" if x["w"] else " 🔥" if x["b"] else " ⚠️" if x["a"] else "") for x in rc]
    ec=[T_COL.get(x["t"],MUT) for x in rc]; ht=["" if x["val"] else "//" for x in rc]
    fig,ax=plt.subplots(figsize=(13,max(6,len(rc)*0.42)),facecolor=BG)
    _sax(ax,f"Comfort Score | {sub}  [Hatched=Discovery]","Score","")
    y=np.arange(len(rc)); mh_=max(x["h"] for x in rc) or 1; mn_=max(x["sz"] for x in rc) or 1
    hw=[x["h"]/mh_*.35 for x in rc]; sw=[1/(1+(x["sd"]/x["tc"] if x["tc"] else 0))*.30 for x in rc]
    ls_=[a+b for a,b in zip(hw,sw)]; aw=[np.log10(x["sz"]+1)/np.log10(mn_+1)*.20 for x in rc]
    le_=[a+b for a,b in zip(ls_,aw)]; ew=[np.clip(x["e"]/x["tc"] if x["tc"] else 0,0,3)/3*.15 for x in rc]
    ax.barh(y,hw,color="#06b6d4",label="HR×0.35"); ax.barh(y,sw,left=hw,color="#10b981",label="Stability×0.30")
    ax.barh(y,aw,left=ls_,color="#8b5cf6",label="Sample×0.20"); ax.barh(y,ew,left=le_,color="#f59e0b",label="EV/Cost×0.15")
    for i,x in enumerate(rc):
        ax.barh(i,x["c"],fill=False,edgecolor=ec[i],hatch=ht[i],lw=1.5,zorder=5)
        ax.text(x["c"]+0.006,i,f" {x['c']:.3f}  HR:{x['h']:.0f}%  N:{x['sz']:,}  EV:{'+' if x['e']>=0 else ''}{x['e']:.2f}  MaxBet:${x['mb']:.2f}",
                color=TXT,va="center",fontsize=7,fontfamily="monospace")
    ax.set_yticks(y); ax.set_yticklabels(nms,color=MUT,fontsize=7.5); ax.invert_yaxis()
    ax.set_xlim(0,1.15); ax.axvline(0.5,color=RED,lw=1,ls="--",alpha=0.6)
    ax.legend(facecolor=BG,edgecolor=GRD,labelcolor=TXT,fontsize=7.5,loc="lower right")
    plt.tight_layout(); plt.savefig("Gauntlet_Comfort.png",dpi=150,facecolor=BG,bbox_inches="tight"); plt.close()

    # Chart 4: Phase Transitions
    fig,ax=plt.subplots(figsize=(13,5),facecolor=BG)
    _sax(ax,f"Phase Transitions | {sub}","Race Events","% of Paths")
    mxr=max(len(x) for x in plg); pm_=np.zeros((len(PHS),mxr))
    for pl in plg:
        for r,ph in enumerate(pl):
            if ph<len(PHS): pm_[ph,r]+=1
    dt=pm_.sum(axis=0); dt[dt==0]=1; pct=pm_/dt*100; yb=np.zeros(mxr)
    for i,(_,_,lbl,col) in enumerate(PHS):
        yt=yb+pct[i]; ax.fill_between(range(mxr),yb,yt,color=col,alpha=0.85,label=lbl.replace("\n"," ")); yb=yt
    ax.set_xlim(0,mxr); ax.set_ylim(0,100)
    ax.legend(facecolor=BG,edgecolor=GRD,labelcolor=TXT,loc="upper right",fontsize=9)
    plt.tight_layout(); plt.savefig("Gauntlet_Phases.png",dpi=150,facecolor=BG,bbox_inches="tight"); plt.close()
    print("  📊 4 Charts saved: Gauntlet_Paths / Dist / Comfort / Phases")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════
def r_rpt(pmap, st, slbl, so):
    SEP="═"*80
    print(f"\n{SEP}\n  GAUNTLET v6.3.4 CRYSTAL OMNI | ${S_BR:,.0f} → ${TGT:,.0f} | {st['n']} paths\n{SEP}")
    print(f"  {so['emo']} {so['lbl']} | Adv: {st['adv']:.1f}% | MaxBet: ${MAX_BET:.0f}")
    print(f"\n  {'OUTCOME':<20} {'COUNT':>6}  {'RATE':>7}")
    print("  "+"─"*36)
    for lb,kn,kr in [("Success","ns","sr"),("Ruin","nr","rr"),("Timeout","nt","tr")]:
        print(f"  {lb:<20} {st[kn]:>6,}  {st[kr]:>6.1f}%")
    print(f"\n  Median to target : {st['mrs'] or 'N/A'}"+(f" races (~{st['mds']}d)" if st['mds'] else ""))
    if st.get("p1s") and st.get("p9s"): print(f"  P10-P90          : {st['p1s']:,} – {st['p9s']:,} races")
    print(f"  Median to ruin   : {st['mrr'] or 'N/A'}"+(f" races (~{st['mdr']}d)" if st['mdr'] else ""))
    print(f"  Median Max DD    : ${st['mdd']:,.0f}")
    print(f"  Median Final BR  : ${st['mbr']:,.0f}")
    print(f"\n  Bankroll percentiles:")
    for pv,v in st["pct"].items(): print(f"    P{pv}: ${v:,.0f}")
    rc=_cscr(pmap)
    val=[x for x in rc if x["val"]]
    if val:
        print(f"\n  ── Validated ({len(val)}) ────────────────────────────────────────────────")
        print(f"  {'Rk':<3} {'Name':<28} {'Score':>6} {'HR%':>5} {'N':>6} {'EV':>8} {'MaxBet':>8} Tier")
        print("  "+"─"*72)
        for i,x in enumerate(val,1):
            wf=" 🔍" if x["w"] else ""; ico={"T1":"🏆","T2":"✅","T3":"💎"}.get(x["t"],"")
            print(f"  {i:<3} {x['n']:<28} {x['c']:.3f} {x['h']:>4.0f}% {x['sz']:>6,} {'+' if x['e']>=0 else ''}{x['e']:>7.2f} ${x['mb']:>6.2f}  {ico}{x['t']}{wf}")
    disc=[x for x in rc if x["b"] or x["a"]]
    if disc:
        pr,mn,el=[],[],[]
        for x in disc:
            if x["sz"]<10 or x["e"]<-x["tc"]: el.append(x["n"])
            elif x["e"]>0 and x["sz"]>=20 and x["h"]>15: pr.append(x["n"])
            else: mn.append(x["n"])
        print(f"\n  ── 🔥 Discovery ({len(disc)}) ─────────────────────────────────────────────")
        if pr:  print(f"  🟢 PROMOTE  ({len(pr):2d}): {', '.join(pr)}")
        if mn:  print(f"  🟡 MONITOR  ({len(mn):2d}): {', '.join(mn)}")
        if el:  print(f"  🔴 ELIMINATE({len(el):2d}): {', '.join(el)}")
        print("  ⚠️  All BONUS/APEX require ASV IS/OOS validation before promotion.")
    wl=[x for x in rc if x["w"]]
    if wl:
        print(f"\n  ── 🔍 Watch List ────────────────────────────────────────────────────────")
        for x in wl:
            pool=pmap.get(x["n"])
            if pool is not None:
                print(f"  🔍 {x['n']:<30} EV: {pool.mean():>+8.2f}  N: {len(pool):>5}")
                print(f"      {INS[x['n']].get('wr','')}")
    print(SEP+"\n")

# ══════════════════════════════════════════════════════════════════════════════
# LADDER TXT EXPORT  — full per-rung detail
# ══════════════════════════════════════════════════════════════════════════════
def _ladder_txt(results, so, slbl, cut_dt):
    now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    adv=1-(1-so["c"])*(1-so["fm"])*(1-so["s"])
    lines=["="*72,
           "  GAUNTLET v6.3.4 CRYSTAL OMNI — 10x VELOCITY LADDER",
           "="*72,
           f"  Generated : {now}",
           f"  Strategy  : {slbl}",
           f"  Stress    : {so['emo']} {so['lbl']}",
           f"  Data cut  : {cut_dt}",
           f"  Paths/rung: {N_PTH:,}",
           f"  Max bet   : ${MAX_BET:.0f}",
           f"  Adverse   : {adv:.1%}","",
           f"  {'Rung':<6} {'Range':<22} {'SR%':>6} {'RR%':>6} {'TR%':>6} {'MedRace':>9} {'MedDay':>8} {'P10':>8} {'P90':>8}",
           "  "+"─"*82]
    cum=1.0
    for i,r in enumerate(results):
        rng=f"${r['s']:,}→${r['t']:,}"
        p10=f"{r['p10']:,}" if r.get("p10") else "N/A"
        p90=f"{r['p90']:,}" if r.get("p90") else "N/A"
        lines.append(f"  {i+1:<6} {rng:<22} {r['sr']:>5.1f}% {r['rr']:>5.1f}% {r.get('tr',0):>5.1f}% {r['m']:>9,} {r['d'] or 'N/A':>8} {p10:>8} {p90:>8}")
        if r["sr"]>0: cum*=r["sr"]/100.0
    lines+=["  "+"─"*82,"",
            f"  Compound SR (independence): {cum*100:.2f}%","",
            "  RUNG DETAIL:","  "+"─"*40]
    for i,r in enumerate(results):
        lines+=[f"  Rung {i+1}: ${r['s']:,} → ${r['t']:,}",
                f"    Success rate : {r['sr']:.1f}%",
                f"    Ruin rate    : {r['rr']:.1f}%",
                f"    Timeout rate : {r.get('tr',0):.1f}%",
                f"    Median races : {r['m']:,}",
                f"    Median days  : {r['d'] or 'N/A'}",
                f"    P10 races    : {r.get('p10') or 'N/A'}",
                f"    P90 races    : {r.get('p90') or 'N/A'}",
                f"    Ruin floor   : ${r['rfl']:,.0f}",""]
    lines.append("="*72)
    with open("Ladder_Results.txt","w") as f: f.write("\n".join(lines))
    print("  💾 Ladder_Results.txt")

# ══════════════════════════════════════════════════════════════════════════════
# SEASONAL SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def _run_seasonal_sweep(df, an, so):
    print(f"\n🗓  SEASONAL SWEEP — {len(SSN_STS)} entry points"); res=[]
    for st_d in SSN_STS:
        sdf=_ssn_filt(df,st_d,SSN_W)
        pm={k:p for k in an if k in INS and len(p:=_x_pnl(sdf,k,INS[k],so["f"]))>0}
        if not pm: res.append({"start":st_d,"sr":0,"rr":100,"med":None,"danger":"🔴 NO DATA"}); print(f"  {st_d} | 🔴 NO DATA"); continue
        tp,sl,tt,pb=_prep(pm); np.random.seed(SEED)
        ap,aph,oc,ra=run_gauntlet_core(SSN_N,50000,S_BR,TGT,R_FLR,MAX_BET,
            float(so["c"]),float(so["fm"]),float(so["s"]),
            1.0 if so["mi"] else 0.0,float(so["mt"]),float(so["ms"]),
            1.0 if so["ti"] else 0.0,float(so["td"]),float(so["tdu"]),tp,sl,tt,pb)
        sr_=ra[oc==1]; m=int(np.median(sr_)) if len(sr_) else None
        sr_pct=np.sum(oc==1)/SSN_N*100; rr_pct=np.sum(oc==2)/SSN_N*100
        danger="🔴 DANGER" if rr_pct>10 else "🟡 CAUTION" if rr_pct>3 else "🟢 SAFE"
        thin=sum(1 for v in pm.values() if len(v)<20)
        print(f"  {st_d} | {danger} Instr:{len(pm)} Thin:{thin} SR:{sr_pct:.1f}% RR:{rr_pct:.1f}% Med:{m or 'N/A'}")
        res.append({"start":st_d,"n_inst":len(pm),"thin":thin,"sr":sr_pct,"rr":rr_pct,"med":m,"danger":danger})
    pd.DataFrame(res).to_csv("Gauntlet_SeasonalSweep.csv",index=False)
    print("  💾 Gauntlet_SeasonalSweep.csv")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*72+"\n 🏁 THE 100x GAUNTLET v6.3.4 CRYSTAL OMNI EDITION\n"+"="*72)
    md=input("\n [1] Gauntlet ($250 → $88k)\n [2] 10x Ladder ($10 → $1M)\n Choice [1]: ").strip() or "1"
    is_lad=md=="2"
    tdy=datetime.date.today()
    e_mo=(tdy-datetime.timedelta(days=548)).strftime("%Y-%m-%d")
    t_mo=(tdy-datetime.timedelta(days=1096)).strftime("%Y-%m-%d")
    ci=input(f"\n Window: [1] 18mo ({e_mo})  [2] 36mo ({t_mo})  [2]: ").strip() or "2"
    cut_dt=e_mo if ci=="1" else t_mo
    si=input("\n Strategy: [1] SAFEST  [2] STANDARD  [3] FULL  [4] TURBO  [5] APEX\n Choice [3]: ").strip() or "3"
    ri=input("\n Stress: [1] VANILLA  [2] HALF  [3] FULL  [4] NIGHTMARE\n Choice [2]: ").strip() or "2"

    # FIX: STRESS_PRESETS (not STS_PRE)
    so=STRESS_PRESETS[{"1":"VANILLA","2":"HALF","3":"FULL","4":"NIGHTMARE"}[ri]]
    an=M_A if si=="1" else M_B if si=="2" else M_C if si=="3" else M_C|BNS if si=="4" else M_C|BNS|APX
    if si=="5" and input("\n APEX quarantined. Type APEX to confirm: ").strip().upper()!="APEX":
        an=M_C|BNS; si="4"
    slbl=["SAFEST","STANDARD","FULL","TURBO","APEX"][int(si)-1]
    an={k for k in an if INST_EN.get(k,True) and k in INS}

    print("\nLoading data...")
    df=_get_df(TRAIN_FILE,TEST_FILE,cut_dt)
    if df.empty: print("❌ No data loaded."); return

    if SSN_SWP and not is_lad:
        _run_seasonal_sweep(df,an,so); return

    print("Extracting PnL pools...")
    pmap={}
    for k in tqdm(sorted(an), desc="Instruments", leave=False):
        if k not in INS: continue
        pnl=_x_pnl(df,k,INS[k],so["f"])
        if len(pnl)>0:
            hm=pnl>HR_MIN; mb=min(int(MAX_BET/INS[k]["tc"]),10)*INS[k]["tc"]
            ico={"T1":"🏆","T2":"✅","T3":"💎","BNS":"🔥","APX":"⚠️"}.get(INS[k]["t"],"")
            disc="" if INS[k].get("v",0) else " [DISC]"
            print(f"  {ico} {k:<32}: {len(pnl):>5,} events  EV:{pnl.mean():>+8.2f}  HR:{hm.mean()*100:>5.1f}%  MaxBet:${mb:.2f}{disc}")
            pmap[k]=pnl
        else:
            print(f"  ⚠  {k:<32}: 0 events — skipped")
    if not pmap: print("❌ No PnL events extracted."); return

    tp,sl,tt,pb=_prep(pmap)

    if is_lad:
        print(f"\n🚀 Running 2x Velocity Ladder ({N_PTH} paths/rung, {len(LDR_RNGS)} rungs)...\n")
        results=[]
        for i,b in tqdm(enumerate(LDR_RNGS), total=len(LDR_RNGS), desc="Rungs", unit="rung", ncols=60):
            tgt_l=float(b*10); rfl_l=max(5.0,b/2.0)
            print(f"  Rung {i+1}: ${b:,} → ${b*10:,}  (floor: ${rfl_l:,.0f})  ",end="",flush=True)
            np.random.seed(SEED)
            ap,aph,oc,ra=run_gauntlet_core(
                N_PTH,50000,float(b),tgt_l,rfl_l,MAX_BET,
                float(so["c"]),float(so["fm"]),float(so["s"]),
                1.0 if so["mi"] else 0.0,float(so["mt"]),float(so["ms"]),
                1.0 if so["ti"] else 0.0,float(so["td"]),float(so["tdu"]),
                tp,sl,tt,pb)
            ns=int(np.sum(oc==1)); nr=int(np.sum(oc==2)); nt=int(np.sum(oc==0))
            sr_=ra[oc==1]
            m=int(np.median(sr_)) if len(sr_) else 0
            p10=int(np.percentile(sr_,10)) if len(sr_)>1 else None
            p90=int(np.percentile(sr_,90)) if len(sr_)>1 else None
            d=m//RPD if m else None
            sr_p=ns/N_PTH*100; rr_p=nr/N_PTH*100; tr_p=nt/N_PTH*100
            results.append({"s":b,"t":b*10,"rfl":rfl_l,"sr":sr_p,"rr":rr_p,"tr":tr_p,
                            "m":m,"d":d,"p10":p10,"p90":p90,"ns":ns,"nr":nr,"nt":nt})
            print(f"SR:{sr_p:.1f}%  RR:{rr_p:.1f}%  Med:{m:,} races (~{d or '?'}d)")
        # Summary table
        print(f"\n{'═'*72}\n  10x LADDER SUMMARY\n{'═'*72}")
        print(f"  {'Rng':<4} {'Range':<20} {'SR%':>6} {'RR%':>6} {'Median':>8} {'Days':>6}")
        print("  "+"─"*52)
        cum=1.0
        for i,r in enumerate(results):
            print(f"  {i+1:<4} ${r['s']:>12,}→${r['t']:>12,} {r['sr']:>6.1f}%  {r['rr']:>6.1f}%  {r['m']:>10,}  {r['d'] or '?':>6}")
            if r["sr"]>0: cum*=r["sr"]/100.0
        print(f"  {'─'*52}\n  Compound SR: {cum*100:.2f}%\n{'═'*72}")
        _ladder_txt(results,so,slbl,cut_dt)
        pd.DataFrame(results).to_csv("Ladder_Results.csv",index=False)
        print("  💾 Ladder_Results.csv\n\n✅ LADDER COMPLETE.")
        return

    # GAUNTLET MODE
    print(f"\n⚔️  Running Gauntlet ({N_PTH} paths, max {M_RCS:,} races)...")
    np.random.seed(SEED)
    ap,aph,oc,ra=run_gauntlet_core(
        N_PTH,M_RCS,S_BR,TGT,R_FLR,MAX_BET,
        float(so["c"]),float(so["fm"]),float(so["s"]),
        1.0 if so["mi"] else 0.0,float(so["mt"]),float(so["ms"]),
        1.0 if so["ti"] else 0.0,float(so["td"]),float(so["tdu"]),
        tp,sl,tt,pb)
    outcomes=["timeout" if c==0 else "success" if c==1 else "ruin" for c in oc]
    fbr=[float(ap[i,ra[i]]) for i in range(N_PTH)]
    paths=[ap[i,:ra[i]+1] for i in range(N_PTH)]
    plg=[aph[i,:ra[i]+1].tolist() for i in range(N_PTH)]
    st_o=_stat(paths,outcomes,ra.tolist(),fbr,slbl,so)

    print("🎨 Rendering charts...")
    charts(paths,outcomes,fbr,st_o,pmap,plg,slbl,so)
    r_rpt(pmap,st_o,slbl,so)

    print("💾 Saving outputs...")
    out={k:(v if not isinstance(v,dict) else str(v)) for k,v in st_o.items()}
    with open("Gauntlet_Results.json","w") as f: json.dump(out,f,indent=4)
    flat={k:v for k,v in out.items() if k!="pct"}
    for pv,v in st_o["pct"].items(): flat[f"br_P{pv}"]=v
    pd.DataFrame([flat]).to_csv("Gauntlet_Results.csv",index=False)
    print("  ✓ Gauntlet_Results.json\n  ✓ Gauntlet_Results.csv\n\n✅ THE 100x GAUNTLET v6.3.4 CRYSTAL OMNI COMPLETE.")

if __name__ == "__main__": main()
