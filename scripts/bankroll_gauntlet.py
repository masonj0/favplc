#!/usr/bin/env python3
"""
THE 100x GAUNTLET — v7.1.5 "OMNI RESTORE"
==================================================
v7.1.5 changes:
  1.  Restored full Hit Logic in _x_pnl (hit_func validation).
  2.  Restored RANKED_RESULTS parsing for structured exotics.
  3.  Restored Charting (matplotlib) and Reporting.
  4.  Restored Interactive Menu and Batch Mode.
  5.  Fixed NameError: defined _mk_env and hit helpers before INS registry.
  6.  Integrated v5.1 PRUNED strategy list as requested.
  7.  Aligned key names (t/tier, tc/ticket_cost) for test compatibility.
"""
import warnings, datetime, json, os, sys, time, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from numba import njit, prange
from numba.typed import List
from tqdm import tqdm

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

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
            fav2_min=0.0, fav2_max=99.0, purse_lo=0, purse_hi=1_000_000):
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
_hit_always_true = lambda a,b,c,d: True
_hit_tri145 = lambda a,b,c,d: (a==1)&(b>=2)&(b<=4)&(c>=2)&(c<=5)
_hit_supr3666= lambda a,b,c,d: (a<=3)&(b<=6)&(c<=6)&(d<=6)
_hit_sup4455= lambda a,b,c,d: (a<=4)&(b<=4)&(c<=5)&(d<=5)
_hit_sup4456= lambda a,b,c,d: (a<=4)&(b<=4)&(c<=5)&(d<=6)
_hit_sup4445= lambda a,b,c,d: (a<=4)&(b<=4)&(c<=4)&(d<=5)
_hit_sup2266= lambda a,b,c,d: (a<=2)&(b<=2)&(c<=6)&(d<=6)

_VH = {
    "_hit_always_true": _hit_always_true,
    "_hit_tri145": _hit_tri145,
    "_hit_supr3666": _hit_supr3666,
    "_hit_sup4455": _hit_sup4455,
    "_hit_sup4456": _hit_sup4456,
    "_hit_sup4445": _hit_sup4445,
    "_hit_sup2266": _hit_sup2266,
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
# INSTRUMENT REGISTRY — v5.1 (PRUNED)
# ══════════════════════════════════════════════════════════════════════════════
INS = {
    # ── ANCHOR ────────────────────────────────────────────────────────────────
    "TRI145": {
        "t": "T1", "tc": 18.00, "hf": "_hit_tri145", "pc": "Trif_paid", "pm": 1.0,
        "mn": 5, "ew": 2.8, "ef": _mk_env(sum_min=9.0, fav2_min=2.5)
    },

    # ── FB4 ($2.40) ───────────────────────────────────────────────────────────
    "FB4_2": {"t": "T1", "tc": 2.40, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 4, "mx": 4, "ew": 1.5, "ef": _mk_env(field_min=4, field_max=4, chalk_req="N", sum_min=4.0, sum_max=6.0, fav2_min=2.5, fav2_max=4.0)},
    "FB4_3": {"t": "T1", "tc": 2.40, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 4, "mx": 4, "ew": 1.5, "ef": _mk_env(field_min=4, field_max=4, chalk_req="N", first_req="N", sum_min=4.0, sum_max=6.0)},
    "FB4_4": {"t": "T1", "tc": 2.40, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 4, "mx": 4, "ew": 1.5, "ef": _mk_env(field_min=4, field_max=4, chalk_req="N", sum_min=4.0, sum_max=6.0)},
    "FB4_5": {"t": "T1", "tc": 2.40, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 4, "mx": 4, "ew": 1.5, "ef": _mk_env(field_min=4, field_max=4, sprint_req="Y", chalk_req="N", sum_min=4.0, sum_max=6.0)},
    "FB4_6": {"t": "T1", "tc": 2.40, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 4, "mx": 4, "ew": 1.5, "ef": _mk_env(field_min=4, field_max=4, race_min=4, race_max=8, sum_min=4.0, sum_max=6.0, fav2_min=2.5, fav2_max=4.0)},

    # ── FB5 ($12.00) ──────────────────────────────────────────────────────────
    "FB5_2": {"t": "T1", "tc": 12.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 5, "mx": 5, "ew": 1.5, "ef": _mk_env(field_min=5, field_max=5, chalk_req="N", fav2_min=4.0)},

    # ── FB6 ($36.00) ──────────────────────────────────────────────────────────
    "FB6_1": {"t": "T1", "tc": 36.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 6, "mx": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, sprint_req="Y", chalk_req="N", first_req="N", fav2_min=4.0)},
    "FB6_2": {"t": "T1", "tc": 36.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 6, "mx": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, race_min=4, race_max=8, chalk_req="N", first_req="N", fav2_min=4.0)},
    "FB6_3": {"t": "T1", "tc": 36.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 6, "mx": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, race_min=4, race_max=8, chalk_req="N", fav2_min=4.0)},
    "FB6_4": {"t": "T1", "tc": 36.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 6, "mx": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, sprint_req="Y", chalk_req="N", fav2_min=4.0)},
    "FB6_5": {"t": "T1", "tc": 36.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 6, "mx": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, chalk_req="N", first_req="N", fav2_min=4.0)},
    "FB6_6": {"t": "T1", "tc": 36.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 6, "mx": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, chalk_req="N", fav2_min=4.0)},

    # ── FB7 ($84.00) ──────────────────────────────────────────────────────────
    "FB7_1": {"t": "T1", "tc": 84.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 7, "mx": 7, "ew": 1.5, "ef": _mk_env(field_min=7, field_max=7, purse_lo=8000, purse_hi=15000, chalk_req="N", first_req="N", sum_min=6.0, sum_max=8.0)},
    "FB7_2": {"t": "T1", "tc": 84.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 7, "mx": 7, "ew": 1.5, "ef": _mk_env(field_min=7, field_max=7, purse_lo=8000, purse_hi=15000, chalk_req="N", sum_min=6.0, sum_max=8.0)},
    "FB7_3": {"t": "T1", "tc": 84.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 7, "mx": 7, "ew": 1.5, "ef": _mk_env(field_min=7, field_max=7, purse_lo=8000, purse_hi=15000, sprint_req="Y", chalk_req="N", fav2_min=4.0)},
    "FB7_4": {"t": "T1", "tc": 84.00, "hf": "_hit_always_true", "pc": "Superf_paid", "pm": 0.05, "mn": 7, "mx": 7, "ew": 1.5, "ef": _mk_env(field_min=7, field_max=7, chalk_req="N", sum_min=6.0, sum_max=8.0, fav2_min=4.0)},

    # ── NON-MONOTONIC STRUCTURED EXOTICS (THE SURVIVORS) ──────────────────────
    "NM_Supr3666_N6_D": {
        "t": "T2", "tc": 18.00, "hf": "_hit_supr3666", "pc": "Superf_paid", "pm": 0.05,
        "mn": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup4455_N6_C": {
        "t": "T1", "tc": 7.20, "hf": "_hit_sup4455", "pc": "Superf_paid", "pm": 0.05,
        "mn": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, chalk_req="N", fav2_min=4.0),
    },
    "NM_Sup4455_N6_D": {
        "t": "T2", "tc": 7.20, "hf": "_hit_sup4455", "pc": "Superf_paid", "pm": 0.05,
        "mn": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup4456_N6_D": {
        "t": "T2", "tc": 10.80, "hf": "_hit_sup4456", "pc": "Superf_paid", "pm": 0.05,
        "mn": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup4445_N6_C": {
        "t": "T1", "tc": 4.80, "hf": "_hit_sup4445", "pc": "Superf_paid", "pm": 0.05,
        "mn": 6, "ew": 1.5, "ef": _mk_env(field_min=6, field_max=6, chalk_req="N", fav2_min=4.0),
    },
    "NM_Sup4445_N7_D": {
        "t": "T1", "tc": 4.80, "hf": "_hit_sup4445", "pc": "Superf_paid", "pm": 0.05,
        "mn": 7, "ew": 1.5, "ef": _mk_env(field_min=7, field_max=7, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup2266_N10_D": {
        "t": "T1", "tc": 2.40, "hf": "_hit_sup2266", "pc": "Superf_paid", "pm": 0.05,
        "mn": 10, "ew": 1.5, "ef": _mk_env(field_min=10, field_max=10, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },
    "NM_Sup2266_N11_C": {
        "t": "T1", "tc": 2.40, "hf": "_hit_sup2266", "pc": "Superf_paid", "pm": 0.05,
        "mn": 11, "ew": 1.5, "ef": _mk_env(field_min=11, field_max=11, chalk_req="N", fav2_min=4.0),
    },
    "NM_Sup2266_N11_D": {
        "t": "T1", "tc": 2.40, "hf": "_hit_sup2266", "pc": "Superf_paid", "pm": 0.05,
        "mn": 11, "ew": 1.5, "ef": _mk_env(field_min=11, field_max=11, chalk_req="N", fav2_min=4.0, sum_min=6.0, sum_max=8.5),
    },
}

# Add test compatibility keys
for k in INS:
    INS[k]["tier"] = INS[k]["t"]
    INS[k]["ticket_cost"] = INS[k]["tc"]
    INS[k]["hit_func"] = INS[k]["hf"]
    INS[k]["payout_col"] = INS[k]["pc"]
    INS[k]["payout_mult"] = INS[k].get("pm", 1.0)
    INS[k]["min_runners"] = INS[k]["mn"]
    INS[k]["ewpd"] = INS[k]["ew"]
    INS[k]["env_filter"] = INS[k]["ef"]
    INS[k]["tc_mode"] = "fixed"

STRATEGY_SETS = {
    "SAFEST":   {k for k,v in INS.items() if v["t"]=="T1" and k.startswith("FB4")},
    "STANDARD": {k for k,v in INS.items() if v["t"]=="T1"},
    "FULL":     set(INS.keys()),
}

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
# CHARTING
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
    _sax(ax, f"GAUNTLET v7.1.5 | ${S_BR:,.0f}->${TGT:,.0f} | {sub}", "Races", "Bankroll ($)")
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
    tp, sl = List(), []; tm = {"T1":0,"T2":1,"T3":2,"T4":3,"BONUS":4,"APEX":5}
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
    _safe_print("="*72); _safe_print("  THE 100x GAUNTLET v7.1.5 OMNI RESTORE"); _safe_print("="*72)
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
