#!/usr/bin/env python3
"""FULL-FIELD BOX ANALYZER — Tri333 through Sup8888
=====================================================
For each field size N, boxes ALL N horses.

n=3  → Trifecta  box: N×(N-1)×(N-2)        =   6 combos → $0.60 dime
n=4  → Superfecta box: N×(N-1)×(N-2)×(N-3) =  24 combos → $2.40 dime
n=5  → Superfecta box:                      = 120 combos → $12.00 dime
n=6  → Superfecta box:                      = 360 combos → $36.00 dime
n=7  → Superfecta box:                      = 840 combos → $84.00 dime
n=8  → Superfecta box:                      =1680 combos → $168.00 dime

Since ALL horses are boxed, EVERY race is degenerate (100% HR).
Edge comes entirely from payout size vs ticket cost.

Filter dimensions (all combinations, depth 1-4):
  Purse_Bin, Miles_Bin, Race_Bin,
  Chalk_Bin, First_Bin, Sum_Bin, Fav2_Bin

Outputs:
  Console   : degenerate report · field audit · gate rankings
  CSVs      : FullBox_n{N}_Gates.csv    (one per field size)
              FullBox_n{N}_Equity.csv   (one per field size)
              FullBox_Combined_TopGates.csv
  JSON      : FullBox_Diagnostics.json
  PNGs      : FullBox_nn_n{N}_RecFact.png
              FullBox_nn_n{N}_Equity.png
              FullBox_nn_n{N}_Heatmap.png
              FullBox_nn_n{N}_MonteCarlo.png
              FullBox_Dashboard.png
"""

import itertools, json, math, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_FILE = "RaceRecords_Output_2025.csv"
TEST_FILE  = "RaceRecords_Output_2026.csv"

DIME_UNIT  = 0.10
BASE_UNIT  = 2.00      # published payout base (academic reference)

# ── Field sizes under investigation ───────────────────────────────────────────
# Each field size N boxes ALL N horses.
# n=3 → Trifecta (top-3 finish). n=4+ → Superfecta (top-4 finish).
TARGET_FIELD_SIZES = [3, 4, 5, 6, 7, 8]

# ── Bet mechanics — computed per field size ────────────────────────────────────
def get_bet_config(n):
    """
    Return (bet_type, positions_required, n_perms, ticket_cost_dime, label).

    n=3  → Trifecta:  top-3 finish,  6 combos
    n≥4  → Superfecta: top-4 finish,  n×(n-1)×(n-2)×(n-3) combos

    Since all N horses are boxed, HR = 100% for every field size.
    """
    if n == 3:
        bet_type  = "Trifecta"
        positions = 3
        n_perms   = n * (n - 1) * (n - 2)           # 6
    else:
        bet_type  = "Superfecta"
        positions = 4
        n_perms   = n * (n - 1) * (n - 2) * (n - 3)

    ticket_cost = n_perms * DIME_UNIT
    label       = f"{bet_type[:3].upper()}{n}{n}{n}" + (
        f"{n}" if bet_type == "Superfecta" else "")
    return {
        "bet_type":    bet_type,
        "positions":   positions,
        "n_perms":     n_perms,
        "ticket_cost": ticket_cost,
        "label":       label,
        "payout_mult": DIME_UNIT / BASE_UNIT,   # scale $2 payout → dime
    }

# Pre-compute configs for easy reference
BET_CONFIGS = {n: get_bet_config(n) for n in TARGET_FIELD_SIZES}

# ── Data quality ──────────────────────────────────────────────────────────────
SPRINT_CUTOFF            = 0.875
MILES_SENTINEL_THRESHOLD = 6.66

# ── Gate thresholds ───────────────────────────────────────────────────────────
MIN_ELIGIBLE      = 30
MIN_ELIGIBLE_THIN = 10
# No MIN_HITS needed — every race is degenerate (100% HR by construction)
MAX_COMBO_DEPTH   = 4

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_N_SIMS          = 1_000
# Ruin thresholds scale with cost — use 50× and 200× ticket cost as anchors
# These will be overridden per field size in the MC section
MC_RUIN_BASE       = [-50, -200, -500, -2000]   # multiplied by ticket_cost
MC_PRACTICAL_BK    = None    # computed per field size: 500× ticket_cost
MC_BK_MULT         = 3.0

# ── Output limits ─────────────────────────────────────────────────────────────
TOP_N_CONSOLE  = 25
TOP_N_EQUITY   = 8
TOP_N_MC       = 6

# ── Sanity filter ─────────────────────────────────────────────────────────────
SANE_RULES = {
    "Race12+":  lambda r: "Race12+" in r,
    "Fld:<3":   lambda r: any(f"Fld:{x}" in r for x in range(1, 3)),
    "Fld:9+":   lambda r: any(f"Fld:{x}" in r
                              for x in [9,10,11,12]) or "Fld:12+" in r,
}
def _sane(rule):
    return all(not fn(rule) for fn in SANE_RULES.values())

# ── Visual theme ──────────────────────────────────────────────────────────────
BG, GRID, TEXT, MUT     = "#0f172a", "#334155", "#f8fafc", "#475569"
ACCENT, RED, CYAN, GOLD = "#10b981", "#f43f5e", "#06b6d4", "#f59e0b"
PURPLE                  = "#a855f7"
PALETTE = [ACCENT, GOLD, CYAN, RED, PURPLE,
           "#ec4899", "#84cc16", "#f97316", "#818cf8", "#34d399"]

# ══════════════════════════════════════════════════════════════════════════════
# BET CONFIG DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
def print_bet_schedule():
    """Print the full-field box cost schedule at startup."""
    print(f"\n  {'─'*68}")
    print(f"  {'n':>3}  {'Bet':^12}  {'Label':^10}  "
          f"{'Perms':>6}  {'Dime Cost':>10}  {'$2 Cost':>12}")
    print(f"  {'─'*68}")
    for n, cfg in BET_CONFIGS.items():
        print(f"  {n:>3}  {cfg['bet_type']:^12}  {cfg['label']:^10}  "
              f"{cfg['n_perms']:>6,}  "
              f"${cfg['ticket_cost']:>9.2f}  "
              f"${cfg['n_perms'] * BASE_UNIT:>11,.2f}")
    print(f"  {'─'*68}\n")

# ══════════════════════════════════════════════════════════════════════════════
# REQUIRED COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_COLS = {
    "RANKED_RESULTS": "Finishing order by odds rank — hit detection.",
    "Runners":        "Field size — primary filter.",
    "ChalkYN":        "Chalk flag — Chalk_Bin dimension.",
    "WhichRace":      "Race number on card — Race_Bin dimension.",
    "Purse":          "Purse amount — Purse_Bin dimension.",
    "SumOf1st2Odds":  "Sum of top-2 favorite odds — Sum_Bin dimension.",
}

# Payout column varies by bet type — detected dynamically
PAYOUT_COL_CANDIDATES = {
    "Trifecta":  ["Tri_paid", "Trifecta_paid", "Trifecta_Paid",
                  "TriPaid", "tri_paid", "Trif_paid"],
    "Superfecta": ["Superf_paid", "Superfecta_paid", "Superfecta_Paid",
                   "SuperfPaid", "superf_paid", "SF_paid"],
}

def detect_payout_col(df, bet_type):
    """Find the payout column for the given bet type."""
    for c in PAYOUT_COL_CANDIDATES[bet_type]:
        if c in df.columns:
            return c
    # Fallback: search for any column with bet-type keyword
    kw = "tri" if bet_type == "Trifecta" else "sup"
    matches = [c for c in df.columns if kw in c.lower() and "paid" in c.lower()]
    if matches:
        return matches[0]
    raise ValueError(
        f"\n  HARD STOP: No payout column found for {bet_type}.\n"
        f"  Searched: {PAYOUT_COL_CANDIDATES[bet_type]}\n"
        f"  Paid-like columns: "
        f"{[c for c in df.columns if 'paid' in c.lower() or 'pay' in c.lower()]}"
    )

def hard_stop(df, label):
    missing = {c: r for c, r in REQUIRED_COLS.items() if c not in df.columns}
    if not missing:
        return
    lines = [f"\n{'='*70}",
             f"  HARD STOP [{label}]: {len(missing)} required column(s) missing.", ""]
    for col, reason in missing.items():
        similar = [c for c in df.columns
                   if col.lower()[:5] in c.lower()
                   or c.lower()[:5] in col.lower()]
        lines += [f"  MISSING : {col}",
                  f"    Reason  : {reason}",
                  f"    Similar : {similar or 'none'}", ""]
    lines.append(f"{'='*70}")
    raise ValueError("\n".join(lines))

# ==========================================
# FAV2 DETECTION
# ==========================================
def detect_fav2(df):
    candidates = ["Fav2Exact","Fav2_odds","2ndFavOdds",
                  "SecondFavOdds","MLE2","MLOdds2","Fav2ML"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"\n  HARD STOP: No Fav2 column found.\n"
        f"  Searched: {candidates}\n"
        f"  Odds-like columns present: "
        f"{[c for c in df.columns if 'fav' in c.lower() or 'odds' in c.lower()]}"
    )

# ==========================================
# SPRINT FLAG
# ==========================================
def compute_sprint_flag(df, warn):
    def _clean(series):
        raw = pd.to_numeric(series, errors="coerce")
        bad = int((raw >= MILES_SENTINEL_THRESHOLD).sum())
        if bad:
            warn.append(f"{bad:,} Miles sentinel rows (≥{MILES_SENTINEL_THRESHOLD}) "
                        f"nulled — Camarero/bad-data excluded.")
        return raw.where(raw < MILES_SENTINEL_THRESHOLD, np.nan)

    if "Miles" in df.columns:
        df["Under7Furl_flag"] = _clean(df["Miles"]) < SPRINT_CUTOFF
        return df, "Miles"
    if "Distance" in df.columns:
        df["Under7Furl_flag"] = (
            _clean(pd.to_numeric(df["Distance"], errors="coerce") / 8.0)
            < SPRINT_CUTOFF)
        return df, "Distance÷8"
    if "Under7Furl" in df.columns:
        df["Under7Furl_flag"] = (
            df["Under7Furl"].astype(str).str.strip().str.upper() == "Y")
        return df, "Under7Furl pre-computed"

    dist_like = [c for c in df.columns
                 if any(k in c.lower()
                        for k in ["mile","dist","furl","length"])]
    raise ValueError(
        f"\n  HARD STOP: No distance column.\n"
        f"  Searched: Miles, Distance, Under7Furl\n"
        f"  Distance-like columns: {dist_like or 'none'}"
    )

# ==========================================
# PREP DATAFRAME
# ==========================================
def prep_df(df, fav2_col, warn):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().reset_index(drop=True)
    hard_stop(df, "prep")

    for col in ["WhichRace","Purse","Runners","SumOf1st2Odds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df, _ = compute_sprint_flag(df, warn)

    if "FirstRaceYN" not in df.columns:
        df["FirstRaceYN"] = "N"
        warn.append("FirstRaceYN missing — defaulted to 'N'.")

    # ── Bins ──────────────────────────────────────────────────────────────────
    df["Fld_Bin"] = df["Runners"].apply(
        lambda x: f"Fld:{int(x)}" if 3 <= x <= 8 else
                  ("Fld:9+" if x > 8 else "Fld:<3"))

    df["Purse_Bin"] = pd.cut(df["Purse"],
        bins=[-1, 8_000, 15_000, 30_000, np.inf],
        labels=["Purse:<8k","Purse:8-15k","Purse:15-30k","Purse:30k+"])

    df["Miles_Bin"] = df["Under7Furl_flag"].apply(
        lambda x: (f"Sprint(<{SPRINT_CUTOFF}mi)" if x is True
                   else (f"Route(>={SPRINT_CUTOFF}mi)" if x is False
                         else "Miles:unknown")))

    df["Race_Bin"] = pd.cut(df["WhichRace"],
        bins=[-1, 3, 8, 11, np.inf],
        labels=["Early(1-3)","Mid(4-8)","Late(9-11)","Race12+"])

    df["Chalk_Bin"] = df["ChalkYN"].apply(
        lambda x: "Chalk:Y" if str(x).strip().upper() == "Y" else "Chalk:N")

    df["First_Bin"] = df["FirstRaceYN"].apply(
        lambda x: "First:Y" if str(x).strip().upper() == "Y" else "First:N")

    df["Sum_Bin"] = pd.cut(df["SumOf1st2Odds"],
        bins=[-0.01, 4.0, 6.0, 8.0, 11.0, 14.0, np.inf],
        labels=["Sum:<4","Sum:4-6","Sum:6-8",
                "Sum:8-11","Sum:11-14","Sum:14+"])

    if fav2_col and fav2_col in df.columns:
        df["Fav2_Bin"] = pd.cut(
            pd.to_numeric(df[fav2_col], errors="coerce"),
            bins=[-0.01, 2.5, 4.0, np.inf],
            labels=["Fav2:<2.5","Fav2:2.5-4","Fav2:4+"])
    else:
        df["Fav2_Bin"] = "Fav2:unknown"

    return df

# ==========================================
# HIT DETECTION — full-field box, dynamic positions required
# ==========================================
def parse_result(raw, n_positions):
    """
    Parse RANKED_RESULTS into list of integer odds-rank positions.
    We only need the first n_positions tokens (3 for trifecta, 4 for superfecta).
    """
    if pd.isna(raw):
        return None
    tokens = str(raw).strip().split()
    if len(tokens) < n_positions:
        return None
    try:
        return [int(t) for t in tokens[:n_positions]]
    except ValueError:
        return None

def is_fullbox_hit(positions, field_size, n_positions):
    """
    Full-field box hit: the top n_positions finishers are ALL within
    odds ranks 1 through field_size (which is trivially true when
    every horse is boxed — included for explicit correctness).

    For a TRUE full-field box, this is always True when positions is valid.
    We keep the check explicit so the logic is transparent and auditable.
    """
    if positions is None or len(positions) < n_positions:
        return False
    valid_ranks = set(range(1, field_size + 1))
    return set(positions[:n_positions]).issubset(valid_ranks)

# ==========================================
# APPLY BET — per field size, using that size's full-field config
# ==========================================
def apply_fullbox(df, field_size, payout_col):
    """
    Filter to one field size, compute payout and PnL.

    All races are degenerate (100% HR) because we box all horses.
    Payout column differs for n=3 (trifecta) vs n≥4 (superfecta).
    Published payout is on $2.00 base → scale by DIME_UNIT/BASE_UNIT.
    """
    cfg = BET_CONFIGS[field_size]
    sub = df[df["Runners"] == field_size].copy().reset_index(drop=True)
    if sub.empty:
        return sub

    # Payout: use the provided payout_col (detected upstream per bet type)
    if payout_col not in sub.columns:
        sub["Payout"] = 0.0
    else:
        raw_payouts   = pd.to_numeric(sub[payout_col], errors="coerce").fillna(0)
        sub["Payout"] = raw_payouts * cfg["payout_mult"]

    # Hit detection — always True for full-field box, but computed explicitly
    hits = [
        is_fullbox_hit(
            parse_result(r, cfg["positions"]),
            field_size,
            cfg["positions"]
        )
        for r in sub["RANKED_RESULTS"]
    ]
    sub["Is_Hit"]        = hits
    sub["Is_Degenerate"] = True    # always True — full-field box
    sub["Net_PnL"]       = np.where(
        sub["Is_Hit"],
        sub["Payout"] - cfg["ticket_cost"],
        -cfg["ticket_cost"]          # miss only if result unparseable
    )
    sub["Ticket_Cost"]   = cfg["ticket_cost"]
    sub["Bet_Label"]     = cfg["label"]
    return sub

# ==========================================
# DEGENERATE REPORT (every field size is degenerate here)
# ==========================================
def degenerate_report(df_is, df_oos, field_size):
    """
    Full baseline report for one field size.
    Payout buckets scale automatically with ticket cost.
    """
    cfg         = BET_CONFIGS[field_size]
    ticket_cost = cfg["ticket_cost"]
    label_str   = cfg["label"]

    print(f"\n  {'─'*70}")
    print(f"  DEGENERATE — n={field_size}  [{label_str}]  "
          f"(100% hit rate — all {field_size} horses boxed)")
    print(f"  {cfg['n_perms']} perms × ${DIME_UNIT:.2f} = ${ticket_cost:.2f}/race  "
          f"| ${cfg['n_perms'] * BASE_UNIT:,.2f} at $2 base (academic)")
    print(f"  {'─'*70}")

    results = {}
    for split_label, sub in [("IS", df_is), ("OOS", df_oos)]:
        if sub is None or sub.empty:
            continue
        pays  = sub["Payout"].values
        nets  = sub["Net_PnL"].values
        elig  = len(sub)
        avg_p = float(pays.mean())
        med_p = float(np.median(pays))
        pnl   = float(nets.sum())
        cost  = elig * ticket_cost
        roi   = pnl / cost * 100
        above = float((pays > ticket_cost).mean() * 100)

        cum    = np.cumsum(nets)
        max_dd = float((np.maximum.accumulate(cum) - cum).max())
        rf     = pnl / max_dd if max_dd > 1e-9 else (
            float("inf") if pnl > 0 else 0.0)

        print(f"\n  [{split_label}]  n={elig:,} races  |  "
              f"Wagered: ${cost:,.2f}  |  "
              f"Cost/race: ${ticket_cost:.2f}")
        print(f"    Avg payout  : ${avg_p:>10.2f}    "
              f"Edge/race : ${avg_p - ticket_cost:>+.2f}")
        print(f"    Med payout  : ${med_p:>10.2f}    "
              f"% above cost: {above:.1f}%")
        print(f"    Total PnL   : ${pnl:>+12,.2f}")
        print(f"    ROI         : {roi:>+7.1f}%")
        print(f"    MaxDD       : ${max_dd:>10,.0f}    "
              f"RecFact   : "
              f"{'∞' if math.isinf(rf) else f'{rf:.2f}'}")

        # Payout distribution — buckets relative to ticket cost
        tc = ticket_cost
        buckets = [0, tc*0.25, tc*0.5, tc, tc*2, tc*5, tc*10, math.inf]
        b_labels = [
            f"<${tc*0.25:.0f}",
            f"${tc*0.25:.0f}-{tc*0.5:.0f}",
            f"${tc*0.5:.0f}-{tc:.0f}",
            f"${tc:.0f}-{tc*2:.0f}",
            f"${tc*2:.0f}-{tc*5:.0f}",
            f"${tc*5:.0f}-{tc*10:.0f}",
            f"${tc*10:.0f}+",
        ]
        print(f"    Payout distribution (relative to ${tc:.2f} cost):")
        for lo, hi, lbl in zip(buckets, buckets[1:], b_labels):
            cnt = int(((pays >= lo) & (pays < hi)).sum())
            pct = cnt / elig * 100
            bar = "█" * min(40, int(pct))
            print(f"      {lbl:>16}  {cnt:>5,}  {pct:>5.1f}%  {bar}")

        results[split_label] = {
            "eligible":        elig,
            "avg_payout":      round(avg_p, 2),
            "med_payout":      round(med_p, 2),
            "total_pnl":       round(pnl, 2),
            "total_wagered":   round(cost, 2),
            "roi_pct":         round(roi, 1),
            "max_dd":          round(max_dd, 2),
            "rec_fact":        round(rf, 3) if not math.isinf(rf) else None,
            "pct_above_cost":  round(above, 1),
        }

    print(f"  {'─'*70}\n")
    return results

# ==========================================
# MASK BUILDER
# ==========================================
def build_mask(df, rule_str):
    conditions = [c.strip() for c in rule_str.split(" + ")]
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        if   cond.startswith("Fld:"):   mask &= df["Fld_Bin"].astype(str)   == cond
        elif cond.startswith("Purse:"): mask &= df["Purse_Bin"].astype(str) == cond
        elif "Sprint" in cond or "Route" in cond:
                                         mask &= df["Miles_Bin"]             == cond
        elif cond in ("Early(1-3)","Mid(4-8)","Late(9-11)","Race12+"):
                                         mask &= df["Race_Bin"].astype(str)  == cond
        elif cond.startswith("Chalk:"): mask &= df["Chalk_Bin"]             == cond
        elif cond.startswith("First:"): mask &= df["First_Bin"]             == cond
        elif cond.startswith("Sum:"):   mask &= df["Sum_Bin"].astype(str)   == cond
        elif cond.startswith("Fav2:"):  mask &= df["Fav2_Bin"].astype(str)  == cond
    return mask

# ==========================================
# GATE STATS HELPER
# ==========================================
def gate_stats(sub, ticket_cost):
    """
    Compute stats for a gate subset.
    All races are degenerate: every race is a hit, every race has a payout.
    Edge comes purely from payout magnitude vs cost.
    """
    elig = len(sub)
    if elig == 0:
        return None

    # For full-field boxes, hits == elig (always), but we check Is_Hit
    # to handle any races with unparseable RANKED_RESULTS
    hits     = int(sub["Is_Hit"].sum())
    pays     = sub.loc[sub["Is_Hit"], "Payout"]
    avg_p    = float(pays.mean()) if hits > 0 else 0.0
    pnl      = float(sub["Net_PnL"].sum())
    cost     = elig * ticket_cost
    roi      = pnl / cost * 100 if cost > 0 else 0.0
    hr       = hits / elig * 100
    be_hr    = 100.0   # always 100% for full-field box — field is fully covered

    # Scale MC practical BK with ticket cost
    practical_bk = max(ticket_cost * 500, 10_000)

    eq      = sub["Net_PnL"].values
    cum_eq  = np.cumsum(eq)
    max_dd  = float((np.maximum.accumulate(cum_eq) - cum_eq).max())
    rf      = (pnl / max_dd) if max_dd > 1e-9 else (
        float("inf") if pnl > 0 else 0.0)
    req_bk  = MC_BK_MULT * max_dd
    prac    = req_bk < practical_bk

    return {
        "Eligible":  elig,
        "Hits":      hits,
        "HR%":       round(hr, 2),
        "BE_HR%":    100.0,           # trivially 100% — full-field
        "AvgPay":    round(avg_p, 2),
        "PnL":       round(pnl, 2),
        "Cost":      round(cost, 2),
        "ROI%":      round(roi, 1),
        "EV_r":      round(pnl / elig, 4),
        "MaxDD":     round(max_dd, 2),
        "RecFact":   round(rf, 3) if not math.isinf(rf) else None,
        "Req_BK":    round(req_bk, 0),
        "Practical": prac,
        "_eq":       eq.tolist(),
    }

# ==========================================
# COMBINATORIC GATE SEARCH
# ==========================================
def run_gates(df_is, df_oos, field_size, label):
    """
    Full combinatoric gate search for one field size.
    All races are degenerate — gates isolate payout-fat environments.
    No hit-rate variation exists (100% HR always).
    """
    cfg         = BET_CONFIGS[field_size]
    ticket_cost = cfg["ticket_cost"]

    dims = ["Purse_Bin","Miles_Bin","Race_Bin",
            "Chalk_Bin","First_Bin","Sum_Bin","Fav2_Bin"]

    active_dims = [
        d for d in dims
        if d in df_is.columns
        and not (df_is[d].astype(str).str.contains("unknown")).all()
    ]

    results = []
    blocked = 0
    thin    = 0

    print(f"  Scanning {label} gates "
          f"(depth 1–{MAX_COMBO_DEPTH}, {len(active_dims)} active dims)…")

    for depth in range(1, MAX_COMBO_DEPTH + 1):
        for combo in itertools.combinations(active_dims, depth):
            combo = list(combo)
            try:
                grp = df_is.groupby(combo, observed=True).agg(
                    _n=("Net_PnL","count")).reset_index()
            except Exception:
                continue

            for _, row in grp.iterrows():
                rule_parts = [str(row[c]) for c in combo]
                rule       = " + ".join(rule_parts)

                if not _sane(rule):
                    blocked += 1
                    continue

                mask_is = build_mask(df_is, rule)
                sub_is  = df_is[mask_is]
                n_is    = len(sub_is)

                if n_is < MIN_ELIGIBLE_THIN:
                    continue

                st_is = gate_stats(sub_is, ticket_cost)
                if st_is is None:
                    continue

                # OOS evaluation
                st_oos = None
                if df_oos is not None and not df_oos.empty:
                    mask_oos = build_mask(df_oos, rule)
                    sub_oos  = df_oos[mask_oos]
                    if len(sub_oos) >= MIN_ELIGIBLE_THIN:
                        st_oos = gate_stats(sub_oos, ticket_cost)

                is_thin = (n_is < MIN_ELIGIBLE)
                if is_thin:
                    thin += 1

                row_out = {
                    "Rule":           rule,
                    "Complexity":     depth,
                    "Thin":           is_thin,
                    "BetLabel":       cfg["label"],
                    "TicketCost":     ticket_cost,
                    # IS stats
                    "IS_Elig":        st_is["Eligible"],
                    "IS_Hits":        st_is["Hits"],
                    "IS_HR%":         st_is["HR%"],
                    "IS_AvgPay":      st_is["AvgPay"],
                    "IS_PnL":         st_is["PnL"],
                    "IS_ROI%":        st_is["ROI%"],
                    "IS_EV_r":        st_is["EV_r"],
                    "IS_MaxDD":       st_is["MaxDD"],
                    "IS_RecFact":     st_is["RecFact"],
                    "IS_Req_BK":      st_is["Req_BK"],
                    "IS_Practical":   st_is["Practical"],
                    # OOS stats
                    "OOS_Elig":       st_oos["Eligible"]  if st_oos else None,
                    "OOS_Hits":       st_oos["Hits"]       if st_oos else None,
                    "OOS_PnL":        st_oos["PnL"]        if st_oos else None,
                    "OOS_ROI%":       st_oos["ROI%"]       if st_oos else None,
                    "OOS_RecFact":    st_oos["RecFact"]    if st_oos else None,
                    "OOS_MaxDD":      st_oos["MaxDD"]      if st_oos else None,
                    "OOS_Req_BK":     st_oos["Req_BK"]     if st_oos else None,
                    # Internal equity series (stripped before CSV export)
                    "_eq_is":         st_is["_eq"],
                }
                results.append(row_out)

    print(f"    Raw gates: {len(results):,}  "
          f"Blocked by sanity: {blocked}  "
          f"Thin (<{MIN_ELIGIBLE}): {thin}")

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    df_res["_rf_sort"] = df_res["IS_RecFact"].apply(
        lambda x: x if x is not None else -999.0)
    df_res = (df_res.sort_values("_rf_sort", ascending=False)
                    .drop(columns=["_rf_sort"])
                    .reset_index(drop=True))
    return df_res

# ==========================================
# PRINT GATE TABLE
# ==========================================
def print_gates(df, title, ticket_cost,
                top_n=TOP_N_CONSOLE, min_is_elig=MIN_ELIGIBLE):
    if df is None or df.empty:
        print(f"  No gates: {title}\n")
        return

    display = df[df["IS_Elig"] >= min_is_elig].head(top_n)
    if display.empty:
        print(f"  No gates meeting MIN_ELIGIBLE={min_is_elig}: {title}\n")
        return

    has_oos = "OOS_PnL" in display.columns and display["OOS_PnL"].notna().any()

    print(f"\n  {'═'*108}")
    print(f"  {title}   [Cost/race: ${ticket_cost:.2f}]")
    print(f"  {'═'*108}")
    hdr = (f"  {'#':>3}  {'Rule':<50}  {'Cx':>2} {'Thn':>3}  "
           f"{'Elig':>5} {'AvgPay':>8} "
           f"{'PnL':>10} {'ROI%':>6} {'MaxDD':>8} "
           f"{'RecFact':>8} {'ReqBK':>9} {'P':>1}")
    if has_oos:
        hdr += f"  {'OEl':>4} {'OPNL':>10} {'OROI':>6} {'ORF':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for rank, (_, r) in enumerate(display.iterrows(), 1):
        rf_s  = (f"{r['IS_RecFact']:>7.2f}"
                 if r["IS_RecFact"] is not None else
                 ("    inf" if (r["IS_PnL"] or 0) > 0 else "    ---"))
        prac  = "✅" if r.get("IS_Practical") else "  "
        thin  = "⚠" if r.get("Thin") else " "
        line  = (f"  {rank:>3}  {str(r['Rule']):<50}  "
                 f"{int(r['Complexity']):>2} {thin:>3}  "
                 f"{int(r['IS_Elig']):>5,} "
                 f"${r['IS_AvgPay']:>7.2f} "
                 f"${r['IS_PnL']:>9,.0f} {r['IS_ROI%']:>5.1f}% "
                 f"${r['IS_MaxDD']:>7,.0f} "
                 f"{rf_s} "
                 f"${r['IS_Req_BK']:>8,.0f} {prac}")
        if has_oos:
            oe  = r.get("OOS_Elig")
            op  = r.get("OOS_PnL")
            or_ = r.get("OOS_ROI%")
            orf = r.get("OOS_RecFact")
            v_icon = ("✅" if pd.notna(op) and (op or 0) > 0
                              and pd.notna(or_) and (or_ or 0) > 0
                       else "🟡" if pd.notna(op) and (op or 0) > 0
                       else "❌" if pd.notna(op) else "⬜")
            line += (
                f"  {int(oe) if pd.notna(oe) else '--':>4} "
                f"{'$'+f'{op:>9,.0f}' if pd.notna(op) else 'N/A':>10} "
                f"{f'{or_:>5.1f}%' if pd.notna(or_) else 'N/A':>6} "
                f"{f'{orf:>6.2f}' if pd.notna(orf) and orf is not None else 'N/A':>7}"
                f" {v_icon}"
            )
        print(line)

    total = len(df[df["IS_Elig"] >= min_is_elig])
    print(f"\n  Showing {min(top_n, total)} of {total} qualifying gates "
          f"(IS_Elig≥{min_is_elig})  "
          f"⚠ = thin (<{MIN_ELIGIBLE} races)\n")

# ==========================================
# MONTE CARLO — ruin thresholds scale with ticket cost
# ==========================================
def get_mc_thresholds(ticket_cost):
    """Scale ruin thresholds proportionally to ticket cost."""
    return [round(m * ticket_cost, 0) for m in MC_RUIN_BASE]

def run_mc(eq_list, ticket_cost, label=""):
    eq = np.array(eq_list, dtype=float)
    n  = len(eq)
    if n < 10:
        return {"error": f"Too few points ({n})", "label": label}

    ruin_thresholds = get_mc_thresholds(ticket_cost)
    practical_bk    = max(ticket_cost * 500, 10_000)

    rng        = np.random.default_rng(seed=42)
    finals     = np.zeros(MC_N_SIMS)
    max_dds    = np.zeros(MC_N_SIMS)
    ruin_flags = {t: np.zeros(MC_N_SIMS, dtype=bool)
                  for t in ruin_thresholds}
    all_paths  = np.zeros((MC_N_SIMS, n))

    for s in range(MC_N_SIMS):
        path         = rng.choice(eq, size=n, replace=True)
        cum          = np.cumsum(path)
        all_paths[s] = cum
        finals[s]    = cum[-1]
        max_dds[s]   = (np.maximum.accumulate(cum) - cum).max()
        for t in ruin_thresholds:
            ruin_flags[t][s] = bool((cum <= t).any())

    sort_idx = np.argsort(finals)
    med_dd   = float(np.median(max_dds))

    return {
        "label":             label,
        "n_races":           n,
        "ticket_cost":       ticket_cost,
        "ruin_thresholds":   ruin_thresholds,
        "ruin_probs":        {t: round(float(ruin_flags[t].mean()*100), 1)
                              for t in ruin_thresholds},
        "median_maxdd":      round(med_dd, 0),
        "p5_final":          round(float(np.percentile(finals,  5)), 0),
        "p50_final":         round(float(np.percentile(finals, 50)), 0),
        "p95_final":         round(float(np.percentile(finals, 95)), 0),
        "required_bankroll": round(MC_BK_MULT * med_dd, 0),
        "practical_deploy":  (MC_BK_MULT * med_dd) < practical_bk,
        "paths_p5":          all_paths[sort_idx[int(0.05*MC_N_SIMS)]].tolist(),
        "paths_p50":         all_paths[sort_idx[int(0.50*MC_N_SIMS)]].tolist(),
        "paths_p95":         all_paths[sort_idx[int(0.95*MC_N_SIMS)]].tolist(),
    }

def print_mc(mc):
    if mc.get("error"):
        print(f"    ⚠ {mc['error']}")
        return
    rp   = mc.get("ruin_probs", {})
    rts  = mc.get("ruin_thresholds", [])
    prac = mc.get("practical_deploy", False)
    tc   = mc.get("ticket_cost", 0)
    print(f"    n={mc['n_races']:,}  cost/race=${tc:.2f}  "
          f"p5=${mc['p5_final']:+,.0f}  "
          f"p50=${mc['p50_final']:+,.0f}  "
          f"p95=${mc['p95_final']:+,.0f}")
    print(f"    MedMaxDD=${mc['median_maxdd']:,.0f}  "
          f"ReqBK=${mc['required_bankroll']:,.0f}  "
          f"{'✅ Practical' if prac else '⚠ High BK'}")
    print(f"    Ruin: "
          + "  ".join(f"@${abs(t):,.0f}={rp.get(t,0):.1f}%"
                      for t in rts))

# ==========================================
# CHARTING UTILITIES
# ==========================================
def _ax(ax, grid_axis="both"):
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linestyle=":", alpha=0.4)

def _save(fname):
    plt.savefig(fname, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  📊 {fname}")

def _fix_cols(df):
    return df.rename(columns={
        "IS_ROI%":  "IS_ROI_",
        "OOS_ROI%": "OOS_ROI_",
        "IS_HR%":   "IS_HR_",
        "OOS_HR%":  "OOS_HR_",
    })

# ==========================================
# CHART A — RecFact bar
# ==========================================
def chart_recfact(gates_df, field_size, fname):
    if gates_df is None or gates_df.empty:
        return
    cfg  = BET_CONFIGS[field_size]
    plot = (gates_df[
        gates_df["IS_RecFact"].notna() &
        (gates_df["IS_PnL"] > 0) &
        (gates_df["IS_Elig"] >= MIN_ELIGIBLE)
    ].head(TOP_N_CONSOLE).iloc[::-1].reset_index(drop=True))
    if plot.empty:
        return

    labels = plot["Rule"].tolist()
    rfs    = plot["IS_RecFact"].tolist()
    pracs  = plot["IS_Practical"].tolist()
    has_oos= plot["OOS_PnL"].notna().any()
    oos_ok = (plot["OOS_PnL"].fillna(-1) > 0).tolist() if has_oos else [False]*len(plot)
    colors = [ACCENT if p and o else GOLD if p else RED
              for p, o in zip(pracs, oos_ok)]

    fig, ax = plt.subplots(
        figsize=(14, max(6, len(labels)*0.42)), facecolor=BG)
    _ax(ax, "x")
    ax.barh(range(len(labels)), rfs, color=colors, height=0.65, alpha=0.88)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=TEXT, fontsize=8)
    ax.axvline(1.0, color=CYAN, lw=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Recovery Factor  (Net PnL ÷ Max Drawdown)",
                  color=MUT, fontsize=10)
    ax.set_title(
        f"n={field_size}  [{cfg['label']}]  Full-Field Box — "
        f"Top Gates by Recovery Factor\n"
        f"${cfg['ticket_cost']:.2f}/race  "
        f"({cfg['n_perms']} perms × ${DIME_UNIT:.2f})  "
        f"Green=practical+OOS✅  Gold=practical  Red=high BK",
        color=TEXT, fontsize=11, fontweight="bold")

    x_max    = max(rfs) if rfs else 1.0
    plot_fix = _fix_cols(plot)
    for i, (rf, r) in enumerate(zip(rfs, plot_fix.itertuples())):
        oe    = getattr(r, "OOS_PnL", None)
        oe_s  = f"OOS:${oe:+,.0f}" if pd.notna(oe) else "OOS:—"
        roi_v = getattr(r, "IS_ROI_", 0) or 0
        ax.text(rf + x_max*0.01, i,
                f"RF={rf:.2f}  ROI={roi_v:+.1f}%  {oe_s}",
                color=TEXT, va="center", fontsize=7)

    patches = [
        plt.Rectangle((0,0),1,1,color=ACCENT,alpha=0.88,label="Practical+OOS✅"),
        plt.Rectangle((0,0),1,1,color=GOLD,  alpha=0.88,label="Practical"),
        plt.Rectangle((0,0),1,1,color=RED,   alpha=0.88,label="High BK"),
    ]
    ax.legend(handles=patches, facecolor=BG, edgecolor=GRID,
              labelcolor=TEXT, fontsize=8, loc="lower right")
    plt.tight_layout()
    _save(fname)

# ==========================================
# CHART B — Equity curves
# ==========================================
def chart_equity(gates_df, df_is, field_size, fname):
    if gates_df is None or gates_df.empty or df_is.empty:
        return
    cfg = BET_CONFIGS[field_size]

    top = (gates_df[
        (gates_df["IS_PnL"] > 0) &
        (gates_df["IS_Elig"] >= MIN_ELIGIBLE) &
        (gates_df["IS_RecFact"].notna())
    ].head(TOP_N_EQUITY))
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    _ax(ax, "both")

    for i, (_, r) in enumerate(top.iterrows()):
        mask   = build_mask(df_is, r["Rule"])
        eq     = df_is.loc[mask, "Net_PnL"].values
        cum_eq = np.cumsum(eq)
        color  = PALETTE[i % len(PALETTE)]
        rf_s   = f"{r['IS_RecFact']:.2f}" if r["IS_RecFact"] is not None else "∞"
        oos_s  = (f" OOS:${r['OOS_PnL']:+,.0f}"
                  if pd.notna(r.get("OOS_PnL")) else "")
        lbl    = (f"[RF={rf_s}]{oos_s} {r['Rule'][:45]}"
                  + ("…" if len(r["Rule"]) > 45 else ""))
        ax.plot(cum_eq, color=color, lw=1.6, alpha=0.9, label=lbl)

    ax.axhline(0, color=MUT, lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Race Index (within gate)", color=MUT, fontsize=10)
    ax.set_ylabel("Cumulative P&L ($)",        color=MUT, fontsize=10)
    ax.set_title(
        f"n={field_size} [{cfg['label']}] — "
        f"Equity Curves: Top {TOP_N_EQUITY} Gates by RecFact",
        color=TEXT, fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
              fontsize=7, loc="upper left", framealpha=0.85)
    plt.tight_layout()
    _save(fname)

# ==========================================
# CHART C — Heatmap: Purse × Sum
# ==========================================
def chart_heatmap_purse_sum(df_is, field_size, fname):
    if df_is.empty:
        return
    cfg = BET_CONFIGS[field_size]

    purse_order = ["Purse:<8k","Purse:8-15k","Purse:15-30k","Purse:30k+"]
    sum_order   = ["Sum:<4","Sum:4-6","Sum:6-8","Sum:8-11","Sum:11-14","Sum:14+"]

    purse_vals  = df_is["Purse_Bin"].astype(str).unique()
    sum_vals    = df_is["Sum_Bin"].astype(str).unique()
    purse_order = [p for p in purse_order if p in purse_vals]
    sum_order   = [s for s in sum_order   if s in sum_vals]

    nrow, ncol = len(purse_order), len(sum_order)
    if nrow == 0 or ncol == 0:
        return

    roi_mat = np.full((nrow, ncol), np.nan)
    rf_mat  = np.full((nrow, ncol), np.nan)
    pnl_mat = np.full((nrow, ncol), np.nan)
    n_mat   = np.zeros((nrow, ncol), dtype=int)
    tc      = cfg["ticket_cost"]

    for pi, pur in enumerate(purse_order):
        for si, sm in enumerate(sum_order):
            sub = df_is[(df_is["Purse_Bin"].astype(str) == pur) &
                        (df_is["Sum_Bin"].astype(str)   == sm)]
            if len(sub) < MIN_ELIGIBLE_THIN:
                continue
            pnl    = float(sub["Net_PnL"].sum())
            cost   = len(sub) * tc
            roi    = pnl / cost * 100 if cost > 0 else np.nan
            cum    = np.cumsum(sub["Net_PnL"].values)
            max_dd = float((np.maximum.accumulate(cum) - cum).max())
            rf     = (pnl / max_dd) if max_dd > 1e-9 else (
                float("inf") if pnl > 0 else np.nan)
            roi_mat[pi, si] = roi
            rf_mat[pi, si]  = min(rf, 50.0) if not math.isinf(rf) else 50.0
            pnl_mat[pi, si] = pnl
            n_mat[pi, si]   = len(sub)

    fig, ax = plt.subplots(
        figsize=(max(10, ncol*1.6), max(5, nrow*1.0)), facecolor=BG)
    ax.set_facecolor(BG)
    vmin = np.nanmin(roi_mat) if not np.all(np.isnan(roi_mat)) else -50
    vmax = np.nanmax(roi_mat) if not np.all(np.isnan(roi_mat)) else 50

    im   = ax.imshow(roi_mat, aspect="auto", cmap="RdYlGn",
                     vmin=min(vmin, -20), vmax=max(vmax, 20))
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors=MUT)
    cbar.set_label("ROI%", color=MUT, fontsize=9)

    ax.set_xticks(range(ncol))
    ax.set_yticks(range(nrow))
    ax.set_xticklabels(sum_order,   color=TEXT, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(purse_order, color=TEXT, fontsize=9)
    ax.set_xlabel("Sum_Bin (SumOf1st2Odds)", color=MUT, fontsize=10)
    ax.set_ylabel("Purse_Bin",               color=MUT, fontsize=10)
    ax.set_title(
        f"n={field_size} [{cfg['label']}] — ROI% Heatmap: Purse × Sum\n"
        f"Cell: ROI% / RF / PnL / N  (blank = <{MIN_ELIGIBLE_THIN} races)  "
        f"Cost/race: ${tc:.2f}",
        color=TEXT, fontsize=11, fontweight="bold")

    for pi in range(nrow):
        for si in range(ncol):
            roi_v = roi_mat[pi, si]
            if np.isnan(roi_v):
                ax.text(si, pi, "—", ha="center", va="center",
                        color=MUT, fontsize=8)
            else:
                rf_s = (f"{rf_mat[pi,si]:.1f}"
                        if not np.isnan(rf_mat[pi,si]) else "—")
                txt  = (f"{roi_v:+.1f}%\nRF={rf_s}\n"
                        f"${pnl_mat[pi,si]:+,.0f}\nn={n_mat[pi,si]}")
                tc_c = "black" if roi_v > 10 else TEXT
                ax.text(si, pi, txt, ha="center", va="center",
                        fontsize=7, color=tc_c, fontweight="bold")

    plt.tight_layout()
    _save(fname)

# ==========================================
# CHART D — Monte Carlo fan chart
# ==========================================
def chart_mc(mc_list, field_size, fname):
    valid = [m for m in mc_list
             if m and "paths_p50" in m and not m.get("error")]
    if not valid:
        return
    cfg = BET_CONFIGS[field_size]

    ncols = min(3, len(valid))
    nrows = math.ceil(len(valid) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7*ncols, 5*nrows), facecolor=BG)
    fig.suptitle(
        f"n={field_size} [{cfg['label']}] — Monte Carlo ({MC_N_SIMS:,} sims)\n"
        f"Cost/race=${cfg['ticket_cost']:.2f}  |  Shaded=90% CI",
        color=TEXT, fontsize=12, fontweight="bold", y=1.01)

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, mc in zip(axes_flat, valid):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.grid(color=GRID, linestyle=":", alpha=0.3)

        p5  = np.array(mc["paths_p5"])
        p50 = np.array(mc["paths_p50"])
        p95 = np.array(mc["paths_p95"])
        x   = np.arange(len(p50))

        ax.fill_between(x, p5, p95, alpha=0.15, color=CYAN)
        ax.plot(x, p50, color=ACCENT, lw=2.0, label="Median")
        ax.plot(x, p5,  color=RED,    lw=1.2, linestyle="--", label="5th pct")
        ax.plot(x, p95, color=CYAN,   lw=1.2, linestyle=":",  label="95th pct")

        rts  = mc.get("ruin_thresholds", [])
        ruin_lvl = rts[1] if len(rts) > 1 else -500
        ax.axhline(ruin_lvl, color=RED, lw=0.8, linestyle="-.", alpha=0.6)
        ax.axhline(0,         color=MUT, lw=0.6, alpha=0.4)

        rp   = mc.get("ruin_probs", {})
        prac = mc.get("practical_deploy", False)
        ax.text(0.02, 0.97,
                f"Ruin@${abs(ruin_lvl):,.0f}: {rp.get(ruin_lvl,0):.1f}%\n"
                f"MedDD: ${mc['median_maxdd']:,.0f}\n"
                f"ReqBK: ${mc['required_bankroll']:,.0f}\n"
                f"{'✅ Practical' if prac else '⚠ High BK'}",
                transform=ax.transAxes, color=RED, fontsize=8,
                va="top", ha="left",
                bbox=dict(facecolor=BG, alpha=0.7, edgecolor=GRID, pad=3))

        ax.set_title(mc.get("label","")[:55], color=TEXT,
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Race index",          color=MUT, fontsize=7)
        ax.set_ylabel("Cumulative P&L ($)",  color=MUT, fontsize=7)
        ax.tick_params(colors=MUT, labelsize=6)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        if ax == list(axes_flat)[0]:
            ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
                      fontsize=6, loc="lower left")

    for ax in list(axes_flat)[len(valid):]:
        ax.set_visible(False)

    plt.tight_layout()
    _save(fname)

# ==========================================
# CHART E — Master Dashboard
# ==========================================
def chart_dashboard(results_by_size, degen_by_size):
    n_sizes = len(results_by_size)
    nrows   = n_sizes + 1

    fig = plt.figure(figsize=(24, 7 * nrows), facecolor=BG)
    fig.suptitle(
        f"Full-Field Box Analyzer — Tri333 through Sup8888\n"
        f"Each field size N boxes ALL N horses  |  "
        f"Every race degenerate (100% HR)  |  "
        f"Edge = payout vs cost",
        color=TEXT, fontsize=14, fontweight="bold", y=0.995)

    gs        = gridspec.GridSpec(nrows, 3, figure=fig, hspace=0.55, wspace=0.38)
    mc_colors = [GOLD, ACCENT, CYAN, PURPLE, RED, "#ec4899"]

    for row_idx, (fsize, res) in enumerate(results_by_size.items()):
        cfg      = BET_CONFIGS[fsize]
        g_profit = res.get("gates_profit", pd.DataFrame())
        df_is    = res.get("df_is",        pd.DataFrame())
        degen    = degen_by_size.get(fsize, {})

        # Left+centre: RecFact bar
        ax_bar = fig.add_subplot(gs[row_idx, :2])
        ax_bar.set_facecolor(BG)
        _ax(ax_bar, "x")
        ax_bar.set_title(
            f"n={fsize} [{cfg['label']}]  ${cfg['ticket_cost']:.2f}/race — "
            f"Top Gates by Recovery Factor",
            color=TEXT, fontsize=11, fontweight="bold")

        if not g_profit.empty:
            top = (g_profit[
                (g_profit["IS_RecFact"].notna()) &
                (g_profit["IS_PnL"] > 0) &
                (g_profit["IS_Elig"] >= MIN_ELIGIBLE)
            ].head(12).iloc[::-1])
            if not top.empty:
                rfs    = top["IS_RecFact"].tolist()
                labels = top["Rule"].str[:45].tolist()
                pracs  = top["IS_Practical"].tolist()
                oos_ok = (top["OOS_PnL"].fillna(-1) > 0).tolist()
                colors = [ACCENT if p and o else GOLD if p else RED
                          for p, o in zip(pracs, oos_ok)]
                ax_bar.barh(range(len(labels)), rfs, color=colors,
                            height=0.65, alpha=0.88)
                ax_bar.set_yticks(range(len(labels)))
                ax_bar.set_yticklabels(labels, color=TEXT, fontsize=8)
                ax_bar.axvline(1.0, color=CYAN, lw=1.2,
                               linestyle="--", alpha=0.7)
                ax_bar.set_xlabel("Recovery Factor", color=MUT, fontsize=9)

        # Right: degenerate summary stats
        ax_r = fig.add_subplot(gs[row_idx, 2])
        ax_r.set_facecolor(BG)
        ax_r.set_title(
            f"n={fsize} Baseline Stats  (100% HR)",
            color=TEXT, fontsize=11, fontweight="bold")
        for sp in ax_r.spines.values(): sp.set_color(GRID)
        ax_r.set_xticks([]); ax_r.set_yticks([])

        tc   = cfg["ticket_cost"]
        is_d  = degen.get("IS",  {})
        oos_d = degen.get("OOS", {})
        stats = [
            ("IS  Eligible",   f"{is_d.get('eligible', 0):,}"),
            ("IS  Avg Pay",    f"${is_d.get('avg_payout', 0):.2f}"),
            ("IS  Edge/race",
             f"${is_d.get('avg_payout',0) - tc:+.2f}"),
            ("IS  ROI%",       f"{is_d.get('roi_pct', 0):+.1f}%"),
            ("IS  RecFact",    f"{is_d.get('rec_fact') or '—'}"),
            ("─"*22, ""),
            ("OOS Eligible",   f"{oos_d.get('eligible', 0):,}"),
            ("OOS Avg Pay",    f"${oos_d.get('avg_payout', 0):.2f}"),
            ("OOS ROI%",       f"{oos_d.get('roi_pct', 0):+.1f}%"),
        ]
        for i, (k, v) in enumerate(stats):
            color = (ACCENT if "+" in v and "—" not in v
                     else RED if "-" in v else TEXT)
            ax_r.text(0.05, 0.93 - i*0.10, f"{k:<18} {v}",
                      transform=ax_r.transAxes,
                      color=color, fontsize=9, fontfamily="monospace")

    # Bottom row: MC comparison across all field sizes
    ax_mc = fig.add_subplot(gs[n_sizes, :])
    ax_mc.set_facecolor(BG)
    _ax(ax_mc, "both")
    ax_mc.set_title(
        f"Monte Carlo — Best Gate per Field Size  [{MC_N_SIMS:,} sims each]",
        color=TEXT, fontsize=11, fontweight="bold")

    plotted = False
    for (fsize, res), color in zip(results_by_size.items(), mc_colors):
        valid = [m for m in res.get("mc", [])
                 if m and "paths_p50" in m and not m.get("error")]
        if valid:
            mc  = valid[0]
            p50 = np.array(mc["paths_p50"])
            p5  = np.array(mc["paths_p5"])
            p95 = np.array(mc["paths_p95"])
            x   = np.arange(len(p50))
            cfg = BET_CONFIGS[fsize]
            ax_mc.fill_between(x, p5, p95, alpha=0.10, color=color)
            ax_mc.plot(x, p50, color=color, lw=2.0,
                       label=f"n={fsize}[{cfg['label']}]: "
                             f"{mc.get('label','')[:30]}")
            ax_mc.plot(x, p5,  color=color, lw=1.0, linestyle="--", alpha=0.6)
            ax_mc.plot(x, p95, color=color, lw=1.0, linestyle=":",  alpha=0.6)
            plotted = True

    if plotted:
        ax_mc.axhline(0, color=MUT, lw=0.6, alpha=0.4)
        ax_mc.set_xlabel("Race index",          color=MUT, fontsize=9)
        ax_mc.set_ylabel("Cumulative P&L ($)",  color=MUT, fontsize=9)
        ax_mc.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_mc.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT,
                     fontsize=8, loc="upper left")

    _save("FullBox_Dashboard.png")

# ==========================================
# JSON EXPORT
# ==========================================
def save_json(payload, path):
    def _clean(obj):
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return obj
    with open(path, "w") as f:
        json.dump(_clean(payload), f, indent=2)
    print(f"  💾 {path}")

# ==========================================
# MAIN
# ==========================================
def main():
    SEP = "=" * 78

    print(f"\n{SEP}")
    print(f"  FULL-FIELD BOX ANALYZER — Tri333 through Sup8888")
    print(f"  Each field size N: box ALL N horses → 100% hit rate")
    print(f"  Edge = payout magnitude vs ticket cost")
    print(f"{SEP}")
    print_bet_schedule()

    warnings_list = []

    # ── Load data ─────────────────────────────────────────────────────────────
    print("  Loading data…")
    try:
        raw_is = pd.read_csv(TRAIN_FILE, low_memory=False)
        print(f"  ✔ IS  : {len(raw_is):,} rows — {TRAIN_FILE}")
    except FileNotFoundError:
        raise FileNotFoundError(f"HARD STOP: Cannot find '{TRAIN_FILE}'")

    try:
        raw_oos = pd.read_csv(TEST_FILE, low_memory=False)
        print(f"  ✔ OOS : {len(raw_oos):,} rows — {TEST_FILE}")
    except FileNotFoundError:
        raw_oos = None
        print("  ⚠  OOS file not found — IS-only mode.")

    hard_stop(raw_is, "IS")
    if raw_oos is not None:
        hard_stop(raw_oos, "OOS")

    fav2_col = detect_fav2(raw_is)
    print(f"  ✔ Fav2 column: {fav2_col}\n")

    df_is  = prep_df(raw_is,  fav2_col, warnings_list)
    df_oos = (prep_df(raw_oos, fav2_col, warnings_list)
              if raw_oos is not None else None)

    if warnings_list:
        print(f"  ⚠  {len(warnings_list)} warning(s):")
        for w in warnings_list: print(f"     {w}")
        print()

    # ── Pre-detect payout columns for each bet type ───────────────────────────
    # Trifecta payout col (for n=3)
    tri_pay_col  = detect_payout_col(raw_is, "Trifecta")
    sup_pay_col  = detect_payout_col(raw_is, "Superfecta")
    print(f"  ✔ Trifecta payout col : {tri_pay_col}")
    print(f"  ✔ Superfecta payout col: {sup_pay_col}\n")

    # Merge payout columns into prepped df
    if tri_pay_col in raw_is.columns:
        df_is[tri_pay_col] = pd.to_numeric(
            raw_is[tri_pay_col], errors="coerce").fillna(0).values
        if df_oos is not None and tri_pay_col in raw_oos.columns:
            df_oos[tri_pay_col] = pd.to_numeric(
                raw_oos[tri_pay_col], errors="coerce").fillna(0).values

    if sup_pay_col in raw_is.columns:
        df_is[sup_pay_col] = pd.to_numeric(
            raw_is[sup_pay_col], errors="coerce").fillna(0).values
        if df_oos is not None and sup_pay_col in raw_oos.columns:
            df_oos[sup_pay_col] = pd.to_numeric(
                raw_oos[sup_pay_col], errors="coerce").fillna(0).values

    # ── Field-size audit ──────────────────────────────────────────────────────
    print("  Field-size distribution (IS):")
    counts = df_is["Runners"].value_counts().sort_index()
    for n, c in counts.items():
        tag = " ← targeted" if n in TARGET_FIELD_SIZES else ""
        print(f"     n={int(n):>2}  {c:>6,} races{tag}")
    print()

    # ==========================================
    # Main loop — one section per field size
    # ==========================================
    degen_by_size   = {}
    results_by_size = {}
    all_combined    = []

    for sec_num, fsize in enumerate(TARGET_FIELD_SIZES, start=1):
        cfg         = BET_CONFIGS[fsize]
        ticket_cost = cfg["ticket_cost"]
        payout_col  = tri_pay_col if fsize == 3 else sup_pay_col

        print(f"\n{'█'*78}")
        print(f"  SECTION {sec_num} — n={fsize}  [{cfg['label']}]")
        print(f"  {cfg['bet_type']}  |  "
              f"{cfg['n_perms']} perms  |  "
              f"${ticket_cost:.2f}/race dime  |  "
              f"${cfg['n_perms']*BASE_UNIT:,.2f}/race $2-base (academic)")
        print(f"  ALL {fsize} horses boxed → 100% HR guaranteed")
        print(f"{'█'*78}\n")

        # Slice and compute
        df_n_is  = apply_fullbox(df_is, fsize, payout_col)
        df_n_oos = (apply_fullbox(df_oos, fsize, payout_col)
                    if df_oos is not None else pd.DataFrame())

        print(f"  IS n={fsize}: {len(df_n_is):,} races   "
              f"OOS n={fsize}: {len(df_n_oos):,} races\n")

        if df_n_is.empty:
            print(f"  ⚠  No IS races for n={fsize} — skipping.\n")
            results_by_size[fsize] = {
                "df_is": df_n_is, "df_oos": df_n_oos,
                "gates": pd.DataFrame(),
                "gates_profit": pd.DataFrame(),
                "gates_deploy": pd.DataFrame(),
                "mc": [],
            }
            degen_by_size[fsize] = {}
            continue

        # Degenerate baseline (every size is degenerate)
        degen = degenerate_report(
            df_n_is,
            df_n_oos if not df_n_oos.empty else None,
            fsize)
        degen_by_size[fsize] = degen

        # Gate search
        gates = run_gates(
            df_n_is,
            df_n_oos if not df_n_oos.empty else None,
            fsize,
            f"n={fsize} [{cfg['label']}]")

        g_profit = pd.DataFrame()
        if not gates.empty:
            g_profit = gates[
                (gates["IS_PnL"] > 0) &
                (gates["IS_Elig"] >= MIN_ELIGIBLE)
            ].copy()

        print_gates(g_profit,
                    f"n={fsize} [{cfg['label']}] — "
                    f"ALL PROFITABLE IS GATES (sorted by RecFact)",
                    ticket_cost)

        # Deployable gates (IS+ AND OOS+)
        g_deploy = pd.DataFrame()
        if not g_profit.empty and "OOS_PnL" in g_profit.columns:
            g_deploy = g_profit[g_profit["OOS_PnL"] > 0].copy()
            if not g_deploy.empty:
                print_gates(g_deploy,
                            f"n={fsize} [{cfg['label']}] — "
                            f"DEPLOYABLE GATES (IS+ AND OOS+)",
                            ticket_cost)

        # Thin watch list
        if not gates.empty:
            g_thin = gates[
                (gates["IS_PnL"] > 0) &
                (gates["IS_Elig"] >= MIN_ELIGIBLE_THIN) &
                (gates["IS_Elig"] < MIN_ELIGIBLE)
            ].copy()
            if not g_thin.empty:
                print_gates(
                    g_thin,
                    f"n={fsize} [{cfg['label']}] — THIN WATCH LIST",
                    ticket_cost,
                    top_n=15, min_is_elig=MIN_ELIGIBLE_THIN)

        # Monte Carlo
        mc_results = []
        mc_source  = g_deploy if not g_deploy.empty else g_profit
        if not mc_source.empty:
            print(f"  Monte Carlo — n={fsize} [{cfg['label']}]  "
                  f"cost/race=${ticket_cost:.2f}\n")
            for _, gr in mc_source.head(TOP_N_MC).iterrows():
                mask = build_mask(df_n_is, gr["Rule"])
                eq   = df_n_is.loc[mask, "Net_PnL"].values.tolist()
                if len(eq) < 10:
                    continue
                mc_r = run_mc(eq, ticket_cost, label=gr["Rule"][:55])
                mc_results.append(mc_r)
                print(f"  {gr['Rule'][:62]}")
                print_mc(mc_r)
                print()

        # Accumulate for combined table
        if not g_profit.empty:
            top = g_profit.head(10).copy()
            top.insert(0, "FieldSize",  fsize)
            top.insert(1, "BetConfig",  cfg["label"])
            all_combined.append(top)

        results_by_size[fsize] = {
            "df_is":        df_n_is,
            "df_oos":       df_n_oos,
            "gates":        gates,
            "gates_profit": g_profit,
            "gates_deploy": g_deploy,
            "mc":           mc_results,
        }

    # ==========================================
    # COMBINED COMPARISON
    # ==========================================
    sec = len(TARGET_FIELD_SIZES) + 1
    print(f"\n{'█'*78}\n  SECTION {sec} — COMBINED TOP-GATE COMPARISON\n{'█'*78}\n")

    combined        = pd.DataFrame()
    combined_sorted = pd.DataFrame()

    if all_combined:
        combined        = pd.concat(all_combined, ignore_index=True)
        combined_sorted = combined.sort_values(
            "IS_RecFact", ascending=False, na_position="last")

        print(f"  {'─'*96}")
        print(f"  COMBINED TOP GATES — all field sizes, sorted by Recovery Factor")
        print(f"  {'─'*96}")
        print(f"  {'n':>3}  {'Label':^10}  {'Rule':<48}  "
              f"{'Cx':>2}  {'Elig':>5} {'ROI%':>6} {'RecFact':>8}  "
              f"{'OOS_PnL':>10} {'OOS_ROI':>8}")
        print(f"  {'─'*96}")
        for _, r in combined_sorted.head(20).iterrows():
            rf_s  = (f"{r['IS_RecFact']:>7.2f}"
                     if r["IS_RecFact"] is not None else "    ---")
            op    = r.get("OOS_PnL")
            or_   = r.get("OOS_ROI%")
            oos_p = f"${op:>9,.0f}" if pd.notna(op) else "        N/A"
            oos_r = f"{or_:>7.1f}%"  if pd.notna(or_) else "      N/A"
            lbl   = str(r.get("BetConfig",""))
            print(f"  {int(r['FieldSize']):>3}  {lbl:^10}  "
                  f"{str(r['Rule']):<48}  "
                  f"{int(r['Complexity']):>2}  "
                  f"{int(r['IS_Elig']):>5,} {r['IS_ROI%']:>5.1f}% "
                  f"{rf_s}  {oos_p} {oos_r}")
        print()

    # ==========================================
    # EXPORTS
    # ==========================================
    sec += 1
    print(f"{'█'*78}\n  SECTION {sec} — EXPORTS\n{'█'*78}\n")

    def _save_csv(df, path, internal_cols=("_eq_is",)):
        if df is None or df.empty:
            return
        save_cols = [c for c in df.columns if c not in internal_cols]
        df[save_cols].to_csv(path, index=False)
        print(f"  💾 {path}  ({len(df):,} rows)")

    for fsize, res in results_by_size.items():
        _save_csv(res["gates"], f"FullBox_n{fsize}_Gates.csv")

    for fsize, res in results_by_size.items():
        g_profit = res["gates_profit"]
        df_n_is  = res["df_is"]
        if g_profit.empty or df_n_is.empty:
            continue
        tc = BET_CONFIGS[fsize]["ticket_cost"]
        eq_rows = []
        for _, gr in g_profit.head(TOP_N_EQUITY).iterrows():
            mask = build_mask(df_n_is, gr["Rule"])
            eq   = df_n_is.loc[mask, "Net_PnL"].values
            for idx_i, val in enumerate(eq):
                eq_rows.append({
                    "FieldSize":   fsize,
                    "BetLabel":    BET_CONFIGS[fsize]["label"],
                    "TicketCost":  tc,
                    "Gate":        gr["Rule"],
                    "Race_idx":    idx_i,
                    "Net_PnL":     val,
                    "Cum_PnL":     float(np.cumsum(eq)[idx_i]),
                })
        if eq_rows:
            path = f"FullBox_n{fsize}_Equity.csv"
            pd.DataFrame(eq_rows).to_csv(path, index=False)
            print(f"  💾 {path}  ({len(eq_rows):,} rows)")

    if not combined_sorted.empty:
        _save_csv(combined_sorted, "FullBox_Combined_TopGates.csv")

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_num = 1
    for fsize, res in results_by_size.items():
        g_profit = res["gates_profit"]
        df_n_is  = res["df_is"]
        mc_list  = res["mc"]
        g_fixed  = (_fix_cols(g_profit)
                    if not g_profit.empty else pd.DataFrame())

        chart_recfact(g_fixed, fsize,
                      f"FullBox_{chart_num:02d}_n{fsize}_RecFact.png")
        chart_num += 1

        chart_equity(g_profit, df_n_is, fsize,
                     f"FullBox_{chart_num:02d}_n{fsize}_Equity.png")
        chart_num += 1

        chart_heatmap_purse_sum(df_n_is, fsize,
                                f"FullBox_{chart_num:02d}_n{fsize}_Heatmap.png")
        chart_num += 1

        if mc_list:
            chart_mc(mc_list, fsize,
                     f"FullBox_{chart_num:02d}_n{fsize}_MonteCarlo.png")
            chart_num += 1

    chart_dashboard(results_by_size, degen_by_size)

    # ── Diagnostics JSON ──────────────────────────────────────────────────────
    diag = {
        "generated":          datetime.now().isoformat(timespec="seconds"),
        "script":             "Full-Field Box Analyzer — Tri333 through Sup8888",
        "train_file":         TRAIN_FILE,
        "test_file":          TEST_FILE,
        "bet_schedule":       {n: {k: v for k, v in cfg.items()
                                   if k != "payout_mult"}
                               for n, cfg in BET_CONFIGS.items()},
        "config": {
            "target_field_sizes": TARGET_FIELD_SIZES,
            "dime_unit":          DIME_UNIT,
            "base_unit":          BASE_UNIT,
            "sprint_cutoff":      SPRINT_CUTOFF,
            "miles_sentinel":     MILES_SENTINEL_THRESHOLD,
            "min_eligible":       MIN_ELIGIBLE,
            "min_eligible_thin":  MIN_ELIGIBLE_THIN,
            "max_combo_depth":    MAX_COMBO_DEPTH,
            "mc_n_sims":          MC_N_SIMS,
            "mc_ruin_base_mult":  MC_RUIN_BASE,
            "mc_bk_mult":         MC_BK_MULT,
            "note":               "All field sizes are degenerate. "
                                  "HR=100% always. Edge = payout vs cost.",
        },
        "degenerate_baselines": degen_by_size,
        "warnings":             warnings_list,
    }
    for fsize, res in results_by_size.items():
        diag[f"n{fsize}_is_races"]    = len(res["df_is"])
        diag[f"n{fsize}_oos_races"]   = len(res["df_oos"])
        diag[f"n{fsize}_gates_found"] = len(res["gates"])
        diag[f"n{fsize}_deployable"]  = len(res["gates_deploy"])
        diag[f"n{fsize}_mc"]          = res["mc"]

    save_json(diag, "FullBox_Diagnostics.json")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    total_pngs = sum(
        3 + (1 if res["mc"] else 0)
        for res in results_by_size.values()
    ) + 1

    total_csvs = (len(TARGET_FIELD_SIZES) * 2
                  + (1 if not combined_sorted.empty else 0))

    print(f"\n{SEP}")
    print(f"  ✅  Full-Field Box Analysis Complete")
    print(f"      {'n':>3}  {'Label':^10}  {'Cost/race':>10}  "
          f"{'IS':>7}  {'OOS':>7}  {'Gates':>7}  {'Deploy':>7}  {'MC':>5}")
    for fsize, res in results_by_size.items():
        cfg = BET_CONFIGS[fsize]
        print(f"      {fsize:>3}  {cfg['label']:^10}  "
              f"${cfg['ticket_cost']:>9.2f}  "
              f"{len(res['df_is']):>7,}  "
              f"{len(res['df_oos']):>7,}  "
              f"{len(res['gates']):>7,}  "
              f"{len(res['gates_deploy']):>7,}  "
              f"{len(res['mc']):>5,}")
    print(f"      Outputs: {total_pngs} PNGs · {total_csvs} CSVs · 1 JSON")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
