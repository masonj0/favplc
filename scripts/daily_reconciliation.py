#!/usr/bin/env python3
"""
DAILY RACE RECONCILIATION REPORT — v8.0.3
==================================================================
Companion to THE 100x GAUNTLET v8.0.2 GOLD MASTER

Changes from v8.0.2:
  - Removed CSV/JSON export (unused)
  - Added TXT file export: captures full console output
  - Added full RANKED_RESULTS detail block for every fired match
"""

import sys, os, argparse, datetime, json, csv, io
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# SCALING CONSTANTS — must match Gauntlet selections
# ══════════════════════════════════════════════════════════════════════════════
WOW_ABS = 25000.0
WOW_RAT = 200.0
UNIVERSAL_MAX_RUNNERS = 12
UNIVERSAL_MAX_SUM     = 12.0
SPRINT_THRESHOLD_MILES = 0.875

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT TEES — writes to both console AND a buffer for TXT export
# ══════════════════════════════════════════════════════════════════════════════
_output_buffer = io.StringIO()

def _safe_print(text="", **kwargs):
    """Print to console (safe encoding) AND capture to buffer."""
    # Write to buffer first (always UTF-8 safe)
    _output_buffer.write(str(text) + "\n")

    # Console: handle encoding issues
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        safe = str(text)
        for s, d in [("=>","->"),("✓","[HIT]"),("✗","[MISS]"),("·",".")]:
            safe = safe.replace(s, d)
        print(safe, **kwargs)

def _reset_buffer():
    """Clear the output buffer (call at start of each run)."""
    global _output_buffer
    _output_buffer = io.StringIO()

def _flush_to_txt(filename):
    """Write the captured buffer to a TXT file."""
    try:
        content = _output_buffer.getvalue()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        # Print this directly (not via _safe_print to avoid recursion)
        print(f"  [TXT] Report saved => {filename} "
              f"({len(content.splitlines())} lines)")
    except Exception as e:
        print(f"  [TXT] Export failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# HIT FUNCTIONS — exact match to Gauntlet v8.0.2
# ══════════════════════════════════════════════════════════════════════════════
def _h_al(p, n=None):    return True
def _h_t145(p, n=None):  return (len(p) > 2 and p[0] == 1
                                  and p[1] in {2,3,4} and p[2] in {2,3,4,5})
def _h_t245(p, n=None):  return (len(p) > 2 and p[0] <= 2
                                  and p[1] <= 4 and p[1] != p[0]
                                  and p[2] <= 5 and p[2] != p[0] and p[2] != p[1])
def _h_s3214(p, n=None): return (len(p) > 3 and p[0] <= 3 and p[1] <= 2
                                  and p[2] <= 1 and p[3] <= 4)
def _h_s3225(p, n=None): return (len(p) > 3 and p[0] <= 3 and p[1] <= 2
                                  and p[2] <= 2 and p[3] <= 5)
def _h_s4455(p, n=None): return (len(p) > 3 and p[0] <= 4 and p[1] <= 4
                                  and max(p[2:4]) <= 5)
def _h_s4456(p, n=None): return (len(p) > 3 and p[0] <= 4 and p[1] <= 4
                                  and p[2] <= 5 and p[3] <= 6)
def _h_s4466(p, n=None): return (len(p) > 3 and p[0] <= 4 and p[1] <= 4
                                  and max(p[2:4]) <= 6)
def _h_s4444(p, n=None): return (len(p) > 3 and p[0] <= 4 and p[1] <= 4
                                  and p[2] <= 4 and p[3] <= 4)
def _h_s5556(p, n=None): return (len(p) > 3 and max(p[:3]) <= 5 and p[3] <= 6)
def _h_s5567(p, n=None): return (len(p) > 3 and p[0] <= 5 and p[1] <= 5
                                  and p[2] <= 6 and p[3] <= 7)
def _h_s2444(p, n=None): return (len(p) > 3 and p[0] <= 2 and p[1] <= 4
                                  and p[2] <= 4 and p[3] <= 4)
def _h_s2555(p, n=None): return (len(p) > 3 and p[0] <= 2 and p[1] <= 5
                                  and p[2] <= 5 and p[3] <= 5)
def _h_s3666(p, n=None): return (len(p) > 3 and p[0] <= 3 and max(p[1:4]) <= 6)
def _h_s6667(p, n=None): return (len(p) > 3 and max(p[:3]) <= 6 and p[3] <= 7)
def _h_s1234(p, n=None): return (len(p) > 3 and p[0] == 1 and p[1] == 2
                                  and p[2] == 3 and p[3] == 4)
def _h_t135(p, n=None):  return (len(p) > 2 and p[0] == 1
                                  and p[1] <= 3 and p[2] <= 5)
def _h_tria22(p, n=None):
    return (len(p) > 2 and p[0] >= 3
            and p[1] <= 2 and p[2] <= 2 and p[1] != p[2])

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT REGISTRY — v8.0.2 GOLD MASTER (all 45 instruments)
# ══════════════════════════════════════════════════════════════════════════════
INSTRUMENTS = {

    # ── T1A ──────────────────────────────────────────────────────────────────
    "FB4_2": {
        "tier": "T1A",
        "description": "Full-Box Superfecta 4-runner (Non-chalk, Sum 4-7.5, Fav2 2.5-4)",
        "canonical_cost": 48.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 4, "runners_max": 4,
            "chalk": "N",
            "sum_min": 4.0, "sum_max": 7.5,
            "fav2_min": 2.5, "fav2_max": 4.0,
        },
    },
    "FB4_3": {
        "tier": "T1A",
        "description": "Full-Box Superfecta 4-runner (Non-chalk, Non-first, Sum 4-6)",
        "canonical_cost": 48.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 4, "runners_max": 4,
            "chalk": "N", "first_race": "N",
            "sum_min": 4.0, "sum_max": 6.0,
        },
    },
    "FB4_4": {
        "tier": "T1A",
        "description": "Full-Box Superfecta 4-runner (Non-chalk, Sum 4-6)",
        "canonical_cost": 48.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 4, "runners_max": 4,
            "chalk": "N",
            "sum_min": 4.0, "sum_max": 6.0,
        },
    },
    "FB4_5": {
        "tier": "T1A",
        "description": "Full-Box Superfecta 4-runner (Non-chalk, Sprint, Sum 4-6)",
        "canonical_cost": 48.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 4, "runners_max": 4,
            "chalk": "N", "sprint": "Y",
            "sum_min": 4.0, "sum_max": 6.0,
        },
    },
    "FB4_6": {
        "tier": "T1A",
        "description": "Full-Box Superfecta 4-runner (Sum 4-6, Fav2 2.5-4)",
        "canonical_cost": 48.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 4, "runners_max": 4,
            "sum_min": 4.0, "sum_max": 6.0,
            "fav2_min": 2.5, "fav2_max": 4.0,
        },
    },
    "Sup3214_R5": {
        "tier": "T1A",
        "description": "Pattern Superfecta 3-2-1-4 (5-runner, Non-chalk, Sum>=3)",
        "canonical_cost": 2.0, "pool": "Superf_paid", "hf": _h_s3214,
        "gates": {
            "runners_min": 5, "runners_max": 5,
            "chalk": "N",
            "sum_min": 3.0,
        },
    },
    "Sup3214_R6": {
        "tier": "T1A",
        "description": "Pattern Superfecta 3-2-1-4 (6-runner, Sum>=3)",
        "canonical_cost": 2.0, "pool": "Superf_paid", "hf": _h_s3214,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "sum_min": 3.0,
        },
    },
    "TRI145": {
        "tier": "T1A",
        "description": "Pattern Trifecta 1-4-5 (Sum>=9, Fav2>=2.5)",
        "canonical_cost": 18.0, "pool": "Trif_paid", "hf": _h_t145,
        "gates": {
            "runners_min": 5, "runners_max": 13,
            "sum_min": 9.0,
            "fav2_min": 2.5,
        },
    },
    "TRI245": {
        "tier": "T1A",
        "description": "Pattern Trifecta 2-4-5 Top-2-Fav win (Sum>=9, Fav2>=2.5)",
        "canonical_cost": 36.0, "pool": "Trif_paid", "hf": _h_t245,
        "gates": {
            "runners_min": 5, "runners_max": 13,
            "sum_min": 9.0,
            "fav2_min": 2.5,
        },
    },

    # ── T1B ──────────────────────────────────────────────────────────────────
    "SB6_S5556_A": {
        "tier": "T1B",
        "description": "Pattern 5-5-5-6 (6-runner, Non-chalk, Sum 6-8.5, Purse 8k-25k)",
        "canonical_cost": 360.0, "pool": "Superf_paid", "hf": _h_s5556,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "sum_min": 6.0, "sum_max": 8.5,
            "purse_min": 8000, "purse_max": 25000,
        },
    },
    "SB6_S5556_B": {
        "tier": "T1B",
        "description": "Pattern 5-5-5-6 (6-runner, Non-chalk, Non-first, Sum 6-8.5, Purse 8k-25k)",
        "canonical_cost": 360.0, "pool": "Superf_paid", "hf": _h_s5556,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N", "first_race": "N",
            "sum_min": 6.0, "sum_max": 8.5,
            "purse_min": 8000, "purse_max": 25000,
        },
    },
    "NM_Sup4455_N6_C": {
        "tier": "T1B",
        "description": "Pattern 4-4-5-5 (6-runner, Non-chalk, Fav2>=4)",
        "canonical_cost": 144.0, "pool": "Superf_paid", "hf": _h_s4455,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
        },
    },
    "FB5_2": {
        "tier": "T1B",
        "description": "Full-Box Superfecta 5-runner (Non-chalk, Fav2>=4, Sum>=6)",
        "canonical_cost": 240.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 5, "runners_max": 5,
            "chalk": "N",
            "fav2_min": 4.0,
            "sum_min": 6.0,
        },
    },
    "NM_Sup4455_N6_D": {
        "tier": "T1B",
        "description": "Pattern 4-4-5-5 (6-runner, Non-chalk, Fav2>=4, Sum 6-8.5)",
        "canonical_cost": 144.0, "pool": "Superf_paid", "hf": _h_s4455,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
            "sum_min": 6.0,
            "sum_max": 8.5,
        },
    },
    "TRI_Sum8": {
        "tier": "T1B",
        "description": "Pattern Trifecta 1-4-5 (Sum>=8, Fav2>=2.5)",
        "canonical_cost": 18.0, "pool": "Trif_paid", "hf": _h_t145,
        "gates": {
            "runners_min": 5, "runners_max": 13,
            "sum_min": 8.0,
            "fav2_min": 2.5,
        },
    },
    "FB5_Fav3": {
        "tier": "T1B",
        "description": "Full-Box Superfecta 5-runner (Non-chalk, Fav2>=3)",
        "canonical_cost": 240.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 5, "runners_max": 5,
            "chalk": "N",
            "fav2_min": 3.0,
        },
    },
    "Supr4444_High": {
        "tier": "T1B",
        "description": "Pattern 4-4-4-4 (4-6 runners, Sum>=7.5)",
        "canonical_cost": 48.0, "pool": "Superf_paid", "hf": _h_s4444,
        "gates": {
            "runners_min": 4, "runners_max": 6,
            "sum_min": 7.5,
        },
    },
    "Sup2444_High": {
        "tier": "T1B",
        "description": "Pattern 2-4-4-4 (4-6 runners, Sum>=7.5)",
        "canonical_cost": 24.0, "pool": "Superf_paid", "hf": _h_s2444,
        "gates": {
            "runners_min": 4, "runners_max": 6,
            "sum_min": 7.5,
        },
    },
    "SB6_S5556_SumWide": {
        "tier": "T1B",
        "description": "Pattern 5-5-5-6 (6-runner, Non-chalk, Sum 6-9.5, Purse 8k-25k)",
        "canonical_cost": 360.0, "pool": "Superf_paid", "hf": _h_s5556,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "sum_min": 6.0, "sum_max": 9.5,
            "purse_min": 8000, "purse_max": 25000,
        },
    },
    "NM_Sup4456_N6_D": {
        "tier": "T1B",
        "description": "Pattern 4-4-5-6 (6-runner, Non-chalk, Fav2>=4, Sum 6-9.5)",
        "canonical_cost": 216.0, "pool": "Superf_paid", "hf": _h_s4456,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
            "sum_min": 6.0,
            "sum_max": 9.5,
        },
    },
    "NM_Supr3666_N6_D": {
        "tier": "T1B",
        "description": "Pattern 3-6-6-6 (6-runner, Non-chalk, Fav2>=4, Sum 6-9.5)",
        "canonical_cost": 360.0, "pool": "Superf_paid", "hf": _h_s3666,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
            "sum_min": 6.0,
            "sum_max": 9.5,
        },
    },

    # ── T2 ───────────────────────────────────────────────────────────────────
    "SB6_S5556_C": {
        "tier": "T2",
        "description": "Pattern 5-5-5-6 (6-runner, Non-chalk, Sprint, Sum 6-8.5, Purse 8k-35k)",
        "canonical_cost": 360.0, "pool": "Superf_paid", "hf": _h_s5556,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N", "sprint": "Y",
            "sum_min": 6.0, "sum_max": 8.5,
            "purse_min": 8000, "purse_max": 35000,
        },
    },
    "SB6_S5556_A_P35": {
        "tier": "T2",
        "description": "Pattern 5-5-5-6 (6-runner, Non-chalk, Sum 6-8.5, Purse 8k-35k)",
        "canonical_cost": 360.0, "pool": "Superf_paid", "hf": _h_s5556,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "sum_min": 6.0, "sum_max": 8.5,
            "purse_min": 8000, "purse_max": 35000,
        },
    },
    "TRI_First": {
        "tier": "T2",
        "description": "Pattern Trifecta 1-4-5 (First race, Sum>=9, Fav2>=2.5)",
        "canonical_cost": 18.0, "pool": "Trif_paid", "hf": _h_t145,
        "gates": {
            "runners_min": 5, "runners_max": 13,
            "first_race": "Y",
            "sum_min": 9.0,
            "fav2_min": 2.5,
        },
    },
    "FB6_2": {
        "tier": "T2",
        "description": "Full-Box Superfecta 6-runner (Non-chalk, Fav2>=4)",
        "canonical_cost": 720.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
        },
    },
    "FB6_3": {
        "tier": "T2",
        "description": "Full-Box Superfecta 6-runner (Non-chalk, Fav2>=4)",
        "canonical_cost": 720.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
        },
    },
    "FB6_5": {
        "tier": "T2",
        "description": "Full-Box Superfecta 6-runner (Non-chalk, Fav2>=4)",
        "canonical_cost": 720.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
        },
    },
    "FB6_6": {
        "tier": "T2",
        "description": "Full-Box Superfecta 6-runner (Non-chalk, Fav2>=4)",
        "canonical_cost": 720.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
        },
    },
    "NM_Sup4466": {
        "tier": "T2",
        "description": "Pattern 4-4-6-6 (6-runner, Non-chalk, Fav2>=4)",
        "canonical_cost": 288.0, "pool": "Superf_paid", "hf": _h_s4466,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
        },
    },
    "NM_Sup5567": {
        "tier": "T2",
        "description": "Pattern 5-5-6-7 (6-runner, Non-chalk, Fav2>=4)",
        "canonical_cost": 480.0, "pool": "Superf_paid", "hf": _h_s5567,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
        },
    },

    # ── T3 ───────────────────────────────────────────────────────────────────
    "FB7_1": {
        "tier": "T3",
        "description": "Full-Box Superfecta 7-runner (Non-chalk, Non-first, Sprint, Purse 8k-15k, Sum 6-8)",
        "canonical_cost": 1680.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 7, "runners_max": 7,
            "chalk": "N", "first_race": "N", "sprint": "Y",
            "purse_min": 8000, "purse_max": 15000,
            "sum_min": 6.0, "sum_max": 8.0,
        },
    },
    "FB7_3": {
        "tier": "T3",
        "description": "Full-Box Superfecta 7-runner (Non-chalk, Sprint, Purse 8k-20k, Fav2>=4, Sum>=6)",
        "canonical_cost": 1680.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 7, "runners_max": 7,
            "chalk": "N", "sprint": "Y",
            "purse_min": 8000, "purse_max": 20000,
            "fav2_min": 4.0,
            "sum_min": 6.0,
        },
    },
    "FB7_4": {
        "tier": "T3",
        "description": "Full-Box Superfecta 7-runner (Non-chalk, Sum 6-8, Fav2>=4, Purse>=10k)",
        "canonical_cost": 1680.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 7, "runners_max": 7,
            "chalk": "N",
            "sum_min": 6.0, "sum_max": 8.0,
            "fav2_min": 4.0,
            "purse_min": 10000,
        },
    },
    "FB6_1": {
        "tier": "T3",
        "description": "Full-Box Superfecta 6-runner (Non-chalk, Sprint, Fav2>=4)",
        "canonical_cost": 720.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N", "sprint": "Y",
            "fav2_min": 4.0,
        },
    },
    "FB6_4": {
        "tier": "T3",
        "description": "Full-Box Superfecta 6-runner (Non-chalk, Sprint, Fav2>=4)",
        "canonical_cost": 720.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N", "sprint": "Y",
            "fav2_min": 4.0,
        },
    },

    # ── BONUS ─────────────────────────────────────────────────────────────────
    "BNS_TRI_HighPurse": {
        "tier": "BONUS",
        "description": "Pattern Trifecta 1-4-5 (Sum>=9, Fav2>=2.5, Purse>=25k)",
        "canonical_cost": 18.0, "pool": "Trif_paid", "hf": _h_t145,
        "gates": {
            "runners_min": 5, "runners_max": 13,
            "sum_min": 9.0,
            "fav2_min": 2.5,
            "purse_min": 25000,
        },
    },
    "BNS_Sup3225_R56": {
        "tier": "BONUS",
        "description": "Pattern 3-2-2-5 (5-6 runners, Sum>=4)",
        "canonical_cost": 24.0, "pool": "Superf_paid", "hf": _h_s3225,
        "gates": {
            "runners_min": 5, "runners_max": 6,
            "sum_min": 4.0,
        },
    },
    "BNS_Supr2555_Sniper": {
        "tier": "BONUS",
        "description": "Pattern 2-5-5-5 (5-8 runners, Sum>=7)",
        "canonical_cost": 96.0, "pool": "Superf_paid", "hf": _h_s2555,
        "gates": {
            "runners_min": 5, "runners_max": 8,
            "sum_min": 7.0,
        },
    },
    "BNS_Tri135": {
        "tier": "BONUS",
        "description": "Pattern Trifecta 1-3-5 (Sum>=7, Fav2>=2.5)",
        "canonical_cost": 12.0, "pool": "Trif_paid", "hf": _h_t135,
        "gates": {
            "runners_min": 5, "runners_max": 13,
            "sum_min": 7.0,
            "fav2_min": 2.5,
        },
    },
    "BNS_SB6_RouteOnly": {
        "tier": "BONUS",
        "description": "Pattern 5-5-5-6 (6-runner, Non-chalk, Route, Sum 6-8.5, Purse 8k-25k)",
        "canonical_cost": 360.0, "pool": "Superf_paid", "hf": _h_s5556,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N", "sprint": "N",
            "sum_min": 6.0, "sum_max": 8.5,
            "purse_min": 8000, "purse_max": 25000,
        },
    },
    "SB12_S6667_A": {
        "tier": "BONUS",
        "description": "Pattern 6-6-6-7 (12-runner, Non-chalk, Route, Sum 4-8.5, Fav2>=4, Purse 8k-25k)",
        "canonical_cost": 960.0, "pool": "Superf_paid", "hf": _h_s6667,
        "gates": {
            "runners_min": 12, "runners_max": 12,
            "chalk": "N", "sprint": "N",
            "sum_min": 4.0, "sum_max": 8.5,
            "fav2_min": 4.0,
            "purse_min": 8000, "purse_max": 25000,
        },
    },
    "FB6_First": {
        "tier": "BONUS",
        "description": "Full-Box Superfecta 6-runner (Non-chalk, First race, Fav2>=4)",
        "canonical_cost": 720.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N", "first_race": "Y",
            "fav2_min": 4.0,
        },
    },
    "FB6_HighPurse": {
        "tier": "BONUS",
        "description": "Full-Box Superfecta 6-runner (Non-chalk, Fav2>=4, Purse>=25k)",
        "canonical_cost": 720.0, "pool": "Superf_paid", "hf": _h_al,
        "gates": {
            "runners_min": 6, "runners_max": 6,
            "chalk": "N",
            "fav2_min": 4.0,
            "purse_min": 25000,
        },
    },
    "BNS_TriA22": {
        "tier": "BONUS",
        "description": "Trifecta ALL/Top2/Top2 — upset winner, top-2 favs fill/show (Sum>=6, Fav2>=2.5)",
        "canonical_cost": 12.0, "pool": "Trif_paid", "hf": _h_tria22,
        "gates": {
            "runners_min": 5, "runners_max": 13,
            "sum_min": 6.0,
            "fav2_min": 2.5,
        },
    },

    # ── APEX ─────────────────────────────────────────────────────────────────
    "APX_Sup1234": {
        "tier": "APEX",
        "description": "Straight Superfecta 1-2-3-4 (5-12 runners, Sum>=6)",
        "canonical_cost": 2.0, "pool": "Superf_paid", "hf": _h_s1234,
        "gates": {
            "runners_min": 5, "runners_max": 12,
            "sum_min": 6.0,
        },
    },
}

# ── Strategy sets ─────────────────────────────────────────────────────────────
_T1A = {"FB4_2","FB4_3","FB4_4","FB4_5","FB4_6",
        "Sup3214_R5","Sup3214_R6","TRI145","TRI245"}
_T1B = {"SB6_S5556_A","SB6_S5556_B","NM_Sup4455_N6_C","FB5_2",
        "NM_Sup4455_N6_D","TRI_Sum8","FB5_Fav3","Supr4444_High",
        "Sup2444_High","SB6_S5556_SumWide","NM_Sup4456_N6_D","NM_Supr3666_N6_D"}
_T1  = _T1A | _T1B
_T2  = {"SB6_S5556_C","SB6_S5556_A_P35","TRI_First","FB6_2","FB6_3",
        "FB6_5","FB6_6","NM_Sup4466","NM_Sup5567"}
_T3  = {"FB7_1","FB7_3","FB7_4","FB6_1","FB6_4"}
_BONUS = {"BNS_TRI_HighPurse","BNS_Sup3225_R56","BNS_Supr2555_Sniper",
          "BNS_Tri135","BNS_SB6_RouteOnly","SB12_S6667_A",
          "FB6_First","FB6_HighPurse","BNS_TriA22"}
_APEX = {"APX_Sup1234"}

STRATEGY_SETS = {
    "SANITY":    {"TRI145","TRI245","TRI_Sum8","FB5_2","FB5_Fav3",
                  "NM_Sup4455_N6_C","Supr4444_High","TRI_First",
                  "Sup3214_R6","Sup3214_R5","BNS_Sup3225_R56"},
    "SAFEST":    _T1,
    "STANDARD":  _T1 | _T2,
    "ALL_TIERS": _T1 | _T2 | _T3,
    "TURBO":     _T1 | _T2 | _T3 | _BONUS | _APEX,
}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_number(s):
    try:
        return float(str(s).replace("$","").replace(",","").strip())
    except:
        return None

def _get_yesterday():
    return (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

def _load_csv(csv_file):
    try:
        return pd.read_csv(csv_file, low_memory=False)
    except Exception as e:
        _safe_print(f"ERROR: Could not load CSV: {e}")
        return pd.DataFrame()

def _parse_date_column(df):
    priority = ["RaceDate","Date","race_date","Race_Date","RACEDATE"]
    date_col = next((c for c in priority if c in df.columns), None)
    if date_col is None:
        date_col = next((c for c in df.columns
                         if "date" in c.lower() or "racedate" in c.lower()), None)
    if not date_col:
        _safe_print("ERROR: No date column found.")
        return None, df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return date_col, df

def _parse_ranked_results(val):
    if pd.isna(val):
        return []
    try:
        return [int(float(x))
                for x in str(val).replace(",", " ").split()
                if x.strip()]
    except:
        return []

def _is_wow(row):
    try:
        sp = float(row.get("Superf_paid", 0) or 0)
        ex = float(row.get("Exacta_paid", 0) or 0)
        tp = float(row.get("Trif_paid", 0) or 0)
        if sp > WOW_ABS:
            return True
        if ex > 0 and sp / ex > WOW_RAT:
            return True
        if tp > WOW_ABS:
            return True
    except:
        pass
    return False

def _get_fav2(row):
    for col in ["Fav2Exact", "Fav2_odds", "FavExact"]:
        if col in row.index:
            v = _parse_number(row[col])
            if v is not None:
                return v
    return None

def _is_sprint(row):
    for col in ["Miles", "Distance"]:
        if col in row.index:
            v = _parse_number(row[col])
            if v is not None:
                miles = v / 8.0 if col == "Distance" else v
                return miles < SPRINT_THRESHOLD_MILES
    return None

def _check_universal_gates(row):
    runners = _parse_number(row.get("Runners"))
    if runners is not None and runners > UNIVERSAL_MAX_RUNNERS:
        return False, f"Runners={runners:.0f} > {UNIVERSAL_MAX_RUNNERS}"
    for col in ["SumOf1st2Odds", "SI"]:
        if col in row.index:
            s = _parse_number(row[col])
            if s is not None and s > UNIVERSAL_MAX_SUM:
                return False, f"{col}={s:.1f} > {UNIVERSAL_MAX_SUM}"
            break
    return True, ""

def _check_instrument_gates(row, gates: dict):
    violations = []

    runners = _parse_number(row.get("Runners"))
    rmin = gates.get("runners_min")
    rmax = gates.get("runners_max")
    if runners is None:
        violations.append("Runners: column missing")
    else:
        if rmin is not None and runners < rmin:
            violations.append(f"Runners={runners:.0f} < {rmin}")
        if rmax is not None and runners > rmax:
            violations.append(f"Runners={runners:.0f} > {rmax}")

    si_val = None
    for col in ["SumOf1st2Odds", "SI"]:
        if col in row.index:
            si_val = _parse_number(row[col])
            break
    smin = gates.get("sum_min")
    smax = gates.get("sum_max")
    if smin is not None or smax is not None:
        if si_val is None:
            violations.append("SumOf1st2Odds: column missing")
        else:
            if smin is not None and si_val < smin:
                violations.append(f"Sum={si_val:.2f} < {smin}")
            if smax is not None and si_val > smax:
                violations.append(f"Sum={si_val:.2f} > {smax}")

    fav2 = _get_fav2(row)
    f2min = gates.get("fav2_min")
    f2max = gates.get("fav2_max")
    if f2min is not None or f2max is not None:
        if fav2 is None:
            violations.append("Fav2Exact: column missing")
        else:
            if f2min is not None and fav2 < f2min:
                violations.append(f"Fav2={fav2:.2f} < {f2min}")
            if f2max is not None and fav2 > f2max:
                violations.append(f"Fav2={fav2:.2f} > {f2max}")

    purse = _parse_number(row.get("Purse"))
    pmin = gates.get("purse_min")
    pmax = gates.get("purse_max")
    if pmin is not None or pmax is not None:
        if purse is None:
            violations.append("Purse: column missing")
        else:
            if pmin is not None and purse < pmin:
                violations.append(f"Purse={purse:.0f} < {pmin}")
            if pmax is not None and purse > pmax:
                violations.append(f"Purse={purse:.0f} > {pmax}")

    chalk_req = gates.get("chalk")
    if chalk_req is not None:
        if "ChalkYN" not in row.index:
            violations.append("ChalkYN: column missing")
        else:
            chalk_val = str(row["ChalkYN"]).strip().upper()
            if chalk_req == "N" and chalk_val == "Y":
                violations.append(f"ChalkYN=Y (need non-chalk)")
            elif chalk_req == "Y" and chalk_val != "Y":
                violations.append(f"ChalkYN={chalk_val} (need chalk)")

    fr_req = gates.get("first_race")
    if fr_req is not None:
        if "FirstRaceYN" not in row.index:
            violations.append("FirstRaceYN: column missing")
        else:
            fr_val = str(row["FirstRaceYN"]).strip().upper()
            if fr_req == "Y" and fr_val != "Y":
                violations.append(f"FirstRaceYN={fr_val} (need Y)")
            elif fr_req == "N" and fr_val == "Y":
                violations.append(f"FirstRaceYN=Y (need non-first)")

    spr_req = gates.get("sprint")
    if spr_req is not None:
        is_spr = _is_sprint(row)
        if is_spr is None:
            violations.append("Miles/Distance: column missing (sprint gate skipped)")
        else:
            if spr_req == "Y" and not is_spr:
                violations.append("Race is route (need sprint)")
            elif spr_req == "N" and is_spr:
                violations.append("Race is sprint (need route)")

    race_min = gates.get("race_min", 1)
    race_max = gates.get("race_max", 11)
    if "WhichRace" in row.index:
        wr = _parse_number(row["WhichRace"])
        if wr is not None:
            if wr < race_min or wr > race_max:
                violations.append(f"WhichRace={wr:.0f} outside {race_min}-{race_max}")

    return len(violations) == 0, violations

def _extract_payout(row, pool_col):
    aliases = {
        "Trif_paid":   ["Trif_paid", "Trifecta_paid"],
        "Superf_paid": ["Superf_paid", "Superfecta_paid"],
    }
    cols_to_try = aliases.get(pool_col, [pool_col])
    for col in cols_to_try:
        if col in row.index:
            v = _parse_number(row[col])
            if v is not None and v > 0:
                return v, True
    return 0.0, False

# ══════════════════════════════════════════════════════════════════════════════
# RANKED RESULTS DETAIL BLOCK
# ══════════════════════════════════════════════════════════════════════════════

# All columns we try to show in the detail block, in display order.
# Each entry: (label, column_name_or_list_of_candidates)
_DETAIL_FIELDS = [
    ("Track",           ["Track", "TrackCode", "Track Name"]),
    ("Race #",          ["WhichRace"]),
    ("Date",            ["RaceDate", "Date", "race_date", "Race_Date", "RACEDATE"]),
    ("Runners",         ["Runners"]),
    ("Miles",           ["Miles", "Distance"]),
    ("Sprint/Route",    None),   # computed
    ("Purse",           ["Purse"]),
    ("ChalkYN",         ["ChalkYN"]),
    ("FirstRaceYN",     ["FirstRaceYN"]),
    ("SumOf1st2Odds",   ["SumOf1st2Odds", "SI"]),
    ("Fav2 odds",       ["Fav2Exact", "Fav2_odds", "FavExact"]),
    ("Fav1 odds",       ["Fav1Exact", "Fav1_odds"]),
    ("RANKED_RESULTS",  ["RANKED_RESULTS"]),
    ("Win odds",        ["Win_odds", "WinOdds", "Winner_odds"]),
    ("Place odds",      ["Place_odds", "PlaceOdds"]),
    ("Show odds",       ["Show_odds", "ShowOdds"]),
    ("4th odds",        ["Fourth_odds", "FourthOdds"]),
    ("Exacta paid",     ["Exacta_paid"]),
    ("Trifecta paid",   ["Trif_paid", "Trifecta_paid"]),
    ("Superfecta paid", ["Superf_paid", "Superfecta_paid"]),
]

def _build_detail_block(row, inst_name, inst, ticket_cost, raw_payout,
                        net_payout, pnl, is_hit, ranks):
    """
    Build a multi-line detail string for one fired match.
    Shows every available RANKED_RESULTS-related field.
    """
    lines = []
    inst_def = INSTRUMENTS.get(inst_name, {})
    mark = "HIT " if is_hit else "MISS"

    lines.append(f"    ┌─ [{mark}] {inst_name}  [{inst_def.get('tier','')}]")
    lines.append(f"    │  {inst_def.get('description','')}")
    lines.append(f"    │")

    # Gate values
    for label, candidates in _DETAIL_FIELDS:
        if candidates is None:
            # Computed field: Sprint/Route
            is_spr = _is_sprint(row)
            if is_spr is None:
                val = "unknown (no Miles/Distance col)"
            else:
                val = "SPRINT" if is_spr else "ROUTE"
            lines.append(f"    │  {label:<22}: {val}")
            continue

        found = False
        for col in candidates:
            if col in row.index:
                raw = row[col]
                # Pretty-format numbers
                if label in ("Purse",):
                    n = _parse_number(raw)
                    val = f"${n:,.0f}" if n is not None else str(raw)
                elif label in ("Exacta paid","Trifecta paid","Superfecta paid"):
                    n = _parse_number(raw)
                    val = f"${n:,.2f}" if n is not None else str(raw)
                elif label == "RANKED_RESULTS":
                    val = str(raw)
                    if ranks:
                        val += f"  => parsed: {ranks}"
                else:
                    val = str(raw)
                lines.append(f"    │  {label:<22}: {val}")
                found = True
                break
        # Skip silently if column not in dataset at all

    lines.append(f"    │")
    lines.append(f"    │  Ticket cost   : ${ticket_cost:,.2f}")
    lines.append(f"    │  Raw payout    : ${raw_payout:,.2f}")
    lines.append(f"    │  Net payout    : ${net_payout:,.2f}")
    pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"(${abs(pnl):,.2f})"
    lines.append(f"    │  P&L           : {pnl_str}")
    lines.append(f"    └{'─'*70}")

    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# CORE REPORT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(csv_file, target_date, scale=1.0,
                    strategy="TURBO", verbose_misses=False):
    base_bet_factor = 2.0 * scale
    unit            = 2.0 * scale

    active_instruments = STRATEGY_SETS.get(strategy.upper(), set())
    if not active_instruments:
        _safe_print(f"  [WARN] Unknown strategy '{strategy}'. Using TURBO.")
        active_instruments = STRATEGY_SETS["TURBO"]

    df = _load_csv(csv_file)
    if df.empty:
        return {}, 0.0

    date_col, df = _parse_date_column(df)
    if date_col is None:
        return {}, 0.0

    if isinstance(target_date, str):
        target_dt = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    else:
        target_dt = target_date
    day_df = df[df[date_col].dt.date == target_dt].copy()

    if day_df.empty:
        _safe_print(f"\n  [INFO] No races found for {target_date}")
        return {}, 0.0

    for col in ["Runners","Purse","WhichRace","SumOf1st2Odds",
                "Superf_paid","Trif_paid","Exacta_paid"]:
        if col in day_df.columns:
            day_df[col] = pd.to_numeric(day_df[col], errors="coerce").fillna(0)

    track_col = next((c for c in ["Track","TrackCode","Track Name"]
                      if c in day_df.columns), None)

    _safe_print("\n" + "="*100)
    _safe_print(f"  DAILY RACE RECONCILIATION REPORT v8.0.3")
    _safe_print(f"  Date     : {target_date}")
    _safe_print(f"  Strategy : {strategy.upper()}  "
                f"({len(active_instruments)} instruments)")
    _safe_print(f"  Base bet : ${unit:.2f}   Scale: {scale:.2f}x   "
                f"Payout factor: {base_bet_factor:.2f}/2")
    _safe_print(f"  Races    : {len(day_df)}")
    _safe_print("="*100)

    if track_col:
        tracks = day_df[track_col].value_counts()
        _safe_print(f"\n  Tracks:")
        for track, count in tracks.items():
            _safe_print(f"    {track}: {count} races")

    fired_gates  = {}
    wow_skipped  = 0
    univ_skipped = 0

    for _, row in day_df.iterrows():
        if _is_wow(row):
            wow_skipped += 1
            continue

        univ_ok, univ_reason = _check_universal_gates(row)
        if not univ_ok:
            univ_skipped += 1
            continue

        wr = _parse_number(row.get("WhichRace"))
        if wr is not None and (wr < 1 or wr > 11):
            continue

        ranks = _parse_ranked_results(row.get("RANKED_RESULTS", ""))

        for inst_name, inst in INSTRUMENTS.items():
            if inst_name not in active_instruments:
                continue

            passed, violations = _check_instrument_gates(row, inst["gates"])
            if not passed:
                continue

            raw_payout, pool_ok = _extract_payout(row, inst["pool"])
            if not pool_ok:
                continue

            n_runners = int(_parse_number(row.get("Runners")) or 0)
            hf = inst["hf"]
            is_hit = hf(ranks, n_runners) if ranks else False

            ticket_cost = inst["canonical_cost"] * scale
            net_payout  = raw_payout * (base_bet_factor / 2.0)
            pnl         = (net_payout - ticket_cost) if is_hit else -ticket_cost

            race_num = (int(_parse_number(row.get("WhichRace")) or 0)
                        if "WhichRace" in row.index else "?")
            track    = (str(row[track_col]) if track_col else "?")

            if inst_name not in fired_gates:
                fired_gates[inst_name] = []

            fired_gates[inst_name].append({
                "race_num":       race_num,
                "track":          track,
                "is_hit":         is_hit,
                "raw_payout":     raw_payout,
                "net_payout":     net_payout if is_hit else 0.0,
                "cost":           ticket_cost,
                "pnl":            pnl,
                "tier":           inst["tier"],
                "ranked_results": str(row.get("RANKED_RESULTS","?")),
                "ranks_parsed":   ranks,
                # Store the full row for detail block
                "_row":           row,
            })

    if wow_skipped:
        _safe_print(f"\n  [WowSuperWow] {wow_skipped} row(s) excluded "
                    f"(payout >${WOW_ABS:,.0f} or >{WOW_RAT:.0f}x exacta)")
    if univ_skipped:
        _safe_print(f"  [Universal gates] {univ_skipped} row(s) excluded "
                    f"(Runners>{UNIVERSAL_MAX_RUNNERS} or "
                    f"Sum>{UNIVERSAL_MAX_SUM})")

    if not fired_gates:
        _safe_print(f"\n  [INFO] No instruments fired on {target_date}.\n")
        return {}, 0.0

    # ── Print results ────────────────────────────────────────────────────────
    total_pnl = 0.0
    _safe_print(f"\n  {'─'*98}")
    _safe_print(f"  {'INSTRUMENT':<30} {'TIER':<8} {'R#':<5} "
                f"{'TRACK':<10} {'COST':>8} {'PAYOUT':>10} "
                f"{'P&L':>10}  RESULT")
    _safe_print(f"  {'─'*98}")

    for inst_name in sorted(fired_gates.keys()):
        races    = fired_gates[inst_name]
        inst     = INSTRUMENTS[inst_name]
        hits     = sum(1 for r in races if r["is_hit"])
        inst_pnl = sum(r["pnl"] for r in races)
        total_pnl += inst_pnl

        _safe_print(f"\n  {inst_name}  [{inst['tier']}]  "
                    f"— {inst['description']}")
        _safe_print(f"  Fired: {len(races)}  Hits: {hits}  "
                    f"Inst P&L: {'+'if inst_pnl>=0 else ''}${inst_pnl:,.2f}")

        for r in races:
            mark    = "✓ HIT " if r["is_hit"] else "✗ MISS"
            pnl_str = (f"+${r['pnl']:,.2f}" if r["pnl"] >= 0
                       else f"(${abs(r['pnl']):,.2f})")

            # ── Summary line (same as before) ──────────────────────────────
            _safe_print(
                f"    {mark}  R{r['race_num']:<3} @ {r['track']:<10}  "
                f"Cost=${r['cost']:>8,.2f}  "
                f"RawPay=${r['raw_payout']:>10,.2f}  "
                f"NetPay=${r['net_payout']:>10,.2f}  "
                f"P&L={pnl_str:>10}  "
                f"Pos:{r['ranked_results']}")

            # ── Full detail block ───────────────────────────────────────────
            detail = _build_detail_block(
                row         = r["_row"],
                inst_name   = inst_name,
                inst        = inst,
                ticket_cost = r["cost"],
                raw_payout  = r["raw_payout"],
                net_payout  = r["net_payout"],
                pnl         = r["pnl"],
                is_hit      = r["is_hit"],
                ranks       = r["ranks_parsed"],
            )
            _safe_print(detail)

    _safe_print(f"\n  {'='*98}")
    total_str = (f"✓ PROFIT: +${total_pnl:,.2f}"
                 if total_pnl >= 0 else f"✗ LOSS: (${abs(total_pnl):,.2f})")
    _safe_print(f"  DAILY TOTAL  {target_date}:  {total_str}")
    _safe_print(f"  Tickets fired: {sum(len(v) for v in fired_gates.values())}")
    _safe_print(f"  {'='*98}\n")

    return fired_gates, total_pnl

# ══════════════════════════════════════════════════════════════════════════════
# TXT EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _make_txt_filename(dates, strategy):
    """Generate a descriptive TXT filename."""
    if len(dates) == 1:
        date_part = dates[0]
    else:
        date_part = f"{dates[0]}_to_{dates[-1]}"
    return f"Reconciliation_{date_part}_{strategy}.txt"

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-DATE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_period_summary(all_fired, all_pnl_by_date, dates, scale, strategy):
    if len(dates) <= 1:
        return
    _safe_print("\n" + "="*100)
    _safe_print(f"  PERIOD SUMMARY — {len(dates)} dates  "
                f"[{dates[0]} to {dates[-1]}]")
    _safe_print(f"  Strategy: {strategy}   Scale: {scale:.2f}x  "
                f"Base bet: ${2.0*scale:.2f}")
    _safe_print("="*100)

    grand_total   = sum(all_pnl_by_date.values())
    total_tickets = sum(len(v) for v in all_fired.values())
    total_hits    = sum(sum(1 for r in v if r["is_hit"]) for v in all_fired.values())

    _safe_print(f"\n  {'Instrument':<30} {'Tier':<8} {'Fires':>7} "
                f"{'Hits':>6} {'HR%':>6}  P&L")
    _safe_print(f"  {'─'*30} {'─'*8} {'─'*7} {'─'*6} {'─'*6}  {'─'*20}")

    for inst_name in sorted(all_fired.keys()):
        races = all_fired[inst_name]
        hits  = sum(1 for r in races if r["is_hit"])
        pnl   = sum(r["pnl"] for r in races)
        hr    = hits / len(races) * 100 if races else 0.0
        tier  = INSTRUMENTS.get(inst_name, {}).get("tier","")
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"(${abs(pnl):,.2f})"
        _safe_print(f"  {inst_name:<30} {tier:<8} {len(races):>7} "
                    f"{hits:>6} {hr:>5.1f}%  {pnl_str}")

    _safe_print(f"\n  {'─'*100}")
    grand_str = (f"✓ PERIOD PROFIT: +${grand_total:,.2f}"
                 if grand_total >= 0
                 else f"✗ PERIOD LOSS: (${abs(grand_total):,.2f})")
    _safe_print(f"  {grand_str}")
    _safe_print(f"  Total tickets : {total_tickets}")
    if total_tickets:
        _safe_print(f"  Total hits    : {total_hits} "
                    f"({100*total_hits/total_tickets:.1f}%)")

    _safe_print(f"\n  Daily P&L breakdown:")
    for d in dates:
        p = all_pnl_by_date.get(d, 0.0)
        marker = "+" if p >= 0 else "-"
        _safe_print(f"    {d}  {marker}${abs(p):,.2f}")

    _safe_print("="*100 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ══════════════════════════════════════════════════════════════════════════════

def _show_menu():
    _safe_print("\n" + "="*100)
    _safe_print("  DAILY RACE RECONCILIATION REPORT v8.0.3 — DATE SELECTOR")
    _safe_print("="*100)
    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    week_ago  = today - datetime.timedelta(days=7)
    _safe_print(f"\n  [1] Yesterday  ({yesterday.strftime('%A, %B %d, %Y')})")
    _safe_print(f"  [2] Previous 7 days  "
                f"({week_ago.strftime('%b %d')} – {yesterday.strftime('%b %d, %Y')})")
    _safe_print("  [3] Specific date")
    _safe_print("  [4] Custom date range")
    _safe_print("  [0] Exit\n")
    choice = input("  Choice [1]: ").strip() or "1"

    if choice == "0":
        sys.exit(0)
    elif choice == "1":
        return [yesterday.strftime("%Y-%m-%d")]
    elif choice == "2":
        return [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(7, 0, -1)]
    elif choice == "3":
        ds = input("  Date (YYYY-MM-DD): ").strip()
        try:
            return [datetime.datetime.strptime(ds, "%Y-%m-%d")
                    .date().strftime("%Y-%m-%d")]
        except ValueError:
            _safe_print("  ERROR: Invalid date."); sys.exit(1)
    elif choice == "4":
        s = input("  Start date (YYYY-MM-DD): ").strip()
        e = input("  End date   (YYYY-MM-DD): ").strip()
        try:
            start = datetime.datetime.strptime(s, "%Y-%m-%d").date()
            end   = datetime.datetime.strptime(e, "%Y-%m-%d").date()
            if start > end:
                _safe_print("  ERROR: Start after end."); sys.exit(1)
            dates = []
            cur = start
            while cur <= end:
                dates.append(cur.strftime("%Y-%m-%d"))
                cur += datetime.timedelta(days=1)
            return dates
        except ValueError:
            _safe_print("  ERROR: Invalid date format."); sys.exit(1)
    else:
        _safe_print("  Invalid choice."); sys.exit(1)

def _select_strategy():
    _safe_print("\n  SELECT STRATEGY:")
    opts = list(STRATEGY_SETS.keys())
    for i, s in enumerate(opts):
        n = len(STRATEGY_SETS[s])
        _safe_print(f"    [{i+1}] {s:<12} ({n} instruments)")
    choice = input("  Choice [5=TURBO]: ").strip() or "5"
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(opts):
            return opts[idx]
    except ValueError:
        pass
    return "TURBO"

def _select_scale():
    _safe_print("\n  SELECT BASE BET (must match Gauntlet setting):")
    opts = [
        ("0", "$0.50", 0.25),
        ("1", "$1.00", 0.50),
        ("2", "$2.00", 1.00),
        ("3", "$3.00", 1.50),
    ]
    for k, lbl, sc in opts:
        _safe_print(f"    [{k}] {lbl} base  (scale={sc:.2f})")
    choice = input("  Choice [2=$2.00]: ").strip() or "2"
    lkp = {k: sc for k, _, sc in opts}
    return lkp.get(choice, 1.00)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Daily race reconciliation — v8.0.3 companion to Gauntlet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scale mapping (must match Gauntlet base-bet selection):
  --scale 0.25  =>  $0.50 base
  --scale 0.50  =>  $1.00 base
  --scale 1.00  =>  $2.00 base  [default / canonical]
  --scale 1.50  =>  $3.00 base

Strategy choices: SANITY  SAFEST  STANDARD  ALL_TIERS  TURBO
        """
    )
    parser.add_argument("--csv",       default="RaceRecords_Output_2026.csv")
    parser.add_argument("--date",      default=None,  help="YYYY-MM-DD")
    parser.add_argument("--range",     default=None,  help="YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--yesterday", action="store_true")
    parser.add_argument("--week",      action="store_true")
    parser.add_argument("--scale",     type=float, default=None)
    parser.add_argument("--strategy",  default=None)
    parser.add_argument("--no-txt",    action="store_true",
                        help="Suppress TXT file output")
    args = parser.parse_args()

    # ── Reset output buffer ───────────────────────────────────────────────────
    _reset_buffer()

    # ── Resolve dates ─────────────────────────────────────────────────────────
    dates = []
    if args.date:
        try:
            dates = [datetime.datetime.strptime(args.date, "%Y-%m-%d")
                     .date().strftime("%Y-%m-%d")]
        except ValueError:
            _safe_print("ERROR: Invalid --date format."); sys.exit(1)
    elif args.range:
        try:
            s, e = args.range.split(":")
            start = datetime.datetime.strptime(s, "%Y-%m-%d").date()
            end   = datetime.datetime.strptime(e, "%Y-%m-%d").date()
            cur   = start
            while cur <= end:
                dates.append(cur.strftime("%Y-%m-%d"))
                cur += datetime.timedelta(days=1)
        except Exception:
            _safe_print("ERROR: Use --range YYYY-MM-DD:YYYY-MM-DD"); sys.exit(1)
    elif args.yesterday:
        dates = [_get_yesterday()]
    elif args.week:
        today = datetime.date.today()
        dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                 for i in range(7, 0, -1)]
    else:
        dates = _show_menu()

    # ── Resolve scale and strategy ────────────────────────────────────────────
    scale    = args.scale    if args.scale    is not None else _select_scale()
    strategy = args.strategy if args.strategy is not None else _select_strategy()

    _safe_print(f"\n  CSV      : {args.csv}")
    _safe_print(f"  Dates    : {len(dates)}  ({dates[0]} to {dates[-1]})")
    _safe_print(f"  Strategy : {strategy}")
    _safe_print(f"  Scale    : {scale:.2f}x  (${2.0*scale:.2f} base bet)")

    # ── Run ───────────────────────────────────────────────────────────────────
    all_fired:       dict = {}
    all_pnl_by_date: dict = {}

    for target_date in dates:
        fired, daily_pnl = generate_report(
            args.csv, target_date, scale=scale, strategy=strategy)
        all_pnl_by_date[target_date] = daily_pnl
        for inst_name, races in fired.items():
            all_fired.setdefault(inst_name, []).extend(races)

    print_period_summary(all_fired, all_pnl_by_date, dates, scale, strategy)

    # ── TXT export ────────────────────────────────────────────────────────────
    if not args.no_txt:
        txt_file = _make_txt_filename(dates, strategy)
        _flush_to_txt(txt_file)


if __name__ == "__main__":
    main()
