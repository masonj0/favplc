#!/usr/bin/env python3
"""
SUPERFECTA / TRIFECTA BOX ANALYZER - v7.5
==========================================
Fixes vs v7.4:
  1. Payout warning threshold raised — superfecta medians of $500-800
     are normal per-ticket values, not pool totals.
  2. MIN_HITS filter added — strategies need enough hits to be
     statistically meaningful, not just lucky on one giant payout.
  3. Sort options: by ROI (default), Net_PnL, or Hit_Rate.
  4. IS/OOS hit rate comparison shown — the key robustness signal.
  5. Expected value per race shown — easier to compare strategies.
  6. Breakeven hit rate shown — how often must it hit to cover cost?
"""

import itertools
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_FILE         = "RaceRecords_Output_2025.csv"
TEST_FILE          = "RaceRecords_Output_2026.csv"
RANKED_RESULTS_COL = "RANKED_RESULTS"
BASE_BET           = 2.00

# ── Minimum thresholds ────────────────────────────────────────────────────────
MIN_ELIGIBLE = 30    # minimum races held (raised from 10)
MIN_HITS     = 3     # minimum hits required — filters lucky 1-2 hit flukes

# ── Display ───────────────────────────────────────────────────────────────────
TOP_N_DISPLAY = 40
# Sort options: "roi", "net_pnl", "hit_rate", "ev_per_race"
SORT_BY = "roi"

# ── Field size range per bet type ─────────────────────────────────────────────
FIELD_MIN = {3: 4,  4: 5}
FIELD_MAX = {3: 12, 4: 12}

# ── Payout sanity thresholds (per-$2-ticket) ──────────────────────────────────
# Raised to realistic values after diagnostic confirmed columns are correct
PAYOUT_WARN: dict[int, float] = {
    3: 2_000.0,    # trifecta: warn if median > $2,000
    4: 10_000.0,   # superfecta: warn if median > $10,000
}


# ==========================================
# HELPERS
# ==========================================

def parse_ranked_result(raw) -> list[int] | None:
    if pd.isna(raw):
        return None
    tokens = str(raw).strip().split()
    if len(tokens) < 3:
        return None
    try:
        return [int(t) for t in tokens]
    except ValueError:
        return None


def is_match(positions: list[int], key: int,
             rest: frozenset[int], n_rest: int) -> bool:
    if len(positions) < n_rest:
        return False
    if positions[0] != key:
        return False
    return set(positions[1:n_rest]).issubset(rest)


def format_rank(rank: int, n_runners: int) -> str:
    if rank == n_runners:
        return "L"
    if rank == n_runners - 1:
        return "S"
    return f"{rank:02d}"


def label_combo(key: int, rest_combo: tuple[int, ...],
                n_runners: int) -> tuple[str, str]:
    key_lbl  = format_rank(key, n_runners)
    rest_lbl = "[" + ", ".join(
        format_rank(r, n_runners) for r in sorted(rest_combo)
    ) + "]"
    return key_lbl, rest_lbl


def get_payout_array(df: pd.DataFrame, n_rest: int) -> np.ndarray:
    candidates = (
        ["Superf_paid", "Superfecta_paid"] if n_rest == 4
        else ["Trif_paid", "Trifecta_paid"]
    )
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    print(f"  ⚠  No payout column found for n_rest={n_rest}.")
    return np.zeros(len(df))


def breakeven_hit_rate(ticket_cost: float, avg_payout: float) -> float:
    """
    Minimum hit rate needed to cover ticket cost.
    breakeven = ticket_cost / avg_payout_per_hit
    """
    if avg_payout <= 0:
        return float("inf")
    return ticket_cost / avg_payout * 100  # as percentage


# ==========================================
# PAYOUT DIAGNOSTIC
# ==========================================

def run_payout_diagnostic(df: pd.DataFrame, label: str) -> None:
    print(f"\n  {'─'*62}")
    print(f"  PAYOUT DIAGNOSTIC — {label}")
    print(f"  {'─'*62}")

    for n_rest, col_candidates in [
        (3, ["Trif_paid", "Trifecta_paid"]),
        (4, ["Superf_paid", "Superfecta_paid"]),
    ]:
        for col in col_candidates:
            if col not in df.columns:
                continue
            vals    = pd.to_numeric(df[col], errors="coerce").dropna()
            nonzero = vals[vals > 0]
            if nonzero.empty:
                print(f"  {col}: no non-zero values")
                continue

            med       = nonzero.median()
            warn_lvl  = PAYOUT_WARN.get(n_rest, 5000.0)
            warn_flag = " ⚠" if med > warn_lvl else " ✅"

            print(f"\n  {col}{warn_flag}")
            print(f"    Non-zero rows  : {len(nonzero):,} / {len(vals):,}")
            print(f"    Min            : ${nonzero.min():>12,.2f}")
            print(f"    Median         : ${med:>12,.2f}")
            print(f"    Mean           : ${nonzero.mean():>12,.2f}")
            print(f"    Max            : ${nonzero.max():>12,.2f}")
            print(f"    p25 / p75      : "
                  f"${nonzero.quantile(0.25):>10,.2f} / "
                  f"${nonzero.quantile(0.75):>10,.2f}")
            print(f"    Sample (first 8): "
                  f"{[round(v, 2) for v in nonzero.head(8).tolist()]}")
            if med > warn_lvl:
                print(f"  ⚠  Median ${med:,.2f} > threshold ${warn_lvl:,.0f}.")
                print(f"     Verify this is a per-$2-ticket payout.")

    print(f"  {'─'*62}\n")


# ==========================================
# CORE SCOUT
# ==========================================

def run_true_scout(df: pd.DataFrame, n_rest: int) -> pd.DataFrame:
    """
    Single-pass scout with correct eligibility model.

    For each parseable race with n_runners in [FIELD_MIN, FIELD_MAX]:
      - Every (key, rest_combo) where all ranks exist in 1..n_runners
        gets eligible += 1 (you hold this ticket this race)
      - If the combo matches the actual result, hits += 1, paid += payout

    Eligibility is automatically correct:
      A combo requiring rank-10 only appears in races with n_runners >= 10.
      In smaller fields it simply isn't enumerated.
    """
    ticket_cost = math.perm(n_rest, n_rest - 1) * BASE_BET
    f_min       = FIELD_MIN.get(n_rest, n_rest + 1)
    f_max       = FIELD_MAX.get(n_rest, 20)

    payouts_arr = get_payout_array(df, n_rest)
    raw_list    = (df[RANKED_RESULTS_COL].tolist()
                   if RANKED_RESULTS_COL in df.columns
                   else [None] * len(df))

    stats: dict[tuple, dict] = {}

    for i, raw in enumerate(raw_list):
        positions = parse_ranked_result(raw)
        if positions is None:
            continue
        n_runners = len(positions)
        if not (f_min <= n_runners <= f_max):
            continue

        payout    = float(payouts_arr[i])
        all_ranks = list(range(1, n_runners + 1))

        for key in all_ranks:
            rest_pool = [r for r in all_ranks if r != key]
            for rest_combo in itertools.combinations(rest_pool, n_rest):
                sk = (n_runners, key, rest_combo)
                if sk not in stats:
                    stats[sk] = {"eligible": 0, "hits": 0, "paid": 0.0,
                                 "payouts": []}
                stats[sk]["eligible"] += 1
                if is_match(positions, key, frozenset(rest_combo), n_rest):
                    stats[sk]["hits"]  += 1
                    stats[sk]["paid"]  += payout
                    stats[sk]["payouts"].append(payout)

    rows = []
    for (n_runners, key, rest_combo), d in stats.items():
        eligible = d["eligible"]
        hits     = d["hits"]
        paid     = d["paid"]

        # Apply both filters
        if eligible < MIN_ELIGIBLE:
            continue
        if hits < MIN_HITS:
            continue

        cost        = eligible * ticket_cost
        net         = paid - cost
        roi         = (net / cost * 100)         if cost > 0     else 0.0
        hit_pct     = (hits / eligible * 100)    if eligible > 0 else 0.0
        ev_per_race = net / eligible             if eligible > 0 else 0.0
        avg_pay     = paid / hits                if hits > 0     else 0.0
        med_pay     = (float(np.median(d["payouts"]))
                       if d["payouts"] else 0.0)
        be_rate     = breakeven_hit_rate(ticket_cost, avg_pay)

        rows.append({
            "n_runners":    n_runners,
            "key_rank":     key,
            "rest_tuple":   rest_combo,
            "eligible":     eligible,
            "hits":         hits,
            "paid":         round(paid, 2),
            "HR_pct":       round(hit_pct, 4),
            "Cost":         round(cost, 2),
            "Net_PnL":      round(net, 2),
            "ROI_pct":      round(roi, 2),
            "EV_per_race":  round(ev_per_race, 4),
            "avg_payout":   round(avg_pay, 2),
            "med_payout":   round(med_pay, 2),
            "be_hit_rate":  round(be_rate, 4),   # % needed to break even
        })

    if not rows:
        return pd.DataFrame()

    sort_col_map = {
        "roi":         "ROI_pct",
        "net_pnl":     "Net_PnL",
        "hit_rate":    "HR_pct",
        "ev_per_race": "EV_per_race",
    }
    sort_col = sort_col_map.get(SORT_BY, "ROI_pct")

    return (
        pd.DataFrame(rows)
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )


# ==========================================
# L/S RE-AGGREGATION
# ==========================================

def reaggregate_ls(abs_df: pd.DataFrame, n_rest: int) -> pd.DataFrame:
    """
    Translate absolute ranks → L/S labels, re-aggregate, re-derive metrics.
    MIN_HITS applied again after aggregation (larger pool may now qualify).
    """
    if abs_df.empty:
        return pd.DataFrame()

    ticket_cost = math.perm(n_rest, n_rest - 1) * BASE_BET

    key_lbls, rest_lbls = [], []
    for _, r in abs_df.iterrows():
        kl, rl = label_combo(
            int(r["key_rank"]),
            tuple(r["rest_tuple"]),
            int(r["n_runners"]),
        )
        key_lbls.append(kl)
        rest_lbls.append(rl)

    work = abs_df.copy()
    work["Key_Label"]  = key_lbls
    work["Rest_Label"] = rest_lbls

    grouped = (
        work
        .groupby(["Key_Label", "Rest_Label"], sort=False)
        .agg(
            eligible      = ("eligible",   "sum"),
            hits          = ("hits",       "sum"),
            paid          = ("paid",       "sum"),
            n_field_sizes = ("n_runners",  "nunique"),
        )
        .reset_index()
    )

    # Re-derive all metrics from summed raw counts
    grouped["Cost"]       = grouped["eligible"] * ticket_cost
    grouped["Net_PnL"]    = grouped["paid"] - grouped["Cost"]
    grouped["ROI_pct"]    = (
        (grouped["Net_PnL"] / grouped["Cost"] * 100)
        .where(grouped["Cost"] > 0, 0.0)
    )
    grouped["HR_pct"]     = (
        (grouped["hits"] / grouped["eligible"] * 100)
        .where(grouped["eligible"] > 0, 0.0)
    )
    grouped["EV_per_race"] = (
        (grouped["Net_PnL"] / grouped["eligible"])
        .where(grouped["eligible"] > 0, 0.0)
    )
    grouped["avg_payout"] = (
        (grouped["paid"] / grouped["hits"])
        .where(grouped["hits"] > 0, 0.0)
    )
    grouped["be_hit_rate"] = grouped.apply(
        lambda r: breakeven_hit_rate(ticket_cost, r["avg_payout"]), axis=1
    )

    # Apply filters to aggregated pool
    grouped = grouped[
        (grouped["eligible"] >= MIN_ELIGIBLE) &
        (grouped["hits"]     >= MIN_HITS)
    ]

    sort_col_map = {
        "roi":         "ROI_pct",
        "net_pnl":     "Net_PnL",
        "hit_rate":    "HR_pct",
        "ev_per_race": "EV_per_race",
    }
    sort_col = sort_col_map.get(SORT_BY, "ROI_pct")

    return (
        grouped
        .round({"HR_pct": 4, "Net_PnL": 2, "ROI_pct": 2,
                "Cost": 2, "EV_per_race": 4, "be_hit_rate": 4})
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )


# ==========================================
# OOS MERGE
# ==========================================

def merge_oos(is_ls: pd.DataFrame, oos_ls: pd.DataFrame) -> pd.DataFrame:
    if oos_ls.empty:
        return is_ls.copy()

    oos_sub = (
        oos_ls[["Key_Label", "Rest_Label",
                "eligible", "hits", "HR_pct",
                "Net_PnL", "ROI_pct", "EV_per_race"]]
        .rename(columns={
            "eligible":    "OOS_Elig",
            "hits":        "OOS_Hits",
            "HR_pct":      "OOS_HR_pct",
            "Net_PnL":     "OOS_PnL",
            "ROI_pct":     "OOS_ROI",
            "EV_per_race": "OOS_EV",
        })
    )
    return pd.merge(is_ls, oos_sub,
                    on=["Key_Label", "Rest_Label"],
                    how="left")


# ==========================================
# REPORT PRINTER
# ==========================================

def print_report(final_df: pd.DataFrame, n_rest: int) -> None:
    if final_df is None or final_df.empty:
        print("  No results to display.")
        return

    ticket_cost = math.perm(n_rest, n_rest - 1) * BASE_BET
    has_oos     = "OOS_PnL" in final_df.columns

    # ── Header ────────────────────────────────────────────────────────────────
    print(
        f"\n  {'Key':>4}  {'Rest Pool (L/S)':<24} | "
        f"{'Elig':>6} {'Hits':>5} {'HR%':>6} "
        f"{'BE%':>6} {'AvgPay':>8} "
        f"{'IS_PnL':>10} {'IS_ROI':>8} {'IS_EV':>7}"
    )
    if has_oos:
        print(
            f"  {'':>4}  {'':24} | "
            f"{'':>6} {'':>5} {'OOS_HR%':>6} "
            f"{'':>6} {'':>8} "
            f"{'OOS_PnL':>10} {'OOS_ROI':>8} {'OOS_EV':>7}"
        )
    print("  " + "─" * (88 + (35 if has_oos else 0)))

    for _, r in final_df.head(TOP_N_DISPLAY).iterrows():
        nfs     = int(r.get("n_field_sizes", 1))
        nfs_tag = f"[{nfs}fs]" if nfs > 1 else "     "

        # IS line
        print(
            f"  {r['Key_Label']:>4}  {r['Rest_Label']:<24} | "
            f"{int(r['eligible']):>6} {int(r['hits']):>5} "
            f"{r['HR_pct']:>5.2f}% "
            f"{r['be_hit_rate']:>5.2f}% "
            f"${r['avg_payout']:>7,.0f} "
            f"${r['Net_PnL']:>9,.0f} {r['ROI_pct']:>7.1f}% "
            f"${r['EV_per_race']:>6.2f} {nfs_tag}"
        )

        # OOS line (indented, same row group)
        if has_oos:
            oos_el   = int(r["OOS_Elig"])    if pd.notna(r.get("OOS_Elig"))   else None
            oos_hits = int(r["OOS_Hits"])    if pd.notna(r.get("OOS_Hits"))   else None
            oos_hr   = float(r["OOS_HR_pct"])if pd.notna(r.get("OOS_HR_pct")) else None
            oos_pnl  = float(r["OOS_PnL"])  if pd.notna(r.get("OOS_PnL"))    else None
            oos_roi  = float(r["OOS_ROI"])  if pd.notna(r.get("OOS_ROI"))    else None
            oos_ev   = float(r["OOS_EV"])   if pd.notna(r.get("OOS_EV"))     else None

            # Hit rate persistence: did OOS HR stay above breakeven?
            be  = float(r["be_hit_rate"])
            st  = "  "
            if oos_hr is not None:
                if oos_hr >= be:
                    st = "✅"  # OOS hit rate still covers cost
                elif oos_hits is not None and oos_hits > 0:
                    st = "⚠️ "  # hit but below breakeven rate
                else:
                    st = "❌"  # zero hits in OOS

            el_s   = f"{oos_el:>6}"        if oos_el   is not None else f"{'N/A':>6}"
            hits_s = f"{oos_hits:>5}"      if oos_hits is not None else f"{'N/A':>5}"
            hr_s   = f"{oos_hr:>5.2f}%"   if oos_hr   is not None else f"{'N/A':>6}"
            pnl_s  = f"${oos_pnl:>9,.0f}" if oos_pnl  is not None else f"{'N/A':>10}"
            roi_s  = f"{oos_roi:>7.1f}%"  if oos_roi  is not None else f"{'N/A':>8}"
            ev_s   = f"${oos_ev:>6.2f}"   if oos_ev   is not None else f"{'N/A':>7}"

            print(
                f"  {'':>4}  {'  → OOS':<24} | "
                f"{el_s} {hits_s} "
                f"{hr_s} "
                f"{'':>6} {'':>8} "
                f"{pnl_s} {roi_s} "
                f"{ev_s} {st}"
            )
        print()   # blank line between strategies for readability

    # ── Summary ───────────────────────────────────────────────────────────────
    pos_is = int((final_df["Net_PnL"] > 0).sum())
    print(f"  IS profitable: {pos_is}/{len(final_df)}")

    if has_oos:
        oos_pnl_col = final_df["OOS_PnL"]
        oos_hr_col  = final_df["OOS_HR_pct"]
        be_col      = final_df["be_hit_rate"]

        pos_oos  = int(oos_pnl_col.gt(0).sum())
        tot_oos  = int(oos_pnl_col.notna().sum())
        # Count strategies where OOS hit rate stayed at or above breakeven
        hr_pass  = int(
            (oos_hr_col.notna() & (oos_hr_col >= be_col)).sum()
        )
        print(f"  OOS positive P&L     : {pos_oos}/{tot_oos}")
        print(f"  OOS HR ≥ breakeven   : {hr_pass}/{tot_oos}  "
              f"← structural robustness signal")

    print(f"\n  Ticket: ${ticket_cost:.0f}/race | "
          f"MIN_ELIGIBLE={MIN_ELIGIBLE} | MIN_HITS={MIN_HITS} | "
          f"Sort={SORT_BY}")
    print(f"  BE% = breakeven hit rate needed to cover ${ticket_cost:.0f} ticket")
    print(f"  EV  = expected value per race (Net_PnL / Eligible)")
    print(f"  [Nfs] = number of distinct field sizes merged into label")


# ==========================================
# MAIN
# ==========================================

def main() -> None:
    print("\n" + "=" * 80)
    print("  SUPERFECTA / TRIFECTA BOX ANALYZER v7.5")
    print("  True P&L | L/S grouping | MIN_HITS filter | Breakeven rate")
    print("=" * 80)

    print("\n📂 Loading data...")
    try:
        df_train = pd.read_csv(TRAIN_FILE, low_memory=False)
        print(f"  Train (IS) : {len(df_train):,} races")
    except FileNotFoundError:
        print(f"  ❌ Could not find {TRAIN_FILE}")
        return

    try:
        df_test = pd.read_csv(TEST_FILE, low_memory=False)
        print(f"  Test  (OOS): {len(df_test):,} races")
    except FileNotFoundError:
        df_test = pd.DataFrame()
        print("  ⚠  No OOS file found — IS-only mode.")

    if RANKED_RESULTS_COL not in df_train.columns:
        print(f"  ❌ Column '{RANKED_RESULTS_COL}' not found.")
        return

    # Payout diagnostic (informational only — no prompt)
    run_payout_diagnostic(df_train, "TRAIN")

    for n_rest, bet_name in [(3, "TRIFECTA  ($12 box)"),
                             (4, "SUPERFECTA ($48 box)")]:
        print(f"\n{'═'*80}")
        print(f"  {bet_name}")
        print("═" * 80)

        # IS
        print(f"  Scanning IS data ...")
        is_abs = run_true_scout(df_train, n_rest)
        if is_abs.empty:
            print(f"  No combos passed MIN_ELIGIBLE={MIN_ELIGIBLE} "
                  f"and MIN_HITS={MIN_HITS}.")
            continue

        is_ls = reaggregate_ls(is_abs, n_rest)
        if is_ls.empty:
            print("  No L/S groups passed filters after aggregation.")
            continue

        top_is_ls = (
            is_ls[is_ls["Net_PnL"] > 0]
            .reset_index(drop=True)
        )

        if top_is_ls.empty:
            print("  ❌ No L/S strategies showed positive Net P&L.")
            continue

        print(f"  IS: {len(is_ls):,} L/S groups | "
              f"{len(top_is_ls):,} profitable | "
              f"sorted by {SORT_BY}")

        # OOS
        final_df = top_is_ls.copy()
        if not df_test.empty and RANKED_RESULTS_COL in df_test.columns:
            print(f"  Scanning OOS data ...")
            oos_abs = run_true_scout(df_test, n_rest)
            if not oos_abs.empty:
                oos_ls   = reaggregate_ls(oos_abs, n_rest)
                final_df = merge_oos(top_is_ls, oos_ls)
            else:
                print("  ⚠  OOS scout: no results.")

        print_report(final_df, n_rest)

        csv_name = (f"Box_Results_"
                    f"{'Trifecta' if n_rest == 3 else 'Superfecta'}_v75.csv")
        save_df = final_df.copy()
        if "rest_tuple" in save_df.columns:
            save_df["rest_tuple"] = save_df["rest_tuple"].astype(str)
        save_df.to_csv(csv_name, index=False)
        print(f"\n  💾 Saved → {csv_name}")

    print("\n  ✅ Analysis complete.\n")


if __name__ == "__main__":
    main()
