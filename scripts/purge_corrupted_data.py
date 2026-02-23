#!/usr/bin/env python3
"""
purge_corrupted_data.py
━━━━━━━━━━━━━━━━━━━━━━
Clean all corrupted, unreliable, and stale data from Fortuna DB and
its sibling caches / artifacts.

Addresses:
  1. ATR tips with race titles stuffed into venue name (all R1)
  2. ATRG audited tips with corrupted odds (parse_currency_value on SP)
  3. Tips where venue normalizes to "unknown" (unmatchable)
  4. Stale unaudited tips past the results lookback window
  5. Harvest logs recording bad metrics
  6. Stale JSON / artifact files

Usage:
  python scripts/purge_corrupted_data.py                  # Dry run — report only
  python scripts/purge_corrupted_data.py --execute        # Purge + reset
  python scripts/purge_corrupted_data.py --execute --aggressive  # Also delete stale tips
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── locate project root ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "fortuna.db"
STALE_DAYS = 4  # tips older than this with no audit are likely unmatchable
INSIGHT_UPGRADE_DATE = "2026-02-22"  # delete all records before this date

# ── output helpers ───────────────────────────────────────────────────
_buf: list[str] = []
_is_gha = "GITHUB_STEP_SUMMARY" in os.environ


def emit(line: str = "") -> None:
    _buf.append(line)
    print(line)


def flush_summary() -> None:
    if _is_gha:
        summary_path = os.environ["GITHUB_STEP_SUMMARY"]
        with open(summary_path, "a") as f:
            f.write("\n".join(_buf) + "\n")


# ── known good venues (populated from fortuna if available) ──────────
_CANONICAL_CACHE: Dict[str, str] = {}


def canonical(venue: str) -> str:
    """Resolve venue through fortuna.get_canonical_venue with cache."""
    if venue in _CANONICAL_CACHE:
        return _CANONICAL_CACHE[venue]
    try:
        import fortuna
        result = fortuna.get_canonical_venue(venue)
    except ImportError:
        # Fallback: lowercase, strip spaces/hyphens
        result = re.sub(r"[\s\-]+", "", venue.lower())
    _CANONICAL_CACHE[venue] = result
    return result


def _is_known_venue(venue: str) -> bool:
    """True if venue maps to something other than 'unknown'."""
    return canonical(venue) != "unknown"


# ── detection functions ──────────────────────────────────────────────

def detect_atr_venue_corruption(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """ATR thoroughbred tips where race title is stuffed into venue.

    Signature: race_id starts with 'atr_' (not 'atrg_'), and EITHER:
      - venue normalizes to 'unknown', OR
      - multiple tips share the same real venue + date + time but have
        different stored venues (race title fragments), OR
      - venue has 3+ words and the first 1-2 words are a known venue
    """
    rows = conn.execute("""
        SELECT id, race_id, venue, race_number, start_time, discipline,
               audit_completed, verdict
        FROM tips
        WHERE race_id LIKE 'atr\_%' ESCAPE '\\'
          AND race_id NOT LIKE 'atrg\_%' ESCAPE '\\'
    """).fetchall()

    corrupted: List[Dict[str, Any]] = []
    for row in rows:
        venue = row["venue"]
        words = venue.split()

        # Check 1: venue is unknown
        if not _is_known_venue(venue):
            corrupted.append(dict(row))
            continue

        # Check 2: first N words are a known venue, but full name is different
        # e.g. "Ludlow Farm" — "Ludlow" is known, but "Ludlow Farm" ≠ "Ludlow"
        if len(words) >= 2:
            for end in range(min(3, len(words)), 0, -1):
                prefix = " ".join(words[:end])
                if _is_known_venue(prefix) and canonical(prefix) != canonical(venue):
                    corrupted.append(dict(row))
                    break

    return corrupted


def detect_atr_race_number_collisions(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """ATR tips where multiple distinct start_times share venue + race_number + date.

    This catches the 'all R1' problem even for tips with correct venue names.
    """
    rows = conn.execute("""
        SELECT venue, race_number, DATE(start_time) as race_date,
               COUNT(DISTINCT start_time) as distinct_times,
               GROUP_CONCAT(id) as tip_ids
        FROM tips
        WHERE race_id LIKE 'atr\_%' ESCAPE '\\'
          AND race_id NOT LIKE 'atrg\_%' ESCAPE '\\'
        GROUP BY venue, race_number, DATE(start_time)
        HAVING COUNT(DISTINCT start_time) > 1
    """).fetchall()

    colliding_ids: Set[int] = set()
    for row in rows:
        for tid in row["tip_ids"].split(","):
            colliding_ids.add(int(tid))

    if not colliding_ids:
        return []

    placeholders = ",".join("?" * len(colliding_ids))
    return [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM tips WHERE id IN ({placeholders})",
            list(colliding_ids),
        ).fetchall()
    ]


def detect_atrg_bad_odds(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """ATRG audited tips — ALL have corrupted odds from parse_currency_value bug."""
    return [
        dict(r)
        for r in conn.execute("""
            SELECT id, race_id, venue, actual_2nd_fav_odds, verdict, net_profit
            FROM tips
            WHERE race_id LIKE 'atrg\_%' ESCAPE '\\'
              AND audit_completed = 1
        """).fetchall()
    ]


def detect_unknown_venues(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Tips where venue normalizes to 'unknown' — can never match results."""
    rows = conn.execute("SELECT * FROM tips").fetchall()
    return [dict(r) for r in rows if canonical(r["venue"]) == "unknown"]


def detect_pre_insight_records(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Records before 2026-02-20 (Insight Upgrade Date)."""
    return [
        dict(r)
        for r in conn.execute("""
            SELECT id, race_id, venue, start_time, report_date
            FROM tips
            WHERE report_date < ? OR start_time < ?
        """, (INSIGHT_UPGRADE_DATE, INSIGHT_UPGRADE_DATE)).fetchall()
    ]


def detect_stale_tips(
    conn: sqlite3.Connection,
    max_age_days: int = STALE_DAYS,
) -> List[Dict[str, Any]]:
    """Unaudited tips older than the results lookback window."""
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
    return [
        dict(r)
        for r in conn.execute("""
            SELECT id, race_id, venue, start_time
            FROM tips
            WHERE audit_completed = 0
              AND start_time < ?
        """, (cutoff,)).fetchall()
    ]


def detect_bad_harvest_logs(conn: sqlite3.Connection) -> int:
    """Harvest logs with maxOdds values that are clearly wrong (> 100)
    from the parse_currency_value bug on fractional odds."""
    row = conn.execute("""
        SELECT COUNT(*) FROM harvest_logs
        WHERE adapter_name = 'AtTheRacesGreyhoundResults'
          AND max_odds > 100
    """).fetchone()
    return row[0] if row else 0


def detect_pre_insight_logs(conn: sqlite3.Connection) -> int:
    """Harvest logs before the insight upgrade date."""
    row = conn.execute("""
        SELECT COUNT(*) FROM harvest_logs
        WHERE timestamp < ?
    """, (INSIGHT_UPGRADE_DATE,)).fetchone()
    return row[0] if row else 0


# ── purge functions ──────────────────────────────────────────────────

def delete_tips(conn: sqlite3.Connection, tip_ids: List[int], reason: str) -> int:
    if not tip_ids:
        return 0
    placeholders = ",".join("?" * len(tip_ids))
    conn.execute(
        f"DELETE FROM tips WHERE id IN ({placeholders})",
        tip_ids,
    )
    emit(f"  🗑️  Deleted {len(tip_ids)} tips — {reason}")
    return len(tip_ids)


def reset_audit(conn: sqlite3.Connection, tip_ids: List[int], reason: str) -> int:
    if not tip_ids:
        return 0
    placeholders = ",".join("?" * len(tip_ids))
    conn.execute(f"""
        UPDATE tips SET
            audit_completed = 0,
            verdict = NULL,
            net_profit = NULL,
            actual_2nd_fav_odds = NULL,
            selection_position = NULL,
            actual_top_5 = NULL,
            audit_timestamp = NULL,
            trifecta_payout = NULL,
            trifecta_combination = NULL,
            superfecta_payout = NULL,
            superfecta_combination = NULL,
            top1_place_payout = NULL,
            top2_place_payout = NULL,
            field_size = NULL
        WHERE id IN ({placeholders})
    """, tip_ids)
    emit(f"  🔄 Reset audit on {len(tip_ids)} tips — {reason}")
    return len(tip_ids)


def purge_harvest_logs(conn: sqlite3.Connection) -> int:
    """Delete harvest log entries with corrupted metrics."""
    # 1. Corrupted ATRG odds
    result = conn.execute("""
        DELETE FROM harvest_logs
        WHERE adapter_name = 'AtTheRacesGreyhoundResults'
          AND max_odds > 100
    """)
    count = result.rowcount
    if count:
        emit(f"  🗑️  Deleted {count} harvest_logs with corrupted maxOdds")

    # 2. Pre-insight records
    result3 = conn.execute("""
        DELETE FROM harvest_logs
        WHERE timestamp < ?
    """, (INSIGHT_UPGRADE_DATE,))
    if result3.rowcount:
        emit(f"  🗑️  Deleted {result3.rowcount} harvest_logs before {INSIGHT_UPGRADE_DATE}")
        count += result3.rowcount

    # 3. Also delete logs where race_count = 0 (no useful info)
    result2 = conn.execute("""
        DELETE FROM harvest_logs
        WHERE race_count = 0
    """)
    if result2.rowcount:
        emit(f"  🗑️  Deleted {result2.rowcount} harvest_logs with zero races")
        count += result2.rowcount
    return count


def reset_all_audited(conn: sqlite3.Connection) -> int:
    """Nuclear option: reset every audited tip for clean re-audit."""
    row = conn.execute(
        "SELECT COUNT(*) FROM tips WHERE audit_completed = 1",
    ).fetchone()
    count = row[0] if row else 0
    if count == 0:
        return 0
    conn.execute("""
        UPDATE tips SET
            audit_completed = 0,
            verdict = NULL,
            net_profit = NULL,
            actual_2nd_fav_odds = NULL,
            selection_position = NULL,
            actual_top_5 = NULL,
            audit_timestamp = NULL,
            trifecta_payout = NULL,
            trifecta_combination = NULL,
            superfecta_payout = NULL,
            superfecta_combination = NULL,
            top1_place_payout = NULL,
            top2_place_payout = NULL,
            field_size = NULL
        WHERE audit_completed = 1
    """)
    emit(f"  🔄 Reset ALL {count} audited tips for clean re-audit")
    return count


# ── file cleanup ─────────────────────────────────────────────────────

def clean_artifact_files(execute: bool) -> int:
    """Remove stale JSON artifacts and shadow DB copies."""
    patterns = [
        "races_*.json",
        "raw_races.json",
        "fortuna_*.db",           # regional shadow DBs
        "results_harvest_*.json",
        "discovery_harvest_*.json",
        "analytics_report.txt",
        "prediction_history.jsonl",
        "fortuna_report.html",
        "summary_grid.txt",
        "field_matrix.txt",
        "goldmine_report.txt",
        "race_data.json",
    ]
    # Files to reset (not delete — just empty them)
    reset_files = [
        "results_harvest.json",
        "discovery_harvest.json",
    ]

    cleaned = 0
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            # Don't delete the main DB!
            if path.name == "fortuna.db":
                continue
            if execute:
                path.unlink(missing_ok=True)
                emit(f"  🗑️  Deleted {path.name}")
            else:
                emit(f"  Would delete {path.name} ({path.stat().st_size / 1024:.1f} KB)")
            cleaned += 1

    for fname in reset_files:
        path = PROJECT_ROOT / fname
        if path.exists():
            if execute:
                path.write_text("{}", encoding="utf-8")
                emit(f"  📝 Reset {fname} to empty JSON")
            else:
                emit(f"  Would reset {fname}")
            cleaned += 1

    return cleaned


# ── GHA cache management ────────────────────────────────────────────

def clear_gha_caches(execute: bool) -> None:
    """Clear GitHub Actions caches so next run starts fresh."""
    emit("\n## 🗄️ GitHub Actions Cache\n")

    # Check if gh CLI is available
    gh = shutil.which("gh")
    if not gh:
        emit("⚠️ `gh` CLI not found — manual cache clearing required.")
        emit("")
        emit("Run these commands locally:")
        emit("```bash")
        emit("gh cache list --repo masonj0/favplc | grep fortuna-db")
        emit("# Then for each cache key:")
        emit("gh cache delete <key> --repo masonj0/favplc")
        emit("```")
        emit("")
        emit("Or clear all fortuna caches at once:")
        emit("```bash")
        emit('gh cache list --repo masonj0/favplc --json key -q ".[].key" | \\')
        emit('  grep "fortuna-db" | \\')
        emit("  xargs -I{} gh cache delete {} --repo masonj0/favplc")
        emit("```")
        return

    # List existing caches
    try:
        result = subprocess.run(
            [gh, "cache", "list", "--json", "key,sizeInBytes,createdAt"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            emit(f"⚠️ Failed to list caches: {result.stderr[:200]}")
            return

        caches = json.loads(result.stdout)
        db_caches = [
            c for c in caches
            if "fortuna-db" in c.get("key", "")
        ]

        if not db_caches:
            emit("✅ No fortuna-db caches found")
            return

        emit(f"Found **{len(db_caches)}** fortuna-db cache(s):\n")
        emit("```")
        for c in db_caches:
            size_kb = c.get("sizeInBytes", 0) / 1024
            emit(f"  {c['key'][:70]:70s}  {size_kb:>8.1f} KB  {c.get('createdAt', '?')[:19]}")
        emit("```\n")

        if execute:
            deleted = 0
            for c in db_caches:
                try:
                    del_result = subprocess.run(
                        [gh, "cache", "delete", c["key"]],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if del_result.returncode == 0:
                        emit(f"  🗑️  Deleted cache: `{c['key'][:60]}`")
                        deleted += 1
                    else:
                        emit(f"  ⚠️ Failed to delete `{c['key'][:60]}`: {del_result.stderr[:80]}")
                except Exception as e:
                    emit(f"  ⚠️ Error deleting `{c['key'][:60]}`: {e}")
            emit(f"\nDeleted **{deleted}/{len(db_caches)}** caches")
        else:
            emit(f"Would delete **{len(db_caches)}** caches (use `--execute`)")

    except Exception as e:
        emit(f"⚠️ Cache enumeration failed: {e}")


# ── main report + purge ─────────────────────────────────────────────

def run(
    db_path: Path,
    *,
    execute: bool = False,
    aggressive: bool = False,
) -> None:
    mode = "🔴 EXECUTE" if execute else "🟡 DRY RUN"
    emit(f"# 🧹 Fortuna Data Purge — {mode}\n")
    emit(f"**Database:** `{db_path}` ({db_path.stat().st_size / 1024:.1f} KB)")
    emit(f"**Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    if not db_path.exists():
        emit("❌ Database not found!")
        flush_summary()
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # ── pre-purge stats ──────────────────────────────────────────
    total = conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
    audited = conn.execute(
        "SELECT COUNT(*) FROM tips WHERE audit_completed = 1",
    ).fetchone()[0]
    logs = conn.execute("SELECT COUNT(*) FROM harvest_logs").fetchone()[0]

    emit("## 📊 Pre-Purge State\n")
    emit(f"- **Total tips:** {total}")
    emit(f"- **Audited:** {audited}")
    emit(f"- **Unaudited:** {total - audited}")
    emit(f"- **Harvest logs:** {logs}")
    emit("")

    # ── detect all issues ────────────────────────────────────────
    emit("## 🔍 Issues Detected\n")

    atr_corrupted = detect_atr_venue_corruption(conn)
    atr_collisions = detect_atr_race_number_collisions(conn)
    atrg_bad = detect_atrg_bad_odds(conn)
    unknown_venue = detect_unknown_venues(conn)
    stale = detect_stale_tips(conn) if aggressive else []
    stuck_t = [t for t in stale if t.get("discipline") == "Thoroughbred"]
    pre_insight = detect_pre_insight_records(conn)
    bad_logs = detect_bad_harvest_logs(conn)
    pre_insight_logs = detect_pre_insight_logs(conn)

    # Deduplicate: tips may appear in multiple categories
    all_delete_ids: Set[int] = set()
    all_reset_ids: Set[int] = set()

    # Category 0: Pre-Insight Records → DELETE
    if pre_insight:
        emit(f"### 🔴 Pre-Insight Records: {len(pre_insight)} tips\n")
        emit(f"Records captured before the {INSIGHT_UPGRADE_DATE} upgrades. No confidence in accuracy.\n")
        all_delete_ids.update(t["id"] for t in pre_insight)
    else:
        emit(f"### ✅ Pre-Insight Records: None before {INSIGHT_UPGRADE_DATE}\n")

    # Category 1: ATR venue corruption → DELETE
    if atr_corrupted:
        emit(f"### 🔴 ATR Venue Corruption: {len(atr_corrupted)} tips\n")
        emit("Race titles stuffed into venue field. Cannot be salvaged.\n")
        venue_groups: Dict[str, int] = defaultdict(int)
        for t in atr_corrupted:
            venue_groups[t["venue"]] += 1
        emit("```")
        for v, n in sorted(venue_groups.items(), key=lambda x: -x[1]):
            emit(f"  {v:55s} {n:3d} tips")
        emit("```\n")
        all_delete_ids.update(t["id"] for t in atr_corrupted)
    else:
        emit("### ✅ ATR Venue Corruption: None detected\n")

    # Category 2: ATR R1 collisions → DELETE remaining colliders
    collision_ids = {t["id"] for t in atr_collisions} - all_delete_ids
    if collision_ids:
        emit(f"### 🔴 ATR Race Number Collisions: {len(collision_ids)} additional tips\n")
        emit("Multiple races stored as R1 at same venue+date. Verdicts unreliable.\n")
        all_delete_ids.update(collision_ids)
    else:
        emit("### ✅ ATR Race Number Collisions: None remaining\n")

    # Category 3: ATRG bad odds → RESET (not delete — tips themselves are fine)
    if atrg_bad:
        emit(f"### 🟡 ATRG Corrupted Odds: {len(atrg_bad)} audited tips\n")
        emit("SP parsed with `parse_currency_value` instead of `parse_fractional_odds`.")
        emit("All `actual_2nd_fav_odds` values are wrong. Will reset for re-audit.\n")
        # Show the damage
        bad_odds = [
            t["actual_2nd_fav_odds"]
            for t in atrg_bad
            if t["actual_2nd_fav_odds"] is not None
        ]
        if bad_odds:
            emit(f"- Odds range: {min(bad_odds):.1f} – {max(bad_odds):.1f}")
            emit(f"- Mean: {sum(bad_odds) / len(bad_odds):.1f}")
            gt10 = sum(1 for o in bad_odds if o > 10)
            emit(f"- Values > 10.0 (clearly wrong): {gt10}/{len(bad_odds)}\n")
        all_reset_ids.update(t["id"] for t in atrg_bad)
    else:
        emit("### ✅ ATRG Odds: No corrupted audited tips\n")

    # Category 4: Unknown venue → DELETE
    unknown_not_already = [
        t for t in unknown_venue
        if t["id"] not in all_delete_ids
    ]
    if unknown_not_already:
        emit(f"### 🔴 Unknown Venue Tips: {len(unknown_not_already)} tips\n")
        emit("Venue normalizes to 'unknown' — can never match results.\n")
        venue_groups2: Dict[str, int] = defaultdict(int)
        for t in unknown_not_already:
            venue_groups2[t["venue"]] += 1
        emit("```")
        for v, n in sorted(venue_groups2.items(), key=lambda x: -x[1]):
            emit(f"  {v:55s} {n:3d} tips  (→ {canonical(v)})")
        emit("```\n")
        all_delete_ids.update(t["id"] for t in unknown_not_already)
    else:
        emit("### ✅ Unknown Venues: None\n")

    # Category 5: Stale unaudited (aggressive only)
    if aggressive and stale:
        stale_not_already = [t for t in stale if t["id"] not in all_delete_ids]
        if stale_not_already:
            stuck_t_ids = {t["id"] for t in stuck_t}
            t_only = [t for t in stale_not_already if t["id"] in stuck_t_ids]
            other_stale = [t for t in stale_not_already if t["id"] not in stuck_t_ids]

            if t_only:
                emit(f"### 🟠 Stuck Thoroughbred Tips: {len(t_only)} tips\n")
                emit(f"Thoroughbred races older than {STALE_DAYS} days with no result. Likely missing RP/Equibase coverage.\n")

            if other_stale:
                emit(f"### 🟠 Other Stale Unaudited Tips: {len(other_stale)} tips\n")
                emit(f"Older than {STALE_DAYS} days, unaudited. Results no longer available.\n")

            all_delete_ids.update(t["id"] for t in stale_not_already)
    elif aggressive:
        emit("### ✅ Stale Tips: None\n")

    # Category 6: Bad harvest logs
    if bad_logs or pre_insight_logs:
        emit(f"### 🟡 Corrupted/Stale Harvest Logs: {bad_logs + pre_insight_logs} entries\n")
        if bad_logs:
            emit(f"- {bad_logs} ATRG entries with maxOdds > 100\n")
        if pre_insight_logs:
            emit(f"- {pre_insight_logs} entries before {INSIGHT_UPGRADE_DATE}\n")

    # Category 7: Also reset ALL other audited tips
    # because the ATR R1 collision may have caused wrong-race matching
    # even for tips with correct venue names
    other_audited = conn.execute("""
        SELECT COUNT(*) FROM tips
        WHERE audit_completed = 1
          AND id NOT IN ({placeholders})
    """.format(
        placeholders=",".join(str(i) for i in (all_delete_ids | all_reset_ids)) or "0",
    )).fetchone()[0]

    if other_audited > 0:
        emit(f"### 🟡 Other Audited Tips: {other_audited}\n")
        emit("Will reset for clean re-audit (ATR collisions may have")
        emit("caused cross-contamination via relaxed key matching).\n")

    # ── summary ──────────────────────────────────────────────────
    emit("## 📋 Action Plan\n")
    emit(f"| Action | Count |")
    emit(f"|--------|-------|")
    emit(f"| DELETE (corrupted/unmatchable tips) | {len(all_delete_ids)} |")
    emit(f"| RESET audit (ATRG bad odds) | {len(all_reset_ids)} |")
    emit(f"| RESET audit (all remaining) | {other_audited} |")
    emit(f"| DELETE harvest logs | {bad_logs}+ |")
    emit(f"| **Tips retained after purge** | **{total - len(all_delete_ids)}** |")
    emit("")

    if not execute:
        emit("> ⚠️ **DRY RUN** — no changes made. Run with `--execute` to apply.\n")
        # Still do file and cache analysis
        emit("## 📁 Artifact Files\n")
        clean_artifact_files(execute=False)
        clear_gha_caches(execute=False)
        flush_summary()
        return

    # ── execute purge ────────────────────────────────────────────
    emit("## 🔧 Executing Purge\n")

    total_deleted = 0
    total_reset = 0

    # Delete corrupted tips
    if all_delete_ids:
        total_deleted += delete_tips(
            conn,
            list(all_delete_ids),
            "corrupted venues / unknown venues / stale / pre-insight",
        )

    # Reset ATRG audited tips
    atrg_reset = [tid for tid in all_reset_ids if tid not in all_delete_ids]
    if atrg_reset:
        total_reset += reset_audit(
            conn,
            atrg_reset,
            "ATRG corrupted odds",
        )

    # Reset ALL remaining audited tips
    total_reset += reset_all_audited(conn)

    # Purge harvest logs
    purge_harvest_logs(conn)

    # Vacuum
    conn.commit()
    emit("\n  🧹 Running VACUUM...")
    conn.execute("VACUUM")
    conn.close()

    new_size = db_path.stat().st_size / 1024
    emit(f"  📦 Database size after purge: {new_size:.1f} KB")

    # ── post-purge verification ──────────────────────────────────
    emit("\n## ✅ Post-Purge Verification\n")
    conn2 = sqlite3.connect(str(db_path))
    post_total = conn2.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
    post_audited = conn2.execute(
        "SELECT COUNT(*) FROM tips WHERE audit_completed = 1",
    ).fetchone()[0]
    post_logs = conn2.execute("SELECT COUNT(*) FROM harvest_logs").fetchone()[0]

    emit(f"- **Tips remaining:** {post_total} (was {total})")
    emit(f"- **Audited:** {post_audited} (should be 0)")
    emit(f"- **Harvest logs:** {post_logs} (was {logs})")
    emit("")

    # Sanity checks
    issues = []
    if post_audited > 0:
        issues.append(f"⚠️ {post_audited} tips still marked audited")

    # Check for remaining unknown venues
    remaining_unknown = conn2.execute("SELECT venue FROM tips").fetchall()
    unknowns = [r[0] for r in remaining_unknown if canonical(r[0]) == "unknown"]
    if unknowns:
        issues.append(f"⚠️ {len(unknowns)} tips still have unknown venues")

    # Check for remaining R1 collisions
    collisions = conn2.execute("""
        SELECT venue, race_number, DATE(start_time),
               COUNT(DISTINCT start_time) as n
        FROM tips
        WHERE race_id LIKE 'atr\\_%' ESCAPE '\\'
          AND race_id NOT LIKE 'atrg\\_%' ESCAPE '\\'
        GROUP BY venue, race_number, DATE(start_time)
        HAVING n > 1
    """).fetchall()
    if collisions:
        issues.append(f"⚠️ {len(collisions)} ATR R1 collision groups remain")

    if issues:
        for issue in issues:
            emit(f"- {issue}")
    else:
        emit("✅ All checks passed — database is clean")

    # Show surviving venue distribution
    emit("\n### Surviving tips by venue")
    emit("```")
    for row in conn2.execute("""
        SELECT venue, discipline, COUNT(*) as n
        FROM tips
        GROUP BY venue, discipline
        ORDER BY n DESC
        LIMIT 20
    """):
        disc = {'H': 'Harness', 'T': 'Thorough', 'G': 'Greyhound'}.get(
            row[1], row[1] or '?')
        c = canonical(row[0])
        flag = " ⚠️" if c == "unknown" else ""
        emit(f"  {row[0]:30s} [{disc:10s}] {row[2]:3d} tips  (→ {c}){flag}")
    emit("```")
    conn2.close()

    # ── clean files ──────────────────────────────────────────────
    emit("\n## 📁 Artifact Cleanup\n")
    clean_artifact_files(execute=True)

    # ── GHA caches ───────────────────────────────────────────────
    clear_gha_caches(execute=True)

    # ── final summary ────────────────────────────────────────────
    emit(f"\n## 🏁 Done\n")
    emit(f"- **Deleted:** {total_deleted} corrupted tips")
    emit(f"- **Reset:** {total_reset} audited tips (ready for clean re-audit)")
    emit(f"- **Tips remaining:** {post_total}")
    emit("")
    emit("**Next steps:**")
    emit("1. Deploy the adapter fixes (ATR venue, ATR race numbers, ATRG SP parsing)")
    emit("2. Run the pipeline — all tips will be freshly audited with correct logic")
    emit("3. Check the new scoreboard for trustworthy numbers")

    flush_summary()


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge corrupted data from Fortuna DB and caches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/purge_corrupted_data.py                  # See what would happen
  python scripts/purge_corrupted_data.py --execute        # Do it
  python scripts/purge_corrupted_data.py --execute --aggressive  # Also delete stale tips
  python scripts/purge_corrupted_data.py --db other.db    # Use different DB
        """,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually purge (default is dry run)",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help=f"Also delete unaudited tips older than {STALE_DAYS} days",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DB_PATH),
        help=f"Path to database (default: {DB_PATH})",
    )
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"❌ Database not found: {db}")
        sys.exit(1)

    run(db, execute=args.execute, aggressive=args.aggressive)


if __name__ == "__main__":
    main()
