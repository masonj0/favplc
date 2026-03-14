#!/usr/bin/env python3
import asyncio
import os
import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure PYTHONPATH=.
try:
    from fortuna import FortunaDB
    from fortuna_utils import (
        now_eastern, from_storage_format, to_storage_format, resolve_daypart, DATE_FORMAT
    )
except ImportError:
    sys.path.append(os.getcwd())
    from fortuna import FortunaDB
    from fortuna_utils import (
        now_eastern, from_storage_format, to_storage_format, resolve_daypart, DATE_FORMAT
    )

EMOJI = {"CASHED": "✅", "CASHED_ESTIMATED": "✅~", "BURNED": "❌", "VOID": "⚪"}

def _mtp(start_time_str: Any) -> float:
    if not start_time_str: return 9999.0
    try:
        # from_storage_format handles YYMMDDTHH:MM:SS and ISO
        st = from_storage_format(str(start_time_str))
        if not st: return 9999.0
        now = now_eastern()
        # Both are expected to be tz-aware (Eastern)
        return (st - now).total_seconds() / 60
    except Exception as e:
        # Fallback to naive comparison if tz issues occur (should not happen with fortuna_utils)
        try:
            st = datetime.fromisoformat(str(start_time_str).replace('Z', '+00:00'))
            now = datetime.now(st.tzinfo)
            return (st - now).total_seconds() / 60
        except Exception:
            return 9999.0

def _mtp_str(minutes: float) -> str:
    if minutes < 0: return "OFF"
    if minutes < 60: return f"{int(minutes)}m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h{mins}m"

def load_snapshot_races(snapshot_dir: str = "snapshots") -> list:
    """Load races from the most recent quarter snapshot file."""
    now = now_eastern()
    daypart = resolve_daypart()
    tag = f"{daypart}_{now.strftime(DATE_FORMAT)}"
    path = Path(snapshot_dir) / f"{tag}_races.json"

    if not path.exists():
        # Fallback: grab the most recently modified snapshot
        snapshots = sorted(Path(snapshot_dir).glob("*_races.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshots:
            return []
        path = snapshots[0]

    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []

def get_fav_odds(race: dict) -> Tuple[Optional[float], Optional[float]]:
    """Returns (fav_odds, second_fav_odds) from snapshot runner data."""
    runners = [r for r in race.get('runners', []) if not r.get('scratched')]
    # Extraction logic for win_odds
    with_odds = []
    for r in runners:
        odds = r.get('win_odds')
        if odds and odds > 0:
            with_odds.append((r, float(odds)))

    if not with_odds:
        return None, None

    with_odds.sort(key=lambda x: x[1])
    fav = with_odds[0][1]
    sec = with_odds[1][1] if len(with_odds) >= 2 else None
    return fav, sec

async def get_snapshot_data(db: FortunaDB):
    def _get():
        conn = db._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        def get_one(query, default=0):
            try:
                cursor.execute(query)
                row = cursor.fetchone()
                return row[0] if row else default
            except sqlite3.OperationalError:
                return default

        total_tips   = get_one("SELECT COUNT(*) FROM tips")
        audited      = get_one("SELECT COUNT(*) FROM tips WHERE audit_completed = 1")
        unverified   = get_one("SELECT COUNT(*) FROM tips WHERE audit_completed = 0")
        goldmines    = get_one("SELECT COUNT(*) FROM tips WHERE is_goldmine = 1")
        best_bets    = get_one("SELECT COUNT(*) FROM tips WHERE is_best_bet = 1")
        harvest_rows = get_one("SELECT COUNT(*) FROM harvest_logs")

        last_harvest = None
        try:
            row = cursor.execute(
                "SELECT timestamp, adapter_name, race_count FROM harvest_logs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                last_harvest = dict(row)
        except sqlite3.OperationalError:
            pass

        recent_audited = []
        try:
            rows = cursor.execute(
                "SELECT audit_timestamp, venue, race_number, verdict, net_profit "
                "FROM tips WHERE audit_completed = 1 ORDER BY audit_timestamp DESC LIMIT 5"
            ).fetchall()
            recent_audited = [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass

        # Also get grades for unaudited tips to join with snapshot
        tip_grades = {}
        try:
            rows = cursor.execute(
                "SELECT race_id, qualification_grade FROM tips WHERE audit_completed = 0"
            ).fetchall()
            tip_grades = {row['race_id']: row['qualification_grade'] for row in rows}
        except sqlite3.OperationalError:
            pass

        return {
            "total_tips": total_tips,
            "audited": audited,
            "unverified": unverified,
            "goldmines": goldmines,
            "best_bets": best_bets,
            "harvest_rows": harvest_rows,
            "last_harvest": last_harvest,
            "recent_audited": recent_audited,
            "tip_grades": tip_grades
        }
    return await db._run_in_executor(_get)

async def main():
    db_path = Path(os.environ.get("FORTUNA_DB_PATH", "fortuna.db"))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "/dev/stdout")

    # Use 'a' for GHA summary
    mode = 'a' if "GITHUB_STEP_SUMMARY" in os.environ else 'w'

    db = FortunaDB(str(db_path))
    if db_path.exists():
        await db.initialize()
        snapshot = await get_snapshot_data(db)
    else:
        snapshot = {
            "total_tips": 0, "audited": 0, "unverified": 0, "goldmines": 0,
            "best_bets": 0, "harvest_rows": 0, "last_harvest": None,
            "recent_audited": [], "tip_grades": {}
        }

    # 1. Load Upcoming Races from Snapshot (Source of Truth for future races)
    all_races = load_snapshot_races()

    upcoming_races = []
    for r in all_races:
        m = _mtp(r.get('start_time'))
        # 60 mins ago to 90 mins ahead
        if -60 <= m <= 90:
            r['_mtp_val'] = m
            # Overlay grade from DB if available
            rid = r.get('id')
            r['_grade'] = snapshot['tip_grades'].get(rid, "—")
            upcoming_races.append(r)

    upcoming_races.sort(key=lambda x: x['_mtp_val'])

    with open(summary_path, mode, encoding="utf-8") as f:
        # --- FortunaDB Snapshot Section ---
        f.write("## 🐎 FortunaDB Snapshot\n\n")
        if not db_path.exists():
            f.write(f"**Database:** `{db_path}` _(not found in workspace)_\n\n")
            f.write("⚠️ No database artifact available yet.\n")
        else:
            f.write(f"- **Database path:** `{db_path}`\n")
            f.write(f"- **Total tips stored:** {snapshot['total_tips']}\n")
            f.write(f"- **Audited tips:** {snapshot['audited']}\n")
            f.write(f"- **Unverified tips:** {snapshot['unverified']}\n")
            f.write(f"- **Goldmines:** {snapshot['goldmines']}\n")
            f.write(f"- **Best bets:** {snapshot['best_bets']}\n")
            f.write(f"- **Harvest log entries:** {snapshot['harvest_rows']}\n")

            last_harvest = snapshot['last_harvest']
            if last_harvest:
                f.write(f"- **Last harvest:** {last_harvest['timestamp']} · {last_harvest['adapter_name']} ({last_harvest['race_count']} races)\n")

            f.write("\n### 🔍 Most Recent Audits\n\n")
            recent_audited = snapshot['recent_audited']
            if not recent_audited:
                f.write("No audited tips yet.\n")
            else:
                f.write("| Time | Venue | Race | Verdict | Profit |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                for row in recent_audited:
                    ts      = (row["audit_timestamp"] or "N/A")[:16].replace("T", " ")
                    venue   = row["venue"] or "Unknown"
                    rn      = row["race_number"] or "?"
                    verdict = row["verdict"] or "?"
                    emoji   = EMOJI.get(verdict, "")
                    profit  = f"${(row['net_profit'] or 0.0):+.2f}"
                    f.write(f"| {ts} | {venue} | R{rn} | {emoji} {verdict} | {profit} |\n")

        f.write("\n---\n")

        # --- Upcoming Races Section ---
        f.write("# 🏇 Upcoming Races (0-90 MTP)\n\n")
        f.write(f"Generated: {now_eastern().strftime('%y%m%d %H:%M:%S')} ET\n\n")

        if not upcoming_races:
            f.write("No upcoming races found in current snapshot (-60 to +90 MTP).\n")
        else:
            f.write("| MTP | Venue | R# | 1st Fav | 2nd Fav | Grade |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in upcoming_races:
                mtp_s = _mtp_str(r['_mtp_val'])
                venue = r.get('venue', 'Unknown')
                rnum = r.get('race_number', '?')

                # Get odds from snapshot runners
                fav1, fav2 = get_fav_odds(r)
                f1_s = f"{fav1:.2f}" if fav1 else ".unk."
                f2_s = f"{fav2:.2f}" if fav2 else ".unk."

                grade = r.get('_grade', '—')

                f.write(f"| {mtp_s} | {venue} | {rnum} | {f1_s} | {f2_s} | {grade} |\n")

        # --- Adapter Health Section (P2-ENH-6) ---
        health_path = Path("adapter_health_report.txt")
        if health_path.exists():
            f.write("\n## 📡 Adapter Health Dashboard\n\n")
            f.write("<details>\n<summary>Click to view detailed adapter status</summary>\n\n")
            f.write("```text\n")
            f.write(health_path.read_text(encoding="utf-8"))
            f.write("\n```\n")
            f.write("</details>\n")

        f.write("\n---\n*Refreshes every hour*\n")

    if db_path.exists():
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())
