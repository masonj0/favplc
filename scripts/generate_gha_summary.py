#!/usr/bin/env python3
import asyncio
import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure PYTHONPATH=.
try:
    from fortuna import FortunaDB
    from fortuna_utils import (
        now_eastern, from_storage_format, to_storage_format
    )
except ImportError:
    sys.path.append(os.getcwd())
    from fortuna import FortunaDB
    from fortuna_utils import (
        now_eastern, from_storage_format, to_storage_format
    )

EMOJI = {"CASHED": "✅", "CASHED_ESTIMATED": "✅~", "BURNED": "❌", "VOID": "⚪"}

def _mtp(start_time_str: Any) -> float:
    if not start_time_str: return 9999.0
    try:
        st = from_storage_format(str(start_time_str))
        if not st: return 9999.0
        now = now_eastern()
        return (st - now).total_seconds() / 60
    except Exception:
        return 9999.0

def _mtp_str(minutes: float) -> str:
    if minutes < 0: return "OFF"
    if minutes < 60: return f"{int(minutes)}m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h{mins}m"

async def get_snapshot_data(db: FortunaDB):
    def _get():
        conn = db._get_conn()
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

        return {
            "total_tips": total_tips,
            "audited": audited,
            "unverified": unverified,
            "goldmines": goldmines,
            "best_bets": best_bets,
            "harvest_rows": harvest_rows,
            "last_harvest": last_harvest,
            "recent_audited": recent_audited
        }
    return await db._run_in_executor(_get)

async def main():
    db_path = Path(os.environ.get("FORTUNA_DB_PATH", "fortuna.db"))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "/dev/stdout")

    # Use 'a' for GHA summary as per user's provided logic
    mode = 'a' if "GITHUB_STEP_SUMMARY" in os.environ else 'w'

    snapshot = None
    if not db_path.exists():
        with open(summary_path, mode, encoding="utf-8") as f:
            f.write("## 🐎 FortunaDB Snapshot\n\n")
            f.write(f"**Database:** `{db_path}` _(not found in workspace)_\n\n")
            f.write("⚠️ No database artifact available yet.\n")
    else:
        db = FortunaDB(str(db_path))
        await db.initialize()
        snapshot = await get_snapshot_data(db)

    with open(summary_path, mode, encoding="utf-8") as f:
        if snapshot:
            # --- FortunaDB Snapshot Section (User's preferred order) ---
            f.write("## 🐎 FortunaDB Snapshot\n\n")
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
            f.write("# 🏇 Upcoming Races\n\n")
            f.write(f"Generated: {now_eastern().strftime('%y%m%d %H:%M:%S')} ET\n\n")
            f.write(f"*Total unaudited tips in database: {snapshot['unverified']}*\n\n")

            # Get all unaudited tips
            unaudited_tips = await db.get_tips(audited=False)

            # Enrich with MTP and filter/sort
            upcoming_races = []
            for r in unaudited_tips:
                m = _mtp(r.get('start_time'))
                if -60 <= m <= 1440:
                    r['_mtp_val'] = m
                    upcoming_races.append(r)

            upcoming_races.sort(key=lambda x: x['_mtp_val'])

            if not upcoming_races:
                f.write("No upcoming races within the next 24 hours discovered yet.\n")
            else:
                f.write("| MTP | Venue | R# | 1st Fav | 2nd Fav | Grade |\n")
                f.write("|---|---|---|---|---|---|\n")
                for r in upcoming_races:
                    mtp_s = _mtp_str(r['_mtp_val'])
                    venue = r.get('venue', 'Unknown')
                    rnum = r.get('race_number', '?')

                    fav1 = r.get('predicted_fav_odds')
                    fav2 = r.get('predicted_2nd_fav_odds')

                    f1_s = f"{fav1:.2f}" if fav1 and fav1 > 0 else ".unk."
                    f2_s = f"{fav2:.2f}" if fav2 and fav2 > 0 else ".unk."

                    grade = r.get('qualification_grade', '-')
                    f.write(f"| {mtp_s} | {venue} | {rnum} | {f1_s} | {f2_s} | {grade} |\n")

            f.write("\n---\n*Refreshes every hour*\n")
            await db.close()

if __name__ == "__main__":
    asyncio.run(main())
