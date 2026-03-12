#!/usr/bin/env python3
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

async def main():
    db = FortunaDB()
    await db.initialize()

    # Get all unaudited tips
    unaudited = await db.get_tips(audited=False)

    # Enrich with MTP and filter/sort
    races = []
    for r in unaudited:
        m = _mtp(r.get('start_time'))
        # Broaden window: 60 mins ago to 24 hours in future (Fix for AU/NZ coverage)
        if -60 <= m <= 1440:
            r['_mtp_val'] = m
            races.append(r)

    races.sort(key=lambda x: x['_mtp_val'])

    output_path = os.environ.get("GITHUB_STEP_SUMMARY", "/dev/stdout")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write("# 🏇 Upcoming Races\n\n")
        f.write(f"Generated: {now_eastern().strftime('%y%m%d %H:%M:%S')} ET\n\n")
        f.write(f"*Total unaudited tips in database: {len(unaudited)}*\n\n")

        if not races:
            f.write("No upcoming races within the next 24 hours discovered yet.\n")
        else:
            f.write("| MTP | Venue | R# | 1st Fav | 2nd Fav | Grade |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in races:
                mtp_s = _mtp_str(r['_mtp_val'])
                venue = r.get('venue', 'Unknown')
                rnum = r.get('race_number', '?')

                # Odds logic
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
