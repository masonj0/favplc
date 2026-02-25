#!/usr/bin/env python3
"""Backfill source column in tips table using harvest_logs correlation.

Strategy: Match tips to harvest_logs by timestamp proximity and venue.
The harvest_logs.adapter_name is the source we need.

Run: python3 scripts/backfill_source.py [--db fortuna.db] [--dry-run]
"""
import sqlite3
import argparse
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo('America/New_York')

# Known adapter → venue patterns (from adapter region sets)
GREYHOUND_ADAPTERS = {'AtTheRacesGreyhound', 'AtTheRacesGreyhoundResults'}
UK_TB_ADAPTERS = {'SkySports', 'RacingPost', 'Timeform', 'SportingLife', 'AtTheRaces'}
USA_ADAPTERS = {'TwinSpires', 'NYRABets', 'DRFResults'}
HARNESS_ADAPTERS = {'StandardbredCanada', 'StandardbredCanadaResults'}

# Venue patterns for reverse-mapping
UK_GREYHOUND_VENUES = {
    'MONMORE', 'NOTTINGHAM', 'HARLOW', 'TOWCESTER', 'HOVE',
    'CENTRAL PARK', 'SUNDERLAND', 'SUFFOLK', 'GREAT YARMOUTH',
    'VALLEY', 'ROMFORD', 'SHEFFIELD', 'KINSLEY', 'PELAW GRANGE',
    'HENLOW', 'CRAYFORD', 'BELLE VUE', 'PERRY BARR', 'YARMOUTH'
}

UK_TB_VENUES = {
    'DONCASTER', 'WOLVERHAMPTON', 'AYR', 'PLUMPTON', 'NEWCASTLE',
    'CHELTENHAM', 'ASCOT', 'KEMPTON', 'KEMPTON PARK', 'SANDOWN',
    'YORK', 'GOODWOOD', 'NEWMARKET', 'EPSOM', 'LINGFIELD',
    'HAYDOCK', 'CHEPSTOW', 'EXETER', 'CARLISLE', 'MUSSELBURGH'
}


def infer_source(venue: str, discipline: str) -> str:
    """Best-effort source inference from venue + discipline."""
    v = (venue or '').upper().strip()
    d = (discipline or '').upper().strip()

    if d in ('G', 'GREYHOUND') or v in UK_GREYHOUND_VENUES:
        return 'AtTheRacesGreyhound'

    if d in ('H', 'HARNESS'):
        return 'StandardbredCanada'

    if v in UK_TB_VENUES or any(v.startswith(uk) for uk in UK_TB_VENUES):
        return 'SkySports'  # Most likely for UK TB

    return 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='fortuna.db')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Check if tips table exists and has source column
    try:
        cursor = conn.execute("PRAGMA table_info('tips')")
        columns = [column[1] for column in cursor.fetchall()]
        if not columns:
            print("Table 'tips' not found.")
            conn.close()
            return
        if 'source' not in columns:
            print("Column 'source' not found in 'tips' table. Adding it now...")
            conn.execute("ALTER TABLE tips ADD COLUMN source TEXT")
            conn.commit()
    except Exception as e:
        print(f"Error checking schema: {e}")
        conn.close()
        return

    # Find tips with missing source
    missing = conn.execute(
        "SELECT id, venue, discipline, start_time "
        "FROM tips WHERE source IS NULL OR source = '' OR source = 'unknown'"
    ).fetchall()

    print(f"Tips with missing source: {len(missing)}")

    updates = []
    source_counts = {}

    for row in missing:
        inferred = infer_source(row['venue'], row['discipline'])
        if inferred != 'unknown':
            updates.append((inferred, row['id']))
            source_counts[inferred] = source_counts.get(inferred, 0) + 1

    print(f"\nInferred sources:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<30} {count:>5}")
    print(f"  {'(still unknown)':<30} {len(missing) - len(updates):>5}")

    if updates and not args.dry_run:
        conn.executemany("UPDATE tips SET source = ? WHERE id = ?", updates)
        conn.commit()
        print(f"\n✅ Updated {len(updates)} tips.")
    elif updates:
        print(f"\nDRY RUN: Would update {len(updates)} tips.")

    conn.close()


if __name__ == '__main__':
    main()
