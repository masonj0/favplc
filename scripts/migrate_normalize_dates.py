#!/usr/bin/env python3
"""One-time migration: Normalize all start_time values to STORAGE_FORMAT.

Run: python3 scripts/migrate_normalize_dates.py [--db fortuna.db] [--dry-run]
"""
import sqlite3
import argparse
import re
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo('America/New_York')
STORAGE_FORMAT = '%y%m%dT%H:%M:%S'

def parse_any(raw: str) -> str:
    """Convert any date format to STORAGE_FORMAT string."""
    if not raw:
        return raw
    s = str(raw).strip()

    # Already STORAGE_FORMAT
    if re.match(r'^\d{6}T\d{2}:\d{2}:\d{2}$', s):
        return s

    # ISO with timezone
    try:
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=EASTERN)
        else:
            dt = dt.astimezone(EASTERN)
        return dt.strftime(STORAGE_FORMAT)
    except (ValueError, TypeError):
        pass

    # ISO date only
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        try:
            dt = datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=EASTERN)
            return dt.strftime(STORAGE_FORMAT)
        except ValueError: pass

    # YYMMDD only
    if re.match(r'^\d{6}$', s):
        try:
            dt = datetime.strptime(s, '%y%m%d').replace(tzinfo=EASTERN)
            return dt.strftime(STORAGE_FORMAT)
        except ValueError: pass

    return s  # Can't parse — leave as-is


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='fortuna.db')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Columns that store dates
    date_cols = ['start_time', 'report_date', 'audit_timestamp']

    for col in date_cols:
        # Check column exists
        try:
            cursor = conn.execute(f"PRAGMA table_info('tips')")
            columns = [column[1] for column in cursor.fetchall()]
            if col not in columns:
                print(f"  Column '{col}' not found — skipping")
                continue
        except Exception as e:
            print(f"  Error checking column {col}: {e}")
            continue

        rows = conn.execute(f"SELECT id, {col} FROM tips WHERE {col} IS NOT NULL").fetchall()
        updates = []

        for row in rows:
            old_val = str(row[col])
            new_val = parse_any(old_val)
            if old_val != new_val:
                updates.append((new_val, row['id']))

        print(f"  {col}: {len(updates)}/{len(rows)} values need normalization")

        if updates and not args.dry_run:
            conn.executemany(f"UPDATE tips SET {col} = ? WHERE id = ?", updates)
            conn.commit()
            print(f"    → Updated {len(updates)} rows")
        elif updates and args.dry_run:
            print(f"    → DRY RUN: Would update {len(updates)} rows")
            for new_val, rid in updates[:5]:
                old = conn.execute(f"SELECT {col} FROM tips WHERE id=?", (rid,)).fetchone()[0]
                print(f"      id={rid}: '{old}' → '{new_val}'")

    conn.close()
    print("Done.")


if __name__ == '__main__':
    main()
