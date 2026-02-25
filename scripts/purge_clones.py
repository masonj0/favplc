#!/usr/bin/env python3
"""Purge duplicate selection clones from tips table.

A clone = same venue + same selection_name + same date but different race_number.
Keeps the entry with the lowest race_number (first occurrence).

Run: python3 scripts/purge_clones.py [--db fortuna.db] [--dry-run]
"""
import sqlite3
import argparse
import re
from collections import defaultdict


def extract_date(start_time: str) -> str:
    """Extract YYMMDD from either format."""
    if not start_time:
        return ''
    s = str(start_time).strip()
    # STORAGE_FORMAT: '260225T14:44:00'
    if re.match(r'^\d{6}T', s):
        return s[:6]
    # ISO: '2026-02-25T14:44:00'
    if re.match(r'^\d{4}-\d{2}-\d{2}', s):
        return s[2:4] + s[5:7] + s[8:10]
    return s[:6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='fortuna.db')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            "SELECT id, race_id, venue, race_number, selection_name, start_time, "
            "audit_completed, verdict "
            "FROM tips WHERE selection_name IS NOT NULL "
            "ORDER BY venue, start_time, race_number"
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"Error reading tips: {e}")
        conn.close()
        return

    # Group by (venue, date, selection_name)
    groups = defaultdict(list)
    for row in rows:
        date_key = extract_date(row['start_time'])
        key = (
            (row['venue'] or '').upper(),
            date_key,
            (row['selection_name'] or '').upper()
        )
        groups[key].append(dict(row))

    to_delete = []
    for key, entries in groups.items():
        if len(entries) <= 1:
            continue

        # Keep the one with lowest race_number; if tied, keep audited one
        entries.sort(key=lambda e: (
            0 if e['audit_completed'] else 1,  # Prefer audited
            e['race_number'] or 999,            # Prefer lower race number
            e['id']                              # Tiebreak: older record
        ))

        keeper = entries[0]
        duplicates = entries[1:]

        venue, date, sel = key
        print(f"\n  Clone group: {venue} / {date} / {sel}")
        print(f"    KEEP: id={keeper['id']} R{keeper['race_number']} "
              f"audit={keeper['audit_completed']} verdict={keeper['verdict']}")
        for dup in duplicates:
            print(f"    DROP: id={dup['id']} R{dup['race_number']} "
                  f"audit={dup['audit_completed']} verdict={dup['verdict']}")
            to_delete.append(dup['id'])

    print(f"\n{'='*60}")
    print(f"Total clones to purge: {len(to_delete)}")
    print(f"Total tips remaining:  {len(rows) - len(to_delete)}")

    if to_delete and not args.dry_run:
        placeholders = ','.join(['?'] * len(to_delete))
        conn.execute(f"DELETE FROM tips WHERE id IN ({placeholders})", to_delete)
        conn.commit()
        print(f"✅ Purged {len(to_delete)} clone entries.")
    elif to_delete:
        print(f"DRY RUN: Would purge {len(to_delete)} entries.")
    else:
        print("No clones found.")

    conn.close()


if __name__ == '__main__':
    main()
