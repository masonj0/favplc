#!/usr/bin/env python3
import json
import sqlite3
import argparse
import os
import glob
from datetime import datetime
from pathlib import Path

def merge_json_races(input_files, output_file):
    print(f"Merging {len(input_files)} JSON files...")
    all_races = []
    seen_keys = set()

    for file_path in input_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Handle both list of races and dict with 'races' key
                races = data.get('races', []) if isinstance(data, dict) else data
                for r in races:
                    # Simple canonical key for deduplication
                    venue = r.get('venue', 'Unknown').lower().replace(' ', '')
                    num = r.get('raceNumber', r.get('race_number', 0))
                    # Use only date part of start_time if possible
                    st = r.get('startTime', r.get('start_time', ''))
                    date_str = st[:10] if len(st) >= 10 else "Unknown"
                    key = f"{venue}|{num}|{date_str}"

                    if key not in seen_keys:
                        all_races.append(r)
                        seen_keys.add(key)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    with open(output_file, 'w') as f:
        json.dump(all_races, f, indent=2)
    print(f"Merged {len(all_races)} unique races into {output_file}")

def ensure_schema(conn):
    """Ensure the target database has the required tables and indexes."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS harvest_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            region TEXT,
            adapter_name TEXT NOT NULL,
            race_count INTEGER NOT NULL,
            max_odds REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            venue TEXT NOT NULL,
            race_number INTEGER NOT NULL,
            discipline TEXT,
            start_time TEXT NOT NULL,
            report_date TEXT NOT NULL,
            is_goldmine INTEGER NOT NULL,
            gap12 TEXT,
            top_five TEXT,
            selection_number INTEGER,
            selection_name TEXT,
            audit_completed INTEGER DEFAULT 0,
            verdict TEXT,
            net_profit REAL,
            selection_position INTEGER,
            actual_top_5 TEXT,
            actual_2nd_fav_odds REAL,
            trifecta_payout REAL,
            trifecta_combination TEXT,
            superfecta_payout REAL,
            superfecta_combination TEXT,
            top1_place_payout REAL,
            top2_place_payout REAL,
            predicted_2nd_fav_odds REAL,
            audit_timestamp TEXT,
            field_size INTEGER,
            match_confidence TEXT
        )
    """)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_race_id ON tips (race_id)")

def merge_databases(primary_db, secondary_dbs):
    print(f"Merging {len(secondary_dbs)} databases into {primary_db}...")

    # Ensure primary DB exists or use a secondary as base
    if not os.path.exists(primary_db):
        if secondary_dbs:
            print(f"Primary DB {primary_db} does not exist. Using {secondary_dbs[0]} as base.")
            import shutil
            shutil.copy2(secondary_dbs[0], primary_db)
            secondary_dbs = secondary_dbs[1:]
        else:
            print("No databases to merge.")
            return

    conn = sqlite3.connect(primary_db)
    cursor = conn.cursor()

    # Ensure schema in primary
    ensure_schema(conn)

    # Ensure WAL mode
    conn.execute("PRAGMA journal_mode=WAL")

    for sec_db in secondary_dbs:
        if not os.path.exists(sec_db):
            print(f"Secondary DB {sec_db} not found, skipping.")
            continue

        if not os.path.getsize(sec_db) > 0:
            print(f"Secondary DB {sec_db} is empty, skipping.")
            continue

        print(f"Merging {sec_db}...")
        try:
            # Attach secondary DB
            cursor.execute(f"ATTACH DATABASE '{sec_db}' AS sec")

            # Check if 'tips' table exists in secondary
            cursor.execute("SELECT name FROM sec.sqlite_master WHERE type='table' AND name='tips'")
            if not cursor.fetchone():
                print(f"Secondary DB {sec_db} is missing 'tips' table, skipping.")
                cursor.execute("DETACH DATABASE sec")
                continue

            # Merge tips table
            # We want to keep the one that is audited if there is a conflict
            # Or just update the fields if sec has more info.

            # UPSERT strategy for the 'tips' table (Gemini Memo Fix)
            # This ensures that audited rows from secondary overwrite pending rows in primary,
            # but we also preserve audited rows in primary if secondary is pending.
            cursor.execute("""
                INSERT INTO tips (
                    race_id, venue, race_number, discipline, start_time, report_date,
                    is_goldmine, gap12, top_five, selection_number, selection_name,
                    predicted_2nd_fav_odds, audit_completed, verdict, net_profit,
                    selection_position, actual_top_5, actual_2nd_fav_odds,
                    trifecta_payout, trifecta_combination, superfecta_payout,
                    superfecta_combination, top1_place_payout,
                    top2_place_payout, audit_timestamp, field_size, match_confidence
                )
                SELECT
                    race_id, venue, race_number, discipline, start_time, report_date,
                    is_goldmine, gap12, top_five, selection_number, selection_name,
                    predicted_2nd_fav_odds, audit_completed, verdict, net_profit,
                    selection_position, actual_top_5, actual_2nd_fav_odds,
                    trifecta_payout, trifecta_combination, superfecta_payout,
                    superfecta_combination, top1_place_payout,
                    top2_place_payout, audit_timestamp, field_size, match_confidence
                FROM sec.tips
                WHERE true
                ON CONFLICT(race_id) DO UPDATE SET
                    audit_completed = excluded.audit_completed,
                    verdict = excluded.verdict,
                    net_profit = excluded.net_profit,
                    selection_position = excluded.selection_position,
                    actual_top_5 = excluded.actual_top_5,
                    actual_2nd_fav_odds = excluded.actual_2nd_fav_odds,
                    trifecta_payout = excluded.trifecta_payout,
                    trifecta_combination = excluded.trifecta_combination,
                    superfecta_payout = excluded.superfecta_payout,
                    superfecta_combination = excluded.superfecta_combination,
                    top1_place_payout = excluded.top1_place_payout,
                    top2_place_payout = excluded.top2_place_payout,
                    audit_timestamp = excluded.audit_timestamp,
                    field_size = COALESCE(tips.field_size, excluded.field_size),
                    match_confidence = excluded.match_confidence
                WHERE excluded.audit_completed = 1 AND tips.audit_completed = 0
            """)

            # Merge harvest_logs table
            cursor.execute("""
                INSERT INTO harvest_logs (timestamp, region, adapter_name, race_count, max_odds)
                SELECT timestamp, region, adapter_name, race_count, max_odds FROM sec.harvest_logs
            """)

            conn.commit()
            cursor.execute("DETACH DATABASE sec")
        except Exception as e:
            print(f"Error merging {sec_db}: {e}")
            conn.rollback()

    conn.close()
    print("Database merge complete.")

def merge_harvest_jsons(pattern, output_file):
    print(f"Merging harvest JSONs matching {pattern}...")
    merged = {}
    for f_path in glob.glob(pattern):
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    if isinstance(v, dict):
                        if k not in merged:
                            merged[k] = {"count": 0, "max_odds": 0.0, "trust_ratio": 0.0}
                        elif not isinstance(merged[k], dict):
                            # Fallback if mixed types
                            merged[k] = {"count": int(merged[k]), "max_odds": 0.0, "trust_ratio": 0.0}

                        merged[k]["count"] += v.get("count", 0)
                        merged[k]["max_odds"] = max(merged[k].get("max_odds", 0.0), v.get("max_odds", 0.0))
                        merged[k]["trust_ratio"] = max(merged[k].get("trust_ratio", 0.0), v.get("trust_ratio", 0.0))
                    else:
                        if k not in merged:
                            merged[k] = v
                        elif isinstance(merged[k], dict):
                            merged[k]["count"] += v
                        else:
                            merged[k] += v
        except Exception as e:
            print(f"Error merging harvest JSON {f_path}: {e}")

    if merged:
        with open(output_file, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"Saved merged harvest to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Consolidate parallel Fortuna runs")
    parser.add_argument("--json-pattern", type=str, default="races_*.json", help="Glob pattern for race JSON files")
    parser.add_argument("--db-pattern", type=str, default="fortuna_*.db", help="Glob pattern for SQLite DB files")
    parser.add_argument("--harvest-pattern", type=str, default="*_harvest*.json", help="Glob pattern for harvest stats")
    parser.add_argument("--output-json", type=str, default="raw_races.json", help="Output merged JSON file")
    parser.add_argument("--output-db", type=str, default="fortuna.db", help="Final unified database")

    args = parser.parse_args()

    json_files = glob.glob(args.json_pattern)
    if json_files:
        merge_json_races(json_files, args.output_json)
    else:
        print("No JSON files found to merge.")

    db_files = glob.glob(args.db_pattern)
    if db_files:
        merge_databases(args.output_db, db_files)
    else:
        print("No database files found to merge.")

    # Merge harvest files
    # Discovery
    merge_harvest_jsons("discovery_harvest*.json", "discovery_harvest.json")
    # Results
    merge_harvest_jsons("results_harvest*.json", "results_harvest.json")

if __name__ == "__main__":
    main()
