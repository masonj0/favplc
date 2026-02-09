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

def merge_databases(primary_db, secondary_dbs):
    print(f"Merging {len(secondary_dbs)} databases into {primary_db}...")
    if not os.path.exists(primary_db):
        print(f"Primary DB {primary_db} does not exist. Using first secondary as primary.")
        if secondary_dbs:
            import shutil
            shutil.copy2(secondary_dbs[0], primary_db)
            secondary_dbs = secondary_dbs[1:]
        else:
            return

    conn = sqlite3.connect(primary_db)
    cursor = conn.cursor()

    # Ensure WAL mode
    conn.execute("PRAGMA journal_mode=WAL")

    for sec_db in secondary_dbs:
        if not os.path.exists(sec_db):
            print(f"Secondary DB {sec_db} not found, skipping.")
            continue

        print(f"Merging {sec_db}...")
        try:
            # Attach secondary DB
            cursor.execute(f"ATTACH DATABASE '{sec_db}' AS sec")

            # Merge tips table
            # We want to keep the one that is audited if there is a conflict
            # Or just update the fields if sec has more info.

            # 1. Update existing tips in primary if secondary has audit completed
            cursor.execute("""
                UPDATE tips
                SET
                    audit_completed = (SELECT audit_completed FROM sec.tips s WHERE s.race_id = tips.race_id),
                    verdict = (SELECT verdict FROM sec.tips s WHERE s.race_id = tips.race_id),
                    net_profit = (SELECT net_profit FROM sec.tips s WHERE s.race_id = tips.race_id),
                    selection_position = (SELECT selection_position FROM sec.tips s WHERE s.race_id = tips.race_id),
                    actual_top_5 = (SELECT actual_top_5 FROM sec.tips s WHERE s.race_id = tips.race_id),
                    actual_2nd_fav_odds = (SELECT actual_2nd_fav_odds FROM sec.tips s WHERE s.race_id = tips.race_id),
                    trifecta_payout = (SELECT trifecta_payout FROM sec.tips s WHERE s.race_id = tips.race_id),
                    trifecta_combination = (SELECT trifecta_combination FROM sec.tips s WHERE s.race_id = tips.race_id),
                    superfecta_payout = (SELECT superfecta_payout FROM sec.tips s WHERE s.race_id = tips.race_id),
                    superfecta_combination = (SELECT superfecta_combination FROM sec.tips s WHERE s.race_id = tips.race_id),
                    top1_place_payout = (SELECT top1_place_payout FROM sec.tips s WHERE s.race_id = tips.race_id),
                    top2_place_payout = (SELECT top2_place_payout FROM sec.tips s WHERE s.race_id = tips.race_id),
                    audit_timestamp = (SELECT audit_timestamp FROM sec.tips s WHERE s.race_id = tips.race_id)
                WHERE race_id IN (SELECT race_id FROM sec.tips WHERE audit_completed = 1)
            """)

            # 2. Insert new tips from secondary that don't exist in primary
            # We exclude 'id' to let it autoincrement
            cursor.execute("""
                INSERT OR IGNORE INTO tips (
                    race_id, venue, race_number, discipline, start_time, report_date,
                    is_goldmine, gap12, top_five, selection_number, predicted_2nd_fav_odds,
                    audit_completed, verdict, net_profit, selection_position,
                    actual_top_5, actual_2nd_fav_odds, trifecta_payout,
                    trifecta_combination, superfecta_payout,
                    superfecta_combination, top1_place_payout,
                    top2_place_payout, audit_timestamp
                )
                SELECT
                    race_id, venue, race_number, discipline, start_time, report_date,
                    is_goldmine, gap12, top_five, selection_number, predicted_2nd_fav_odds,
                    audit_completed, verdict, net_profit, selection_position,
                    actual_top_5, actual_2nd_fav_odds, trifecta_payout,
                    trifecta_combination, superfecta_payout,
                    superfecta_combination, top1_place_payout,
                    top2_place_payout, audit_timestamp
                FROM sec.tips
                WHERE race_id NOT IN (SELECT race_id FROM tips)
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
                    merged[k] = merged.get(k, 0) + v
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
