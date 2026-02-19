#!/usr/bin/env python3
import sqlite3
import hashlib
import argparse
import sys
import os

def get_table_checksum(conn, table_name, key_columns):
    cursor = conn.cursor()
    # Check if table exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    if not cursor.fetchone():
        return None, 0

    # Sort by key columns to ensure deterministic hash
    order_by = ", ".join(key_columns)
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY {order_by}")

    hasher = hashlib.sha256()
    row_count = 0
    for row in cursor:
        row_count += 1
        hasher.update(str(row).encode('utf-8'))

    return hasher.hexdigest(), row_count

def compare_databases(db1_path, db2_path):
    if not os.path.exists(db1_path) or not os.path.exists(db2_path):
        print(f"Error: One or both database files missing. {db1_path}, {db2_path}")
        return False

    conn1 = sqlite3.connect(db1_path)
    conn2 = sqlite3.connect(db2_path)

    tables_to_check = {
        "tips": ["race_id"],
        "harvest_logs": ["timestamp", "adapter_name"]
    }

    mismatches = []

    print(f"Comparing {db1_path} vs {db2_path}")
    print("-" * 60)

    for table, keys in tables_to_check.items():
        hash1, count1 = get_table_checksum(conn1, table, keys)
        hash2, count2 = get_table_checksum(conn2, table, keys)

        status = "OK"
        if hash1 != hash2:
            status = "MISMATCH"
            mismatches.append(table)

        print(f"Table: {table:15s} | {status:8s} | Count1: {count1:5d} | Count2: {count2:5d}")
        if status == "MISMATCH":
            print(f"  Hash1: {hash1}")
            print(f"  Hash2: {hash2}")

            # Find specific missing race_ids if table is 'tips'
            if table == "tips":
                cursor1 = conn1.cursor()
                cursor2 = conn2.cursor()
                ids1 = set(row[0] for row in cursor1.execute("SELECT race_id FROM tips"))
                ids2 = set(row[0] for row in cursor2.execute("SELECT race_id FROM tips"))

                only_in_1 = ids1 - ids2
                only_in_2 = ids2 - ids1

                if only_in_1:
                    print(f"  Only in {db1_path} (first 10): {list(only_in_1)[:10]}")
                if only_in_2:
                    print(f"  Only in {db2_path} (first 10): {list(only_in_2)[:10]}")

    conn1.close()
    conn2.close()

    return len(mismatches) == 0

def main():
    parser = argparse.ArgumentParser(description="Shadow DB integrity comparison")
    parser.add_argument("db1", help="Path to first database (e.g., cached master)")
    parser.add_argument("db2", help="Path to second database (e.g., fresh snapshot)")
    args = parser.parse_args()

    success = compare_databases(args.db1, args.db2)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
