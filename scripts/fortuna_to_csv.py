#!/usr/bin/env python3
import sqlite3
import pandas as pd
import argparse
import os
from pathlib import Path

def export_to_csv(db_path, output_dir):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)

    # Get list of tables
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

    print(f"Exporting tables to {output_dir}...")
    exported_count = 0
    for table_name in tables['name']:
        if table_name == 'sqlite_sequence':
            continue
        print(f"  Exporting table: {table_name}")
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            csv_file = output_path / f"{table_name}.csv"
            df.to_csv(csv_file, index=False)
            exported_count += 1
        except Exception as e:
            print(f"  Error exporting {table_name}: {e}")

    conn.close()
    if exported_count == 0:
        # Create a dummy file if no tables were exported to avoid GHA upload errors
        (output_path / "empty_db.txt").write_text("Database contained no exportable tables.")

    print(f"Export complete! {exported_count} tables exported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FortunaDB to CSV")
    parser.add_argument("db_path", nargs="?", default="fortuna.db", help="Path to FortunaDB (default: fortuna.db)")
    parser.add_argument("--output", "-o", default="db_export", help="Output directory for CSVs (default: db_export)")

    args = parser.parse_args()
    export_to_csv(args.db_path, args.output)
