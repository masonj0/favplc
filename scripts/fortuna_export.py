#!/usr/bin/env python3
import sqlite3
import pandas as pd
import argparse
import os
from pathlib import Path

def export_db(db_path, output_dir, format_type):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)

    # Get list of tables
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting tables to {output_dir} in {format_type.upper()} format...")
    exported_count = 0

    for table_name in tables['name']:
        if table_name == 'sqlite_sequence':
            continue
        print(f"  Exporting table: {table_name}")
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

            if format_type in ['csv', 'both']:
                csv_path = os.path.join(output_dir, f"{table_name}.csv")
                df.to_csv(csv_path, index=False)
                print(f"    Saved to {csv_path}")

            if format_type in ['tsv', 'both']:
                tsv_path = os.path.join(output_dir, f"{table_name}.tsv")
                df.to_csv(tsv_path, index=False, sep='\t')
                print(f"    Saved to {tsv_path}")

            exported_count += 1
        except Exception as table_err:
            print(f"  Error exporting table {table_name}: {table_err}")

    if exported_count == 0:
        print("No tables found to export.")
    else:
        print(f"Export complete! {exported_count} tables saved to {output_dir}")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FortunaDB to CSV/TSV")
    parser.add_argument("db_path", nargs="?", default="fortuna.db", help="Path to FortunaDB (default: fortuna.db)")
    parser.add_argument("--output-dir", "-o", default="db_export", help="Output directory (default: db_export)")
    parser.add_argument("--format", "-f", choices=['csv', 'tsv', 'both'], default='both', help="Output format (default: both)")

    args = parser.parse_args()
    export_db(args.db_path, args.output_dir, args.format)
