#!/usr/bin/env python3
import sqlite3
import pandas as pd
import argparse
import os
from pathlib import Path

def export_to_excel(db_path, output_file):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)

    # Get list of tables
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

    print(f"Exporting tables to {output_file}...")
    try:
        # We use openpyxl as the engine
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            exported_count = 0
            for table_name in tables['name']:
                if table_name == 'sqlite_sequence':
                    continue
                print(f"  Exporting table: {table_name}")
                try:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    # Excel sheet names have a 31 character limit
                    sheet_name = table_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    exported_count += 1
                except Exception as table_err:
                    print(f"  Error exporting table {table_name}: {table_err}")

        if exported_count == 0:
            print("No tables found to export.")
        else:
            print(f"Export complete! {exported_count} tables saved to {output_file}")

    except Exception as e:
        print(f"Error during export: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FortunaDB to Excel")
    parser.add_argument("db_path", nargs="?", default="fortuna.db", help="Path to FortunaDB (default: fortuna.db)")
    parser.add_argument("--output", "-o", default="fortuna_export.xlsx", help="Output Excel file (default: fortuna_export.xlsx)")

    args = parser.parse_args()
    export_to_excel(args.db_path, args.output)
