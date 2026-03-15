#!/usr/bin/env python3
import sqlite3
import pandas as pd
import argparse
import os
from pathlib import Path

def export_to_excel(db_path, excel_path):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)

    # Get list of tables
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

    print(f"Exporting to {excel_path}...")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for table_name in tables['name']:
            if table_name == 'sqlite_sequence':
                continue
            print(f"  Reading table: {table_name}")
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            # Excel sheet names have a 31 character limit
            sheet_name = table_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    conn.close()
    print("Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FortunaDB to Excel")
    parser.add_argument("db_path", nargs="?", default="fortuna.db", help="Path to FortunaDB (default: fortuna.db)")
    parser.add_argument("--output", "-o", default="fortuna_export.xlsx", help="Output Excel path (default: fortuna_export.xlsx)")

    args = parser.parse_args()
    export_to_excel(args.db_path, args.output)
