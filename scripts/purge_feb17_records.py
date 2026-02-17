import asyncio
import sqlite3
import os
import sys
from pathlib import Path

# Add root to path so we can import fortuna
sys.path.insert(0, str(Path(__file__).parent.parent))

from fortuna import FortunaDB

async def purge_records():
    db = FortunaDB()
    # Initialize ensures tables exist and applies migrations
    await db.initialize()

    db_path = db.db_path
    print(f"Connecting to database at: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Target date: 2026-02-17
        # Since all timestamps are stored in Eastern Time (EST/EDT) as per Project Convention,
        # we can use prefix matching on the ISO strings.
        target_prefix = '2026-02-17'

        print(f"Purging all records NOT from {target_prefix} (USA EST)...")

        # 1. Purge 'tips' table
        cursor.execute("SELECT COUNT(*) FROM tips")
        tips_before = cursor.fetchone()[0]

        cursor.execute(f"DELETE FROM tips WHERE report_date NOT LIKE '{target_prefix}%'")
        tips_deleted = cursor.rowcount

        cursor.execute("SELECT COUNT(*) FROM tips")
        tips_after = cursor.fetchone()[0]

        print(f"Tips: {tips_before} -> {tips_after} (Deleted {tips_deleted})")

        # 2. Purge 'harvest_logs' table
        cursor.execute("SELECT COUNT(*) FROM harvest_logs")
        logs_before = cursor.fetchone()[0]

        cursor.execute(f"DELETE FROM harvest_logs WHERE timestamp NOT LIKE '{target_prefix}%'")
        logs_deleted = cursor.rowcount

        cursor.execute("SELECT COUNT(*) FROM harvest_logs")
        logs_after = cursor.fetchone()[0]

        print(f"Harvest Logs: {logs_before} -> {logs_after} (Deleted {logs_deleted})")

        conn.commit()

        total_deleted = tips_deleted + logs_deleted
        if total_deleted > 0:
            print("VACUUMing database for optimal performance...")
            conn.execute("VACUUM")
            print("VACUUM complete.")

        print("\nCleanup Complete. Only Feb 17, 2026 records remain.")

    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(purge_records())
