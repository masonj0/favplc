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
    # Initialize ensures tables exist
    await db.initialize()

    db_path = db.db_path
    print(f"Connecting to database at: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Check counts before
        cursor.execute("SELECT COUNT(*) FROM tips")
        total_before = cursor.fetchone()[0]

        # We preserve only 2026-02-17 (today in sandbox)
        target_date = '2026-02-17'

        cursor.execute(f"SELECT COUNT(*) FROM tips WHERE report_date LIKE '{target_date}%'")
        today_count = cursor.fetchone()[0]

        print(f"Total records before purge: {total_before}")
        print(f"Records from {target_date}: {today_count}")

        # Delete everything NOT from target date
        cursor.execute(f"DELETE FROM tips WHERE report_date NOT LIKE '{target_date}%'")
        deleted_count = cursor.rowcount
        conn.commit()

        print(f"Deleted {deleted_count} records.")

        # Check counts after
        cursor.execute("SELECT COUNT(*) FROM tips")
        total_after = cursor.fetchone()[0]
        print(f"Total records after purge: {total_after}")

        if deleted_count > 0:
            print("VACUUMing database...")
            conn.execute("VACUUM")

    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(purge_records())
