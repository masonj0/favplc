import sqlite3
import re
from datetime import datetime

def migrate_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Migrate tips table
    print("Migrating tips table...")
    cursor.execute("SELECT id, race_id, start_time, report_date, audit_timestamp FROM tips")
    tips = cursor.fetchall()

    for tip in tips:
        updates = {}

        # Migrate race_id: replace _20YYMMDD_ with _YYMMDD_
        new_race_id = re.sub(r'_20(\d{6})_', r'_\1_', tip['race_id'])
        if new_race_id != tip['race_id']:
            updates['race_id'] = new_race_id

        # Migrate start_time, report_date, audit_timestamp: YYYY-MM-DD -> YYMMDD
        for col in ['start_time', 'report_date', 'audit_timestamp']:
            val = tip[col]
            if val:
                # ISO format: 2026-02-24T12:00:00.000...
                # Target: 260224T12:00:00
                new_val = re.sub(r'^(\d{2})(\d{2})-(\d{2})-(\d{2})', r'\2\3\4', val)
                new_val = re.sub(r'(\d{2}:\d{2}:\d{2})\..*$', r'\1', new_val)
                new_val = re.sub(r'(\d{2}:\d{2}:\d{2})[+-].*$', r'\1', new_val)
                if new_val != val:
                    updates[col] = new_val

        if updates:
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            cursor.execute(f"UPDATE tips SET {set_clause} WHERE id = ?", (*updates.values(), tip['id']))

    # 2. Migrate harvest_logs table
    print("Migrating harvest_logs table...")
    cursor.execute("SELECT id, timestamp FROM harvest_logs")
    logs = cursor.fetchall()
    for log in logs:
        val = log['timestamp']
        if val:
            new_val = re.sub(r'^(\d{2})(\d{2})-(\d{2})-(\d{2})', r'\2\3\4', val)
            new_val = re.sub(r'(\d{2}:\d{2}:\d{2})\..*$', r'\1', new_val)
            new_val = re.sub(r'(\d{2}:\d{2}:\d{2})[+-].*$', r'\1', new_val)
            if new_val != val:
                cursor.execute("UPDATE harvest_logs SET timestamp = ? WHERE id = ?", (new_val, log['id']))

    # 3. Migrate schema_version
    print("Migrating schema_version table...")
    cursor.execute("SELECT version, applied_at FROM schema_version")
    rows = cursor.fetchall()
    for ver, val in rows:
        if val:
            new_val = re.sub(r'^(\d{2})(\d{2})-(\d{2})-(\d{2})', r'\2\3\4', val)
            new_val = re.sub(r'(\d{2}:\d{2}:\d{2})\..*$', r'\1', new_val)
            new_val = re.sub(r'(\d{2}:\d{2}:\d{2})[+-].*$', r'\1', new_val)
            if new_val != val:
                cursor.execute("UPDATE schema_version SET applied_at = ? WHERE version = ?", (new_val, ver))

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    import os
    db_path = os.environ.get("FORTUNA_DB_PATH", "fortuna.db")
    if os.path.exists(db_path):
        migrate_db(db_path)
    else:
        print(f"{db_path} not found.")
