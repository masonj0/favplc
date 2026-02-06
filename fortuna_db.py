from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)

DB_PATH = os.environ.get("FORTUNA_DB_PATH", "fortuna.db")

class FortunaDB:
    """
    SQLite backend for Fortuna to replace JSON storage.
    Handles persistence for tips, predictions, and audit outcomes.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._initialized = False

    async def initialize(self):
        """Creates the database schema if it doesn't exist."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    venue TEXT,
                    race_number INTEGER,
                    start_time TEXT,
                    report_date TEXT,
                    is_goldmine INTEGER,
                    gap12 REAL,
                    top_five TEXT,
                    selection_number INTEGER,
                    audit_completed INTEGER DEFAULT 0,
                    verdict TEXT,
                    net_profit REAL,
                    selection_position INTEGER,
                    actual_top_5 TEXT,
                    actual_2nd_fav_odds REAL,
                    trifecta_payout REAL,
                    trifecta_combination TEXT,
                    audit_timestamp TEXT
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_race_id ON tips (race_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_completed ON tips (audit_completed)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_venue ON tips (venue)")
            await db.commit()

        self._initialized = True
        logger.info("Database initialized", path=self.db_path)

    async def log_tips(self, tips: List[Dict[str, Any]]):
        """
        Logs new tips to the database with deduplication.
        """
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()

            for tip in tips:
                race_id = tip.get("race_id")
                # Deduplication check: same race_id in the last 4 hours
                async with db.execute(
                    "SELECT report_date FROM tips WHERE race_id = ? ORDER BY id DESC LIMIT 1",
                    (race_id,)
                ) as cursor:
                    last_report = await cursor.fetchone()
                    if last_report:
                        try:
                            last_time = datetime.fromisoformat(last_report[0])
                            if (datetime.now(timezone.utc) - last_time).total_seconds() < 4 * 3600:
                                continue # Skip redundant tip
                        except (ValueError, TypeError):
                            pass

                await db.execute("""
                    INSERT INTO tips (
                        race_id, venue, race_number, start_time, report_date,
                        is_goldmine, gap12, top_five, selection_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_id,
                    tip.get("venue"),
                    tip.get("race_number"),
                    tip.get("start_time"),
                    tip.get("report_date", now),
                    1 if tip.get("is_goldmine") else 0,
                    tip.get("1Gap2", 0.0),
                    tip.get("top_five"),
                    tip.get("selection_number")
                ))
            await db.commit()

    async def get_unverified_tips(self, lookback_hours: int = 48) -> List[Dict[str, Any]]:
        """
        Returns tips that haven't been audited yet.
        """
        if not self._initialized:
            await self.initialize()

        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=lookback_hours)).isoformat()

        tips = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tips WHERE audit_completed = 0 AND start_time > ? AND start_time < ?",
                (cutoff, now.isoformat())
            ) as cursor:
                async for row in cursor:
                    tips.append(dict(row))
        return tips

    async def update_audit_result(self, race_id: str, outcome: Dict[str, Any]):
        """
        Updates a tip with its audit outcome.
        """
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE tips SET
                    audit_completed = 1,
                    verdict = ?,
                    net_profit = ?,
                    selection_position = ?,
                    actual_top_5 = ?,
                    actual_2nd_fav_odds = ?,
                    trifecta_payout = ?,
                    trifecta_combination = ?,
                    audit_timestamp = ?
                WHERE race_id = ? AND audit_completed = 0
            """, (
                outcome.get("verdict"),
                outcome.get("net_profit"),
                outcome.get("selection_position"),
                outcome.get("actual_top_5"),
                outcome.get("actual_2nd_fav_odds"),
                outcome.get("trifecta_payout"),
                outcome.get("trifecta_combination"),
                datetime.now(timezone.utc).isoformat(),
                race_id
            ))
            await db.commit()

    async def get_all_audited_tips(self) -> List[Dict[str, Any]]:
        """
        Returns all audited tips for reporting.
        """
        if not self._initialized:
            await self.initialize()

        tips = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tips WHERE audit_completed = 1 ORDER BY start_time DESC"
            ) as cursor:
                async for row in cursor:
                    tips.append(dict(row))
        return tips

    async def migrate_from_json(self, json_path: str = "hot_tips_db.json"):
        """
        Migrates data from existing JSON file to SQLite.
        """
        path = Path(json_path)
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                return

            logger.info("Migrating data from JSON", count=len(data))

            if not self._initialized:
                await self.initialize()

            async with aiosqlite.connect(self.db_path) as db:
                for entry in data:
                    # Check if already exists
                    race_id = entry.get("race_id")
                    report_date = entry.get("report_date")

                    async with db.execute(
                        "SELECT id FROM tips WHERE race_id = ? AND report_date = ?",
                        (race_id, report_date)
                    ) as cursor:
                        if await cursor.fetchone():
                            continue

                    await db.execute("""
                        INSERT INTO tips (
                            race_id, venue, race_number, start_time, report_date,
                            is_goldmine, gap12, top_five, selection_number,
                            audit_completed, verdict, net_profit, selection_position,
                            actual_top_5, actual_2nd_fav_odds, trifecta_payout,
                            trifecta_combination, audit_timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        race_id,
                        entry.get("venue"),
                        entry.get("race_number"),
                        entry.get("start_time"),
                        report_date,
                        1 if entry.get("is_goldmine") else 0,
                        entry.get("1Gap2", 0.0),
                        entry.get("top_five"),
                        entry.get("selection_number"),
                        1 if entry.get("audit_completed") else 0,
                        entry.get("verdict"),
                        entry.get("net_profit"),
                        entry.get("selection_position"),
                        entry.get("actual_top_5"),
                        entry.get("actual_2nd_fav_odds"),
                        entry.get("trifecta_payout"),
                        entry.get("trifecta_combination"),
                        entry.get("audit_timestamp")
                    ))
                await db.commit()
            logger.info("Migration complete")
        except Exception as e:
            logger.error("Migration failed", error=str(e))
