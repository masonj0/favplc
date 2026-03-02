import pytest
import sqlite3
import os
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from fortuna import FortunaDB, Race, Runner, OddsData, refresh_odds_for_races, run_quarter_fetch
from fortuna_utils import EASTERN, to_storage_format, DATE_FORMAT

@pytest.mark.asyncio
async def test_initialize_dedup_runs_once(tmp_path):
    db_file = tmp_path / "test_dedup.db"
    db = FortunaDB(str(db_file))

    # Pre-create the schema correctly but without the index to test the IntegrityError path
    await db.initialize()

    # Drop index and insert duplicates, and downgrade schema version to test migration
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("DROP INDEX idx_race_id")
        conn.execute("INSERT INTO tips (race_id, venue, race_number, start_time, report_date, is_goldmine) VALUES ('r1', 'V', 1, 'T', 'D', 0)")
        # This one will be a duplicate
        conn.execute("INSERT INTO tips (race_id, venue, race_number, start_time, report_date, is_goldmine) VALUES ('r1', 'V', 1, 'T', 'D', 0)")
        conn.execute("DELETE FROM schema_version WHERE version = 8")

    # Re-initialize fresh instance should trigger one-time cleanup
    db2 = FortunaDB(str(db_file))
    await db2.initialize()

    with sqlite3.connect(str(db_file)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM tips WHERE race_id='r1'").fetchone()[0]
        assert count == 1
        # Index should now exist
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_race_id'").fetchone()
        assert res is not None

@pytest.mark.asyncio
async def test_clone_detection_format_independent(tmp_path):
    db_file = tmp_path / "test_clone.db"
    db = FortunaDB(str(db_file))
    await db.initialize()

    now = datetime.now(EASTERN)
    tip1 = {
        "race_id": "race_1",
        "venue": "Ascot",
        "race_number": 1,
        "selection_name": "Horse A",
        "start_time": to_storage_format(now),
        "report_date": to_storage_format(now),
        "is_goldmine": False,
        "1Gap2": 1.0,
        "predicted_fav_odds": 2.0,
        "predicted_2nd_fav_odds": 3.0,
        "is_best_bet": True
    }

    await db.log_tips([tip1])

    # Try to log a clone
    tip2 = tip1.copy()
    tip2["race_id"] = "race_2" # Different ID but same horse/venue/race/day

    await db.log_tips([tip2])

    # Verify only one tip exists
    async with db.get_connection() as conn:
        async with conn.execute("SELECT COUNT(*) FROM tips") as cursor:
            row = await cursor.fetchone()
            assert row[0] == 1

@pytest.mark.asyncio
async def test_odds_refresh_cross_day_isolation():
    # Two races, same venue/number, different dates
    now = datetime.now(EASTERN)
    yesterday = now - timedelta(days=1)

    race_today = Race(
        id="today", venue="Ascot", race_number=1,
        start_time=now, source="Manual", runners=[Runner(name="Runner A", number=1)]
    )
    race_yesterday = Race(
        id="yesterday", venue="Ascot", race_number=1,
        start_time=yesterday, source="Manual", runners=[Runner(name="Runner A", number=1)]
    )

    # Mock fast adapter races
    fresh_race = Race(
        id="fresh", venue="Ascot", race_number=1,
        start_time=now, source="Fast", runners=[Runner(name="Runner A", number=1, win_odds=2.0)]
    )

    with patch("fortuna._find_adapter_class") as mock_find:
        mock_adapter_cls = MagicMock()
        mock_adapter = AsyncMock()
        mock_adapter.get_races.return_value = [fresh_race]
        mock_adapter_cls.return_value = mock_adapter
        mock_find.return_value = mock_adapter_cls

        updated = await refresh_odds_for_races([race_today, race_yesterday], {})

        # Today's race should have odds, yesterday's should not
        assert updated[0].runners[0].win_odds == 2.0
        assert updated[1].runners[0].win_odds is None
