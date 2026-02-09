import pytest
import aiosqlite
from fortuna import FortunaDB, EASTERN
from datetime import datetime

@pytest.mark.asyncio
async def test_fortuna_db_log_harvest(tmp_path):
    db_file = tmp_path / "test_harvest.db"
    db = FortunaDB(db_path=str(db_file))
    await db.initialize()

    harvest_summary = {
        "AdapterA": {"count": 10, "max_odds": 25.5},
        "AdapterB": 5 # Old format support
    }

    await db.log_harvest(harvest_summary, region="USA")

    async with aiosqlite.connect(str(db_file)) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM harvest_logs") as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 2

            row_a = next(r for r in rows if r["adapter_name"] == "AdapterA")
            assert row_a["race_count"] == 10
            assert row_a["max_odds"] == 25.5
            assert row_a["region"] == "USA"

            row_b = next(r for r in rows if r["adapter_name"] == "AdapterB")
            assert row_b["race_count"] == 5
            assert row_b["max_odds"] == 0.0
