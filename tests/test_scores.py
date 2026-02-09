import pytest
import aiosqlite
from fortuna import FortunaDB, EASTERN
from datetime import datetime, timedelta

@pytest.mark.asyncio
async def test_fortuna_db_get_adapter_scores(tmp_path):
    db_file = tmp_path / "test_scores.db"
    db = FortunaDB(db_path=str(db_file))
    await db.initialize()

    now = datetime.now(EASTERN)

    # Mock some harvest logs
    # Adapter A: Good performer
    # Adapter B: Poor performer
    # Adapter C: Old performer (outside 30 days)

    logs = [
        # (timestamp, region, adapter_name, race_count, max_odds)
        ((now - timedelta(days=1)).isoformat(), "USA", "AdapterA", 10, 30.0),
        ((now - timedelta(days=2)).isoformat(), "USA", "AdapterA", 12, 40.0),
        ((now - timedelta(days=1)).isoformat(), "USA", "AdapterB", 2, 5.0),
        ((now - timedelta(days=40)).isoformat(), "USA", "AdapterC", 50, 100.0), # Too old
    ]

    async with aiosqlite.connect(str(db_file)) as conn:
        await conn.executemany(
            "INSERT INTO harvest_logs (timestamp, region, adapter_name, race_count, max_odds) VALUES (?, ?, ?, ?, ?)",
            logs
        )
        await conn.commit()

    scores = await db.get_adapter_scores(days=30)

    assert "AdapterA" in scores
    assert "AdapterB" in scores
    assert "AdapterC" not in scores

    # AdapterA: avg_count=11, avg_max_odds=35. Score = 11 + 35*2 = 81
    # AdapterB: avg_count=2, avg_max_odds=5. Score = 2 + 5*2 = 12
    assert scores["AdapterA"] == 81.0
    assert scores["AdapterB"] == 12.0
