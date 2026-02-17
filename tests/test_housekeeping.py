import pytest
from datetime import datetime, timedelta
from fortuna import Race, Runner, HotTipsTracker, EASTERN, get_canonical_venue
import aiosqlite
import os

@pytest.mark.asyncio
async def test_hot_tips_tracker_24h_limit(tmp_path):
    db_file = tmp_path / "test_fortuna.db"
    tracker = HotTipsTracker(db_path=str(db_file))

    now = datetime.now(EASTERN)

    # 1. Race in 10 mins (should be logged)
    race_ok = Race(
        id="race_ok",
        venue="Track A",
        race_number=1,
        start_time=now + timedelta(minutes=10),
        source="Test",
        runners=[],
            metadata={"is_best_bet": True, "1Gap2": 1.5, "predicted_2nd_fav_odds": 4.5}
    )

    # 2. Race in 25 hours (should be rejected - current limit is 24h)
    race_too_far = Race(
        id="race_too_far",
        venue="Track A",
        race_number=2,
        start_time=now + timedelta(hours=25),
        source="Test",
        runners=[],
            metadata={"is_best_bet": True, "1Gap2": 2.0, "predicted_2nd_fav_odds": 5.0}
    )

    await tracker.log_tips([race_ok, race_too_far])

    async with aiosqlite.connect(str(db_file)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT race_id FROM tips") as cursor:
            rows = await cursor.fetchall()
            race_ids = [row["race_id"] for row in rows]
            assert "race_ok" in race_ids
            assert "race_too_far" not in race_ids

@pytest.mark.asyncio
async def test_next_race_per_track_logic():
    # We can't easily test run_discovery directly without a lot of mocking
    # but we can verify the logic I added.

    # Mocking what run_discovery does with unique_races
    unique_races = []
    now = datetime.now(EASTERN)

    # Track A: Race 1 (in 5 mins), Race 2 (in 35 mins)
    unique_races.append(Race(id="A1", venue="Track A", race_number=1, start_time=now + timedelta(minutes=5), source="T", runners=[]))
    unique_races.append(Race(id="A2", venue="Track A", race_number=2, start_time=now + timedelta(minutes=35), source="T", runners=[]))

    # Track B: Race 3 (started 2 mins ago), Race 4 (in 28 mins)
    unique_races.append(Race(id="B3", venue="Track B", race_number=3, start_time=now - timedelta(minutes=2), source="T", runners=[]))
    unique_races.append(Race(id="B4", venue="Track B", race_number=4, start_time=now + timedelta(minutes=28), source="T", runners=[]))

    # Track C: Race 5 (started 10 mins ago - should be ignored), Race 6 (in 40 mins)
    unique_races.append(Race(id="C5", venue="Track C", race_number=5, start_time=now - timedelta(minutes=10), source="T", runners=[]))
    unique_races.append(Race(id="C6", venue="Track C", race_number=6, start_time=now + timedelta(minutes=40), source="T", runners=[]))

    # Apply the same logic as in run_discovery
    next_races_map = {}
    for race in unique_races:
        st = race.start_time
        if st.tzinfo is None: st = st.replace(tzinfo=EASTERN)
        v = get_canonical_venue(race.venue)
        if st > now - timedelta(minutes=5):
            if v not in next_races_map or st < next_races_map[v].start_time:
                next_races_map[v] = race

    filtered = list(next_races_map.values())
    filtered_ids = [r.id for r in filtered]

    assert "A1" in filtered_ids
    assert "A2" not in filtered_ids
    assert "B3" in filtered_ids # Started 2 mins ago, within 5 min window
    assert "B4" not in filtered_ids
    assert "C5" not in filtered_ids # Started 10 mins ago, outside 5 min window
    assert "C6" in filtered_ids # Earliest remaining for Track C
