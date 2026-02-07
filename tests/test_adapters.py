import pytest
from unittest.mock import AsyncMock, patch
from fortuna import AtTheRacesAdapter, Race, Runner, OddsData
from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_at_the_races_adapter_parsing():
    adapter = AtTheRacesAdapter()

    # Mock HTML response for index
    index_html = '<html><a href="/racecard/newmarket/1234">Newmarket</a></html>'
    # Mock HTML response for race page
    race_html = """
    <html>
        <body>
            <div class="race-header__details">
                <div class="race-header__details--primary">
                    <h2>Newmarket</h2>
                    <span class="race-time">14:30</span>
                </div>
            </div>
            <div class="horse-in-racecard">
                <a href="/form/horse/111">Horse One</a>
                <span class="horse-in-racecard__saddle-cloth-number">1</span>
                <span class="horse-in-racecard__odds">5/1</span>
            </div>
            <div class="horse-in-racecard">
                <a href="/form/horse/222">Horse Two</a>
                <span class="horse-in-racecard__saddle-cloth-number">2</span>
                <span class="horse-in-racecard__odds">10/1</span>
            </div>
        </body>
    </html>
    """

    # We need to mock the response object properly
    class MockResponse:
        def __init__(self, text, status):
            self.text = text
            self.status = status
        def json(self):
            import json
            return json.loads(self.text)

    with patch("fortuna.SmartFetcher.fetch", new_callable=AsyncMock) as mock_fetch:
        # 1. Main index, 2. International index, 3. Race page
        mock_fetch.side_effect = [
            MockResponse(index_html, 200),
            MockResponse("", 200),
            MockResponse(race_html, 200)
        ]

        races = await adapter.get_races("2026-02-05")

        assert len(races) > 0
        race = races[0]
        assert "Newmarket" in race.venue
        assert len(race.runners) == 2
        runner1 = next(r for r in race.runners if r.name == "Horse One")
        assert runner1.win_odds == 6.0
        runner2 = next(r for r in race.runners if r.name == "Horse Two")
        assert runner2.win_odds == 11.0

@pytest.mark.asyncio
async def test_simply_success_analyzer_1Gap2():
    from fortuna import SimplySuccessAnalyzer

    analyzer = SimplySuccessAnalyzer()

    # 2.0 and 5.0 -> 1Gap2 should be 3.0
    # Must use odds dict because _get_best_win_odds uses it
    fav_odds = {"source1": OddsData(win=2.0, source="source1")}
    sec_odds = {"source1": OddsData(win=5.0, source="source1")}

    race = Race(
        id="test_race",
        venue="Test Track",
        race_number=1,
        start_time=datetime.now(timezone.utc),
        source="Test",
        runners=[
            Runner(name="Fav", number=1, odds=fav_odds),
            Runner(name="Sec", number=2, odds=sec_odds)
        ]
    )

    result = analyzer.qualify_races([race])
    qualified = result["races"]
    assert len(qualified) == 1
    assert qualified[0].metadata["1Gap2"] == 3.0
    assert qualified[0].metadata["is_goldmine"] is True # 2nd fav 5.0 and field size 2 (<=8)

@pytest.mark.asyncio
async def test_hot_tips_tracker(tmp_path):
    from fortuna import HotTipsTracker, Race, Runner
    import aiosqlite

    db_file = tmp_path / "test_fortuna.db"
    tracker = HotTipsTracker(db_path=str(db_file))

    race = Race(
        id="tip_1",
        venue="Track A",
        race_number=1,
        start_time=datetime.now(timezone.utc),
        source="Test",
        runners=[],
        metadata={"is_goldmine": True, "1Gap2": 1.5},
        top_five_numbers="1, 2, 3"
    )

    await tracker.log_tips([race])

    assert db_file.exists()
    async with aiosqlite.connect(str(db_file)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tips WHERE race_id = 'tip_1'") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row["venue"] == "Track A"
            assert row["is_goldmine"] == 1
            assert float(row["gap12"]) == 1.5

@pytest.mark.asyncio
async def test_runner_number_sanitization():
    from fortuna import Race, Runner, AtTheRacesAdapter
    from datetime import datetime, timezone

    # Simulate a race where runner numbers are clearly IDs (> 100)
    runners = [
        Runner(name="Horse A", number=15234),
        Runner(name="Horse B", number=15235),
    ]
    race = Race(
        id="suspicious_race",
        venue="Test Track",
        race_number=1,
        start_time=datetime.now(timezone.utc),
        source="Test",
        runners=runners
    )

    adapter = AtTheRacesAdapter()
    # _validate_and_parse_races is where the heuristic lives
    # We need to mock _parse_races to return our suspicious race
    with patch.object(AtTheRacesAdapter, "_parse_races", return_value=[race]):
        valid_races = adapter._validate_and_parse_races({"dummy": "data"})

    assert len(valid_races) == 1
    sanitized_runners = valid_races[0].runners
    assert sanitized_runners[0].number == 1
    assert sanitized_runners[1].number == 2

@pytest.mark.asyncio
async def test_sky_racing_world_adapter_parsing():
    from fortuna import SkyRacingWorldAdapter
    adapter = SkyRacingWorldAdapter()

    index_html = '<html><a class="fg-race-link" href="/form-guide/thoroughbred/australia/randwick/2026-02-07/R1">R1</a></html>'
    race_html = """
    <html>
        <body>
            <h1 class="sdc-site-racing-header__name">14:30 RANDWICK</h1>
            <div class="runner_row" data-tab-no="1" data-name="HORSE ONE">
                <span class="horseName">HORSE ONE</span>
                <span class="pa_odds">5/1</span>
            </div>
            <div class="runner_row" data-tab-no="2" data-name="HORSE TWO">
                <span class="horseName">HORSE TWO</span>
                <span class="pa_odds">10/1</span>
            </div>
        </body>
    </html>
    """

    class MockResponse:
        def __init__(self, text, status):
            self.text = text
            self.status = status

    with patch("fortuna.SmartFetcher.fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = [
            MockResponse(index_html, 200),
            MockResponse(race_html, 200)
        ]

        races = await adapter.get_races("2026-02-07")

        assert len(races) > 0
        race = races[0]
        assert "Randwick" in race.venue
        assert len(race.runners) == 2
        assert race.runners[0].win_odds == 6.0
        assert race.runners[1].win_odds == 11.0
