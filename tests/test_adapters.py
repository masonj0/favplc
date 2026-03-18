import pytest
import json
from unittest.mock import AsyncMock, patch
from fortuna import AtTheRacesAdapter, Race, Runner, OddsData, RacingAndSportsAdapter
from datetime import datetime, timezone, timedelta

@pytest.mark.asyncio
async def test_at_the_races_adapter_ajax_movers():
    adapter = AtTheRacesAdapter()

    # Mock AJAX response for market movers - need at least 2 for validation
    movers_html = """
    <div class="market-mover-row">
        <a href="/racecard/newmarket/18-February-2026/1430">14:30 Newmarket</a>
        <span class="track-name">Newmarket</span>
        <span class="horse-name">Mover Horse 1</span>
        <span class="price">4/1</span>
    </div>
    <div class="market-mover-row">
        <a href="/racecard/newmarket/18-February-2026/1430">14:30 Newmarket</a>
        <span class="track-name">Newmarket</span>
        <span class="horse-name">Mover Horse 2</span>
        <span class="price">10/1</span>
    </div>
    """

    class MockResponse:
        def __init__(self, text, status):
            self.text = text
            self.status = status
            self.status_code = status
        def json(self):
            return json.loads(self.text)

    target_date_str = "260218"

    with patch("fortuna.SmartFetcher.fetch", new_callable=AsyncMock) as mock_fetch:
        # 1. bootstrap home, 2. bootstrap index, 3. bootstrap movers, 4. bootstrap movers intl, 5. uk-ire movers, 6. international movers (empty)
        mock_fetch.side_effect = [
            MockResponse("home", 200),
            MockResponse("index", 200),
            MockResponse("movers_boot", 200),
            MockResponse("movers_boot_intl", 200),
            MockResponse(movers_html, 200),
            MockResponse("", 200)
        ]

        races = await adapter.get_races(target_date_str)

        assert len(races) > 0
        race = races[0]
        assert "newmarket" in race.venue.lower()
        assert len(race.runners) == 2
        # Runners might be in a different order depending on map processing
        runner1 = next(r for r in race.runners if r.name == "Mover Horse 1")
        assert runner1.win_odds == 5.0
        runner2 = next(r for r in race.runners if r.name == "Mover Horse 2")
        assert runner2.win_odds == 11.0

@pytest.mark.asyncio
async def test_at_the_races_adapter_fallback_parsing():
    adapter = AtTheRacesAdapter()

    # Mock HTML response for index (with a link that matches window)
    from zoneinfo import ZoneInfo
    now_site = datetime.now(ZoneInfo("Europe/London"))
    future_time = (now_site + timedelta(hours=2))
    target_date_str = future_time.strftime("%y%m%d")
    time_str = future_time.strftime("%H%M")

    index_html = f'<html><a href="/racecard/newmarket/{time_str}">Newmarket</a></html>'

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

    class MockResponse:
        def __init__(self, text, status):
            self.text = text
            self.status = status
            self.status_code = status

    with patch("fortuna.SmartFetcher.fetch", new_callable=AsyncMock) as mock_fetch:
        # 1. bootstrap home, 2. bootstrap index, 3. bootstrap movers, 4. bootstrap movers intl, 5. uk-ire movers (empty), 6. international movers (empty), 7. index re-fetch, 8. race page
        mock_fetch.side_effect = [
            MockResponse("home", 200),
            MockResponse("index", 200),
            MockResponse("movers_boot", 200),
            MockResponse("movers_boot_intl", 200),
            MockResponse("", 200),
            MockResponse("", 200),
            MockResponse(index_html, 200),
            MockResponse(race_html, 200)
        ]

        races = await adapter.get_races(target_date_str)

        assert len(races) > 0
        race = races[0]
        assert "newmarket" in race.venue.lower()
        assert len(race.runners) == 2
        runner1 = next(r for r in race.runners if r.name == "Horse One")
        assert runner1.win_odds == 6.0

@pytest.mark.asyncio
async def test_racing_and_sports_json_v2():
    adapter = RacingAndSportsAdapter()

    json_data = {
        "meetings": [
            {
                "venueName": "Randwick",
                "country": "AUS",
                "races": [
                    {
                        "raceNumber": 1,
                        "raceTime": "14:30",
                        "runners": [
                            {
                                "horseName": "JSON Horse 1",
                                "tabNo": 1,
                                "winOdds": "5.00"
                            },
                            {
                                "horseName": "JSON Horse 2",
                                "tabNo": 2,
                                "winOdds": "10.00"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    class MockResponse:
        def __init__(self, data, status):
            self.data = data
            self.status = status
            self.text = json.dumps(data)
            self.status_code = status
        def json(self):
            return self.data

    with patch("fortuna.SmartFetcher.fetch", new_callable=AsyncMock) as mock_fetch:
        # 1. bootstrap home, 2. JSON API, 3. form-guide (as fallback)
        mock_fetch.side_effect = [
            MockResponse("home", 200),
            MockResponse(json_data, 200),
            MockResponse("no links", 200)
        ]

        races = await adapter.get_races("260218")

        assert len(races) == 1
        race = races[0]
        assert race.venue == "Randwick"
        assert len(race.runners) == 2
        assert race.runners[0].name == "JSON Horse 1"
        assert race.runners[0].win_odds == 5.0

@pytest.mark.asyncio
async def test_simply_success_analyzer_gap_abs():
    from fortuna import SimplySuccessAnalyzer

    analyzer = SimplySuccessAnalyzer()

    # 2.0 and 5.0 -> gap_abs should be 3.0
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
            Runner(name="Fav", number=1, odds=fav_odds, metadata={"odds_source_trustworthy": True}),
            Runner(name="Sec", number=2, odds=sec_odds, metadata={"odds_source_trustworthy": True}),
            Runner(name="T3", number=3, win_odds=10.0, metadata={"odds_source_trustworthy": True}),
            Runner(name="T4", number=4, win_odds=15.0, metadata={"odds_source_trustworthy": True}),
            Runner(name="T5", number=5, win_odds=20.0, metadata={"odds_source_trustworthy": True})
        ]
    )

    result = analyzer.qualify_races([race])
    qualified = result["races"]
    assert len(qualified) == 1
    # Absolute gap: (5.0 - 2.0) = 3.0
    assert qualified[0].metadata["gap_abs"] == 3.0
    assert qualified[0].metadata["is_goldmine"] is True

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
        metadata={
            "is_goldmine": True,
            "gap_abs": 1.5,
            "predicted_fav_odds": 1.5,
            "predicted_2nd_fav_odds": 4.5,
            "is_best_bet": True,
            "qualification_grade": "A"
        },
        top_five_numbers="1, 2, 3"
    )

    await tracker.log_tips([race])

    assert db_file.exists()
    async with aiosqlite.connect(str(db_file)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tips WHERE race_id = 'tip_1'") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert "tracka" in row["venue"].lower() or "track a" in row["venue"].lower()
            assert row["is_goldmine"] == 1
            assert float(row["gap_abs"]) == 1.5

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
            self.status_code = status

    with patch("fortuna.SmartFetcher.fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = [
            MockResponse("bootstrap home", 200),
            MockResponse("bootstrap fg", 200),
            MockResponse("bootstrap thoroughbred", 200),
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

def test_generate_race_id_new_format():
    from fortuna import generate_race_id
    from datetime import datetime, timezone

    st = datetime(2026, 2, 7, 14, 30, tzinfo=timezone.utc)
    # Prefix: srw, Venue: Randwick, R1, Thoroughbred
    rid = generate_race_id("srw", "Randwick", st, 1, "Thoroughbred")

    # Format: {prefix}_{venue_slug}_{date_str}_{time_str}_R{race_number}{disc_suffix}
    # Expected: srw_randwick_260207_1430_R1_t (Shortened year)
    assert rid == "srw_randwick_260207_1430_R1_t"

    # Harness
    rid_h = generate_race_id("ts", "Meadowlands", st, 5, "Harness")
    assert rid_h == "ts_meadowlands_260207_1430_R5_h"
