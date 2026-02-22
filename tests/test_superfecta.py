import pytest
from datetime import datetime
from decimal import Decimal
from fortuna import SimplySuccessAnalyzer, Race, Runner, EASTERN

def test_superfecta_key_trigger():
    # Setup a race that should trigger superfecta key
    # gap12 > 0.75 and runners >= 4
    runners = [
        Runner(number=1, name="Fav", win_odds=1.5),
        Runner(number=2, name="Sec", win_odds=3.0), # (3.0-1.5)/1.5 = 1.0 > 0.75
        Runner(number=3, name="Third", win_odds=5.0),
        Runner(number=4, name="Fourth", win_odds=10.0),
    ]
    # Set odds_source_trustworthy to True so it passes the airlock
    for r in runners:
        r.metadata['odds_source_trustworthy'] = True

    race = Race(id="test1", venue="Test Track", raceNumber=1, runners=runners,
                startTime=datetime.now(EASTERN), source="test")

    analyzer = SimplySuccessAnalyzer(config={})
    results = analyzer.qualify_races([race])

    analyzed_race = results['races'][0]
    assert analyzed_race.metadata.get('is_superfecta_key') is True
    assert analyzed_race.metadata.get('superfecta_key_number') == 1
    assert analyzed_race.metadata.get('superfecta_box_numbers') == [2, 3, 4]

def test_superfecta_key_not_triggered_low_gap():
    runners = [
        Runner(number=1, name="Fav", win_odds=2.0),
        Runner(number=2, name="Sec", win_odds=3.0), # (3.0-2.0)/2.0 = 0.5 < 0.75
        Runner(number=3, name="Third", win_odds=5.0),
        Runner(number=4, name="Fourth", win_odds=10.0),
    ]
    for r in runners:
        r.metadata['odds_source_trustworthy'] = True

    race = Race(id="test2", venue="Test Track", raceNumber=2, runners=runners,
                startTime=datetime.now(EASTERN), source="test")

    analyzer = SimplySuccessAnalyzer(config={})
    results = analyzer.qualify_races([race])

    analyzed_race = results['races'][0]
    assert analyzed_race.metadata.get('is_superfecta_key') is False

def test_superfecta_key_not_triggered_low_runners():
    runners = [
        Runner(number=1, name="Fav", win_odds=1.5),
        Runner(number=2, name="Sec", win_odds=3.0), # gap 1.0 > 0.75
        Runner(number=3, name="Third", win_odds=5.0),
    ]
    for r in runners:
        r.metadata['odds_source_trustworthy'] = True

    race = Race(id="test3", venue="Test Track", raceNumber=3, runners=runners,
                startTime=datetime.now(EASTERN), source="test")

    analyzer = SimplySuccessAnalyzer(config={})
    results = analyzer.qualify_races([race])

    analyzed_race = results['races'][0]
    assert analyzed_race.metadata.get('is_superfecta_key') is False
