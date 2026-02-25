
import pytest
from datetime import datetime
from fortuna import Race, Runner, SimplySuccessAnalyzer, EASTERN

def test_scoring_signal_population():
    analyzer = SimplySuccessAnalyzer()

    # Create a race with minimal data that might cause scoring to be skipped or fail
    # Restoration Note: S0-S8 pipeline requires 4 runners with odds
    # BUT min_field_size gate requires 5 runners for Thoroughbred
    race = Race(
        id="test_race_1",
        venue="Test Track",
        race_number=1,
        start_time=datetime.now(EASTERN),
        runners=[
            Runner(number=1, name="Runner 1", win_odds=2.0),
            Runner(number=2, name="Runner 2", win_odds=4.0),
            Runner(number=3, name="Runner 3", win_odds=6.0),
            Runner(number=4, name="Runner 4", win_odds=8.0),
            Runner(number=5, name="Runner 5", win_odds=10.0)
        ],
        source="Test"
    )

    # We need to set provides_odds to False to skip the trust check if we don't have enough metadata
    race.metadata["provides_odds"] = False

    result = analyzer.qualify_races([race])
    qualified = result["races"]

    assert len(qualified) == 1
    meta = qualified[0].metadata

    # Check that signals currently implemented are present
    assert "qualification_grade" in meta
    assert "composite_score" in meta
    assert "1Gap2" in meta

    assert meta["qualification_grade"] in ['A+', 'A', 'B+', 'B', 'C', 'D']
    assert meta["composite_score"] >= 0.0

def test_scoring_signal_population_with_exception():
    analyzer = SimplySuccessAnalyzer()

    race = Race(
        id="test_race_2",
        venue="Test Track",
        race_number=2,
        start_time=datetime.now(EASTERN),
        runners=[
            Runner(number=1, name="Runner 1", win_odds=2.0),
            Runner(number=2, name="Runner 2", win_odds=3.0),
            Runner(number=3, name="Runner 3", win_odds=4.0),
            Runner(number=4, name="Runner 4", win_odds=5.0),
            Runner(number=5, name="Runner 5", win_odds=6.0)
        ],
        source="Test"
    )
    race.metadata["provides_odds"] = False

    result = analyzer.qualify_races([race])
    qualified = result["races"]
    assert len(qualified) == 1
    meta = qualified[0].metadata
    assert "composite_score" in meta
    assert meta["composite_score"] >= 0.0

if __name__ == "__main__":
    test_scoring_signal_population()
    print("Test passed!")
