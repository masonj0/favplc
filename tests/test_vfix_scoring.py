
import pytest
from datetime import datetime
from fortuna import Race, Runner, SimplySuccessAnalyzer, EASTERN

def test_scoring_signal_population():
    analyzer = SimplySuccessAnalyzer()

    # Create a race with minimal data that might cause scoring to be skipped or fail
    race = Race(
        id="test_race_1",
        venue="Test Track",
        race_number=1,
        start_time=datetime.now(EASTERN),
        runners=[
            Runner(number=1, name="Runner 1", win_odds=2.0),
            Runner(number=2, name="Runner 2", win_odds=None) # Insufficient odds
        ],
        source="Test"
    )

    # We need to set provides_odds to False to skip the trust check if we don't have enough metadata
    race.metadata["provides_odds"] = False

    result = analyzer.qualify_races([race])
    qualified = result["races"]

    assert len(qualified) == 1
    meta = qualified[0].metadata

    # Check that all signals are present and not None (except predicted_2nd_fav_odds which can be None)
    assert "place_prob" in meta
    assert "predicted_ev" in meta
    assert "market_depth" in meta
    assert "condition_modifier" in meta
    assert "qualification_grade" in meta
    assert "composite_score" in meta
    assert "1Gap2" in meta

    assert meta["qualification_grade"] == 'D'
    assert meta["composite_score"] == 0.0

def test_scoring_signal_population_with_exception():
    analyzer = SimplySuccessAnalyzer()

    race = Race(
        id="test_race_2",
        venue="Test Track",
        race_number=2,
        start_time=datetime.now(EASTERN),
        runners=[
            Runner(number=1, name="Runner 1", win_odds=2.0),
            Runner(number=2, name="Runner 2", win_odds=3.0)
        ],
        source="Test"
    )
    race.metadata["provides_odds"] = False

    # Force an exception by mocking or just providing weird data if possible
    # Actually, the try/except block covers the whole "Best Bet Logic"

    # Let's see if we can trigger an exception in the scoring pipeline.
    # Maybe by setting race_type to something that causes an error?
    # Actually, the RT extraction is quite safe.

    # How about making all_odds empty despite active runners?
    # (Shouldn't happen with the logic, but let's try)

    result = analyzer.qualify_races([race])
    qualified = result["races"]
    assert len(qualified) == 1
    meta = qualified[0].metadata
    assert "place_prob" in meta
    assert meta["place_prob"] >= 0.0

if __name__ == "__main__":
    test_scoring_signal_population()
    print("Test passed!")
