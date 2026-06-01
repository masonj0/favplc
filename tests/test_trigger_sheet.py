import subprocess
import pytest
import os

def test_hourly_trigger_sheet_execution():
    """Test that the hourly trigger sheet script runs without error for a known date."""
    result = subprocess.run(
        ["python3", "scripts/generate_hourly_trigger_sheet.py", "--date", "2026-06-01"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "HOURLY TRIGGER SHEET" in result.stdout
    assert "Skip 1 and 2?" in result.stdout

def test_ruleset_loading():
    """Test that the consensus ruleset file exists and is valid JSON."""
    rules_path = "scripts/consensus_ruleset.json"
    assert os.path.exists(rules_path)
    import json
    with open(rules_path, "r") as f:
        data = json.load(f)
    assert "_meta" in data
    assert "live_bot_config" in data
