import sys
import os
import pytest

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_instrument_registry_integrity():
    """Verify that all instruments in the registry have the required fields."""
    try:
        from scripts.bankroll_gauntlet import INSTRUMENTS, _hit_tri145
    except ImportError as e:
        pytest.fail(f"Could not import INSTRUMENTS from scripts.bankroll_gauntlet: {e}")

    required_keys = {"tier", "ticket_cost", "hit_func", "payout_col", "min_runners", "ewpd", "desc", "env_filter"}

    for name, inst in INSTRUMENTS.items():
        # Check for mandatory keys
        for key in required_keys:
            assert key in inst, f"Instrument '{name}' is missing required key: {key}"

        # Check types/validity
        assert inst["tier"] in {"T1", "T2", "T3", "T4"}, f"Invalid tier for '{name}': {inst['tier']}"
        assert inst["ticket_cost"] > 0, f"Non-positive ticket cost for '{name}'"
        assert callable(inst["hit_func"]), f"hit_func for '{name}' is not callable"
        assert isinstance(inst["min_runners"], int), f"min_runners for '{name}' must be an integer"

        # Verify specific v5.4 fields for SB instruments
        if name.startswith("SB"):
            assert "max_runners" in inst, f"SB instrument '{name}' must have 'max_runners' field"
            assert inst["max_runners"] >= inst["min_runners"], f"SB instrument '{name}' has max_runners < min_runners"

def test_hit_functions_mapping():
    """Verify that all hit functions have a vector mapping for performance."""
    from scripts.bankroll_gauntlet import INSTRUMENTS, _VEC_HIT, _hit_always_true

    for name, inst in INSTRUMENTS.items():
        hf = inst["hit_func"]
        if hf is _hit_always_true:
            continue
        assert hf in _VEC_HIT, f"Hit function for '{name}' is missing a vector mapping in _VEC_HIT"

if __name__ == "__main__":
    pytest.main([__file__])
