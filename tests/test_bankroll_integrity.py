import sys
import os
import pytest

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_instrument_registry_integrity():
    """Verify that all instruments in the registry have the required fields."""
    try:
        from scripts.bankroll_gauntlet import INS
    except ImportError as e:
        pytest.fail(f"Could not import INS from scripts.bankroll_gauntlet: {e}")

    # v6.3.4 internal keys: t, tc, hf, pc, mn, ew, ef
    required_keys = {"t", "tc", "hf", "pc", "mn", "ew", "ef"}

    for name, inst in INS.items():
        # Check for mandatory keys
        for key in required_keys:
            assert key in inst, f"Instrument '{name}' is missing required key: {key}"

        # Check types/validity
        assert inst["t"] in {"T1", "T2", "T3", "T4", "BNS", "APX"}, f"Invalid tier for '{name}': {inst['t']}"
        assert inst["tc"] > 0, f"Non-positive ticket cost for '{name}'"
        assert callable(inst["hf"]), f"hf for '{name}' is not callable"
        assert isinstance(inst["mn"], int), f"mn for '{name}' must be an integer"

        # Verify specific fields for SB instruments (v6.3.4 uses mx)
        if name.startswith("SB"):
            assert "mx" in inst, f"SB instrument '{name}' must have 'mx' field"
            assert inst["mx"] >= inst["mn"], f"SB instrument '{name}' has mx < mn"

def test_hit_functions_mapping():
    """Verify that all hit functions have a vector mapping for performance."""
    from scripts.bankroll_gauntlet import INS, _VH, _h_al

    for name, inst in INS.items():
        hf = inst["hf"]
        if hf is _h_al:
            continue
        assert hf in _VH, f"Hit function for '{name}' is missing a vector mapping in _VH"

if __name__ == "__main__":
    pytest.main([__file__])
