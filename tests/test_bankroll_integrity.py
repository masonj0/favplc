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

    # v7.1.4 internal keys: t, tc_mode, hf, pc, mn, ew, ef
    required_keys = {"t", "tc_mode", "hf", "pc", "mn", "ew", "ef"}

    for name, inst in INS.items():
        # Check for mandatory keys
        for key in required_keys:
            assert key in inst, f"Instrument '{name}' is missing required key: {key}"

        # Check types/validity
        assert inst["t"] in {"T1A", "T1B", "T1", "T2", "T3", "T4", "BONUS", "APEX"}, f"Invalid tier for '{name}': {inst['t']}"
        assert isinstance(inst["mn"], int), f"mn for '{name}' must be an integer"
        # In v7.1.4, mx might not be present if it equals mn (handled by get logic)
        # but let's check it if it exists
        if "mx" in inst:
            assert inst["mx"] >= inst["mn"], f"Instrument '{name}' has mx < mn"

def test_hit_functions_mapping():
    """Verify that all hit functions have a vector mapping for performance."""
    from scripts.bankroll_gauntlet import INS, _VH

    for name, inst in INS.items():
        hf_key = inst["hf"]
        if hf_key == "_h_al":
            continue
        assert hf_key in _VH, f"Hit function key '{hf_key}' for '{name}' is missing a vector mapping in _VH"

if __name__ == "__main__":
    pytest.main([__file__])
