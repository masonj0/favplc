import sys
import os
import pytest

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_instrument_registry_integrity():
    """Verify that all instruments in the registry have the required fields for v7.5.2."""
    try:
        from scripts.bankroll_gauntlet import INS
    except ImportError as e:
        pytest.fail(f"Could not import INS from scripts.bankroll_gauntlet: {e}")

    # v7.5.2 keys: t, tc_mode, unit_price, hf, pc, mn, ew, ef
    required_keys = {"t", "tc_mode", "unit_price", "hf", "pc", "mn", "ew", "ef"}

    for name, inst in INS.items():
        # Check for mandatory keys
        for key in required_keys:
            assert key in inst, f"Instrument '{name}' is missing required key: {key}"

        # Check types/validity
        assert inst["t"] in {"T1A", "T1B", "T1", "T2", "T3", "BONUS", "APEX"}, f"Invalid tier for '{name}': {inst['t']}"
        assert isinstance(inst["mn"], int), f"mn for '{name}' must be an integer"
        assert isinstance(inst["ew"], (int, float)), f"ew for '{name}' must be numeric"
        assert isinstance(inst["unit_price"], (int, float)), f"unit_price for '{name}' must be numeric"

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
