import math

from sdfmpneo.geometry_research import (
    _GEOMETRY_SEED_BATCH,
    _GEOMETRY_SEED_MIN_NODES,
    _GEOMETRY_SEED_MIN_RELATIVE_GAIN,
    _GEOMETRY_SEED_STALE_BATCHES,
    _geometry_seed_budget,
    _geometry_seed_relative_gain,
)


def test_geometry_seed_policy_defaults():
    assert _GEOMETRY_SEED_BATCH == 8
    assert _GEOMETRY_SEED_MIN_NODES == 16
    assert _GEOMETRY_SEED_MIN_RELATIVE_GAIN == 1e-3
    assert _GEOMETRY_SEED_STALE_BATCHES == 2
    assert _geometry_seed_budget(64) == 48


def test_geometry_seed_relative_gain_and_invalid_prefixes():
    assert math.isclose(_geometry_seed_relative_gain(1.0, 0.8), 0.2)
    assert _geometry_seed_relative_gain(1.0, 0.9995) < _GEOMETRY_SEED_MIN_RELATIVE_GAIN
    assert _geometry_seed_relative_gain(1.0, float('inf')) == -float('inf')
    assert _geometry_seed_relative_gain(float('inf'), 1.0) == float('inf')
