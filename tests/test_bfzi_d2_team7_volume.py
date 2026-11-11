"""Minimal checks for frozen-volume numerical comparison."""
import numpy as np
from examples.bfzi_d2_team7_volume import pts, pair


def test_fixed_volume_grid():
    p = pts(((0, 1), (2, 3), (4, 5)), 5)
    assert p.shape == (125, 3)
    assert np.allclose(p.min(axis=0), (0, 2, 4))
    assert np.allclose(p.max(axis=0), (1, 3, 5))


def test_same_and_scaled_field():
    a = {"B": np.tile([2., 1., 3.], (9, 1)),
         "J": np.tile([3., 4., 0.], (9, 1)),
         "valid": np.ones(9, dtype=bool)}
    b = {"B": a["B"].copy(), "J": a["J"].copy(), "valid": a["valid"].copy()}
    same = pair(a, b, "plate_bulk")
    assert same["B_L2_rel"] == same["J_L2_rel"] == 0.
    a["B"] *= 1.25
    a["J"] *= 1.25
    diff = pair(a, b, "plate_bulk")
    assert np.isclose(diff["B_L2_rel"], .25)
    assert np.isclose(diff["B_rms_rel"], .25)
    assert np.isclose(diff["J_L2_rel"], .25)

