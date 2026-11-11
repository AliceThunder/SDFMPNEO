"""Small direct checks for the numerical-only TEAM-7 observables."""
import numpy as np
from examples.bfzi_d2_team7_conv import cmp


def fields(scale=1.0):
    x = np.linspace(-0.1, 0.1, 101)
    out = {}
    for k in range(1, 5):
        a = np.zeros((101, 27))
        a[:, 4] = x
        a[:, 20] = 0.5 * scale
        a[:, 21] = 0.8 * scale
        a[:, 22] = (1 + x) * scale
        a[:, 24] = (2 + x) * scale
        out[k] = a
    return out


def test_identical_fields():
    r = cmp(fields(), fields())
    assert r["A1_Bz_rel"] == 0
    assert r["A1_Bz_mean_delta_over_rms"] == 0
    assert r["Jy_top_rel"] == 0


def test_reference_normalization():
    r = cmp(fields(1.5), fields())
    assert np.isclose(r["A1_Bz_rel"], 0.5)
    assert np.isclose(r["A1_Bxyz_rel"], 0.5)
    assert np.isclose(r["Jy_top_rel"], 0.5)
    assert np.isclose(r["A1_Bz_mean_delta_T"], 0.5)
    assert 0 < r["A1_Bz_mean_delta_over_rms"] < 0.5


def test_coordinate_mismatch():
    a, b = fields(), fields()
    b[2][0, 4] += 1e-3
    try:
        cmp(a, b)
    except ValueError as e:
        assert "coordinates mismatch" in str(e)
    else:
        raise AssertionError("Expected coordinate mismatch")

