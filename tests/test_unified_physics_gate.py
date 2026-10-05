import numpy as np

from sdfmpneo.unified_physics_gate import _power_contraction_relative_error


def _passive_matrix():
    return np.array(
        [
            [2.0, 0.30 + 0.10j],
            [0.30 - 0.10j, 1.4],
        ],
        dtype=complex,
    )


def test_power_contraction_error_is_zero_for_identical_dissipation_tensor():
    d = _passive_matrix()
    assert _power_contraction_relative_error(d, d) == 0.0


def test_power_contraction_error_detects_diagonal_power_change():
    reference = _passive_matrix()
    value = reference.copy()
    value[0, 0] *= 1.1
    error = _power_contraction_relative_error(value, reference)
    assert np.isclose(error, 0.1, rtol=1e-13, atol=1e-13)


def test_power_contraction_error_detects_imaginary_mutual_loss_change():
    reference = _passive_matrix()
    value = reference.copy()
    value[0, 1] += 0.05j
    value[1, 0] -= 0.05j
    assert _power_contraction_relative_error(value, reference) > 0.0
