import numpy as np

from sdfmpneo.unified_terminal_contact_source import (
    regularized_path_vector,
    terminal_contact_length,
    terminal_segment_weights,
)


class _Coil:
    conductor_width = 2.0e-3
    conductor_thickness = 0.5e-3


def _straight_points(step):
    length = 0.12
    n = int(round(length / float(step)))
    return np.column_stack(
        (
            np.linspace(0.0, length, n + 1),
            np.zeros(n + 1),
            np.zeros(n + 1),
        )
    )


def test_terminal_contact_length_is_physical_not_mesh_dependent():
    coil = _Coil()
    expected = 3.0 * coil.conductor_width
    assert np.isclose(terminal_contact_length(coil, 0.12), expected)
    for step in (4.0e-3, 2.0e-3, 1.0e-3):
        weights, contact = terminal_segment_weights(coil, _straight_points(step))
        assert np.isclose(contact, expected)
        assert 0.0 < weights[0] < 1.0
        assert 0.0 < weights[-1] < 1.0
        assert np.isclose(np.max(weights), 1.0)
        assert np.allclose(weights, weights[::-1])


def test_regularized_path_vector_converges_to_distributed_contact_integral():
    coil = _Coil()
    contact = 3.0 * coil.conductor_width
    # Cubic smoothstep has mean 1/2 on each contact.  A straight unit-current
    # interior therefore has effective vector length L-contact (two contacts,
    # each losing contact/2 from the hard-endpoint path integral).
    expected = 0.12 - contact
    values = []
    for step in (2.0e-3, 1.0e-3, 0.5e-3):
        vector, measured_contact = regularized_path_vector(coil, _straight_points(step))
        assert np.isclose(measured_contact, contact)
        values.append(float(vector[0]))
        assert abs(vector[1]) < 1e-15
        assert abs(vector[2]) < 1e-15
    assert abs(values[-1] - expected) < 2e-6
    assert abs(values[-1] - values[-2]) < abs(values[0] - expected)
