import numpy as np

from sdfmpneo.unified_mesh_gate_terminal_dissipation import (
    _apply_diagonal_shift,
    _terminal_modal,
)


def test_terminal_mesh_gate_shift_preserves_mutual_and_power_pair():
    z = np.array([[2.0 + 3.0j, 0.5 + 0.2j], [0.5 + 0.2j, 4.0 + 5.0j]], complex)
    d = np.array([[1.5, 0.25], [0.25, 3.5]], complex)
    shift = np.array([7.0, -2.0])
    zc, dc = _apply_diagonal_shift(z, d, shift)

    np.testing.assert_allclose(np.diag(np.real(zc - z)), shift)
    np.testing.assert_allclose(np.diag(np.real(dc - d)), shift)
    np.testing.assert_allclose(np.imag(np.diag(zc)), np.imag(np.diag(z)))
    np.testing.assert_allclose(zc - np.diag(np.diag(zc)), z - np.diag(np.diag(z)))
    np.testing.assert_allclose(dc - np.diag(np.diag(dc)), d - np.diag(np.diag(d)))


def test_terminal_modal_extracts_only_terminal_dissipative_terms():
    audit = {
        "terminal_dissipative_ports": [
            {
                "port": 0,
                "terminals": [
                    {"delta_modal_h": [1.0, 2.0, 3.0]},
                    {"delta_modal_h": [4.0, 5.0, 6.0]},
                ],
            },
            {
                "port": 1,
                "terminals": [
                    {"delta_modal_h": [7.0, 8.0, 9.0]},
                    {"delta_modal_h": [10.0, 11.0, 12.0]},
                ],
            },
        ]
    }
    out = _terminal_modal(audit, 3, 2)
    np.testing.assert_allclose(out[:, 0], [5.0, 7.0, 9.0])
    np.testing.assert_allclose(out[:, 1], [17.0, 19.0, 21.0])
