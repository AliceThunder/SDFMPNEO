import numpy as np

import sdfmpneo.unified_self_correction as correction


class _Background:
    background_config = {"fine_step": 0.012}
    self_correction_config = {"enabled": True, "fine_step": 0.003}


def test_local_self_defect_uses_localized_response_and_preserves_power_partition(monkeypatch):
    background = _Background()
    z = np.array([[5.0 + 0.2j, 0.3 + 0.1j], [0.3 + 0.1j, 6.0 + 0.25j]], complex)
    d = np.array([[4.9, 0.2 + 0.03j], [0.2 - 0.03j, 5.9]], complex)
    d_out = 0.5 * (z + z.conj().T) - d
    modal = np.array(
        [
            [[1.0, 0.1j], [-0.1j, 1.4]],
            [[-0.4, 0.02], [0.02, 0.7]],
        ],
        complex,
    )
    phi = np.ones((3, 2))

    def fake_local(_background, _geometry, port, step, phi=None):
        p = int(port) + 1
        coarse = np.isclose(step, 0.012)
        # Raw terminal response deliberately changes by an unrelated huge
        # amount.  Production correction must use only localized_* values.
        raw_scale = 100.0 if coarse else 10000.0
        raw_modal = None if phi is None else np.array([50.0 * p, -20.0 * p]) * raw_scale
        if coarse:
            localized_z = complex(0.2 * p, 0.02 * p)
            localized_d = 0.16 * p
            localized_out = 0.04 * p
            localized_modal = None if phi is None else np.array([0.05 * p, -0.02 * p])
        else:
            localized_z = complex(0.8 * p, 0.07 * p)
            localized_d = 0.66 * p
            localized_out = 0.14 * p
            localized_modal = None if phi is None else np.array([0.25 * p, -0.12 * p])
        return {
            "z": complex(raw_scale * p, -0.3 * raw_scale * p),
            "d_vol": 0.9 * raw_scale * p,
            "d_out": 0.1 * raw_scale * p,
            "modal_h": raw_modal,
            "localized_z": localized_z,
            "localized_d_vol": localized_d,
            "localized_d_out": localized_out,
            "localized_modal_h": localized_modal,
            "localized_power_balance_relative_error": 0.0,
            "linear_relative_residual": 0.0,
            "power_balance_relative_error": 0.0,
            "joule_total_power_relative_error": 2e-13 if coarse else 3e-13,
            "joule_modal_contraction_relative_error": 4e-13 if phi is not None else 0.0,
            "n_cells": 10,
            "n_edges": 20,
            "fine_step": float(step),
        }

    monkeypatch.setattr(correction, "_solve_local", fake_local)
    result = correction.apply_local_self_correction(
        background, {"unused": True}, z, d, d_out, phi=phi, modal_h=modal
    )

    assert np.allclose(result.z[0, 1], z[0, 1])
    assert np.allclose(result.d_vol[0, 1], d[0, 1])
    assert np.allclose(result.modal_h[:, 0, 1], modal[:, 0, 1])

    # These are exactly the localized fine-minus-coarse defects.  The huge raw
    # values above must not leak into the production correction.
    assert np.allclose(np.diag(result.z - z), [0.6 + 0.05j, 1.2 + 0.10j])
    assert np.allclose(np.real(np.diag(result.d_vol - d)), [0.5, 1.0])
    assert np.allclose(np.real(np.diag(result.d_out - d_out)), [0.1, 0.2])
    assert np.allclose(np.real(result.modal_h[:, 0, 0] - modal[:, 0, 0]), [0.2, -0.1])
    assert np.allclose(np.real(result.modal_h[:, 1, 1] - modal[:, 1, 1]), [0.4, -0.2])
    assert result.audit["model"] == "canonical_local_transverse_fine_minus_coarse_self_defect_v2"
    assert result.audit["corrected_power_balance_relative_error"] < 1e-14
    assert result.audit["maximum_localized_power_balance_relative_error"] == 0.0
    assert result.audit["maximum_joule_total_power_relative_error"] == 3e-13
    assert result.audit["maximum_joule_modal_contraction_relative_error"] == 4e-13


def test_disabled_self_correction_is_identity():
    background = _Background()
    background.self_correction_config = {"enabled": False}
    z = np.eye(2, dtype=complex)
    d = np.eye(2, dtype=complex) * 0.8
    d_out = np.eye(2, dtype=complex) * 0.2
    result = correction.apply_local_self_correction(background, {}, z, d, d_out)
    assert np.array_equal(result.z, z)
    assert np.array_equal(result.d_vol, d)
    assert np.array_equal(result.d_out, d_out)
    assert result.audit["enabled"] is False
