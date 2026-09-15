import numpy as np

import sdfmpneo.unified_self_correction as correction


class _Background:
    background_config = {"fine_step": 0.012}
    self_correction_config = {"enabled": True, "fine_step": 0.003}


def test_local_self_defect_uses_complete_fine_minus_coarse_response(monkeypatch):
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
        raw_scale = 100.0 if coarse else 120.0
        raw_modal = None if phi is None else np.array([0.5 * p, -0.2 * p]) * raw_scale
        # Keep a deliberately unrelated localized diagnostic so the test proves
        # v3 is using the complete local defect rather than the old v2 object.
        localized_scale = 1.0 if coarse else 1.1
        return {
            "z": complex(raw_scale * p, -0.3 * raw_scale * p),
            "d_vol": 0.9 * raw_scale * p,
            "d_out": 0.1 * raw_scale * p,
            "modal_h": raw_modal,
            "localized_z": complex(0.2 * localized_scale * p, 0.02 * localized_scale * p),
            "localized_d_vol": 0.16 * localized_scale * p,
            "localized_d_out": 0.04 * localized_scale * p,
            "localized_modal_h": (
                None if phi is None else np.array([0.05 * p, -0.02 * p]) * localized_scale
            ),
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

    # Complete fine-minus-coarse defect: raw_scale changes by 20.
    assert np.allclose(np.diag(result.z - z), [20.0 - 6.0j, 40.0 - 12.0j])
    assert np.allclose(np.real(np.diag(result.d_vol - d)), [18.0, 36.0])
    assert np.allclose(np.real(np.diag(result.d_out - d_out)), [2.0, 4.0])
    assert np.allclose(np.real(result.modal_h[:, 0, 0] - modal[:, 0, 0]), [10.0, -4.0])
    assert np.allclose(np.real(result.modal_h[:, 1, 1] - modal[:, 1, 1]), [20.0, -8.0])
    assert result.audit["model"] == "canonical_local_full_fine_minus_coarse_self_defect_v3"
    assert result.audit["corrected_power_balance_relative_error"] < 1e-14
    assert result.audit["maximum_localized_power_balance_relative_error"] == 0.0
    assert result.audit["maximum_full_local_power_balance_relative_error"] == 0.0
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
