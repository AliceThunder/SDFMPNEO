import numpy as np

import sdfmpneo.unified_self_correction as correction
import sdfmpneo.unified_self_correction_audit as audit


class _Background:
    coil_materials = ("tx",)
    background_config = {"fine_step": 0.012}
    self_correction_config = {
        "fine_step": 0.003,
        "validation_fine_step": 0.00225,
        "relative_tolerance": 0.1,
        "joule_identity_tolerance": 1e-10,
        "linear_relative_residual_tolerance": 1e-9,
    }


def _local(step, *, joule_error=0.0, residual=0.0, linear_converged=True):
    step = float(step)
    if np.isclose(step, 0.012):
        raw_z = 120.0 + 4.0j
        raw_d = 119.9
        raw_out = 0.1
        local_z = 1.0 + 0.10j
        local_d = 0.90
        local_out = 0.10
    elif np.isclose(step, 0.003):
        raw_z = 56.0 + 0.30j
        raw_d = 55.99
        raw_out = 0.01
        local_z = 1.50 + 0.20j
        local_d = 1.38
        local_out = 0.12
    else:
        # The absolute local response may differ strongly from the coarse seed;
        # v3 requires the *fine-minus-seed defect* itself to have converged.
        raw_z = 55.6 + 0.28j
        raw_d = 55.59
        raw_out = 0.01
        local_z = 1.505 + 0.201j
        local_d = 1.382
        local_out = 0.121

    return {
        "z": raw_z,
        "d_vol": raw_d,
        "d_out": raw_out,
        "modal_h": None,
        "localized_z": local_z,
        "localized_d_vol": local_d,
        "localized_d_out": local_out,
        "localized_modal_h": None,
        "localized_power_balance_relative_error": 0.0,
        "linear_relative_residual": residual,
        "linear_solver_converged": linear_converged,
        "power_balance_relative_error": 0.0,
        "joule_total_power_relative_error": joule_error,
        "joule_modal_contraction_relative_error": 0.0,
        "n_cells": 1,
        "n_edges": 1,
        "fine_step": step,
    }


def test_local_self_audit_gates_complete_defect(monkeypatch):
    background = _Background()
    calls = []

    def fake_local(_b, _g, _p, step, phi=None):
        calls.append(float(step))
        return _local(step)

    monkeypatch.setattr(correction, "_solve_local", fake_local)
    report = audit.audit_local_self_correction(background, [{"case": 0}])
    port = report["samples"][0]["ports"][0]

    assert np.allclose(calls, [0.012, 0.003, 0.00225])
    assert port["warm_start_seed_step"] == 0.012
    assert report["self_correction_model"] == (
        "canonical_local_full_fine_minus_coarse_self_defect_v3"
    )
    assert port["defect_model"] == "full_local_fine_minus_coarse_v3"
    assert port["relative_z_error"] < report["relative_tolerance"]
    assert port["relative_d_vol_error"] < report["relative_tolerance"]
    assert port["relative_outward_partition_significance"] < report["relative_tolerance"]
    assert port["diagnostic_localized_defect_relative_error"] < report["relative_tolerance"]
    assert report["maximum_joule_total_power_relative_error"] == 0.0
    assert report["maximum_linear_relative_residual"] == 0.0
    assert report["linear_solver_converged"]
    assert report["converged"]


def test_local_self_audit_rejects_nonconverged_longitudinal_defect(monkeypatch):
    background = _Background()

    def fake_local(_b, _g, _p, step, phi=None):
        value = _local(step)
        if np.isclose(step, 0.00225):
            # localized part remains converged, but the complete terminal defect
            # is not; v3 must fail rather than hiding it behind the v2 diagnostic.
            value["z"] = 9.8 - 3.0e6j
            value["d_vol"] = 9.798
            value["d_out"] = 0.002
        return value

    monkeypatch.setattr(correction, "_solve_local", fake_local)
    report = audit.audit_local_self_correction(background, [{"case": 0}])
    port = report["samples"][0]["ports"][0]
    assert port["diagnostic_localized_defect_relative_error"] < report["relative_tolerance"]
    assert port["relative_z_error"] > report["relative_tolerance"]
    assert not report["converged"]


def test_local_self_audit_rejects_bad_joule_identity(monkeypatch):
    background = _Background()
    monkeypatch.setattr(
        correction,
        "_solve_local",
        lambda _b, _g, _p, step, phi=None: _local(step, joule_error=1e-4),
    )
    report = audit.audit_local_self_correction(background, [{"case": 0}])
    assert report["maximum_relative_error"] < report["relative_tolerance"]
    assert report["maximum_joule_total_power_relative_error"] > report["joule_identity_tolerance"]
    assert not report["converged"]


def test_local_self_audit_rejects_uncertified_linear_solve(monkeypatch):
    background = _Background()

    def fake_local(_b, _g, _p, step, phi=None):
        if np.isclose(step, 0.00225):
            return _local(step, residual=4e-6, linear_converged=False)
        return _local(step, residual=2e-11, linear_converged=True)

    monkeypatch.setattr(correction, "_solve_local", fake_local)
    report = audit.audit_local_self_correction(background, [{"case": 0}])
    assert report["maximum_relative_error"] < report["relative_tolerance"]
    assert report["maximum_linear_relative_residual"] == 4e-6
    assert report["maximum_linear_relative_residual"] > report["linear_relative_residual_tolerance"]
    assert not report["linear_solver_converged"]
    assert not report["converged"]
