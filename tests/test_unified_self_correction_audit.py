import numpy as np

import sdfmpneo.unified_self_correction_audit as audit


class _Background:
    coil_materials = ("tx",)
    self_correction_config = {
        "fine_step": 0.003,
        "validation_fine_step": 0.00225,
        "relative_tolerance": 0.1,
        "joule_identity_tolerance": 1e-10,
    }


def _local(step, *, joule_error=0.0):
    if np.isclose(step, 0.003):
        return {
            "z": 10.0 + 1.0j,
            "d_vol": 9.99,
            "d_out": 0.01,
            "modal_h": None,
            "linear_relative_residual": 0.0,
            "power_balance_relative_error": 0.0,
            "joule_total_power_relative_error": joule_error,
            "joule_modal_contraction_relative_error": 0.0,
            "n_cells": 1,
            "n_edges": 1,
            "fine_step": step,
        }
    return {
        "z": 10.01 + 1.005j,
        "d_vol": 10.008,
        "d_out": 0.002,
        "modal_h": None,
        "linear_relative_residual": 0.0,
        "power_balance_relative_error": 0.0,
        "joule_total_power_relative_error": joule_error,
        "joule_modal_contraction_relative_error": 0.0,
        "n_cells": 1,
        "n_edges": 1,
        "fine_step": step,
    }


def test_local_self_audit_does_not_gate_on_raw_outward_relative_change(monkeypatch):
    background = _Background()
    monkeypatch.setattr(audit, "_solve_local", lambda _b, _g, _p, step, phi=None: _local(step))
    report = audit.audit_local_self_correction(background, [{"case": 0}])
    port = report["samples"][0]["ports"][0]

    assert port["raw_relative_d_out_error"] > 1.0
    assert port["relative_outward_partition_significance"] < 0.01
    assert report["maximum_joule_total_power_relative_error"] == 0.0
    assert report["converged"]


def test_local_self_audit_rejects_bad_joule_identity_even_when_mesh_converges(monkeypatch):
    background = _Background()
    monkeypatch.setattr(
        audit,
        "_solve_local",
        lambda _b, _g, _p, step, phi=None: _local(step, joule_error=1e-4),
    )
    report = audit.audit_local_self_correction(background, [{"case": 0}])
    assert report["maximum_relative_error"] < report["relative_tolerance"]
    assert report["maximum_joule_total_power_relative_error"] > report["joule_identity_tolerance"]
    assert not report["converged"]
