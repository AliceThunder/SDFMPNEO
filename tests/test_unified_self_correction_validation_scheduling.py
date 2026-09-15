import threading

import numpy as np

import sdfmpneo.unified_self_correction as correction
import sdfmpneo.unified_self_correction_audit as audit


class _Background:
    coil_materials = ("tx", "rx")


def _result(step):
    step = float(step)
    return {
        "z": 1.0 + 1.0j,
        "d_vol": 1.0,
        "d_out": 0.0,
        "localized_z": 0.8 + 0.2j,
        "localized_d_vol": 0.75,
        "localized_d_out": 0.05,
        "localized_power_balance_relative_error": 0.0,
        "power_balance_relative_error": 0.0,
        "linear_relative_residual": 1e-12,
        "linear_solver_converged": True,
        "joule_total_power_relative_error": 0.0,
        "joule_modal_contraction_relative_error": 0.0,
        "fine_step": step,
    }


def test_fine_ports_parallel_but_large_validation_ports_serialized(monkeypatch):
    cfg = {
        "fine_step": 0.003,
        "validation_fine_step": 0.00225,
        "relative_tolerance": 0.1,
        "joule_identity_tolerance": 1e-10,
        "linear_relative_residual_tolerance": 1e-9,
        "parallel_preflight_ports": 2,
        "parallel_validation_ports": 1,
        "linear_warm_start_from_parent": False,
    }
    monkeypatch.setattr(correction, "_config", lambda background: dict(cfg))
    monkeypatch.setattr(correction, "_parent_fine_step", lambda background: 0.012)

    calls = []
    lock = threading.Lock()

    def fake_solve(parent, geometry, port, step, phi=None):
        with lock:
            calls.append((int(port), float(step), threading.current_thread().name))
        return _result(step)

    monkeypatch.setattr(correction, "_solve_local", fake_solve)

    result = audit.audit_local_self_correction(_Background(), [object()])

    fine_calls = [row for row in calls if np.isclose(row[1], 0.003)]
    validation_calls = [row for row in calls if np.isclose(row[1], 0.00225)]
    assert len(fine_calls) == 2
    assert len(validation_calls) == 2
    assert all("local-self-fine" in row[2] for row in fine_calls)
    assert all("local-self-validation" not in row[2] for row in validation_calls)
    assert result["parallel_ports"] == 2
    assert result["parallel_validation_ports"] == 1
    assert result["linear_solver_converged"]
    assert result["converged"]
