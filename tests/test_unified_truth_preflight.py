import numpy as np

import sdfmpneo.unified_truth_preflight as preflight


class _Background:
    pass


def _tuple(z, d, d_out):
    return None, None, None, np.asarray(z, complex), np.asarray(d, complex), np.asarray(d_out, complex)


def test_open_boundary_gate_does_not_require_raw_outward_flux_invariance(monkeypatch):
    base = _Background()
    expanded = _Background()
    settings = {
        "BACKGROUND": {
            "bounds": [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            "open_boundary_check": {
                "padding": 0.5,
                "relative_tolerance": 5e-2,
            },
        }
    }
    z0 = np.array([[2.0, 0.20], [0.20, 1.5]], complex)
    z1 = np.array([[2.01, 0.198], [0.198, 1.49]], complex)
    d0 = np.array([[1.990, 0.198], [0.198, 1.490]], complex)
    d1 = np.array([[2.008, 0.197], [0.197, 1.488]], complex)
    # Moving the artificial boundary outward in a lossy medium can reduce the
    # residual outward flux by a large *relative* factor even when that flux is
    # tiny compared with total terminal dissipation.
    o0 = np.eye(2) * 1.0e-2
    o1 = np.eye(2) * 2.0e-3

    monkeypatch.setattr(preflight, "_background_from_settings", lambda *args, **kwargs: expanded)

    def fake_solve(background, geometry):
        return _tuple(z0, d0, o0) if background is base else _tuple(z1, d1, o1)

    monkeypatch.setattr(preflight, "_solve_fields", fake_solve)
    report = preflight.audit_open_boundary_domain(settings, base, [{"case": 0}])
    sample = report["samples"][0]

    assert sample["raw_relative_d_out_error"] > 1.0
    assert sample["relative_outward_partition_significance"] < 5e-2
    assert sample["relative_z_error"] < 5e-2
    assert sample["relative_d_vol_error"] < 5e-2
    assert report["converged"]


def test_em_mesh_gate_reports_self_vs_mutual_without_relaxing_failure(monkeypatch):
    base = _Background()
    refined = _Background()
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "max_step": 0.03,
            "mesh_check": {
                "refinement_factor": 0.75,
                "relative_tolerance": 1e-1,
            },
        }
    }
    z0 = np.array([[2.0, 0.20], [0.20, 1.5]], complex)
    z1 = np.array([[1.4, 0.198], [0.198, 1.0]], complex)
    d0 = np.array([[1.8, 0.10j], [-0.10j, 1.2]], complex)
    d1 = np.array([[1.2, 0.099j], [-0.099j, 0.8]], complex)
    o = np.eye(2) * 0.01

    monkeypatch.setattr(preflight, "_background_from_settings", lambda *args, **kwargs: refined)

    def fake_solve(background, geometry):
        return _tuple(z0, d0, o) if background is base else _tuple(z1, d1, o)

    monkeypatch.setattr(preflight, "_solve_fields", fake_solve)
    report = preflight.audit_em_mesh_preflight(settings, base, [{"case": 0}])
    sample = report["samples"][0]
    diagnosis = preflight._mesh_failure_diagnosis(report)

    assert not report["converged"]
    assert sample["maximum_relative_error"] > 1e-1
    assert sample["diagnostic_z_self_relative_error"] > sample["relative_mutual_impedance_error"]
    assert sample["diagnostic_d_vol_self_relative_error"] > sample["diagnostic_d_vol_mutual_relative_error"]
    assert diagnosis["code"] == "unresolved_source_self_response"
