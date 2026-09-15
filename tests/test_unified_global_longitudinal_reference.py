import types

import numpy as np

import sdfmpneo.unified_global_longitudinal_reference as glr


class _Background:
    n_cells = 1
    nx = ny = nz = 1
    cell_axes = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    cell_centers = np.array([[0.0, 0.0, 0.0]])
    coil_materials = ("tx", "rx")
    background_config = {
        "fine_step": 0.012,
        "max_step": 0.08,
        "mesh_check": {"refinement_factor": 0.75, "relative_tolerance": 0.1},
    }


def test_global_longitudinal_reference_changes_only_self_diagonal(monkeypatch):
    background = _Background()
    z = np.array([[4.0 + 0.2j, 0.4 + 0.1j], [0.4 + 0.1j, 5.0 + 0.3j]], complex)
    d = np.array([[3.7, 0.1 + 0.02j], [0.1 - 0.02j, 4.6]], complex)
    d_out = 0.5 * (z + z.conj().T) - d
    modal = np.array(
        [
            [[1.0, 0.03j], [-0.03j, 1.2]],
            [[0.4, 0.01], [0.01, 0.6]],
        ],
        complex,
    )

    monkeypatch.setattr(
        glr,
        "_correction",
        lambda _b, _g, phi=None: {
            "delta_z": np.array([2.0 + 0.05j, 3.0 + 0.07j]),
            "delta_d_vol": np.array([1.8, 2.7]),
            "delta_d_out": np.array([0.2, 0.3]),
            "delta_modal_h": (
                None
                if phi is None
                else np.array([[0.5, 0.8], [0.2, 0.4]], float)
            ),
            "audit": {"enabled": True, "model": glr._MODEL},
        },
    )

    zc, dc, oc, hc, audit = glr.apply_global_longitudinal_reference(
        background,
        object(),
        z,
        d,
        d_out,
        phi=np.ones((1, 2)),
        modal_h=modal,
    )

    assert np.allclose(zc[0, 1], z[0, 1])
    assert np.allclose(dc[0, 1], d[0, 1])
    assert np.allclose(oc[0, 1], d_out[0, 1])
    assert np.allclose(hc[:, 0, 1], modal[:, 0, 1])
    assert np.allclose(np.diag(zc - z), [2.0 + 0.05j, 3.0 + 0.07j])
    assert np.allclose(np.real(np.diag(dc - d)), [1.8, 2.7])
    assert np.allclose(np.real(np.diag(oc - d_out)), [0.2, 0.3])
    assert np.allclose(np.real(hc[:, 0, 0] - modal[:, 0, 0]), [0.5, 0.2])
    assert np.allclose(np.real(hc[:, 1, 1] - modal[:, 1, 1]), [0.8, 0.4])
    assert audit["model"] == glr._MODEL


def test_resolved_reference_step_is_common_across_refined_backgrounds():
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "mesh_check": {"refinement_factor": 0.75, "relative_tolerance": 0.1},
        }
    }
    base = _Background()
    glr._resolve_settings(settings, base)
    assert np.isclose(
        settings["BACKGROUND"]["mesh_check"]["longitudinal_reference_fine_step"],
        0.009,
    )
    assert np.isclose(
        settings["BACKGROUND"]["mesh_check"]["longitudinal_reference_validation_factor"],
        0.75,
    )
    assert np.isclose(base.background_config["_production_fine_step"], 0.012)

    refined = _Background()
    refined.background_config = {
        "fine_step": 0.009,
        "max_step": 0.06,
        "_production_fine_step": 0.012,
        "mesh_check": dict(settings["BACKGROUND"]["mesh_check"]),
    }
    assert np.isclose(glr._config(base)["reference_fine_step"], 0.009)
    assert np.isclose(glr._config(refined)["reference_fine_step"], 0.009)


def test_scalar_reference_failure_skips_full_em_mesh_gate(monkeypatch):
    calls = {"mesh": 0}

    preflight = types.SimpleNamespace()
    truth = types.SimpleNamespace()
    preflight._SELF_CORRECTION_MODEL = "canonical_local_transverse_fine_minus_coarse_self_defect_v2"
    preflight._solve_fields = lambda *args, **kwargs: None
    preflight._correct = lambda b, g, z, d, o: (z, d, o, {})
    preflight.run_truth_preflight = lambda settings, background, geometries, monitor=None: {}
    preflight._mesh_failure_diagnosis = lambda mesh: {"code": "fallback"}

    def original_mesh(*args, **kwargs):
        calls["mesh"] += 1
        return {"converged": True, "maximum_relative_error": 0.0, "samples": []}

    preflight.audit_em_mesh_preflight = original_mesh
    truth._port_truth_from_context = lambda background, context: None
    truth.solve_port_truth_tensors = lambda background, geometry: None
    truth.solve_truth_tensors = lambda background, geometry: None
    truth._loewner_violation = lambda *args, **kwargs: 0.0

    monkeypatch.setattr(
        glr,
        "audit_reference_convergence",
        lambda background, geometry: {
            "converged": False,
            "maximum_relative_error": 0.42,
            "relative_tolerance": 0.1,
        },
    )
    glr.install(preflight, truth)

    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "mesh_check": {
                "refinement_factor": 0.75,
                "relative_tolerance": 0.1,
                "source_path_relative_tolerance": 1e-10,
            },
        }
    }
    background = _Background()
    report = preflight.audit_em_mesh_preflight(settings, background, [object()])
    assert not report["converged"]
    assert report["skipped_full_em_mesh_gate"]
    assert report["maximum_relative_error"] == 0.42
    assert calls["mesh"] == 0
    diagnosis = preflight._mesh_failure_diagnosis(report)
    assert diagnosis["code"] == "global_longitudinal_reference_not_converged"
