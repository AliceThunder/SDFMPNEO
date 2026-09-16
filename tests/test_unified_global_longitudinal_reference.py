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
        "self_correction": {
            "fine_step": 0.003,
            "validation_fine_step": 0.00225,
            "core_padding": 0.006,
            "boundary_padding": 0.04,
            "growth": 1.5,
            "max_step": 0.02,
        },
        "mesh_check": {"refinement_factor": 0.75, "relative_tolerance": 0.1},
    }
    self_correction_config = dict(background_config["self_correction"])


def test_boundary_conditioned_longitudinal_defect_changes_only_reactive_self_diagonal(monkeypatch):
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
            "delta_z": np.array([0.05j, 0.07j]),
            "delta_d_vol": np.zeros(2),
            "delta_d_out": np.zeros(2),
            "delta_modal_h": (
                None if phi is None else np.zeros((2, 2), float)
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
    assert np.allclose(dc, d)
    assert np.allclose(oc, d_out)
    assert np.allclose(hc, modal)
    assert np.allclose(np.diag(zc - z), [0.05j, 0.07j])
    assert np.allclose(0.5 * (zc + zc.conj().T), 0.5 * (z + z.conj().T))
    assert audit["model"] == glr._MODEL


def test_defect_applies_only_reactive_part_and_keeps_loss_as_diagnostic():
    coarse = {
        "z_reaction": 4.0 + 0.20j,
        "d_vol": 3.7,
        "modal_h": np.array([1.0, 0.4]),
    }
    fine = {
        "z_reaction": 6.5 + 0.26j,
        "d_vol": 5.5,
        "modal_h": np.array([1.5, 0.7]),
    }
    defect = glr._defect(coarse, fine)
    assert np.isclose(defect["delta_d_vol"], 0.0)
    assert np.isclose(defect["delta_z"].real, 0.0)
    assert np.isclose(defect["delta_z"].imag, 0.06)
    assert defect["delta_d_out"] == 0.0
    assert np.allclose(defect["delta_modal_h"], 0.0)
    assert np.isclose(defect["diagnostic_raw_delta_d_vol"], 1.8)
    assert np.isclose(defect["diagnostic_raw_delta_z_real"], 2.5)
    assert np.isclose(defect["reaction_delta_z"].real, 2.5)
    assert np.isclose(defect["interface_real_flux_defect"], 0.7)
    assert defect["correction_semantics"] == "reactive_only_certified_longitudinal_defect"


def test_resolved_patch_settings_follow_certified_local_reference_steps():
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "self_correction": {
                "fine_step": 0.003,
                "validation_fine_step": 0.00225,
                "core_padding": 0.006,
                "boundary_padding": 0.04,
                "growth": 1.5,
                "max_step": 0.02,
            },
            "mesh_check": {"refinement_factor": 0.75, "relative_tolerance": 0.1},
        }
    }
    background = _Background()
    glr._resolve_settings(settings, background)
    own = settings["BACKGROUND"]["global_longitudinal_correction"]
    assert np.isclose(own["fine_step"], 0.003)
    assert np.isclose(own["validation_fine_step"], 0.00225)
    assert np.isclose(own["relative_tolerance"], 0.1)
    cfg = glr._config(background)
    assert np.isclose(cfg["fine_step"], 0.003)
    assert np.isclose(cfg["validation_fine_step"], 0.00225)


def test_patch_coarse_axes_are_exact_parent_subsets():
    parent = types.SimpleNamespace(
        x=np.linspace(-0.2, 0.2, 21),
        y=np.linspace(-0.2, 0.2, 21),
        z=np.linspace(-0.2, 0.2, 21),
    )
    package = types.SimpleNamespace(
        half_extent=np.array([0.03, 0.02, 0.01]),
        pose=types.SimpleNamespace(
            rotation=np.eye(3),
            translation=np.array([0.01, -0.02, 0.03]),
        ),
    )
    axes, _, _ = glr._patch_axes(parent, package, {"boundary_padding": 0.04})
    for source, patch in zip((parent.x, parent.y, parent.z), axes):
        assert patch.size >= 3
        assert all(np.any(np.isclose(source, value)) for value in patch)


def test_scalar_defect_failure_skips_full_em_mesh_gate(monkeypatch):
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
    assert diagnosis["code"] == "boundary_conditioned_longitudinal_defect_not_converged"
