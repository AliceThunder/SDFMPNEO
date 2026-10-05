import types

import numpy as np

import sdfmpneo.unified_global_longitudinal_physics_gate as adapter


def test_post_basis_corrected_fields_apply_global_reference(monkeypatch):
    module = types.SimpleNamespace()
    module._SELF_CORRECTION_MODEL = "canonical_local_transverse_fine_minus_coarse_self_defect_v2"
    context = types.SimpleNamespace()
    z = np.array([[4.0 + 0.2j, 0.4 + 0.1j], [0.4 + 0.1j, 5.0 + 0.3j]], complex)
    d = np.eye(2, dtype=complex)
    d_out = np.eye(2, dtype=complex) * 0.2
    modal = np.stack([np.eye(2), 2.0 * np.eye(2)]).astype(complex)
    module._corrected_fields = lambda background, geometry, phi=None: (
        context,
        z.copy(),
        d.copy(),
        d_out.copy(),
        modal.copy(),
        {"local": True},
    )
    module.audit_mesh_convergence = lambda *args, **kwargs: {
        "converged": True,
        "maximum_relative_error": 0.01,
        "samples": [],
    }

    calls = {"remember": 0, "apply": 0}

    longitudinal = types.SimpleNamespace()
    longitudinal._MODEL = "global_compatible_longitudinal_self_reference_v1"

    def remember(background, geometry, value):
        assert value is context
        calls["remember"] += 1

    def apply(background, geometry, zi, di, oi, *, phi=None, modal_h=None):
        calls["apply"] += 1
        outz = np.asarray(zi, complex).copy()
        outd = np.asarray(di, complex).copy()
        outo = np.asarray(oi, complex).copy()
        outh = np.asarray(modal_h, complex).copy()
        outz[0, 0] += 2.0 + 0.05j
        outd[0, 0] += 1.8
        outo[0, 0] += 0.2
        outh[:, 0, 0] += np.array([0.5, 0.7])
        return outz, outd, outo, outh, {"enabled": True}

    longitudinal._remember_context = remember
    longitudinal.apply_global_longitudinal_reference = apply
    longitudinal.audit_reference_convergence = lambda *args, **kwargs: {
        "converged": True,
        "maximum_relative_error": 0.02,
    }

    adapter.install(module, longitudinal)
    _, zc, dc, oc, hc, audit = module._corrected_fields(object(), object(), np.ones((1, 2)))
    assert calls == {"remember": 1, "apply": 1}
    assert np.allclose(zc[0, 1], z[0, 1])
    assert np.allclose(dc[0, 1], d[0, 1])
    assert np.allclose(oc[0, 1], d_out[0, 1])
    assert np.allclose(hc[:, 0, 1], modal[:, 0, 1])
    assert audit["global_longitudinal_reference"]["enabled"]


def test_post_basis_scalar_failure_skips_full_mesh(monkeypatch):
    module = types.SimpleNamespace()
    module._SELF_CORRECTION_MODEL = "canonical_local_transverse_fine_minus_coarse_self_defect_v2"
    module._corrected_fields = lambda *args, **kwargs: None
    calls = {"mesh": 0}

    def full_mesh(*args, **kwargs):
        calls["mesh"] += 1
        return {"converged": True, "maximum_relative_error": 0.01, "samples": []}

    module.audit_mesh_convergence = full_mesh
    longitudinal = types.SimpleNamespace()
    longitudinal._MODEL = "global_compatible_longitudinal_self_reference_v1"
    longitudinal._remember_context = lambda *args, **kwargs: None
    longitudinal.apply_global_longitudinal_reference = lambda *args, **kwargs: None
    longitudinal.audit_reference_convergence = lambda *args, **kwargs: {
        "converged": False,
        "maximum_relative_error": 0.31,
    }

    adapter.install(module, longitudinal)
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "mesh_check": {"refinement_factor": 0.75, "relative_tolerance": 0.1},
        }
    }
    report = module.audit_mesh_convergence(settings, object(), [object()])
    assert not report["converged"]
    assert report["skipped_full_em_mesh_gate"]
    assert report["maximum_relative_error"] == 0.31
    assert calls["mesh"] == 0
