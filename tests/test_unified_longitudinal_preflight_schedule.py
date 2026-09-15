import types

from sdfmpneo.unified_longitudinal_preflight_schedule import install


def _good_source_row():
    return {
        "finite_support_source": True,
        "terminal_path_conservation": True,
        "material_fraction_closure_error": 0.0,
        "wire_loss_partition_relative_error": 0.0,
        "maximum_terminal_path_integral_relative_error": 0.0,
    }


def _modules(audit_result):
    calls = {"audit": 0, "original": 0}
    global_module = types.SimpleNamespace()
    global_module._MODEL = "terminal-v3"
    global_module._geometry_key = lambda geometry: str(geometry)
    global_module._resolve_settings = lambda settings, background: None

    def audit(background, geometry):
        calls["audit"] += 1
        return dict(audit_result)

    global_module.audit_reference_convergence = audit

    corrected = types.SimpleNamespace()
    corrected._source_and_loss_partition = lambda background, geometry: _good_source_row()
    corrected._skipped_gate_report = lambda reason: {
        "converged": False,
        "skipped": True,
        "skip_reason": reason,
        "samples": [],
    }
    corrected._skipped_mesh_report = lambda settings, reason: {
        "converged": False,
        "skipped": True,
        "skip_reason": reason,
        "samples": [],
        "maximum_relative_error": float("inf"),
    }

    def original(settings, background, geometries, monitor=None):
        calls["original"] += 1
        return {"certified": True, "status": "original-ran"}

    corrected.run_truth_preflight = original
    return corrected, global_module, calls


def test_failed_terminal_scalar_gate_skips_original_expensive_preflight():
    corrected, global_module, calls = _modules(
        {"converged": False, "maximum_relative_error": 0.42, "samples": []}
    )
    install(corrected, global_module)
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "mesh_check": {"samples": 1, "refinement_factor": 0.75},
        }
    }
    report = corrected.run_truth_preflight(settings, object(), ["g0"])
    assert calls["audit"] == 1
    assert calls["original"] == 0
    assert report["certified"] is False
    assert report["failure_diagnosis"]["code"] == (
        "boundary_conditioned_longitudinal_defect_not_converged"
    )
    assert report["em_mesh_convergence"]["skipped_full_em_mesh_gate"] is True
    assert report["local_self_correction_convergence"]["skipped"] is True


def test_passed_terminal_scalar_gate_is_cached_before_original_preflight():
    corrected, global_module, calls = _modules(
        {"converged": True, "maximum_relative_error": 0.02, "samples": []}
    )
    install(corrected, global_module)
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "mesh_check": {"samples": 1, "refinement_factor": 0.75},
        }
    }
    background = object()
    report = corrected.run_truth_preflight(settings, background, ["g0"])
    assert report["status"] == "original-ran"
    assert calls["original"] == 1
    assert calls["audit"] == 1
    # The installed audit surface must now reuse the exact same certificate.
    again = global_module.audit_reference_convergence(background, "g0")
    assert again["converged"] is True
    assert calls["audit"] == 1
