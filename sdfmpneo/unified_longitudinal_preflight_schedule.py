"""Run longitudinal scalar certificates before expensive local Maxwell.

This changes scheduling only.  Source/material continuity is checked first; then
the boundary-conditioned terminal reactive reference and the whole-domain
longitudinal dissipative reference are audited.  A failed scalar certificate
returns the same structured preflight failure without running 118k/254k
local-self Maxwell, open-boundary or MQS gates.  Successful scalar audits are
cached on the background and reused by the later corrected mesh gate.
"""
from __future__ import annotations

import copy
import numpy as np


def _geometry_key(global_module, geometry):
    helper = getattr(global_module, "_geometry_key", None)
    return helper(geometry) if callable(helper) else repr(geometry)


def install(corrected_preflight_module, global_module):
    if bool(getattr(corrected_preflight_module, "_early_longitudinal_screen_installed", False)):
        return corrected_preflight_module

    original_audit = global_module.audit_reference_convergence

    def cached_audit(background, geometry):
        cache = getattr(background, "_sdfmpneo_longitudinal_reference_audit_cache", None)
        if cache is None:
            cache = {}
            background._sdfmpneo_longitudinal_reference_audit_cache = cache
        key = _geometry_key(global_module, geometry)
        if key not in cache:
            cache[key] = copy.deepcopy(original_audit(background, geometry))
        return copy.deepcopy(cache[key])

    global_module.audit_reference_convergence = cached_audit

    original_run = corrected_preflight_module.run_truth_preflight

    def run_truth_preflight(settings, background, geometries, monitor=None):
        global_module._resolve_settings(settings, background)
        geometries = list(geometries)
        if not geometries:
            return original_run(settings, background, geometries, monitor)

        source_rows = [
            corrected_preflight_module._source_and_loss_partition(background, g)
            for g in geometries
        ]
        max_fraction = max(
            float(row["material_fraction_closure_error"]) for row in source_rows
        )
        max_partition = max(
            float(row["wire_loss_partition_relative_error"]) for row in source_rows
        )
        max_terminal = max(
            float(row["maximum_terminal_path_integral_relative_error"])
            for row in source_rows
        )
        source_checks = {
            "finite_support_source_ok": all(
                bool(row["finite_support_source"]) for row in source_rows
            ),
            "terminal_source_continuity_ok": all(
                bool(row["terminal_path_conservation"]) for row in source_rows
            ),
            "material_fraction_closure_ok": max_fraction <= 1e-10,
            "wire_loss_not_double_counted": max_partition <= 1e-12,
        }
        if not all(source_checks.values()):
            return original_run(settings, background, geometries, monitor)

        cfg = settings["BACKGROUND"]
        mesh_n = min(
            len(geometries),
            max(1, int(cfg.get("mesh_check", {}).get("samples", 1))),
        )
        scalar_rows = []
        for index, geometry in enumerate(geometries[:mesh_n]):
            if monitor is not None:
                monitor.checkpoint()
            row = cached_audit(background, geometry)
            row["index"] = int(index)
            scalar_rows.append(row)
        scalar_worst = max(
            (float(row.get("maximum_relative_error", np.inf)) for row in scalar_rows),
            default=0.0,
        )
        scalar_ok = bool(
            scalar_rows and all(bool(row.get("converged", False)) for row in scalar_rows)
        )
        scalar_report = {
            "model": global_module._MODEL,
            "sample_count": len(scalar_rows),
            "maximum_relative_error": float(scalar_worst),
            "converged": scalar_ok,
            "samples": scalar_rows,
        }
        if scalar_ok:
            print(
                "longitudinal scalar prerequisite……passed; continuing local Maxwell audit",
                flush=True,
            )
            return original_run(settings, background, geometries, monitor)

        skip_reason = "longitudinal scalar prerequisite failed"
        print(f"local self correction convergence……skipped ({skip_reason})", flush=True)
        print(f"开放边界域扩展 Gate……skipped ({skip_reason})", flush=True)
        print(f"full-wave ↔ MQS formulation Gate……skipped ({skip_reason})", flush=True)
        print(f"pre-basis corrected EM mesh Gate……skipped ({skip_reason})", flush=True)

        local_self = corrected_preflight_module._skipped_gate_report(skip_reason)
        domain = corrected_preflight_module._skipped_gate_report(skip_reason)
        formulation = corrected_preflight_module._skipped_gate_report(skip_reason)
        mesh = corrected_preflight_module._skipped_mesh_report(settings, skip_reason)
        mesh["global_longitudinal_reference_convergence"] = scalar_report
        mesh["maximum_relative_error"] = float(scalar_worst)
        mesh["maximum_source_path_length_relative_error"] = 0.0
        mesh["source_geometry_invariant"] = True
        mesh["skipped_full_em_mesh_gate"] = True
        mesh["skip_reason"] = skip_reason

        dissipative_failed = False
        dissipative_worst = 0.0
        for row in scalar_rows:
            item = row.get("global_dissipative_reference")
            if isinstance(item, dict) and not bool(item.get("converged", False)):
                dissipative_failed = True
                dissipative_worst = max(
                    dissipative_worst,
                    float(item.get("maximum_relative_error", np.inf)),
                )
        if dissipative_failed:
            diagnosis = {
                "code": "global_longitudinal_dissipative_reference_not_converged",
                "maximum_relative_error": float(dissipative_worst),
                "recommendation": (
                    "Terminal charge continuity and the reactive terminal-scale defect are "
                    "certified, but the whole-domain longitudinal dissipative scalar "
                    "reference is not yet converged. Diagnose the package/seawater loss "
                    "operator or its scalar reference; do not run/refine global Maxwell and "
                    "do not relax the Gate."
                ),
            }
        else:
            diagnosis = {
                "code": "boundary_conditioned_longitudinal_defect_not_converged",
                "maximum_relative_error": float(scalar_worst),
                "recommendation": (
                    "The physical terminal charge/source continuity is certified, but the "
                    "terminal-scale boundary-conditioned reactive scalar defect is not "
                    "converged. Adjust only the terminal-local scalar resolution/model; do "
                    "not run/refine global Maxwell and do not relax the Gate."
                ),
            }

        checks = {
            **source_checks,
            "open_boundary_domain_converged": False,
            "low_frequency_formulation_converged": False,
            "em_mesh_converged": False,
            "local_self_correction_converged": False,
        }
        return {
            **{key: bool(value) for key, value in checks.items()},
            "source_model": getattr(background, "source_model", "unknown"),
            "source_centerline_step": float(
                getattr(background, "source_centerline_step", np.nan)
            ),
            "terminal_model": getattr(background, "terminal_model", "unknown"),
            "maximum_terminal_path_integral_relative_error": float(max_terminal),
            "maximum_material_fraction_closure_error": float(max_fraction),
            "maximum_wire_loss_partition_relative_error": float(max_partition),
            "open_boundary_convergence": domain,
            "formulation_convergence": formulation,
            "em_mesh_convergence": mesh,
            "local_self_correction_convergence": local_self,
            "failure_diagnosis": diagnosis,
            "source_samples": source_rows,
            "certified": False,
            "status": "truth_preflight_failed",
        }

    corrected_preflight_module.run_truth_preflight = run_truth_preflight
    corrected_preflight_module._early_longitudinal_screen_installed = True
    return corrected_preflight_module


__all__ = ["install"]
