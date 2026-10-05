"""Post-basis Physics Gate adapter for the global longitudinal reference."""
from __future__ import annotations

import numpy as np


def install(physics_gate_module, longitudinal_module):
    if bool(getattr(physics_gate_module, "_global_longitudinal_reference_installed", False)):
        return physics_gate_module

    original_fields = physics_gate_module._corrected_fields

    def corrected_fields(background, geometry, phi=None):
        context, z, d, d_out, modal, audit = original_fields(
            background, geometry, phi
        )
        longitudinal_module._remember_context(background, geometry, context)
        zc, dc, oc, hc, global_audit = longitudinal_module.apply_global_longitudinal_reference(
            background,
            geometry,
            z,
            d,
            d_out,
            phi=phi,
            modal_h=modal,
        )
        out = dict(audit)
        out["global_longitudinal_reference"] = global_audit
        return context, zc, dc, oc, hc, out

    physics_gate_module._corrected_fields = corrected_fields

    original_mesh = physics_gate_module.audit_mesh_convergence

    def audit_mesh_convergence(settings, background, geometries, monitor=None):
        geometries = list(geometries)
        scalar_rows = []
        for index, geometry in enumerate(geometries):
            if monitor is not None:
                monitor.checkpoint()
            row = longitudinal_module.audit_reference_convergence(background, geometry)
            row["index"] = int(index)
            scalar_rows.append(row)
        scalar_worst = max((r["maximum_relative_error"] for r in scalar_rows), default=0.0)
        scalar_ok = all(bool(r["converged"]) for r in scalar_rows)
        scalar_report = {
            "model": longitudinal_module._MODEL,
            "sample_count": len(scalar_rows),
            "maximum_relative_error": float(scalar_worst),
            "converged": bool(scalar_ok),
            "samples": scalar_rows,
        }
        if not scalar_ok:
            cfg = dict(settings["BACKGROUND"].get("mesh_check", {}))
            factor = float(cfg.get("refinement_factor", 0.75))
            base = float(settings["BACKGROUND"]["fine_step"])
            print(
                "corrected spatial mesh refinement Gate……skipped "
                "(global longitudinal scalar reference failed)",
                flush=True,
            )
            return {
                "sample_count": len(geometries),
                "refinement_factor": factor,
                "base_fine_step": base,
                "refined_fine_step": factor * base,
                "relative_tolerance": float(cfg.get("relative_tolerance", 1e-1)),
                "maximum_relative_error": float(scalar_worst),
                "self_correction_model": physics_gate_module._SELF_CORRECTION_MODEL,
                "global_longitudinal_reference_convergence": scalar_report,
                "converged": False,
                "skipped_full_em_mesh_gate": True,
                "samples": [],
            }
        report = original_mesh(settings, background, geometries, monitor)
        report["global_longitudinal_reference_convergence"] = scalar_report
        report["maximum_relative_error"] = max(
            float(report.get("maximum_relative_error", 0.0)), float(scalar_worst)
        )
        report["converged"] = bool(report.get("converged", False) and scalar_ok)
        return report

    physics_gate_module.audit_mesh_convergence = audit_mesh_convergence
    physics_gate_module._global_longitudinal_reference_installed = True
    return physics_gate_module


__all__ = ["install"]
