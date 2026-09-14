"""Pre-basis truth preflight for the defect-corrected production EM truth."""
from __future__ import annotations

import copy
import numpy as np

from .unified_physics_gate import (
    _background_from_settings as _raw_background_from_settings,
    _off_diagonal,
    _power_contraction_relative_error,
    _relative,
    _solve_fields,
    audit_low_frequency_formulation,
)
from .unified_self_correction import apply_local_self_correction
from .unified_self_correction_audit import audit_local_self_correction
from .unified_truth_preflight import (
    _diagonal,
    _diag_vector,
    _source_and_loss_partition,
    _source_path_lengths,
    audit_open_boundary_domain,
)


def _background_from_settings(settings, *, fine_step=None, max_step=None):
    bg = _raw_background_from_settings(settings, fine_step=fine_step, max_step=max_step)
    cfg = copy.deepcopy(dict(settings["BACKGROUND"]))
    if fine_step is not None:
        cfg["fine_step"] = float(fine_step)
    if max_step is not None:
        cfg["max_step"] = float(max_step)
    bg.background_config = cfg
    bg.self_correction_config = copy.deepcopy(dict(cfg.get("self_correction", {})))
    return bg


def _correct(background, geometry, z, d, d_out):
    result = apply_local_self_correction(background, geometry, z, d, d_out)
    return result.z, result.d_vol, result.d_out, result.audit


def audit_em_mesh_preflight(settings, background, geometries, monitor=None):
    cfg = dict(settings["BACKGROUND"].get("mesh_check", {}))
    tolerance = float(cfg.get("relative_tolerance", 1e-1))
    factor = float(cfg.get("refinement_factor", 0.75))
    path_tolerance = float(cfg.get("source_path_relative_tolerance", 1e-10))
    if not 0.0 < factor < 1.0:
        raise ValueError("mesh_check.refinement_factor must lie in (0, 1)")
    base_fine = float(settings["BACKGROUND"]["fine_step"])
    base_max = float(settings["BACKGROUND"].get("max_step", 4.0 * base_fine))
    refined_fine = factor * base_fine
    refined = _background_from_settings(
        settings,
        fine_step=refined_fine,
        max_step=max(refined_fine, factor * base_max),
    )
    rows = []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        context0, _, _, z0_raw, d0_raw, o0_raw = _solve_fields(background, geometry)
        context1, _, _, z1_raw, d1_raw, o1_raw = _solve_fields(refined, geometry)
        print(f"local self defect……{index+1}/{len(geometries)} base mesh", flush=True)
        z0, d0, o0, correction0 = _correct(background, geometry, z0_raw, d0_raw, o0_raw)
        print(f"local self defect……{index+1}/{len(geometries)} refined mesh", flush=True)
        z1, d1, o1, correction1 = _correct(refined, geometry, z1_raw, d1_raw, o1_raw)

        path0 = _source_path_lengths(context0)
        path1 = _source_path_lengths(context1)
        path_error = (
            _relative(path0, path1)
            if path0.shape == path1.shape and path0.size and np.all(np.isfinite(path0)) and np.all(np.isfinite(path1))
            else float("inf")
        )
        raw_zdiag0 = _diag_vector(z0_raw)
        raw_zdiag1 = _diag_vector(z1_raw)
        zdiag0 = _diag_vector(z0)
        zdiag1 = _diag_vector(z1)
        row = {
            "index": int(index),
            "geometry": geometry,
            "relative_z_error": _relative(z0, z1),
            "relative_d_vol_error": _relative(d0, d1),
            "relative_p_vol_error": _power_contraction_relative_error(d0, d1),
            "relative_d_out_error": _relative(o0, o1),
            "relative_mutual_impedance_error": _relative(_off_diagonal(z0), _off_diagonal(z1)),
            "relative_source_path_length_error": path_error,
            "diagnostic_z_self_relative_error": _relative(_diagonal(z0), _diagonal(z1)),
            "diagnostic_z_self_resistive_relative_error": _relative(np.real(zdiag0), np.real(zdiag1)),
            "diagnostic_z_self_reactive_relative_error": _relative(np.imag(zdiag0), np.imag(zdiag1)),
            "diagnostic_d_vol_self_relative_error": _relative(_diagonal(d0), _diagonal(d1)),
            "diagnostic_d_vol_mutual_relative_error": _relative(_off_diagonal(d0), _off_diagonal(d1)),
            "diagnostic_raw_z_self_relative_error": _relative(_diagonal(z0_raw), _diagonal(z1_raw)),
            "diagnostic_raw_z_self_resistive_relative_error": _relative(np.real(raw_zdiag0), np.real(raw_zdiag1)),
            "diagnostic_raw_z_self_reactive_relative_error": _relative(np.imag(raw_zdiag0), np.imag(raw_zdiag1)),
            "base_z_self_real": np.real(zdiag0).tolist(),
            "base_z_self_imag": np.imag(zdiag0).tolist(),
            "refined_z_self_real": np.real(zdiag1).tolist(),
            "refined_z_self_imag": np.imag(zdiag1).tolist(),
            "base_d_vol_self": np.real(np.diag(d0)).tolist(),
            "refined_d_vol_self": np.real(np.diag(d1)).tolist(),
            "base_source_path_lengths": path0.tolist(),
            "refined_source_path_lengths": path1.tolist(),
            "base_self_correction": correction0,
            "refined_self_correction": correction1,
        }
        gated = (
            "relative_z_error",
            "relative_d_vol_error",
            "relative_p_vol_error",
            "relative_d_out_error",
            "relative_mutual_impedance_error",
        )
        row["maximum_relative_error"] = max(float(row[key]) for key in gated)
        rows.append(row)
        print(
            "pre-basis corrected EM mesh Gate……"
            f"{index + 1}/{len(geometries)}  max={row['maximum_relative_error']:.3e}  "
            f"self-Z={row['diagnostic_z_self_relative_error']:.3e}  "
            f"raw-self={row['diagnostic_raw_z_self_relative_error']:.3e}  "
            f"mutual-Z={row['relative_mutual_impedance_error']:.3e}  "
            f"path={row['relative_source_path_length_error']:.3e}",
            flush=True,
        )
    worst = max((row["maximum_relative_error"] for row in rows), default=0.0)
    worst_path = max((row["relative_source_path_length_error"] for row in rows), default=0.0)
    return {
        "sample_count": len(rows),
        "refinement_factor": factor,
        "base_fine_step": base_fine,
        "refined_fine_step": refined_fine,
        "relative_tolerance": tolerance,
        "source_path_relative_tolerance": path_tolerance,
        "maximum_relative_error": float(worst),
        "maximum_source_path_length_relative_error": float(worst_path),
        "source_geometry_invariant": bool(worst_path <= path_tolerance),
        "self_correction_model": "canonical_local_fine_minus_coarse_self_defect_v1",
        "converged": bool(worst <= tolerance and worst_path <= path_tolerance),
        "samples": rows,
    }


def _mesh_failure_diagnosis(mesh):
    if bool(mesh.get("converged", False)) or not mesh.get("samples"):
        return None
    row = max(mesh["samples"], key=lambda item: float(item.get("maximum_relative_error", 0.0)))
    path_error = float(row.get("relative_source_path_length_error", np.inf))
    if path_error > float(mesh.get("source_path_relative_tolerance", 1e-10)):
        return {
            "code": "mesh_refinement_changed_physical_source_geometry",
            "geometry": row.get("geometry"),
            "relative_source_path_length_error": path_error,
            "recommendation": "Mesh refinement must compare the same physical source geometry.",
        }
    self_z = float(row.get("diagnostic_z_self_relative_error", np.inf))
    mutual_z = float(row.get("relative_mutual_impedance_error", np.inf))
    return {
        "code": "local_self_correction_not_converged" if self_z > mutual_z else "general_em_mesh_nonconvergence",
        "geometry": row.get("geometry"),
        "corrected_self_z_relative_error": self_z,
        "raw_self_z_relative_error": float(row.get("diagnostic_raw_z_self_relative_error", np.inf)),
        "mutual_z_relative_error": mutual_z,
        "maximum_relative_error": float(row.get("maximum_relative_error", np.inf)),
        "recommendation": (
            "The local self defect is still insufficiently converged. Reduce the configured local self-correction fine step or enlarge its canonical local core; do not relax the global mesh Gate."
            if self_z > mutual_z
            else "Refine the remaining non-self EM truth discretization; do not relax the Gate."
        ),
    }


def run_truth_preflight(settings, background, geometries, monitor=None):
    geometries = list(geometries)
    if not geometries:
        raise ValueError("truth preflight needs at least one held-out geometry")
    cfg = settings["BACKGROUND"]
    domain_n = min(len(geometries), max(1, int(cfg.get("open_boundary_check", {}).get("samples", 1))))
    formulation_n = min(len(geometries), max(1, int(cfg.get("formulation_check", {}).get("samples", 1))))
    mesh_n = min(len(geometries), max(1, int(cfg.get("mesh_check", {}).get("samples", 1))))
    self_n = min(len(geometries), max(1, int(cfg.get("self_correction", {}).get("samples", 1))))
    domain = audit_open_boundary_domain(settings, background, geometries[:domain_n], monitor)
    formulation = audit_low_frequency_formulation(settings, background, geometries[:formulation_n], monitor)
    mesh = audit_em_mesh_preflight(settings, background, geometries[:mesh_n], monitor)
    local_self = audit_local_self_correction(background, geometries[:self_n], monitor)
    source_rows = [_source_and_loss_partition(background, g) for g in geometries]
    max_fraction = max(row["material_fraction_closure_error"] for row in source_rows)
    max_partition = max(row["wire_loss_partition_relative_error"] for row in source_rows)
    max_terminal = max(row["maximum_terminal_path_integral_relative_error"] for row in source_rows)
    checks = {
        "finite_support_source_ok": all(row["finite_support_source"] for row in source_rows),
        "terminal_source_continuity_ok": all(row["terminal_path_conservation"] for row in source_rows),
        "material_fraction_closure_ok": max_fraction <= 1e-10,
        "wire_loss_not_double_counted": max_partition <= 1e-12,
        "open_boundary_domain_converged": bool(domain["converged"]),
        "low_frequency_formulation_converged": bool(formulation["converged"]),
        "em_mesh_converged": bool(mesh["converged"]),
        "local_self_correction_converged": bool(local_self["converged"]),
    }
    certified = all(bool(value) for value in checks.values())
    diagnosis = _mesh_failure_diagnosis(mesh)
    if not local_self["converged"]:
        joule_error = float(local_self.get("maximum_joule_total_power_relative_error", np.inf))
        joule_limit = float(local_self.get("joule_identity_tolerance", 1e-10))
        if joule_error > joule_limit:
            diagnosis = {
                "code": "local_self_joule_identity_failed",
                "maximum_joule_total_power_relative_error": joule_error,
                "joule_identity_tolerance": joule_limit,
                "recommendation": "This is a local defect implementation/scaling failure, not a mesh-convergence failure. Repair the D_vol/q_cell identity; do not relax tolerance or globally refine the mesh.",
            }
        else:
            diagnosis = {
                "code": "local_self_reference_not_converged",
                "maximum_relative_error": float(local_self["maximum_relative_error"]),
                "fine_step": float(local_self["fine_step"]),
                "validation_fine_step": float(local_self["validation_fine_step"]),
                "recommendation": "Refine only the canonical local self problem until its independent fine-grid audit meets tolerance; do not globally refine the UWPT domain.",
            }
    return {
        **{key: bool(value) for key, value in checks.items()},
        "source_model": getattr(background, "source_model", "unknown"),
        "source_centerline_step": float(getattr(background, "source_centerline_step", np.nan)),
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
        "certified": bool(certified),
        "status": "certified" if certified else "truth_preflight_failed",
    }


__all__ = ["audit_em_mesh_preflight", "run_truth_preflight"]
