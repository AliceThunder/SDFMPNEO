"""Pre-basis spatial truth checks required before canonical thermal construction."""
from __future__ import annotations

import numpy as np

from .unified_physics_gate import (
    _background_from_settings,
    _off_diagonal,
    _relative,
    _solve_fields,
    audit_low_frequency_formulation,
    audit_open_boundary_domain,
)


def audit_em_mesh_preflight(settings, background, geometries, monitor=None):
    cfg = dict(settings["BACKGROUND"].get("mesh_check", {}))
    tolerance = float(cfg.get("relative_tolerance", 1e-1))
    factor = float(cfg.get("refinement_factor", 0.75))
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
        _, _, _, z0, d0, o0 = _solve_fields(background, geometry)
        _, _, _, z1, d1, o1 = _solve_fields(refined, geometry)
        row = {
            "index": int(index),
            "relative_z_error": _relative(z0, z1),
            "relative_d_vol_error": _relative(d0, d1),
            "relative_d_out_error": _relative(o0, o1),
            "relative_mutual_impedance_error": _relative(_off_diagonal(z0), _off_diagonal(z1)),
        }
        row["maximum_relative_error"] = max(v for k, v in row.items() if k.startswith("relative_"))
        rows.append(row)
        print(
            "pre-basis EM mesh Gate……"
            f"{index + 1}/{len(geometries)}  max={row['maximum_relative_error']:.3e}",
            flush=True,
        )
    worst = max((row["maximum_relative_error"] for row in rows), default=0.0)
    return {
        "sample_count": len(rows),
        "refinement_factor": factor,
        "base_fine_step": base_fine,
        "refined_fine_step": refined_fine,
        "relative_tolerance": tolerance,
        "maximum_relative_error": float(worst),
        "converged": bool(worst <= tolerance),
        "samples": rows,
    }


def _source_and_loss_partition(background, geometry):
    context = background.geometry_context(geometry, assemble_thermal=False)
    rows = tuple(getattr(context, "source_regularization", ()))
    matching = len(rows) == context.source_shape.shape[1]
    source_ok = bool(
        matching
        and all(
            row.get("model") == getattr(background, "source_model", None)
            and float(row.get("conductor_width", 0.0)) > 0.0
            and float(row.get("conductor_thickness", 0.0)) > 0.0
            and abs(float(row.get("heat_weight_sum", 0.0)) - 1.0) <= 1e-12
            and float(row.get("source_norm", 0.0)) > 0.0
            for row in rows
        )
    )
    terminal_ok = bool(
        matching
        and all(
            row.get("terminal_model") == getattr(background, "terminal_model", None)
            and float(row.get("terminal_separation", 0.0)) > np.finfo(float).tiny
            and float(row.get("terminal_path_integral_relative_error", np.inf)) <= 1e-12
            for row in rows
        )
    )
    sigma = np.asarray(background.cell_properties(context, None, em=True)[0], float)
    expected = np.zeros(background.n_cells, float)
    for name, fraction in context.fractions.items():
        if name in background.coil_materials:
            continue
        local = float(
            background._temperature_material(name, np.asarray(background.ambient_temperature))
        )
        expected += np.asarray(fraction, float) * local
    partition_error = float(
        np.linalg.norm(sigma - expected)
        / max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    )
    return {
        "finite_support_source": source_ok,
        "terminal_path_conservation": terminal_ok,
        "maximum_terminal_path_integral_relative_error": max(
            (float(row.get("terminal_path_integral_relative_error", np.inf)) for row in rows),
            default=float("inf"),
        ),
        "material_fraction_closure_error": float(
            getattr(context, "material_fraction_closure_error", np.inf)
        ),
        "wire_loss_partition_relative_error": partition_error,
    }


def run_truth_preflight(settings, background, geometries, monitor=None):
    geometries = list(geometries)
    if not geometries:
        raise ValueError("truth preflight needs at least one held-out geometry")
    cfg = settings["BACKGROUND"]
    domain_n = min(len(geometries), max(1, int(cfg.get("open_boundary_check", {}).get("samples", 1))))
    formulation_n = min(len(geometries), max(1, int(cfg.get("formulation_check", {}).get("samples", 1))))
    mesh_n = min(len(geometries), max(1, int(cfg.get("mesh_check", {}).get("samples", 1))))
    domain = audit_open_boundary_domain(settings, background, geometries[:domain_n], monitor)
    formulation = audit_low_frequency_formulation(settings, background, geometries[:formulation_n], monitor)
    mesh = audit_em_mesh_preflight(settings, background, geometries[:mesh_n], monitor)
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
    }
    certified = all(bool(value) for value in checks.values())
    return {
        **{key: bool(value) for key, value in checks.items()},
        "source_model": getattr(background, "source_model", "unknown"),
        "terminal_model": getattr(background, "terminal_model", "unknown"),
        "maximum_terminal_path_integral_relative_error": float(max_terminal),
        "maximum_material_fraction_closure_error": float(max_fraction),
        "maximum_wire_loss_partition_relative_error": float(max_partition),
        "open_boundary_convergence": domain,
        "formulation_convergence": formulation,
        "em_mesh_convergence": mesh,
        "source_samples": source_rows,
        "certified": bool(certified),
        "status": "certified" if certified else "truth_preflight_failed",
    }


__all__ = ["audit_em_mesh_preflight", "run_truth_preflight"]