"""Post-basis Physics Gate for the local-self-corrected production truth."""
from __future__ import annotations

import copy
import numpy as np

from .unified_corrected_truth import solve_truth_tensors
from .unified_physics_gate import (
    _background_from_settings as _raw_background_from_settings,
    _fraction_vector,
    _interpolate_cell_field,
    _modal_tensors,
    _off_diagonal,
    _perturb_geometry,
    _power_contraction_relative_error,
    _relative,
    _solve_fields,
)
from .unified_self_correction import apply_local_self_correction


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


def _corrected_fields(background, geometry, phi=None):
    context, X, sigma, z, d, d_out = _solve_fields(background, geometry)
    modal = None
    if phi is not None:
        modal = _modal_tensors(background, X, sigma, phi)
    corrected = apply_local_self_correction(
        background,
        geometry,
        z,
        d,
        d_out,
        phi=phi,
        modal_h=modal,
    )
    return context, corrected.z, corrected.d_vol, corrected.d_out, corrected.modal_h, corrected.audit


def _reduced_steady_outputs(background, context, phi, modal_h):
    phi = np.asarray(phi, float)
    modal_h = np.asarray(modal_h, complex)
    M, K = background.thermal_operator_full(context.fractions)
    Mr = phi.T @ (M @ phi)
    Kr = phi.T @ (K @ phi)
    q = 0.5 * np.real(modal_h[:, 0, 0])
    try:
        a = np.linalg.solve(Kr, q)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(Kr, q, rcond=None)[0]
    theta = phi @ a
    wire = np.asarray(
        [float(np.dot(np.asarray(w, float), theta)) for w in context.line_heat_weights],
        float,
    )
    return {
        "theta": np.asarray(theta, float),
        "maximum_temperature_rise": float(np.max(theta)),
        "wire_temperature_rise": wire,
        "mass": M,
        "reduced_coordinate": np.asarray(a, float),
        "reduced_mass": Mr,
        "reduced_stiffness": Kr,
    }


def audit_mesh_convergence(settings, background, geometries, monitor=None):
    """Refinement of corrected Z/D/H and their reduced thermal response."""
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
        context0 = background.geometry_context(geometry, assemble_thermal=False)
        phi0 = background.thermal_library.basis_for_geometry(background, geometry)
        phi1 = np.column_stack(
            [
                _interpolate_cell_field(background, refined.cell_centers, phi0[:, j])
                for j in range(phi0.shape[1])
            ]
        )
        context0, z0, d0, o0, h0, correction0 = _corrected_fields(background, geometry, phi0)
        context1, z1, d1, o1, h1, correction1 = _corrected_fields(refined, geometry, phi1)
        t0 = _reduced_steady_outputs(background, context0, phi0, h0)
        t1 = _reduced_steady_outputs(refined, context1, phi1, h1)
        theta1_on_base = _interpolate_cell_field(refined, background.cell_centers, t1["theta"])
        M0 = t0["mass"]
        Mr0 = t0["reduced_mass"]
        a1_on_base = np.linalg.solve(Mr0, phi0.T @ (M0 @ theta1_on_base))
        row = {
            "index": int(index),
            "geometry": geometry,
            "relative_z_error": _relative(z0, z1),
            "relative_d_vol_error": _relative(d0, d1),
            "relative_p_vol_error": _power_contraction_relative_error(d0, d1),
            "relative_d_out_error": _relative(o0, o1),
            "relative_mutual_impedance_error": _relative(_off_diagonal(z0), _off_diagonal(z1)),
            "relative_modal_h_error": _relative(h0, h1),
            "relative_tmax_error": abs(
                t0["maximum_temperature_rise"] - t1["maximum_temperature_rise"]
            ) / max(abs(t1["maximum_temperature_rise"]), np.finfo(float).tiny),
            "relative_wire_temperature_error": _relative(
                t0["wire_temperature_rise"], t1["wire_temperature_rise"]
            ),
            "relative_steady_coordinate_error": _relative(
                t0["reduced_coordinate"], a1_on_base
            ),
            "base_self_correction": correction0,
            "refined_self_correction": correction1,
        }
        row["maximum_relative_error"] = max(
            v for k, v in row.items() if k.startswith("relative_")
        )
        rows.append(row)
        print(
            f"corrected spatial mesh refinement Gate……{index+1}/{len(geometries)}  "
            f"max={row['maximum_relative_error']:.3e}  H={row['relative_modal_h_error']:.3e}",
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
        "self_correction_model": "canonical_local_fine_minus_coarse_self_defect_v1",
        "converged": bool(worst <= tolerance),
        "samples": rows,
    }


def audit_geometry_continuity(settings, background, geometries, monitor=None):
    cfg = dict(settings["BACKGROUND"].get("geometry_continuity_check", {}))
    translation_step = float(cfg.get("translation_step", 1e-4))
    angle_step = float(cfg.get("angle_step", 1e-3))
    limit = float(cfg.get("relative_change_limit", 2e-1))
    rows = []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        base = background.validate_geometry(geometry)
        variants = {
            "translation": _perturb_geometry(base, translation=[translation_step, 0.0, 0.0]),
            "rotation": _perturb_geometry(base, yaw=angle_step),
        }
        context0 = background.geometry_context(base, assemble_thermal=True)
        z0, d0, h0, _, _, _ = solve_truth_tensors(background, base)
        for kind, candidate in variants.items():
            try:
                candidate = background.validate_geometry(candidate)
            except ValueError:
                continue
            context1 = background.geometry_context(candidate, assemble_thermal=True)
            z1, d1, h1, _, _, _ = solve_truth_tensors(background, candidate)
            row = {
                "index": int(index),
                "perturbation": kind,
                "source_relative_change": _relative(context1.source_shape, context0.source_shape),
                "material_fraction_relative_change": _relative(
                    _fraction_vector(context1), _fraction_vector(context0)
                ),
                "basis_relative_change": _relative(context1.thermal_basis, context0.thermal_basis),
                "z_relative_change": _relative(z1, z0),
                "d_vol_relative_change": _relative(d1, d0),
                "modal_h_relative_change": _relative(h1, h0),
            }
            row["maximum_relative_change"] = max(
                v for k, v in row.items() if k.endswith("relative_change")
            )
            rows.append(row)
            print(
                f"corrected geometry/transport continuity Gate……{index+1}/{len(geometries)} "
                f"{kind} max={row['maximum_relative_change']:.3e}",
                flush=True,
            )
    worst = max((row["maximum_relative_change"] for row in rows), default=float("inf"))
    return {
        "sample_count": len(rows),
        "translation_step": translation_step,
        "angle_step": angle_step,
        "relative_change_limit": limit,
        "maximum_relative_change": float(worst),
        "converged": bool(rows and worst <= limit),
        "samples": rows,
    }


def run_physics_gate(settings, background, dataset, geometries, *, preflight, monitor=None):
    geometries = list(geometries)
    if not geometries:
        raise ValueError("Physics Gate needs at least one held-out geometry")
    if not bool(preflight.get("certified", False)):
        raise RuntimeError("post-basis Physics Gate requires a certified truth preflight")
    cfg = settings["BACKGROUND"]
    mesh_n = min(len(geometries), max(1, int(cfg.get("mesh_check", {}).get("samples", 1))))
    continuity_n = min(
        len(geometries),
        max(1, int(cfg.get("geometry_continuity_check", {}).get("samples", 1))),
    )
    mesh = audit_mesh_convergence(settings, background, geometries[:mesh_n], monitor)
    continuity = audit_geometry_continuity(
        settings, background, geometries[:continuity_n], monitor
    )
    audit = dict(dataset.audit)
    checks = {
        "linear_solve_ok": audit["maximum_linear_relative_residual"] <= 1e-8,
        "reciprocity_ok": audit["maximum_reciprocity_relative_error"] <= 1e-8,
        "volume_passivity_ok": audit["minimum_d_vol_eigenvalue"] >= -1e-9,
        "physical_outward_passivity_ok": audit["minimum_physical_outward_eigenvalue"] >= -1e-9,
        "implied_outward_passivity_ok": audit["minimum_implied_outward_eigenvalue"] >= -1e-9,
        "independent_poynting_balance_ok": audit["maximum_open_boundary_power_balance_relative_error"] <= 1e-7,
        "modal_loewner_ok": audit["maximum_relative_loewner_violation"] <= 1e-8,
        "joule_total_power_identity_ok": audit["maximum_joule_total_power_relative_error"] <= 1e-10,
        "joule_modal_identity_ok": audit["maximum_joule_modal_contraction_relative_error"] <= 1e-10,
        "local_self_joule_total_identity_ok": audit.get(
            "maximum_local_self_joule_total_power_relative_error", np.inf
        ) <= 1e-10,
        "local_self_joule_modal_identity_ok": audit.get(
            "maximum_local_self_joule_modal_contraction_relative_error", np.inf
        ) <= 1e-10,
        "local_self_correction_available": audit.get("local_self_correction_available", 0.0) >= 0.5,
        "local_self_correction_power_balance_ok": audit.get(
            "maximum_local_self_correction_power_balance_relative_error", np.inf
        ) <= 1e-7,
        "material_fraction_closure_ok": bool(
            preflight["material_fraction_closure_ok"]
            and audit["maximum_material_fraction_closure_error"] <= 1e-10
        ),
        "finite_support_source_ok": bool(
            preflight["finite_support_source_ok"]
            and audit["source_regularization_available"] >= 0.5
        ),
        "terminal_source_continuity_ok": bool(preflight["terminal_source_continuity_ok"]),
        "wire_loss_not_double_counted": bool(preflight["wire_loss_not_double_counted"]),
        "outward_power_form_available": bool(audit["independent_outward_power_available"] >= 0.5),
        "open_boundary_domain_converged": bool(preflight["open_boundary_domain_converged"]),
        "low_frequency_formulation_converged": bool(preflight["low_frequency_formulation_converged"]),
        "em_mesh_converged": bool(preflight["em_mesh_converged"]),
        "full_mesh_converged": bool(mesh["converged"]),
        "geometry_transport_continuous": bool(continuity["converged"]),
    }
    certified = all(bool(v) for v in checks.values())
    return {
        **{k: bool(v) for k, v in checks.items()},
        "reaction_impedance_convention": "negative_source_reaction",
        "phasor_convention": "peak_exp_plus_iwt",
        "source_model": getattr(background, "source_model", "unknown"),
        "self_correction_model": "canonical_local_fine_minus_coarse_self_defect_v1",
        "terminal_model": getattr(background, "terminal_model", "unknown"),
        "boundary_model": getattr(background, "boundary_model", "unknown"),
        "truth_preflight": preflight,
        "mesh_convergence": mesh,
        "geometry_continuity": continuity,
        "certified": bool(certified),
        "status": "certified" if certified else "physics_gate_failed",
        "audit": audit,
    }


__all__ = ["audit_geometry_continuity", "audit_mesh_convergence", "run_physics_gate"]
