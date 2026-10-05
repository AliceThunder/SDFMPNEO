"""Strict spatial/post-basis Physics Gate utilities for the production tensor-ROM."""
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from .unified_background import EPS0, MU0
from .unified_geometry import UnifiedUWPTGeometry
from .unified_open_boundary import OpenBoundaryBackground
from .unified_tensor_surrogate import solve_truth_tensors


def _relative(value, reference):
    a = np.asarray(value)
    b = np.asarray(reference)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), np.finfo(float).tiny))


def _off_diagonal(matrix):
    a = np.asarray(matrix)
    return a - np.diag(np.diag(a))


def _audit_current_vectors(n_ports):
    """Hermitian current-space basis used for scalar quadratic-form audits."""
    n = int(n_ports)
    eye = np.eye(n, dtype=complex)
    vectors = [eye[:, p] for p in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            vectors.append(eye[:, i] + eye[:, j])
            vectors.append(eye[:, i] + 1j * eye[:, j])
    if len(vectors) != n * n:
        raise AssertionError("Hermitian current audit span dimension mismatch")
    return vectors


def _power_contraction_relative_error(value, reference):
    """Worst scalar 1/2 c^H D c error over a complete Hermitian current span."""
    a = np.asarray(value, complex)
    b = np.asarray(reference, complex)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("power-contraction matrices must be square and shape-compatible")
    natural = max(float(np.linalg.norm(b)), np.finfo(float).tiny)
    worst = 0.0
    for current in _audit_current_vectors(a.shape[0]):
        c2 = float(np.linalg.norm(current) ** 2)
        pa = 0.5 * float(np.real(current.conj() @ a @ current))
        pb = 0.5 * float(np.real(current.conj() @ b @ current))
        scale = max(abs(pb), 1e-12 * natural * c2, np.finfo(float).tiny)
        worst = max(worst, abs(pa - pb) / scale)
    return float(worst)


def _passive_sqrt(value):
    root = complex(np.sqrt(value))
    if root.real < 0.0 or (abs(root.real) <= np.finfo(float).eps and root.imag > 0.0):
        root = -root
    return root


def _mqs_boundary_admittance(background):
    material = background.materials[background.seawater_material]
    sigma = float(background._temperature_material(background.seawater_material, background.ambient_temperature))
    mu = MU0 * float(material.get("relative_permeability", 1.0))
    if sigma <= 0.0 or mu <= 0.0:
        raise ValueError("MQS comparison requires positive seawater conductivity/permeability")
    return _passive_sqrt((-1j * sigma / background.omega) / mu)


def _operator(background, context, *, mqs=False):
    if not mqs:
        return background.em_operator(context, None)
    sigma, _, mu_inv, _, _, _ = background.cell_properties(context, None, em=True)
    h2 = np.asarray(background.face_cell_hodge @ mu_inv).reshape(-1)
    hs = np.asarray(background.edge_cell_hodge @ sigma).reshape(-1)
    boundary = 1j * background.omega * _mqs_boundary_admittance(background) * sp.diags(
        background.boundary_edge_hodge
    )
    return (
        background.curl.T @ sp.diags(h2) @ background.curl
        + 1j * background.omega * sp.diags(hs)
        + boundary
    ).tocsr()


def _solve_fields(background, geometry, *, mqs=False):
    context = background.geometry_context(geometry, assemble_thermal=False)
    A = _operator(background, context, mqs=mqs)
    B = background.rhs_matrix(context)
    try:
        lu = spla.splu(A.tocsc())
        X = np.column_stack([lu.solve(B[:, p]) for p in range(B.shape[1])])
    except RuntimeError:
        X = np.column_stack([spla.spsolve(A, B[:, p]) for p in range(B.shape[1])])
    if np.any(~np.isfinite(X)):
        raise FloatingPointError("Physics Gate Maxwell solve produced non-finite fields")
    source = np.asarray(context.source_shape, float)
    reaction = -source.T @ X
    z = 0.5 * (reaction + reaction.T)
    sigma = np.asarray(background.cell_properties(context, None, em=True)[0], float)
    edge_loss = np.asarray(background.edge_cell_hodge @ sigma).reshape(-1)
    raw_d = X.conj().T @ (edge_loss[:, None] * X)
    d = 0.5 * (raw_d + raw_d.conj().T)
    admittance = _mqs_boundary_admittance(background) if mqs else background.boundary_admittance()
    outward = float(admittance.real) * np.asarray(background.boundary_edge_hodge, float)
    raw_out = X.conj().T @ (outward[:, None] * X)
    d_out = 0.5 * (raw_out + raw_out.conj().T)
    return context, X, sigma, z, d, d_out


def _background_from_settings(settings, *, bounds=None, fine_step=None, max_step=None):
    regions = settings["REGIONS"]
    cfg = copy.deepcopy(dict(settings["BACKGROUND"]))
    if bounds is not None:
        cfg["bounds"] = np.asarray(bounds, float).tolist()
    if fine_step is not None:
        cfg["fine_step"] = float(fine_step)
    if max_step is not None:
        cfg["max_step"] = float(max_step)
    return OpenBoundaryBackground.from_config(
        cfg,
        frequency_hz=settings["PHYSICS"]["frequency_hz"],
        materials=settings["MATERIALS"],
        coil_materials=regions["coil_materials"],
        package_materials=regions["package_materials"],
        seawater_material=regions["seawater_material"],
        ambient_temperature=settings["PHYSICS"]["ambient_temperature"],
    )


def audit_open_boundary_domain(settings, background, geometries, monitor=None):
    cfg = dict(settings["BACKGROUND"].get("open_boundary_check", {}))
    tolerance = float(cfg.get("relative_tolerance", 5e-2))
    padding = np.asarray(cfg.get("padding", 0.12), float)
    if padding.ndim == 0:
        padding = np.full(3, float(padding))
    if padding.shape != (3,) or np.any(padding <= 0.0):
        raise ValueError("open_boundary_check.padding must be positive scalar/length-3")
    bounds = np.asarray(settings["BACKGROUND"]["bounds"], float)
    expanded = bounds.copy()
    expanded[:, 0] -= padding
    expanded[:, 1] += padding
    reference = _background_from_settings(settings, bounds=expanded)
    rows = []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        _, _, _, z0, d0, o0 = _solve_fields(background, geometry)
        _, _, _, z1, d1, o1 = _solve_fields(reference, geometry)
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
            f"开放边界域扩展 Gate……{index+1}/{len(geometries)}  max={row['maximum_relative_error']:.3e}",
            flush=True,
        )
    worst = max((row["maximum_relative_error"] for row in rows), default=0.0)
    return {
        "sample_count": len(rows),
        "padding": padding.tolist(),
        "relative_tolerance": tolerance,
        "maximum_relative_error": float(worst),
        "converged": bool(worst <= tolerance),
        "samples": rows,
    }


def audit_low_frequency_formulation(settings, background, geometries, monitor=None):
    cfg = dict(settings["BACKGROUND"].get("formulation_check", {}))
    tolerance = float(cfg.get("relative_tolerance", 2e-2))
    material = background.materials[background.seawater_material]
    sigma = float(material.get("electrical_conductivity", 0.0))
    epsilon = EPS0 * float(material.get("relative_permittivity", 1.0))
    displacement_ratio = float(background.omega * epsilon / max(sigma, np.finfo(float).tiny))
    rows = []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        _, _, _, zf, df, of = _solve_fields(background, geometry, mqs=False)
        _, _, _, zm, dm, om = _solve_fields(background, geometry, mqs=True)
        row = {
            "index": int(index),
            "relative_z_error": _relative(zf, zm),
            "relative_d_vol_error": _relative(df, dm),
            "relative_d_out_error": _relative(of, om),
            "relative_mutual_impedance_error": _relative(_off_diagonal(zf), _off_diagonal(zm)),
        }
        row["maximum_relative_error"] = max(v for k, v in row.items() if k.startswith("relative_"))
        rows.append(row)
        print(
            f"full-wave ↔ MQS formulation Gate……{index+1}/{len(geometries)}  max={row['maximum_relative_error']:.3e}",
            flush=True,
        )
    worst = max((row["maximum_relative_error"] for row in rows), default=0.0)
    return {
        "sample_count": len(rows),
        "relative_tolerance": tolerance,
        "seawater_displacement_to_conduction_ratio": displacement_ratio,
        "maximum_relative_error": float(worst),
        "converged": bool(worst <= tolerance),
        "samples": rows,
    }


def _cell_heat(background, sigma, field):
    edge_energy = np.abs(np.asarray(field, complex).reshape(-1)) ** 2
    cell_edge_energy = np.asarray(background.edge_cell_hodge.T @ edge_energy).reshape(-1)
    return np.asarray(0.5 * np.asarray(sigma, float) * cell_edge_energy, float)


def _interpolate_cell_field(source_background, target_points, values):
    field = np.asarray(values, float).reshape(
        source_background.nx, source_background.ny, source_background.nz
    )
    interpolation = RegularGridInterpolator(
        source_background.cell_axes,
        field,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    result = np.asarray(interpolation(np.asarray(target_points, float)), float).reshape(-1)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("mesh Gate interpolation produced non-finite values")
    return result


def _modal_tensors(background, X, sigma, phi):
    out = []
    for j in range(phi.shape[1]):
        weighted = np.asarray(background.edge_cell_hodge @ (sigma * phi[:, j])).reshape(-1)
        h = X.conj().T @ (weighted[:, None] * X)
        out.append(0.5 * (h + h.conj().T))
    return np.asarray(out, complex)


def _steady_thermal_outputs(background, context, X, sigma, phi=None):
    q = _cell_heat(background, sigma, X[:, 0])
    M, K = background.thermal_operator_full(context.fractions)
    theta = np.asarray(spla.spsolve(K.tocsc(), q), float).reshape(-1)
    wire = np.asarray(
        [float(np.dot(np.asarray(w, float), theta)) for w in context.line_heat_weights],
        float,
    )
    result = {
        "theta": theta,
        "maximum_temperature_rise": float(np.max(theta)),
        "wire_temperature_rise": wire,
        "mass": M,
    }
    if phi is not None:
        Mr = phi.T @ (M @ phi)
        result["reduced_coordinate"] = np.linalg.solve(Mr, phi.T @ (M @ theta))
    return result


def audit_mesh_convergence(settings, background, geometries, monitor=None):
    """Post-basis refinement: H_j, scalar P_vol and thermal outputs."""
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
        context0, X0, sigma0, z0, d0, o0 = _solve_fields(background, geometry)
        context1, X1, sigma1, z1, d1, o1 = _solve_fields(refined, geometry)
        phi0 = background.thermal_library.basis_for_geometry(background, geometry)
        phi1 = np.column_stack(
            [
                _interpolate_cell_field(background, refined.cell_centers, phi0[:, j])
                for j in range(phi0.shape[1])
            ]
        )
        h0 = _modal_tensors(background, X0, sigma0, phi0)
        h1 = _modal_tensors(refined, X1, sigma1, phi1)
        t0 = _steady_thermal_outputs(background, context0, X0, sigma0, phi0)
        t1 = _steady_thermal_outputs(refined, context1, X1, sigma1)
        theta1_on_base = _interpolate_cell_field(refined, background.cell_centers, t1["theta"])
        M0 = t0["mass"]
        Mr0 = phi0.T @ (M0 @ phi0)
        a1_on_base = np.linalg.solve(Mr0, phi0.T @ (M0 @ theta1_on_base))
        row = {
            "index": int(index),
            "relative_z_error": _relative(z0, z1),
            "relative_d_vol_error": _relative(d0, d1),
            "relative_p_vol_error": _power_contraction_relative_error(d0, d1),
            "relative_d_out_error": _relative(o0, o1),
            "relative_mutual_impedance_error": _relative(_off_diagonal(z0), _off_diagonal(z1)),
            "relative_modal_h_error": _relative(h0, h1),
            "relative_tmax_error": abs(
                t0["maximum_temperature_rise"] - t1["maximum_temperature_rise"]
            )
            / max(abs(t1["maximum_temperature_rise"]), np.finfo(float).tiny),
            "relative_wire_temperature_error": _relative(
                t0["wire_temperature_rise"], t1["wire_temperature_rise"]
            ),
            "relative_steady_coordinate_error": _relative(
                t0["reduced_coordinate"], a1_on_base
            ),
        }
        row["maximum_relative_error"] = max(
            v for k, v in row.items() if k.startswith("relative_")
        )
        rows.append(row)
        print(
            f"空间 mesh refinement Gate……{index+1}/{len(geometries)}  max={row['maximum_relative_error']:.3e}",
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


def _perturb_geometry(geometry, *, translation=None, yaw=None):
    g = (
        geometry
        if isinstance(geometry, UnifiedUWPTGeometry)
        else UnifiedUWPTGeometry.from_mapping(geometry)
    )
    mapping = g.to_mapping()
    port = min(1, len(mapping["coils"]) - 1)
    if translation is not None:
        shift = np.asarray(translation, float)
        for section in ("coils", "packages"):
            value = np.asarray(mapping[section][port]["translation"], float) + shift
            mapping[section][port]["translation"] = value.tolist()
    if yaw is not None:
        for section in ("coils", "packages"):
            value = np.asarray(mapping[section][port]["angles"], float)
            value[2] += float(yaw)
            mapping[section][port]["angles"] = value.tolist()
    return mapping


def _fraction_vector(context):
    return np.concatenate(
        [np.asarray(context.fractions[name], float) for name in sorted(context.fractions)]
    )


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
            "translation": _perturb_geometry(
                base, translation=[translation_step, 0.0, 0.0]
            ),
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
                "source_relative_change": _relative(
                    context1.source_shape, context0.source_shape
                ),
                "material_fraction_relative_change": _relative(
                    _fraction_vector(context1), _fraction_vector(context0)
                ),
                "basis_relative_change": _relative(
                    context1.thermal_basis, context0.thermal_basis
                ),
                "z_relative_change": _relative(z1, z0),
                "d_vol_relative_change": _relative(d1, d0),
                "modal_h_relative_change": _relative(h1, h0),
            }
            row["maximum_relative_change"] = max(
                v for k, v in row.items() if k.endswith("relative_change")
            )
            rows.append(row)
            print(
                f"geometry/transport continuity Gate……{index+1}/{len(geometries)} {kind} max={row['maximum_relative_change']:.3e}",
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
    """Run only checks that require the frozen geometry-aware thermal basis/tensors.

    Domain, formulation, source, terminal, loss-partition and EM-only mesh checks are
    intentionally reused from the already-passed pre-basis preflight rather than
    re-solving the same expensive Maxwell problems a second time.
    """
    geometries = list(geometries)
    if not geometries:
        raise ValueError("Physics Gate needs at least one held-out geometry")
    if not bool(preflight.get("certified", False)):
        raise RuntimeError("post-basis Physics Gate requires a certified truth preflight")
    cfg = settings["BACKGROUND"]
    mesh_n = min(
        len(geometries), max(1, int(cfg.get("mesh_check", {}).get("samples", 1)))
    )
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
        "outward_power_form_available": bool(
            audit["independent_outward_power_available"] >= 0.5
        ),
        "open_boundary_domain_converged": bool(
            preflight["open_boundary_domain_converged"]
        ),
        "low_frequency_formulation_converged": bool(
            preflight["low_frequency_formulation_converged"]
        ),
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
        "terminal_model": getattr(background, "terminal_model", "unknown"),
        "boundary_model": getattr(background, "boundary_model", "unknown"),
        "truth_preflight": preflight,
        "mesh_convergence": mesh,
        "geometry_continuity": continuity,
        "certified": bool(certified),
        "status": "certified" if certified else "physics_gate_failed",
        "audit": audit,
    }


__all__ = [
    "audit_geometry_continuity",
    "audit_low_frequency_formulation",
    "audit_mesh_convergence",
    "audit_open_boundary_domain",
    "run_physics_gate",
]