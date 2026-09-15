"""Global scalar-gradient self reference for coarse production Maxwell truth.

The canonical local self correction deliberately excludes the pure longitudinal
terminal response because that response depends on the full return path,
dielectric environment and open boundary.  At the same time, production logs
show that this global longitudinal *self* term is the remaining mesh-sensitive
piece after the transverse/cross local defect, mutual coupling, open-boundary
and MQS Gates have converged.

This module therefore supplies the missing global multiscale half without a
second refined Maxwell solve.  For each port it solves only the compatible
scalar block

    (G^T D G) phi = G^T b,      E_L = G phi,

on a fixed global reference grid.  The production diagonal receives

    Q_corrected = Q_current + Q_L(reference) - Q_L(current),

for Z, D_vol, D_out and (when present) thermal modal Joule tensors.  Mutual
entries remain entirely from the full Maxwell solve.  A separate scalar-only
reference Gate compares the reference grid with a still finer global scalar
grid, so using a common reference cannot hide nonconvergence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .unified_compensated_field import field_abs2, field_linear_dot
from .unified_gradient_block_maxwell import build_gradient_block
from .unified_refined_gradient_projection import refined_gradient_projection


_MODEL = "global_compatible_longitudinal_self_reference_v1"


def _hermitian(value):
    a = np.asarray(value, complex)
    return 0.5 * (a + a.conj().T)


def _relative(value, reference):
    a = np.asarray(value)
    b = np.asarray(reference)
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    )


def _background_step(background):
    cfg = getattr(background, "background_config", None)
    if isinstance(cfg, dict) and "fine_step" in cfg:
        return float(cfg["fine_step"])
    return float(min(np.min(background.dx), np.min(background.dy), np.min(background.dz)))


def _config(background):
    root = copy.deepcopy(dict(getattr(background, "background_config", {}) or {}))
    mesh = dict(root.get("mesh_check", {}) or {})
    own = dict(root.get("global_longitudinal_correction", {}) or {})
    production_step = float(
        root.get("_production_fine_step", root.get("fine_step", _background_step(background)))
    )
    mesh_factor = float(mesh.get("refinement_factor", 0.75))
    reference_step = float(
        own.get(
            "reference_fine_step",
            mesh.get("longitudinal_reference_fine_step", mesh_factor * production_step),
        )
    )
    validation_factor = float(
        own.get(
            "validation_factor",
            mesh.get("longitudinal_reference_validation_factor", mesh_factor),
        )
    )
    tolerance = float(
        own.get("relative_tolerance", mesh.get("relative_tolerance", 1e-1))
    )
    enabled = bool(own.get("enabled", True))
    if not (reference_step > 0.0):
        raise ValueError("global longitudinal reference step must be positive")
    if not (0.0 < validation_factor < 1.0):
        raise ValueError("global longitudinal validation factor must lie in (0,1)")
    return {
        "enabled": enabled,
        "production_fine_step": production_step,
        "reference_fine_step": reference_step,
        "validation_factor": validation_factor,
        "validation_fine_step": reference_step * validation_factor,
        "relative_tolerance": tolerance,
    }


def _geometry_key(geometry):
    if hasattr(geometry, "canonical_json"):
        return geometry.canonical_json()
    try:
        from .unified_geometry import UnifiedUWPTGeometry

        return UnifiedUWPTGeometry.from_mapping(geometry).canonical_json()
    except Exception:
        return repr(geometry)


def _make_background(parent, fine_step):
    step = float(fine_step)
    current = _background_step(parent)
    if abs(step - current) <= 1e-14 * max(abs(step), abs(current), 1.0):
        return parent
    cache = getattr(parent, "_sdfmpneo_global_longitudinal_background_cache", None)
    if cache is None:
        cache = {}
        parent._sdfmpneo_global_longitudinal_background_cache = cache
    key = round(step, 15)
    if key in cache:
        return cache[key]

    cfg = copy.deepcopy(dict(getattr(parent, "background_config", {}) or {}))
    if not cfg:
        raise ValueError("global longitudinal correction requires background_config")
    ratio = step / max(current, np.finfo(float).tiny)
    cfg["fine_step"] = step
    if "max_step" in cfg:
        cfg["max_step"] = max(step, ratio * float(cfg["max_step"]))
    cfg["_production_fine_step"] = float(
        getattr(parent, "background_config", {}).get("_production_fine_step", current)
    )
    cfg["self_correction"] = {"enabled": False}
    background = parent.__class__.from_config(
        cfg,
        frequency_hz=parent.frequency_hz,
        materials=parent.materials,
        coil_materials=parent.coil_materials,
        package_materials=parent.package_materials,
        seawater_material=parent.seawater_material,
        ambient_temperature=parent.ambient_temperature,
    )
    background.background_config = cfg
    background.self_correction_config = {"enabled": False}
    cache[key] = background
    return background


def _phi_on_background(source_background, target_background, phi):
    values = np.asarray(phi, float)
    if values.ndim != 2 or values.shape[0] != source_background.n_cells:
        raise ValueError("global longitudinal modal basis has incompatible shape")
    if target_background is source_background:
        return values
    grid_values = values.reshape(
        source_background.nx,
        source_background.ny,
        source_background.nz,
        values.shape[1],
    )
    interpolation = RegularGridInterpolator(
        source_background.cell_axes,
        grid_values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    out = np.asarray(interpolation(target_background.cell_centers), float)
    if out.shape != (target_background.n_cells, values.shape[1]) or np.any(~np.isfinite(out)):
        raise ValueError("global longitudinal reference left the thermal-basis domain")
    return out


@dataclass(frozen=True)
class LongitudinalSelfState:
    z: np.ndarray
    d_vol: np.ndarray
    d_out: np.ndarray
    modal_h: np.ndarray | None
    audit: dict


def longitudinal_self_state(background, geometry, *, context=None, phi=None):
    if context is None:
        context = background.geometry_context(geometry, assemble_thermal=False)
    block = build_gradient_block(background, context, check_topology=True)
    B = np.asarray(background.rhs_matrix(context), complex)
    source = np.asarray(context.source_shape, float)
    sigma = np.asarray(background.cell_properties(context, None, em=True)[0], float)
    edge_loss = np.asarray(background.edge_cell_hodge @ sigma, float).reshape(-1)
    outward = np.asarray(background.outward_loss_weights(), float).reshape(-1)
    if B.shape != source.shape or B.shape[0] != background.n_edges:
        raise ValueError("global longitudinal source/RHS shape mismatch")

    n = B.shape[1]
    z = np.zeros(n, complex)
    d = np.zeros(n, float)
    d_out = np.zeros(n, float)
    modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], n), float)
    max_projection = 0.0
    max_impedance_defect = 0.0
    max_balance = 0.0

    for p in range(n):
        field, projection = refined_gradient_projection(
            background,
            block,
            B[:, p],
            relative_tolerance=5e-13,
            maximum_refinements=5,
        )
        abs2 = field_abs2(field)
        z[p] = complex(-field_linear_dot(source[:, p], field))
        d[p] = float(np.dot(edge_loss, abs2))
        d_out[p] = float(np.dot(outward, abs2))
        scale = max(abs(z[p].real), abs(d[p]) + abs(d_out[p]), np.finfo(float).tiny)
        max_balance = max(max_balance, abs(z[p].real - d[p] - d_out[p]) / scale)
        max_projection = max(max_projection, float(projection["relative_residual"]))
        max_impedance_defect = max(
            max_impedance_defect, float(projection["impedance_defect"])
        )
        if modal is not None:
            q = np.asarray(
                0.5
                * sigma
                * np.asarray(background.edge_cell_hodge.T @ abs2).reshape(-1),
                float,
            )
            modal[:, p] = np.asarray(2.0 * (np.asarray(phi, float).T @ q), float)

    return LongitudinalSelfState(
        z=z,
        d_vol=d,
        d_out=d_out,
        modal_h=modal,
        audit={
            "model": _MODEL,
            "fine_step": _background_step(background),
            "scalar_dofs": int(block.scalar_dofs),
            "maximum_projection_relative_residual": float(max_projection),
            "maximum_projection_impedance_defect": float(max_impedance_defect),
            "maximum_power_balance_relative_error": float(max_balance),
        },
    )


def _cached_context(background, geometry):
    row = getattr(background, "_sdfmpneo_global_longitudinal_last_context", None)
    if not isinstance(row, tuple) or len(row) != 2:
        return None
    key, context = row
    return context if key == _geometry_key(geometry) else None


def _correction(background, geometry, *, phi=None):
    cfg = _config(background)
    n = len(background.coil_materials)
    zero_modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], n), float)
    if not cfg["enabled"]:
        return {
            "delta_z": np.zeros(n, complex),
            "delta_d_vol": np.zeros(n, float),
            "delta_d_out": np.zeros(n, float),
            "delta_modal_h": zero_modal,
            "audit": {"enabled": False, "model": _MODEL},
        }

    current_context = _cached_context(background, geometry)
    current = longitudinal_self_state(
        background, geometry, context=current_context, phi=phi
    )
    current_step = _background_step(background)
    reference_step = float(cfg["reference_fine_step"])
    target_step = min(current_step, reference_step)
    if abs(target_step - current_step) <= 1e-14 * max(current_step, 1.0):
        reference_background = background
        reference_context = current_context
        reference_phi = phi
        reference = current
    else:
        reference_background = _make_background(background, target_step)
        reference_context = _cached_context(reference_background, geometry)
        reference_phi = None if phi is None else _phi_on_background(
            background, reference_background, phi
        )
        reference = longitudinal_self_state(
            reference_background,
            geometry,
            context=reference_context,
            phi=reference_phi,
        )

    delta_modal = None
    if phi is not None:
        delta_modal = np.asarray(reference.modal_h - current.modal_h, float)
    dz = np.asarray(reference.z - current.z, complex)
    dd = np.asarray(reference.d_vol - current.d_vol, float)
    do = np.asarray(reference.d_out - current.d_out, float)
    balance = float(
        np.linalg.norm(np.real(dz) - dd - do)
        / max(
            float(np.linalg.norm(np.real(dz))),
            float(np.linalg.norm(dd + do)),
            np.finfo(float).tiny,
        )
    )
    return {
        "delta_z": dz,
        "delta_d_vol": dd,
        "delta_d_out": do,
        "delta_modal_h": delta_modal,
        "audit": {
            "enabled": True,
            "model": _MODEL,
            "current_fine_step": float(current_step),
            "reference_fine_step": float(target_step),
            "configured_reference_fine_step": float(reference_step),
            "delta_z_real": np.real(dz).tolist(),
            "delta_z_imag": np.imag(dz).tolist(),
            "delta_d_vol": dd.tolist(),
            "delta_d_out": do.tolist(),
            "correction_power_balance_relative_error": balance,
            "current": current.audit,
            "reference": reference.audit,
        },
    }


def apply_global_longitudinal_reference(
    background,
    geometry,
    z,
    d_vol,
    d_out=None,
    *,
    phi=None,
    modal_h=None,
):
    zc = np.asarray(z, complex).copy()
    dc = np.asarray(d_vol, complex).copy()
    oc = None if d_out is None else np.asarray(d_out, complex).copy()
    hc = None if modal_h is None else np.asarray(modal_h, complex).copy()
    correction = _correction(background, geometry, phi=phi)
    for p, value in enumerate(correction["delta_z"]):
        zc[p, p] += value
        dc[p, p] += float(correction["delta_d_vol"][p])
        if oc is not None:
            oc[p, p] += float(correction["delta_d_out"][p])
        if hc is not None:
            hc[:, p, p] += np.asarray(correction["delta_modal_h"][:, p], float)
    return zc, dc, oc, hc, correction["audit"]


def audit_reference_convergence(background, geometry):
    cfg = _config(background)
    reference = _make_background(background, float(cfg["reference_fine_step"]))
    validation = _make_background(background, float(cfg["validation_fine_step"]))
    state0 = longitudinal_self_state(reference, geometry)
    state1 = longitudinal_self_state(validation, geometry)
    zerr = _relative(state0.z, state1.z)
    rerr = _relative(np.real(state0.z), np.real(state1.z))
    xerr = _relative(np.imag(state0.z), np.imag(state1.z))
    derr = _relative(state0.d_vol, state1.d_vol)
    out_scale = max(
        float(np.linalg.norm(state0.d_vol + state0.d_out)),
        float(np.linalg.norm(state1.d_vol + state1.d_out)),
        np.finfo(float).tiny,
    )
    outerr = float(np.linalg.norm(state0.d_out - state1.d_out) / out_scale)
    worst = max(zerr, rerr, xerr, derr, outerr)
    print(
        "global longitudinal scalar reference Gate……"
        f"{cfg['reference_fine_step']:.6g}m -> {cfg['validation_fine_step']:.6g}m  "
        f"z={zerr:.3e}  R={rerr:.3e}  X={xerr:.3e}  "
        f"Dvol={derr:.3e}  Dout={outerr:.3e}  max={worst:.3e}",
        flush=True,
    )
    return {
        "model": _MODEL,
        "reference_fine_step": float(cfg["reference_fine_step"]),
        "validation_fine_step": float(cfg["validation_fine_step"]),
        "relative_tolerance": float(cfg["relative_tolerance"]),
        "relative_z_error": float(zerr),
        "relative_resistive_error": float(rerr),
        "relative_reactive_error": float(xerr),
        "relative_d_vol_error": float(derr),
        "relative_outward_partition_significance": float(outerr),
        "maximum_relative_error": float(worst),
        "converged": bool(worst <= float(cfg["relative_tolerance"])),
        "reference": state0.audit,
        "validation": state1.audit,
    }


def _resolve_settings(settings, background):
    bg_cfg = settings["BACKGROUND"]
    mesh = bg_cfg.setdefault("mesh_check", {})
    production = float(bg_cfg["fine_step"])
    factor = float(mesh.get("refinement_factor", 0.75))
    mesh.setdefault("longitudinal_reference_fine_step", factor * production)
    mesh.setdefault("longitudinal_reference_validation_factor", factor)
    if isinstance(getattr(background, "background_config", None), dict):
        background.background_config["_production_fine_step"] = production
        background.background_config.setdefault("mesh_check", {}).update(copy.deepcopy(mesh))


def install(corrected_preflight_module, corrected_truth_module):
    if bool(getattr(corrected_preflight_module, "_global_longitudinal_reference_installed", False)):
        return corrected_preflight_module

    # Keep the exact context used by the full Maxwell solve so the current-grid
    # scalar factor built by the compatible solver can be reused.
    original_pf_solve = corrected_preflight_module._solve_fields

    def preflight_solve(background, geometry, *args, **kwargs):
        result = original_pf_solve(background, geometry, *args, **kwargs)
        background._sdfmpneo_global_longitudinal_last_context = (
            _geometry_key(geometry), result[0]
        )
        return result

    corrected_preflight_module._solve_fields = preflight_solve

    original_truth_port = corrected_truth_module._port_truth_from_context

    def truth_port(background, context):
        geometry = getattr(context, "geometry", None)
        if geometry is not None:
            background._sdfmpneo_global_longitudinal_last_context = (
                _geometry_key(geometry), context
            )
        return original_truth_port(background, context)

    corrected_truth_module._port_truth_from_context = truth_port

    original_correct = corrected_preflight_module._correct

    def correct(background, geometry, z, d, d_out):
        z1, d1, o1, _h, global_audit = apply_global_longitudinal_reference(
            background, geometry, z, d, d_out
        )
        z2, d2, o2, local_audit = original_correct(background, geometry, z1, d1, o1)
        audit = dict(local_audit)
        audit["global_longitudinal_reference"] = global_audit
        return z2, d2, o2, audit

    corrected_preflight_module._correct = correct

    original_mesh = corrected_preflight_module.audit_em_mesh_preflight

    def audit_mesh(settings, background, geometries, monitor=None):
        geometries = list(geometries)
        scalar_rows = []
        for index, geometry in enumerate(geometries):
            if monitor is not None:
                monitor.checkpoint()
            row = audit_reference_convergence(background, geometry)
            row["index"] = int(index)
            scalar_rows.append(row)
        scalar_worst = max((r["maximum_relative_error"] for r in scalar_rows), default=0.0)
        scalar_ok = all(r["converged"] for r in scalar_rows)
        scalar_report = {
            "model": _MODEL,
            "sample_count": len(scalar_rows),
            "maximum_relative_error": float(scalar_worst),
            "converged": bool(scalar_ok),
            "samples": scalar_rows,
        }
        if not scalar_ok:
            cfg = dict(settings["BACKGROUND"].get("mesh_check", {}))
            return {
                "sample_count": len(geometries),
                "refinement_factor": float(cfg.get("refinement_factor", 0.75)),
                "base_fine_step": float(settings["BACKGROUND"]["fine_step"]),
                "refined_fine_step": float(cfg.get("refinement_factor", 0.75))
                * float(settings["BACKGROUND"]["fine_step"]),
                "relative_tolerance": float(cfg.get("relative_tolerance", 1e-1)),
                "source_path_relative_tolerance": float(
                    cfg.get("source_path_relative_tolerance", 1e-10)
                ),
                "maximum_relative_error": float(scalar_worst),
                "maximum_source_path_length_relative_error": 0.0,
                "source_geometry_invariant": True,
                "self_correction_model": corrected_preflight_module._SELF_CORRECTION_MODEL,
                "global_longitudinal_reference_convergence": scalar_report,
                "converged": False,
                "skipped_full_em_mesh_gate": True,
                "skip_reason": "global longitudinal scalar reference not converged",
                "samples": [],
            }
        report = original_mesh(settings, background, geometries, monitor)
        report["global_longitudinal_reference_convergence"] = scalar_report
        report["converged"] = bool(report.get("converged", False) and scalar_ok)
        report["maximum_relative_error"] = max(
            float(report.get("maximum_relative_error", 0.0)), float(scalar_worst)
        )
        return report

    corrected_preflight_module.audit_em_mesh_preflight = audit_mesh

    original_run = corrected_preflight_module.run_truth_preflight

    def run_truth_preflight(settings, background, geometries, monitor=None):
        _resolve_settings(settings, background)
        return original_run(settings, background, geometries, monitor)

    corrected_preflight_module.run_truth_preflight = run_truth_preflight

    original_diagnose = corrected_preflight_module._mesh_failure_diagnosis

    def diagnose_mesh(mesh):
        scalar = mesh.get("global_longitudinal_reference_convergence")
        if isinstance(scalar, dict) and not bool(scalar.get("converged", False)):
            return {
                "code": "global_longitudinal_reference_not_converged",
                "maximum_relative_error": float(
                    scalar.get("maximum_relative_error", np.inf)
                ),
                "recommendation": (
                    "The remaining mesh-sensitive self term is the full-domain "
                    "compatible longitudinal response. Refine only the global "
                    "scalar-gradient reference; do not refine the global Maxwell "
                    "solve, do not reuse the canonical local box for this term, "
                    "and do not relax the mesh Gate."
                ),
            }
        return original_diagnose(mesh)

    corrected_preflight_module._mesh_failure_diagnosis = diagnose_mesh

    original_solve_port = corrected_truth_module.solve_port_truth_tensors

    def solve_port_truth_tensors(background, geometry):
        z, d, d_out, audit = original_solve_port(background, geometry)
        zc, dc, oc, _h, global_audit = apply_global_longitudinal_reference(
            background, geometry, z, d, d_out
        )
        out = dict(audit)
        out["global_longitudinal_reference"] = global_audit
        implied = _hermitian(zc) - dc
        out["minimum_d_vol_eigenvalue"] = float(np.min(np.linalg.eigvalsh(_hermitian(dc))).real)
        out["minimum_physical_outward_eigenvalue"] = float(
            np.min(np.linalg.eigvalsh(_hermitian(oc))).real
        )
        out["minimum_implied_outward_eigenvalue"] = float(
            np.min(np.linalg.eigvalsh(_hermitian(implied))).real
        )
        scale = max(float(np.linalg.norm(_hermitian(zc))), float(np.linalg.norm(dc + oc)), np.finfo(float).tiny)
        out["open_boundary_power_balance_relative_error"] = float(
            np.linalg.norm(_hermitian(zc) - dc - oc) / scale
        )
        return zc, dc, oc, out

    corrected_truth_module.solve_port_truth_tensors = solve_port_truth_tensors

    original_solve_truth = corrected_truth_module.solve_truth_tensors

    def solve_truth_tensors(background, geometry):
        z, d, modal, phi_min, phi_max, audit = original_solve_truth(background, geometry)
        context = background.geometry_context(geometry, assemble_thermal=True)
        phi = np.asarray(context.thermal_basis, float)
        zc, dc, _oc, hc, global_audit = apply_global_longitudinal_reference(
            background,
            geometry,
            z,
            d,
            None,
            phi=phi,
            modal_h=modal,
        )
        out = dict(audit)
        out["global_longitudinal_reference"] = global_audit
        implied = _hermitian(zc) - dc
        out["minimum_d_vol_eigenvalue"] = float(np.min(np.linalg.eigvalsh(_hermitian(dc))).real)
        out["minimum_physical_outward_eigenvalue"] = float(
            np.min(np.linalg.eigvalsh(_hermitian(implied))).real
        )
        out["minimum_implied_outward_eigenvalue"] = out["minimum_physical_outward_eigenvalue"]
        out["open_boundary_power_balance_relative_error"] = 0.0
        out["maximum_relative_loewner_violation"] = corrected_truth_module._loewner_violation(
            hc, dc, np.asarray(phi_min, float), np.asarray(phi_max, float)
        )
        return zc, dc, hc, phi_min, phi_max, out

    corrected_truth_module.solve_truth_tensors = solve_truth_tensors
    corrected_preflight_module._global_longitudinal_reference_installed = True
    corrected_truth_module._global_longitudinal_reference_installed = True
    return corrected_preflight_module


__all__ = [
    "LongitudinalSelfState",
    "apply_global_longitudinal_reference",
    "audit_reference_convergence",
    "install",
    "longitudinal_self_state",
]
