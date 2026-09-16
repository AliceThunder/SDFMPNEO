"""Certified whole-domain scalar reference for longitudinal self dissipation.

The boundary-conditioned terminal patch is an appropriate near-field reference
for the reactive open-terminal self response, but its local conductive-loss
integral is not a stable production quantity.  Dissipation remains a whole-domain
loss-operator quantity.  This adapter therefore leaves the certified terminal
reactive correction intact and adds a separate *global* scalar reference for
self resistance / D_vol.

For a production background with fine step h and the existing EM mesh factor r,
the production scalar loss reference is h*r (the same mesh used by the refined
full-Maxwell Gate).  It is independently certified against h*r^2.  Only after
that scalar-only Gate passes do we apply

    delta D_vol,pp = D_L,pp(h*r) - D_L,pp(current),
    Re(delta Z_pp) = delta D_vol,pp,

with the matching Joule modal correction.  D_out and every mutual term are left
untouched.  Setting the real impedance correction exactly equal to the volume
loss correction preserves Herm(Z)-D_vol-D_out power closure.

The refined scalar backgrounds consume the already-certified physical nodal
terminal charge directly, so they do not build the redundant edge-space G.T G
charge-lift factor.
"""
from __future__ import annotations

import copy

import numpy as np

from .unified_charge_regularized_source import terminal_charge_target
from .unified_gradient_block_maxwell import build_gradient_block


_MODEL = "global_longitudinal_dissipative_scalar_reference_v1"


def _geometry_key(module, geometry):
    helper = getattr(module, "_geometry_key", None)
    return helper(geometry) if callable(helper) else repr(geometry)


def _config(background):
    root = dict(getattr(background, "background_config", {}) or {})
    own = dict(root.get("global_dissipative_reference", {}) or {})
    if not own:
        return {
            "enabled": False,
            "reference_step": np.inf,
            "validation_step": np.inf,
            "reference_max_step": np.inf,
            "validation_max_step": np.inf,
            "relative_tolerance": 1e-1,
            "max_cells": 400000,
        }
    return {
        "enabled": bool(own.get("enabled", True)),
        "reference_step": float(own["reference_step"]),
        "validation_step": float(own["validation_step"]),
        "reference_max_step": float(own["reference_max_step"]),
        "validation_max_step": float(own["validation_max_step"]),
        "relative_tolerance": float(own.get("relative_tolerance", 1e-1)),
        "max_cells": int(own.get("max_cells", 400000)),
    }


def _interpolate_basis(parent, target, phi):
    from scipy.interpolate import RegularGridInterpolator

    values = np.asarray(phi, float)
    if values.ndim != 2 or values.shape[0] != parent.n_cells:
        raise ValueError("global dissipative reference basis has incompatible parent shape")
    grid = values.reshape(parent.nx, parent.ny, parent.nz, values.shape[1])
    interpolation = RegularGridInterpolator(
        parent.cell_axes,
        grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    # A finer whole-domain mesh has cell centers closer to the physical boundary
    # than the parent cell-center grid.  Clamp only those outermost coordinates
    # to the parent center envelope before interpolation.  This is the natural
    # constant extension of the boundary-adjacent parent cell and, unlike linear
    # extrapolation, cannot create new extrema that would violate Phi min/max
    # bounds used by the Joule Loewner certificate.
    coords = np.asarray(target.cell_centers, float).copy()
    for axis_index, axis in enumerate(parent.cell_axes):
        values_axis = np.asarray(axis, float)
        coords[:, axis_index] = np.clip(
            coords[:, axis_index], values_axis[0], values_axis[-1]
        )
    out = np.asarray(interpolation(coords), float)
    if out.shape != (target.n_cells, values.shape[1]) or np.any(~np.isfinite(out)):
        raise ValueError("global dissipative reference basis transfer is non-finite")
    return out


def _reference_background(parent, step, max_step, max_cells):
    cfg = copy.deepcopy(dict(getattr(parent, "background_config", {}) or {}))
    required = ("bounds", "core_half_extent")
    if any(name not in cfg for name in required):
        raise ValueError("global dissipative reference requires the production background config")
    cfg["fine_step"] = float(step)
    cfg["max_step"] = float(max(max_step, step))
    ref = parent.__class__.from_config(
        cfg,
        frequency_hz=parent.frequency_hz,
        materials=parent.materials,
        coil_materials=parent.coil_materials,
        package_materials=parent.package_materials,
        seawater_material=parent.seawater_material,
        ambient_temperature=parent.ambient_temperature,
    )
    if int(ref.n_cells) > int(max_cells):
        raise RuntimeError(
            "global dissipative scalar reference exceeds cell budget: "
            f"cells={ref.n_cells}, budget={int(max_cells)}, step={float(step):.6g}m"
        )
    ref.background_config = cfg
    ref.self_correction_config = {"enabled": False}
    # Refined scalar-only truth needs G.T S=q_target, not the explicit edge lift.
    ref._sdfmpneo_scalar_charge_target_only = True
    return ref


def _scalar_state(module, parent, background, geometry, *, phi=None):
    context = None
    if background is parent:
        cached_context = getattr(module, "_cached_context", None)
        if callable(cached_context):
            context = cached_context(background, geometry)
    if context is None:
        context = background.geometry_context(geometry, assemble_thermal=False)
    block = build_gradient_block(background, context, check_topology=True)
    sigma = np.asarray(background.cell_properties(context, None, em=True)[0], float)
    edge_loss = np.asarray(background.edge_cell_hodge @ sigma, float).reshape(-1)
    n = len(background.coil_materials)
    d = np.zeros(n, float)
    z = np.zeros(n, complex)
    modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], n), float)
    local_phi = None if phi is None else _interpolate_basis(parent, background, phi)
    residuals = []
    supports = []
    for p, coil in enumerate(context.geometry.coils):
        q_full, meta = terminal_charge_target(background, coil)
        q = np.asarray(q_full[1:], complex)
        scalar_rhs = (-1j * float(background.omega)) * q
        potential = np.asarray(block.factor.solve(scalar_rhs), complex).reshape(-1)
        residuals.append(
            float(
                np.linalg.norm(scalar_rhs - block.scalar_matrix @ potential)
                / max(float(np.linalg.norm(scalar_rhs)), np.finfo(float).tiny)
            )
        )
        field = np.asarray(block.gradient @ potential, complex).reshape(-1)
        abs2 = np.abs(field) ** 2
        d[p] = float(np.dot(edge_loss, abs2))
        # Gauge node zero has phi=0, hence q.T phi = q[1:].T potential.
        z[p] = complex(-q @ potential)
        supports.append(int(meta.get("terminal_charge_support_nodes", 0)))
        if modal is not None:
            q_cells = np.asarray(
                0.5
                * sigma
                * np.asarray(background.edge_cell_hodge.T @ abs2).reshape(-1),
                float,
            )
            modal[:, p] = np.asarray(2.0 * (local_phi.T @ q_cells), float)
    return {
        "fine_step": float(module._background_step(background)),
        "n_cells": int(background.n_cells),
        "scalar_dofs": int(block.scalar_dofs),
        "d_vol": d,
        "z_reaction": z,
        "modal_h": modal,
        "maximum_scalar_relative_residual": max(residuals, default=0.0),
        "terminal_charge_support_nodes": supports,
    }


def _cache(background):
    value = getattr(background, "_sdfmpneo_global_dissipative_reference_cache", None)
    if value is None:
        value = {}
        background._sdfmpneo_global_dissipative_reference_cache = value
    return value


def _reference_state(module, background, geometry, *, step, max_step, phi=None):
    cfg = _config(background)
    key = (_geometry_key(module, geometry), float(step), float(max_step), phi is None)
    cache = _cache(background)
    if phi is None and key in cache:
        return copy.deepcopy(cache[key])
    ref = _reference_background(background, step, max_step, cfg["max_cells"])
    state = _scalar_state(module, background, ref, geometry, phi=phi)
    print(
        "global longitudinal dissipative scalar: "
        f"step={float(step):.6g}m, cells={state['n_cells']}, "
        f"dofs={state['scalar_dofs']}, residual={state['maximum_scalar_relative_residual']:.3e}",
        flush=True,
    )
    if phi is None:
        cache[key] = copy.deepcopy(state)
    return state


def install(module):
    if bool(getattr(module, "_global_dissipative_reference_installed", False)):
        return module

    original_resolve = module._resolve_settings
    original_correction = module._correction
    original_audit = module.audit_reference_convergence

    def resolve_settings(settings, background):
        original_resolve(settings, background)
        bg = settings["BACKGROUND"]
        mesh = dict(bg.get("mesh_check", {}) or {})
        own = bg.setdefault("global_dissipative_reference", {})
        base = float(bg["fine_step"])
        factor = float(mesh.get("refinement_factor", 0.75))
        if not 0.0 < factor < 1.0:
            raise ValueError("global dissipative reference requires mesh refinement_factor in (0,1)")
        base_max = float(bg.get("max_step", 4.0 * base))
        reference = base * factor
        validation = reference * factor
        own.setdefault("enabled", True)
        own.setdefault("reference_step", reference)
        own.setdefault("validation_step", validation)
        own.setdefault("reference_max_step", max(reference, base_max * factor))
        own.setdefault("validation_max_step", max(validation, base_max * factor * factor))
        own.setdefault("relative_tolerance", float(mesh.get("relative_tolerance", 1e-1)))
        own.setdefault("max_cells", 400000)
        if not 0.0 < float(own["validation_step"]) < float(own["reference_step"]) < base:
            raise ValueError("global dissipative scalar steps must satisfy validation < reference < base")
        if isinstance(getattr(background, "background_config", None), dict):
            background.background_config["global_dissipative_reference"] = copy.deepcopy(own)

    module._resolve_settings = resolve_settings

    def correction(background, geometry, *, phi=None):
        raw = original_correction(background, geometry, phi=phi)
        cfg = _config(background)
        current_step = float(module._background_step(background))
        audit = dict(raw.get("audit", {}))
        audit["global_dissipative_reference_model"] = _MODEL
        if not cfg["enabled"] or current_step <= cfg["reference_step"] * (1.0 + 1e-12):
            audit["global_dissipative_reference_enabled"] = False
            raw["audit"] = audit
            return raw

        reference = _reference_state(
            module,
            background,
            geometry,
            step=cfg["reference_step"],
            max_step=cfg["reference_max_step"],
            phi=phi,
        )
        current_d = np.asarray(audit.get("global_scalar", {}).get("d_vol", ()), float)
        if current_d.shape != np.asarray(reference["d_vol"]).shape:
            current = _scalar_state(module, background, background, geometry, phi=phi)
            current_d = np.asarray(current["d_vol"], float)
        elif phi is not None:
            current = _scalar_state(module, background, background, geometry, phi=phi)
        else:
            current = None
        delta = np.asarray(reference["d_vol"], float) - current_d
        raw["delta_z"] = np.asarray(raw["delta_z"], complex) + delta.astype(complex)
        raw["delta_d_vol"] = np.asarray(raw["delta_d_vol"], float) + delta
        if phi is not None:
            if raw.get("delta_modal_h") is None:
                raw["delta_modal_h"] = np.zeros_like(reference["modal_h"])
            raw["delta_modal_h"] = (
                np.asarray(raw["delta_modal_h"], float)
                + np.asarray(reference["modal_h"], float)
                - np.asarray(current["modal_h"], float)
            )
        audit.update(
            global_dissipative_reference_enabled=True,
            global_dissipative_reference_step=float(cfg["reference_step"]),
            global_dissipative_delta_d_vol=delta.tolist(),
            global_dissipative_delta_z_real=delta.tolist(),
            global_dissipative_reference_scalar={
                "fine_step": float(reference["fine_step"]),
                "n_cells": int(reference["n_cells"]),
                "scalar_dofs": int(reference["scalar_dofs"]),
                "d_vol": np.asarray(reference["d_vol"], float).tolist(),
                "z_real": np.real(reference["z_reaction"]).tolist(),
                "maximum_scalar_relative_residual": float(reference["maximum_scalar_relative_residual"]),
            },
        )
        raw["audit"] = audit
        return raw

    module._correction = correction

    def audit_reference_convergence(background, geometry):
        report = dict(original_audit(background, geometry))
        cfg = _config(background)
        if not cfg["enabled"]:
            return report
        current = np.asarray(report.get("global_scalar", {}).get("d_vol", ()), float)
        if current.size == 0:
            _context, _potentials, global_audit = module._global_scalar_potentials(background, geometry)
            current = np.asarray(global_audit["d_vol"], float)
        reference = _reference_state(
            module,
            background,
            geometry,
            step=cfg["reference_step"],
            max_step=cfg["reference_max_step"],
            phi=None,
        )
        validation = _reference_state(
            module,
            background,
            geometry,
            step=cfg["validation_step"],
            max_step=cfg["validation_max_step"],
            phi=None,
        )
        rows = []
        worst = 0.0
        for p in range(len(current)):
            d0 = float(current[p])
            dr = float(reference["d_vol"][p])
            dv = float(validation["d_vol"][p])
            delta_r = dr - d0
            delta_v = dv - d0
            scale = max(abs(delta_v), abs(dv), abs(d0), np.finfo(float).tiny)
            derr = abs(delta_r - delta_v) / scale
            out_r = float(np.real(reference["z_reaction"][p])) - dr
            out_v = float(np.real(validation["z_reaction"][p])) - dv
            outward = abs(out_r - out_v) / scale
            row_worst = max(float(derr), float(outward))
            worst = max(worst, row_worst)
            rows.append({
                "port": int(p),
                "current_d_vol": d0,
                "reference_d_vol": dr,
                "validation_d_vol": dv,
                "reference_delta_d_vol": delta_r,
                "validation_delta_d_vol": delta_v,
                "relative_d_vol_defect_error": float(derr),
                "relative_outward_partition_significance": float(outward),
                "maximum_relative_error": float(row_worst),
            })
            print(
                "global longitudinal dissipative scalar Gate: "
                f"port={p + 1}, {cfg['reference_step']:.6g}m->{cfg['validation_step']:.6g}m, "
                f"Dvol={derr:.3e}, Dout={outward:.3e}, max={row_worst:.3e}",
                flush=True,
            )
        scalar_ok = bool(worst <= cfg["relative_tolerance"])
        reactive_ok = bool(report.get("converged", False))
        report["global_dissipative_reference"] = {
            "model": _MODEL,
            "reference_step": float(cfg["reference_step"]),
            "validation_step": float(cfg["validation_step"]),
            "relative_tolerance": float(cfg["relative_tolerance"]),
            "maximum_relative_error": float(worst),
            "converged": scalar_ok,
            "reference": {
                "n_cells": int(reference["n_cells"]),
                "scalar_dofs": int(reference["scalar_dofs"]),
                "maximum_scalar_relative_residual": float(reference["maximum_scalar_relative_residual"]),
            },
            "validation": {
                "n_cells": int(validation["n_cells"]),
                "scalar_dofs": int(validation["scalar_dofs"]),
                "maximum_scalar_relative_residual": float(validation["maximum_scalar_relative_residual"]),
            },
            "samples": rows,
        }
        report["global_dissipative_reference_applied"] = True
        report["maximum_global_dissipative_relative_error"] = float(worst)
        report["maximum_relative_error"] = max(
            float(report.get("maximum_relative_error", 0.0)), float(worst)
        )
        report["converged"] = bool(reactive_ok and scalar_ok)
        report["model"] = f"{getattr(module, '_MODEL', 'longitudinal')}+{_MODEL}"
        return report

    module.audit_reference_convergence = audit_reference_convergence
    module._global_dissipative_reference_installed = True
    return module


__all__ = ["install"]
