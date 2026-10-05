"""Certification/adaptation layer for boundary-conditioned scalar patches.

The longitudinal near-field defect is valid only when its coarse Cartesian patch
represents the *same* scalar problem as the parent global grid.  In particular,
a target-port patch must not silently drop another rotated package merely because
the two physical OBBs do not overlap: their Cartesian bounding boxes may overlap.

This layer therefore keeps the complete global geometry/material assembly in
every scalar patch, selects only the target port RHS, and independently checks
that the parent global scalar potential satisfies the coarse patch interior
equations.  The artificial patch boundary receives the global Dirichlet trace;
no Silver-Muller term is added there.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator


def _consistency_tolerance(background):
    root = dict(getattr(background, "background_config", {}) or {})
    cfg = dict(root.get("global_longitudinal_correction", {}) or {})
    return float(cfg.get("coarse_consistency_tolerance", 1e-8))


def _restricted_parent_potential(module, parent, global_potential, patch):
    values = np.asarray(global_potential, complex).reshape(
        parent.nx + 1, parent.ny + 1, parent.nz + 1
    )
    interpolation = RegularGridInterpolator(
        (parent.x, parent.y, parent.z),
        values,
        method="linear",
        bounds_error=True,
    )
    return np.asarray(
        interpolation(module.node_coordinates(patch)), complex
    ).reshape(-1)


def coarse_parent_restriction_residual(
    module,
    parent,
    patch,
    geometry,
    global_potential,
    *,
    source_port=0,
):
    """Return the parent-potential residual in coarse patch interior equations."""
    context = patch.geometry_context(geometry, assemble_thermal=False)
    B = np.asarray(patch.rhs_matrix(context), complex)
    p = int(source_port)
    if B.ndim != 2 or p < 0 or p >= B.shape[1]:
        raise AssertionError("longitudinal consistency source port is invalid")
    rhs = B[:, p]
    G = module.gradient_operator(patch, gauge_fixed=False)
    diagonal, _sigma, _hs = module._volume_edge_diagonal(patch, context)
    scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsr()
    scalar.sum_duplicates()
    scalar.eliminate_zeros()
    scalar_rhs = np.asarray(G.T @ rhs, complex).reshape(-1)

    boundary = module._boundary_node_mask(patch)
    interior = ~boundary
    parent_phi = _restricted_parent_potential(
        module, parent, global_potential, patch
    )
    action = np.asarray(scalar[interior] @ parent_phi, complex).reshape(-1)
    residual = scalar_rhs[interior] - action
    denominator = max(
        float(np.linalg.norm(scalar_rhs[interior])),
        float(np.linalg.norm(action)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(residual) / denominator)


def _padding_candidates(module, background):
    cfg = module._config(background)
    configured = float(cfg["boundary_padding"])
    floor = min(configured, max(float(cfg["core_padding"]), 1e-6))
    values = [
        configured,
        max(floor, 0.5 * configured),
        max(floor, 0.25 * configured),
        floor,
    ]
    out = []
    for value in values:
        value = float(value)
        if not any(
            abs(value - old) <= 1e-13 * max(abs(value), abs(old), 1.0)
            for old in out
        ):
            out.append(value)
    return out


def _as_geometry(module, geometry):
    cls = module.UnifiedUWPTGeometry
    return geometry if isinstance(geometry, cls) else cls.from_mapping(geometry)


def _full_geometry_patch_axes(module, parent, geometry, target_port, cfg):
    """Patch bounds contain every physical package; refinement core targets one."""
    g = _as_geometry(module, geometry)
    lows = []
    highs = []
    for package in g.packages:
        rotation = np.asarray(package.pose.rotation, float)
        half = np.asarray(package.half_extent, float)
        center = np.asarray(package.pose.translation, float)
        aabb_half = np.abs(rotation) @ half
        lows.append(center - aabb_half)
        highs.append(center + aabb_half)
    lo = np.min(np.asarray(lows, float), axis=0)
    hi = np.max(np.asarray(highs, float), axis=0)
    padding = float(cfg["boundary_padding"])
    axes = tuple(
        module._select_axis(axis, lo[k] - padding, hi[k] + padding)
        for k, axis in enumerate((parent.x, parent.y, parent.z))
    )

    target = g.packages[int(target_port)]
    center = np.asarray(target.pose.translation, float)
    aabb_half = np.abs(np.asarray(target.pose.rotation, float)) @ np.asarray(
        target.half_extent, float
    )
    return axes, center, aabb_half, g


def _make_full_patch_background(module, parent, axes, *, fine_step):
    patch = parent.__class__(
        *axes,
        frequency_hz=parent.frequency_hz,
        materials=parent.materials,
        coil_materials=tuple(parent.coil_materials),
        package_materials=tuple(parent.package_materials),
        seawater_material=parent.seawater_material,
        ambient_temperature=parent.ambient_temperature,
    )
    patch.background_config = {
        "fine_step": float(fine_step),
        "self_correction": {"enabled": False},
    }
    patch.self_correction_config = {"enabled": False}
    return patch


def _full_patch_state(
    module,
    parent,
    patch,
    geometry,
    global_potential,
    *,
    source_port,
    phi=None,
    fine_step,
    certify_parent=False,
):
    """Solve one target-port scalar problem while retaining all patch materials."""
    context = patch.geometry_context(geometry, assemble_thermal=False)
    B = np.asarray(patch.rhs_matrix(context), complex)
    source_shape = np.asarray(context.source_shape, float)
    p = int(source_port)
    if B.ndim != 2 or p < 0 or p >= B.shape[1]:
        raise AssertionError("longitudinal scalar patch source port is invalid")
    rhs = B[:, p]
    source = source_shape[:, p]

    G = module.gradient_operator(patch, gauge_fixed=False)
    diagonal, sigma, _hs = module._volume_edge_diagonal(patch, context)
    scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
    scalar.sum_duplicates()
    scalar.eliminate_zeros()
    scalar_rhs = np.asarray(G.T @ rhs, complex).reshape(-1)

    boundary = module._boundary_node_mask(patch)
    interior = ~boundary
    phi_nodes = np.zeros(G.shape[1], complex)
    phi_nodes[boundary] = module._boundary_values(
        parent, global_potential, patch, boundary
    )
    Sii = scalar[interior][:, interior].tocsc()
    Sib = scalar[interior][:, boundary].tocsr()
    rhs_i = scalar_rhs[interior] - Sib @ phi_nodes[boundary]
    try:
        factor = spla.splu(
            Sii,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        factor = spla.splu(Sii)
    phi_nodes[interior] = np.asarray(factor.solve(rhs_i), complex).reshape(-1)
    residual = float(
        np.linalg.norm(rhs_i - Sii @ phi_nodes[interior])
        / max(float(np.linalg.norm(rhs_i)), np.finfo(float).tiny)
    )

    field = np.asarray(G @ phi_nodes, complex).reshape(-1)
    abs2 = np.abs(field) ** 2
    z_reaction = complex(-source @ field)
    edge_loss = np.asarray(patch.edge_cell_hodge @ sigma, float).reshape(-1)
    d_vol = float(np.dot(edge_loss, abs2))
    q_cells = np.asarray(
        0.5 * sigma * np.asarray(patch.edge_cell_hodge.T @ abs2).reshape(-1),
        float,
    )
    modal = None
    if phi is not None:
        local_phi = module._interpolate_cell_basis(parent, patch, phi)
        modal = np.asarray(2.0 * (local_phi.T @ q_cells), float)

    consistency = None
    if certify_parent:
        consistency = coarse_parent_restriction_residual(
            module,
            parent,
            patch,
            geometry,
            global_potential,
            source_port=p,
        )
    return {
        "z_reaction": z_reaction,
        "d_vol": d_vol,
        "modal_h": modal,
        "scalar_relative_residual": residual,
        "n_cells": int(patch.n_cells),
        "scalar_dofs": int(np.count_nonzero(interior)),
        "fine_step": float(fine_step),
        "parent_restriction_relative_residual": (
            None if consistency is None else float(consistency)
        ),
        "full_geometry_material_assembly": True,
        "source_port": p,
    }


def install(module):
    if bool(getattr(module, "_patch_consistency_installed", False)):
        return module

    # Production module exposes these hooks.  Tiny unit-test adapters may not;
    # in that case install only the report/correction certification wrappers.
    can_adapt_patch = all(
        hasattr(module, name)
        for name in (
            "_port_states",
            "_config",
            "_select_axis",
            "_refined_axes",
            "_boundary_values",
            "_interpolate_cell_basis",
            "_volume_edge_diagonal",
            "_boundary_node_mask",
            "gradient_operator",
            "UnifiedUWPTGeometry",
        )
    )

    if can_adapt_patch:
        def port_states(
            background,
            geometry,
            port,
            global_potential,
            target_steps,
            *,
            phi=None,
        ):
            base_cfg = module._config(background)
            tolerance = _consistency_tolerance(background)
            attempts = []
            for padding in _padding_candidates(module, background):
                cfg = dict(base_cfg)
                cfg["boundary_padding"] = float(padding)
                coarse_axes, center, aabb_half, full_geometry = _full_geometry_patch_axes(
                    module, background, geometry, port, cfg
                )
                coarse = _make_full_patch_background(
                    module,
                    background,
                    coarse_axes,
                    fine_step=module._background_step(background),
                )
                coarse_state = _full_patch_state(
                    module,
                    background,
                    coarse,
                    full_geometry,
                    global_potential,
                    source_port=int(port),
                    phi=phi,
                    fine_step=module._background_step(background),
                    certify_parent=True,
                )
                consistency_value = float(
                    coarse_state["parent_restriction_relative_residual"]
                )
                attempts.append({
                    "boundary_padding": float(padding),
                    "parent_restriction_relative_residual": consistency_value,
                })
                print(
                    "boundary-conditioned longitudinal patch consistency: "
                    f"port={int(port) + 1}, padding={padding:.6g}m, "
                    f"residual={consistency_value:.3e}, "
                    f"accepted={'yes' if consistency_value <= tolerance else 'no'}",
                    flush=True,
                )
                if consistency_value > tolerance:
                    continue

                coarse_state = dict(coarse_state)
                coarse_state["selected_boundary_padding"] = float(padding)
                coarse_state["boundary_selection_attempts"] = attempts.copy()
                states = {"coarse": coarse_state}
                for step in target_steps:
                    axes = module._refined_axes(
                        coarse_axes, center, aabb_half, cfg, float(step)
                    )
                    patch = _make_full_patch_background(
                        module, background, axes, fine_step=float(step)
                    )
                    state = dict(
                        _full_patch_state(
                            module,
                            background,
                            patch,
                            full_geometry,
                            global_potential,
                            source_port=int(port),
                            phi=phi,
                            fine_step=float(step),
                            certify_parent=False,
                        )
                    )
                    state["selected_boundary_padding"] = float(padding)
                    states[float(step)] = state
                return states

            raise RuntimeError(
                "no full-geometry boundary-conditioned longitudinal patch "
                "reproduces the parent scalar restriction; attempts=" + repr(attempts)
            )

        module._port_states = port_states

    original_audit = module.audit_reference_convergence

    def audit_reference_convergence(background, geometry):
        try:
            report = dict(original_audit(background, geometry))
        except RuntimeError as exc:
            tolerance = _consistency_tolerance(background)
            print(
                "boundary-conditioned longitudinal coarse consistency: "
                f"accepted=no ({exc})",
                flush=True,
            )
            return {
                "model": module._MODEL,
                "relative_tolerance": float(module._config(background)["relative_tolerance"]),
                "coarse_parent_restriction_tolerance": float(tolerance),
                "maximum_coarse_parent_restriction_relative_residual": float("inf"),
                "coarse_parent_restriction_consistent": False,
                "maximum_relative_error": float("inf"),
                "converged": False,
                "failure": str(exc),
                "samples": [],
            }
        tolerance = _consistency_tolerance(background)
        values = []
        for row in report.get("samples", []):
            value = row.get("coarse", {}).get(
                "parent_restriction_relative_residual"
            )
            if value is not None:
                values.append(float(value))
        maximum = max(values, default=0.0)
        consistent = bool(maximum <= tolerance)
        report["coarse_parent_restriction_tolerance"] = float(tolerance)
        report["maximum_coarse_parent_restriction_relative_residual"] = float(maximum)
        report["coarse_parent_restriction_consistent"] = consistent
        report["converged"] = bool(report.get("converged", False) and consistent)
        print(
            "boundary-conditioned longitudinal coarse consistency: "
            f"max={maximum:.3e}, tol={tolerance:.1e}, "
            f"accepted={'yes' if consistent else 'no'}",
            flush=True,
        )
        return report

    module.audit_reference_convergence = audit_reference_convergence

    original_correction = module._correction

    def correction(background, geometry, *, phi=None):
        result = original_correction(background, geometry, phi=phi)
        audit = dict(result.get("audit", {}))
        if not bool(audit.get("enabled", False)):
            return result
        tolerance = _consistency_tolerance(background)
        values = []
        selected = []
        full_geometry_flags = []
        for row in audit.get("ports", []):
            coarse = row.get("coarse", {})
            value = coarse.get("parent_restriction_relative_residual")
            if value is not None:
                values.append(float(value))
            if coarse.get("selected_boundary_padding") is not None:
                selected.append(float(coarse["selected_boundary_padding"]))
            full_geometry_flags.append(
                bool(coarse.get("full_geometry_material_assembly", False))
            )
        maximum = max(values, default=0.0)
        audit["coarse_parent_restriction_tolerance"] = float(tolerance)
        audit["maximum_coarse_parent_restriction_relative_residual"] = float(maximum)
        audit["coarse_parent_restriction_consistent"] = bool(maximum <= tolerance)
        audit["selected_boundary_padding"] = selected
        audit["full_geometry_material_assembly"] = bool(
            full_geometry_flags and all(full_geometry_flags)
        )
        if maximum > tolerance or not audit["full_geometry_material_assembly"]:
            raise RuntimeError(
                "boundary-conditioned longitudinal patch is not the full-geometry "
                "parent scalar restriction; maximum coarse consistency residual="
                f"{maximum:.3e}, tolerance={tolerance:.3e}. Refuse to apply the "
                "near-field defect."
            )
        out = dict(result)
        out["audit"] = audit
        return out

    module._correction = correction

    original_resolve = module._resolve_settings

    def resolve_settings(settings, background):
        original_resolve(settings, background)
        own = settings["BACKGROUND"].setdefault(
            "global_longitudinal_correction", {}
        )
        own.setdefault("coarse_consistency_tolerance", 1e-8)
        if isinstance(getattr(background, "background_config", None), dict):
            background.background_config.setdefault(
                "global_longitudinal_correction", {}
            )["coarse_consistency_tolerance"] = float(
                own["coarse_consistency_tolerance"]
            )

    module._resolve_settings = resolve_settings
    module._patch_consistency_installed = True
    return module


__all__ = ["coarse_parent_restriction_residual", "install"]
