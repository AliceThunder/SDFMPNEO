"""Certification/adaptation layer for boundary-conditioned scalar patches.

The near-field defect is valid only when its coarse patch is literally the
parent global scalar problem restricted to the shared Cartesian subgrid.  The
parent potential is therefore substituted into every coarse-patch interior
equation.  If the configured artificial boundary includes omitted external
geometry, the patch is deterministically shrunk while retaining the same global
Dirichlet trace.  No defect is accepted unless the parent-restriction residual
meets a strict independent tolerance.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
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
    local_geometry,
    global_potential,
):
    """Return the parent-potential residual in the patch interior equations."""
    context = patch.geometry_context(local_geometry, assemble_thermal=False)
    B = np.asarray(patch.rhs_matrix(context), complex)
    if B.shape[1] != 1:
        raise AssertionError("longitudinal consistency patch must have one port")
    rhs = B[:, 0]
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
    values = [configured, max(floor, 0.5 * configured), max(floor, 0.25 * configured), floor]
    out = []
    for value in values:
        value = float(value)
        if not any(abs(value - old) <= 1e-13 * max(abs(value), abs(old), 1.0) for old in out):
            out.append(value)
    return out


def install(module):
    if bool(getattr(module, "_patch_consistency_installed", False)):
        return module

    original_patch_state = module._patch_state

    def patch_state(parent, patch, local_geometry, global_potential, *, phi=None, fine_step):
        state = original_patch_state(
            parent,
            patch,
            local_geometry,
            global_potential,
            phi=phi,
            fine_step=fine_step,
        )
        parent_step = module._background_step(parent)
        step = float(fine_step)
        if abs(step - parent_step) <= 1e-13 * max(abs(step), abs(parent_step), 1.0):
            consistency = coarse_parent_restriction_residual(
                module,
                parent,
                patch,
                local_geometry,
                global_potential,
            )
        else:
            consistency = None
        state = dict(state)
        state["parent_restriction_relative_residual"] = (
            None if consistency is None else float(consistency)
        )
        return state

    module._patch_state = patch_state

    # The default 40 mm patch inherited from the Maxwell local-self box can
    # overlap the other package even though the physical OBBs do not overlap.
    # Since this scalar patch has exact global Dirichlet data, large padding is
    # not a physical requirement.  Choose the largest deterministic patch that
    # is actually the parent scalar restriction.
    original_port_states = module._port_states

    def port_states(background, geometry, port, global_potential, target_steps, *, phi=None):
        del original_port_states  # implementation below intentionally controls the chosen patch
        base_cfg = module._config(background)
        tolerance = _consistency_tolerance(background)
        global_geometry, local_geometry = module._single_port_geometry(geometry, port)
        package = global_geometry.packages[int(port)]
        attempts = []
        for padding in _padding_candidates(module, background):
            cfg = dict(base_cfg)
            cfg["boundary_padding"] = float(padding)
            coarse_axes, center, aabb_half = module._patch_axes(background, package, cfg)
            coarse = module._make_patch_background(
                background,
                port,
                coarse_axes,
                fine_step=module._background_step(background),
            )
            coarse_state = module._patch_state(
                background,
                coarse,
                local_geometry,
                global_potential,
                phi=phi,
                fine_step=module._background_step(background),
            )
            consistency_value = coarse_state.get("parent_restriction_relative_residual")
            consistency_value = float("inf") if consistency_value is None else float(consistency_value)
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
                patch = module._make_patch_background(
                    background, port, axes, fine_step=float(step)
                )
                state = dict(
                    module._patch_state(
                        background,
                        patch,
                        local_geometry,
                        global_potential,
                        phi=phi,
                        fine_step=float(step),
                    )
                )
                state["selected_boundary_padding"] = float(padding)
                states[float(step)] = state
            return states

        raise RuntimeError(
            "no boundary-conditioned longitudinal patch reproduces the parent "
            "scalar restriction; attempts=" + repr(attempts)
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
        for row in audit.get("ports", []):
            coarse = row.get("coarse", {})
            value = coarse.get("parent_restriction_relative_residual")
            if value is not None:
                values.append(float(value))
            if coarse.get("selected_boundary_padding") is not None:
                selected.append(float(coarse["selected_boundary_padding"]))
        maximum = max(values, default=0.0)
        audit["coarse_parent_restriction_tolerance"] = float(tolerance)
        audit["maximum_coarse_parent_restriction_relative_residual"] = float(maximum)
        audit["coarse_parent_restriction_consistent"] = bool(maximum <= tolerance)
        audit["selected_boundary_padding"] = selected
        if maximum > tolerance:
            raise RuntimeError(
                "boundary-conditioned longitudinal patch is not the parent scalar "
                "restriction; maximum coarse consistency residual="
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
