"""Terminal-component longitudinal dissipative defect correction.

Uniform whole-domain scalar refinement is inefficient for the production source:
the physical terminal charge lives on a sub-millimetre conductor cross section,
while the nonlocal return path is already well represented by the coarse global
scalar solve.  Refining the complete domain therefore spends almost all DOFs far
from the only unresolved feature.

This adapter keeps the already-certified reactive longitudinal correction and
replaces the failed whole-domain dissipative reference by two independently
resolved terminal *self* defects per port:

1. split the physical nodal terminal charge q_target into its negative/feed and
   positive/return components (each has unit total magnitude);
2. solve each component on the current global scalar block, so its far field and
   material environment are exactly the production problem;
3. restrict that component potential to a full-geometry Cartesian patch whose
   coarse nodes are a strict subset of the global grid;
4. retain every coarse patch node and insert fine nodes only around the selected
   terminal contact; and
5. add D_vol(fine)-D_vol(coarse) for that terminal component.

The feed/return cross interaction is deliberately left on the global grid.  It is
a smooth nonlocal term; only the two singular/near-field terminal self energies
are replaced.  Re(delta Z_pp) is set equal to the summed D_vol defect and D_out
is unchanged, preserving the production power identity.  A still finer
terminal-only solve certifies each applied defect before any expensive Maxwell
mesh Gate is allowed to run.
"""
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import unified_longitudinal_patch_consistency as _consistency
from . import unified_terminal_longitudinal_refinement as _terminal_refinement
from .unified_charge_regularized_source import terminal_charge_target
from .unified_gradient_block_maxwell import build_gradient_block, gradient_operator


_MODEL = "global_terminal_component_longitudinal_dissipative_defect_v1"
_TERMINAL_NAMES = ("feed", "return")


def _geometry_key(module, geometry):
    helper = getattr(module, "_geometry_key", None)
    return helper(geometry) if callable(helper) else repr(geometry)


def _config(background):
    root = copy.deepcopy(dict(getattr(background, "background_config", {}) or {}))
    own = copy.deepcopy(dict(root.get("terminal_dissipative_reference", {}) or {}))
    long_cfg = copy.deepcopy(dict(root.get("global_longitudinal_correction", {}) or {}))
    mesh = copy.deepcopy(dict(root.get("mesh_check", {}) or {}))
    base_cells = float(long_cfg.get("terminal_cells_per_support", 1.6))
    reference_cells = float(own.get("reference_cells_per_support", 2.0 * base_cells))
    validation_ratio = float(own.get("validation_ratio", 0.75))
    if not np.isfinite(reference_cells) or reference_cells < 2.0:
        raise ValueError("terminal dissipative reference_cells_per_support must be >= 2")
    if not np.isfinite(validation_ratio) or not 0.0 < validation_ratio < 1.0:
        raise ValueError("terminal dissipative validation_ratio must lie in (0,1)")
    validation_cells = reference_cells / validation_ratio
    return {
        "enabled": bool(own.get("enabled", True)),
        "reference_cells_per_support": reference_cells,
        "validation_cells_per_support": validation_cells,
        "validation_ratio": validation_ratio,
        "core_padding_factor": float(
            own.get(
                "core_padding_factor",
                long_cfg.get("terminal_core_padding_factor", 1.5),
            )
        ),
        "max_cells": int(
            own.get("max_cells", long_cfg.get("terminal_patch_max_cells", 575000))
        ),
        "relative_tolerance": float(
            own.get("relative_tolerance", mesh.get("relative_tolerance", 1e-1))
        ),
        "coarse_consistency_tolerance": float(
            own.get(
                "coarse_consistency_tolerance",
                long_cfg.get("coarse_consistency_tolerance", 1e-8),
            )
        ),
    }


def _component_charge(q_target, terminal):
    q = np.asarray(q_target, float).reshape(-1)
    t = int(terminal)
    if t == 0:
        out = np.minimum(q, 0.0)
        magnitude = float(-np.sum(out))
    elif t == 1:
        out = np.maximum(q, 0.0)
        magnitude = float(np.sum(out))
    else:
        raise ValueError("terminal component index must be 0 or 1")
    if not np.isfinite(magnitude) or magnitude <= np.finfo(float).tiny:
        raise RuntimeError("terminal component charge has zero support")
    # terminal_charge_target already normalizes each sign to unit magnitude.  Do
    # not silently renormalize here; a failed invariant must remain visible.
    if abs(magnitude - 1.0) > 5e-12:
        raise RuntimeError(
            "terminal component charge lost its unit physical normalization: "
            f"terminal={t}, magnitude={magnitude:.16e}"
        )
    return out


def _parent_component_potential(module, background, geometry, port, terminal):
    context = module._cached_context(background, geometry)
    if context is None:
        context = background.geometry_context(geometry, assemble_thermal=False)
        module._remember_context(background, geometry, context)
    block = build_gradient_block(background, context, check_topology=True)
    coil = context.geometry.coils[int(port)]
    q_target, meta = terminal_charge_target(background, coil)
    component = _component_charge(q_target, terminal)
    rhs = (-1j * float(background.omega)) * np.asarray(component[1:], complex)
    potential = np.asarray(block.factor.solve(rhs), complex).reshape(-1)
    residual = float(
        np.linalg.norm(rhs - block.scalar_matrix @ potential)
        / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    )
    full = np.zeros((background.nx + 1) * (background.ny + 1) * (background.nz + 1), complex)
    full[1:] = potential
    return full, {
        "global_component_scalar_relative_residual": residual,
        "terminal_charge_support_nodes": int(meta.get("terminal_charge_support_nodes", 0)),
        "terminal_charge_component": _TERMINAL_NAMES[int(terminal)],
    }


def _component_state(
    module,
    parent,
    patch,
    geometry,
    port,
    terminal,
    parent_component_potential,
    *,
    phi=None,
    fine_step,
    certify_parent=False,
):
    # Pure scalar patches consume q_target directly; never build the redundant
    # compatible edge-space charge lift on these potentially large local grids.
    patch._sdfmpneo_scalar_charge_target_only = True
    context = patch.geometry_context(geometry, assemble_thermal=False)
    p = int(port)
    if p < 0 or p >= len(context.geometry.coils):
        raise AssertionError("terminal dissipative source port is invalid")
    q_target, charge_meta = terminal_charge_target(patch, context.geometry.coils[p])
    component = _component_charge(q_target, terminal)

    G = gradient_operator(patch, gauge_fixed=False)
    diagonal, sigma, _hs = module._volume_edge_diagonal(patch, context)
    scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
    scalar.sum_duplicates()
    scalar.eliminate_zeros()
    scalar_rhs = (-1j * float(patch.omega)) * np.asarray(component, complex)

    boundary = module._boundary_node_mask(patch)
    interior = ~boundary
    restricted = _consistency._restricted_parent_potential(
        module, parent, parent_component_potential, patch
    )
    consistency = None
    if certify_parent:
        action = np.asarray(scalar[interior] @ restricted, complex).reshape(-1)
        residual = np.asarray(scalar_rhs[interior] - action, complex).reshape(-1)
        denominator = max(
            float(np.linalg.norm(scalar_rhs[interior])),
            float(np.linalg.norm(action)),
            np.finfo(float).tiny,
        )
        consistency = float(np.linalg.norm(residual) / denominator)
        # A certified coarse patch is exactly the restricted parent solution;
        # using it directly avoids an unnecessary coarse LU for every terminal.
        phi_nodes = restricted
        solve_residual = consistency
    else:
        phi_nodes = np.zeros(G.shape[1], complex)
        phi_nodes[boundary] = module._boundary_values(
            parent, parent_component_potential, patch, boundary
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
        solve_residual = float(
            np.linalg.norm(rhs_i - Sii @ phi_nodes[interior])
            / max(float(np.linalg.norm(rhs_i)), np.finfo(float).tiny)
        )

    field = np.asarray(G @ phi_nodes, complex).reshape(-1)
    abs2 = np.abs(field) ** 2
    edge_loss = np.asarray(patch.edge_cell_hodge @ np.asarray(sigma, float), float).reshape(-1)
    d_vol = float(np.dot(edge_loss, abs2))
    z_reaction = complex(-np.asarray(component, float) @ phi_nodes)
    q_cells = np.asarray(
        0.5
        * np.asarray(sigma, float)
        * np.asarray(patch.edge_cell_hodge.T @ abs2).reshape(-1),
        float,
    )
    modal = None
    if phi is not None:
        local_phi = module._interpolate_cell_basis(parent, patch, phi)
        modal = np.asarray(2.0 * (local_phi.T @ q_cells), float)

    return {
        "z_reaction": z_reaction,
        "d_vol": d_vol,
        "modal_h": modal,
        "scalar_relative_residual": float(solve_residual),
        "parent_restriction_relative_residual": (
            None if consistency is None else float(consistency)
        ),
        "n_cells": int(patch.n_cells),
        "scalar_dofs": int(np.count_nonzero(interior)),
        "fine_step": float(fine_step),
        "source_port": p,
        "terminal": int(terminal),
        "terminal_name": _TERMINAL_NAMES[int(terminal)],
        "terminal_charge_support_nodes": int(
            charge_meta.get("terminal_charge_support_nodes", 0)
        ),
        "terminal_charge_contact_length": float(
            charge_meta.get("terminal_charge_contact_length", 0.0)
        ),
    }


def _terminal_axes(module, background, geometry, port, terminal, coarse_axes, cells):
    boxes, contact, coil = _terminal_refinement._contact_boxes(
        module, background, geometry, int(port)
    )
    t = int(terminal)
    if t < 0 or t >= len(boxes):
        raise AssertionError("terminal contact box index is invalid")
    requested_upper = float(module._config(background)["fine_step"])
    steps = _terminal_refinement._directional_axis_steps(
        background, coil, float(cells), requested_upper
    )
    cfg = _config(background)
    halo = float(cfg["core_padding_factor"]) * max(
        float(coil.conductor_width), float(coil.conductor_thickness)
    )
    lo, hi = boxes[t]
    axes = []
    for axis in range(3):
        intervals = ((float(lo[axis] - halo), float(hi[axis] + halo)),)
        axes.append(
            _terminal_refinement._subdivide_axis(
                coarse_axes[axis], intervals, float(steps[axis])
            )
        )
    return tuple(axes), np.asarray(steps, float), boxes[t], float(contact)


def _select_coarse_patch(module, background, geometry, port, terminal, parent_phi, *, phi=None):
    long_cfg = module._config(background)
    tolerance = float(_config(background)["coarse_consistency_tolerance"])
    attempts = []
    for padding in _consistency._padding_candidates(module, background):
        cfg = dict(long_cfg)
        cfg["boundary_padding"] = float(padding)
        coarse_axes, _center, _aabb_half, full_geometry = _consistency._full_geometry_patch_axes(
            module, background, geometry, int(port), cfg
        )
        coarse = _consistency._make_full_patch_background(
            module,
            background,
            coarse_axes,
            fine_step=module._background_step(background),
        )
        state = _component_state(
            module,
            background,
            coarse,
            full_geometry,
            int(port),
            int(terminal),
            parent_phi,
            phi=phi,
            fine_step=module._background_step(background),
            certify_parent=True,
        )
        value = float(state["parent_restriction_relative_residual"])
        attempts.append(
            {
                "boundary_padding": float(padding),
                "parent_restriction_relative_residual": value,
            }
        )
        print(
            "terminal dissipative coarse consistency: "
            f"port={int(port) + 1}, terminal={_TERMINAL_NAMES[int(terminal)]}, "
            f"padding={float(padding):.6g}m, residual={value:.3e}, "
            f"accepted={'yes' if value <= tolerance else 'no'}",
            flush=True,
        )
        if value <= tolerance:
            state = dict(state)
            state["selected_boundary_padding"] = float(padding)
            state["boundary_selection_attempts"] = attempts.copy()
            return coarse_axes, full_geometry, state
    raise RuntimeError(
        "no terminal-component coarse patch reproduces the parent scalar restriction; "
        f"port={int(port) + 1}, terminal={_TERMINAL_NAMES[int(terminal)]}, "
        f"attempts={attempts!r}"
    )


def _terminal_reference(
    module,
    background,
    geometry,
    port,
    terminal,
    *,
    cells_per_support,
    phi=None,
):
    parent_phi, parent_meta = _parent_component_potential(
        module, background, geometry, int(port), int(terminal)
    )
    coarse_axes, full_geometry, coarse = _select_coarse_patch(
        module,
        background,
        geometry,
        int(port),
        int(terminal),
        parent_phi,
        phi=phi,
    )
    axes, steps, box, contact = _terminal_axes(
        module,
        background,
        full_geometry,
        int(port),
        int(terminal),
        coarse_axes,
        float(cells_per_support),
    )
    n_cells = int(np.prod([len(axis) - 1 for axis in axes], dtype=np.int64))
    budget = int(_config(background)["max_cells"])
    print(
        "terminal dissipative refinement: "
        f"port={int(port) + 1}, terminal={_TERMINAL_NAMES[int(terminal)]}, "
        f"cells_per_support={float(cells_per_support):.4g}, "
        f"axis_steps={[float(v) for v in steps]}, cells={n_cells}, budget={budget}",
        flush=True,
    )
    if n_cells > budget:
        raise RuntimeError(
            "terminal dissipative scalar patch exceeds the certified cell budget: "
            f"port={int(port) + 1}, terminal={_TERMINAL_NAMES[int(terminal)]}, "
            f"cells={n_cells}, limit={budget}, axis_steps={steps.tolist()}"
        )
    patch = _consistency._make_full_patch_background(
        module,
        background,
        axes,
        fine_step=float(np.min(steps)),
    )
    refined = _component_state(
        module,
        background,
        patch,
        full_geometry,
        int(port),
        int(terminal),
        parent_phi,
        phi=phi,
        fine_step=float(np.min(steps)),
        certify_parent=False,
    )
    delta_d = float(refined["d_vol"] - coarse["d_vol"])
    delta_modal = None
    if phi is not None:
        delta_modal = np.asarray(refined["modal_h"] - coarse["modal_h"], float)
    return {
        "port": int(port),
        "terminal": int(terminal),
        "terminal_name": _TERMINAL_NAMES[int(terminal)],
        "cells_per_support": float(cells_per_support),
        "axis_steps": steps.tolist(),
        "terminal_box": {
            "lo": np.asarray(box[0], float).tolist(),
            "hi": np.asarray(box[1], float).tolist(),
        },
        "terminal_contact_length": float(contact),
        "coarse": coarse,
        "refined": refined,
        "delta_d_vol": delta_d,
        "delta_modal_h": delta_modal,
        **parent_meta,
    }


def _relative_defect(reference, validation):
    dr = float(reference["delta_d_vol"])
    dv = float(validation["delta_d_vol"])
    scale = max(
        abs(dv),
        abs(dr),
        abs(float(validation["refined"]["d_vol"])),
        abs(float(validation["coarse"]["d_vol"])),
        np.finfo(float).tiny,
    )
    return float(abs(dr - dv) / scale)


def install(module, implementation_module):
    if bool(getattr(module, "_terminal_dissipative_defect_installed", False)):
        return module
    required = (
        "_config",
        "_background_step",
        "_cached_context",
        "_remember_context",
        "_volume_edge_diagonal",
        "_boundary_node_mask",
        "_boundary_values",
        "_interpolate_cell_basis",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "terminal dissipative defect requires longitudinal hooks: " + ", ".join(missing)
        )

    original_resolve = module._resolve_settings
    original_correction = module._correction
    original_audit = module.audit_reference_convergence

    def resolve_settings(settings, background):
        original_resolve(settings, background)
        bg = settings["BACKGROUND"]
        # The uniform 9->6.75-mm whole-domain reference has been falsified by
        # production evidence. Keep its implementation available for diagnostics
        # but never apply it in production once the terminal-local defect is on.
        global_cfg = bg.setdefault("global_dissipative_reference", {})
        global_cfg["enabled"] = False
        long_cfg = dict(bg.get("global_longitudinal_correction", {}) or {})
        mesh_cfg = dict(bg.get("mesh_check", {}) or {})
        own = bg.setdefault("terminal_dissipative_reference", {})
        base_cells = float(long_cfg.get("terminal_cells_per_support", 1.6))
        own.setdefault("enabled", True)
        own.setdefault("reference_cells_per_support", 2.0 * base_cells)
        own.setdefault("validation_ratio", 0.75)
        own.setdefault(
            "core_padding_factor",
            float(long_cfg.get("terminal_core_padding_factor", 1.5)),
        )
        own.setdefault(
            "max_cells",
            int(long_cfg.get("terminal_patch_max_cells", 575000)),
        )
        own.setdefault(
            "relative_tolerance",
            float(mesh_cfg.get("relative_tolerance", 1e-1)),
        )
        own.setdefault(
            "coarse_consistency_tolerance",
            float(long_cfg.get("coarse_consistency_tolerance", 1e-8)),
        )
        if isinstance(getattr(background, "background_config", None), dict):
            background.background_config["global_dissipative_reference"] = copy.deepcopy(
                global_cfg
            )
            background.background_config["terminal_dissipative_reference"] = copy.deepcopy(
                own
            )

    module._resolve_settings = resolve_settings

    def correction(background, geometry, *, phi=None):
        raw = original_correction(background, geometry, phi=phi)
        cfg = _config(background)
        audit = dict(raw.get("audit", {}))
        audit["terminal_dissipative_reference_model"] = _MODEL
        if not cfg["enabled"]:
            audit["terminal_dissipative_reference_enabled"] = False
            raw["audit"] = audit
            return raw

        n = len(background.coil_materials)
        delta = np.zeros(n, float)
        modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], n), float)
        rows = []
        reference_cells = float(cfg["reference_cells_per_support"])
        for p in range(n):
            terminal_rows = []
            for terminal in range(2):
                state = _terminal_reference(
                    module,
                    background,
                    geometry,
                    p,
                    terminal,
                    cells_per_support=reference_cells,
                    phi=phi,
                )
                delta[p] += float(state["delta_d_vol"])
                if modal is not None:
                    modal[:, p] += np.asarray(state["delta_modal_h"], float)
                terminal_rows.append(state)
            rows.append(
                {
                    "port": int(p),
                    "delta_d_vol": float(delta[p]),
                    "terminals": terminal_rows,
                }
            )

        raw["delta_z"] = np.asarray(raw["delta_z"], complex) + delta.astype(complex)
        raw["delta_d_vol"] = np.asarray(raw["delta_d_vol"], float) + delta
        if modal is not None:
            if raw.get("delta_modal_h") is None:
                raw["delta_modal_h"] = np.zeros_like(modal)
            raw["delta_modal_h"] = np.asarray(raw["delta_modal_h"], float) + modal
        audit.update(
            terminal_dissipative_reference_enabled=True,
            terminal_dissipative_reference_cells_per_support=reference_cells,
            terminal_dissipative_delta_d_vol=delta.tolist(),
            terminal_dissipative_delta_z_real=delta.tolist(),
            terminal_dissipative_ports=rows,
            terminal_dissipative_semantics=(
                "sum_of_feed_return_component_self_defects_cross_interaction_global"
            ),
        )
        raw["audit"] = audit
        return raw

    module._correction = correction

    def audit_reference_convergence(background, geometry):
        report = dict(original_audit(background, geometry))
        cfg = _config(background)
        if not cfg["enabled"]:
            return report
        # Do not spend additional scalar work when the independently certified
        # terminal reactive prerequisite is already invalid.
        if not bool(report.get("converged", False)):
            return report

        reference_cells = float(cfg["reference_cells_per_support"])
        validation_cells = float(cfg["validation_cells_per_support"])
        rows = []
        worst = 0.0
        max_residual = 0.0
        max_consistency = 0.0
        for p in range(len(background.coil_materials)):
            terminal_rows = []
            port_reference = 0.0
            port_validation = 0.0
            for terminal in range(2):
                reference = _terminal_reference(
                    module,
                    background,
                    geometry,
                    p,
                    terminal,
                    cells_per_support=reference_cells,
                    phi=None,
                )
                validation = _terminal_reference(
                    module,
                    background,
                    geometry,
                    p,
                    terminal,
                    cells_per_support=validation_cells,
                    phi=None,
                )
                error = _relative_defect(reference, validation)
                worst = max(worst, error)
                port_reference += float(reference["delta_d_vol"])
                port_validation += float(validation["delta_d_vol"])
                max_residual = max(
                    max_residual,
                    float(reference["refined"]["scalar_relative_residual"]),
                    float(validation["refined"]["scalar_relative_residual"]),
                )
                max_consistency = max(
                    max_consistency,
                    float(reference["coarse"]["parent_restriction_relative_residual"]),
                    float(validation["coarse"]["parent_restriction_relative_residual"]),
                )
                terminal_rows.append(
                    {
                        "terminal": int(terminal),
                        "terminal_name": _TERMINAL_NAMES[int(terminal)],
                        "reference_delta_d_vol": float(reference["delta_d_vol"]),
                        "validation_delta_d_vol": float(validation["delta_d_vol"]),
                        "relative_d_vol_defect_error": float(error),
                        "reference": reference,
                        "validation": validation,
                    }
                )
                print(
                    "terminal longitudinal dissipative defect Gate: "
                    f"port={p + 1}, terminal={_TERMINAL_NAMES[int(terminal)]}, "
                    f"Dvol={error:.3e}, ref={float(reference['delta_d_vol']):.6e}, "
                    f"val={float(validation['delta_d_vol']):.6e}",
                    flush=True,
                )
            rows.append(
                {
                    "port": int(p),
                    "reference_delta_d_vol": float(port_reference),
                    "validation_delta_d_vol": float(port_validation),
                    "terminals": terminal_rows,
                }
            )

        converged = bool(
            worst <= float(cfg["relative_tolerance"])
            and max_consistency <= float(cfg["coarse_consistency_tolerance"])
            and max_residual <= 1e-9
        )
        report["terminal_dissipative_reference"] = {
            "model": _MODEL,
            "reference_cells_per_support": reference_cells,
            "validation_cells_per_support": validation_cells,
            "relative_tolerance": float(cfg["relative_tolerance"]),
            "coarse_consistency_tolerance": float(cfg["coarse_consistency_tolerance"]),
            "maximum_relative_error": float(worst),
            "maximum_scalar_relative_residual": float(max_residual),
            "maximum_coarse_parent_restriction_relative_residual": float(max_consistency),
            "converged": converged,
            "samples": rows,
            "semantics": "terminal_component_self_only_cross_interaction_global",
        }
        report["maximum_terminal_dissipative_relative_error"] = float(worst)
        report["maximum_relative_error"] = max(
            float(report.get("maximum_relative_error", 0.0)), float(worst)
        )
        report["converged"] = bool(report.get("converged", False) and converged)
        report["model"] = (
            "global_boundary_conditioned_longitudinal_reactive_defect_v4+" + _MODEL
        )
        return report

    module.audit_reference_convergence = audit_reference_convergence
    module._MODEL = (
        "global_boundary_conditioned_longitudinal_reactive_defect_v4+" + _MODEL
    )
    module._terminal_dissipative_defect_installed = True
    return module


__all__ = ["_component_charge", "_config", "install"]
