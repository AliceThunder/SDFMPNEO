"""Balanced full-port terminal-local dissipative defect correction.

The physical terminal charge is globally balanced.  Splitting it into isolated
feed/return net-charge scalar problems would introduce an implicit compensation
at the removed gauge node and is therefore not an admissible production truth.
This module instead keeps the complete balanced q_target in every solve and
localizes only the *energy contraction*.

For each port and each terminal contact:

1. solve/restrict the complete balanced global scalar potential;
2. choose a terminal energy window whose faces are parent coarse-grid nodes;
3. keep the complete global geometry/material problem and complete balanced
   q_target, but insert extra Cartesian nodes only around the selected terminal;
4. integrate Joule heat only inside that fixed terminal window; and
5. add D_local(fine)-D_local(coarse).

Feed and return windows must be disjoint, so their defects can be summed without
double counting.  The smooth feed/return cross field is still present in each
balanced solve and in the local energy where physically relevant; only the
unresolved terminal neighbourhood is replaced.  The refined local reference uses
geometry-resolved sigma/epsilon edge-dual mass, while the coarse baseline is the
exact restriction of the production scalar operator.  Thus the correction also
removes the local cut-cell material error without perturbing the certified full
Maxwell operator.

Re(delta Z_pp) is set exactly equal to the summed local D_vol defect, D_out is
unchanged, and modal Joule heat receives the same local fine-minus-coarse
contraction.  A still finer terminal-local solve independently certifies every
applied defect before expensive Maxwell Gates are allowed to run.
"""
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import unified_longitudinal_patch_consistency as _consistency
from . import unified_terminal_longitudinal_refinement as _terminal_refinement
from .unified_charge_regularized_source import terminal_charge_target
from .unified_gradient_block_maxwell import gradient_operator
from .unified_resolved_admittance_hodge import _build_permittivity_weights
from .unified_resolved_conductive_hodge import _build_conductivity_hodge


_MODEL = "global_balanced_terminal_local_longitudinal_dissipative_defect_v2"
_TERMINAL_NAMES = ("feed", "return")


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
    return {
        "enabled": bool(own.get("enabled", True)),
        "reference_cells_per_support": reference_cells,
        "validation_cells_per_support": reference_cells / validation_ratio,
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


def _snap_window(coarse_axes, box, halo):
    lo, hi = (np.asarray(box[0], float), np.asarray(box[1], float))
    out_lo = np.empty(3, float)
    out_hi = np.empty(3, float)
    for axis_index, axis_values in enumerate(coarse_axes):
        axis = np.asarray(axis_values, float)
        target_lo = float(lo[axis_index] - halo)
        target_hi = float(hi[axis_index] + halo)
        left = max(0, int(np.searchsorted(axis, target_lo, side="right")) - 1)
        right_node = min(
            len(axis) - 1,
            int(np.searchsorted(axis, target_hi, side="left")),
        )
        if right_node <= left:
            right_node = min(len(axis) - 1, left + 1)
        if right_node <= left:
            raise RuntimeError("terminal dissipative energy window has no coarse cell")
        out_lo[axis_index] = float(axis[left])
        out_hi[axis_index] = float(axis[right_node])
    return out_lo, out_hi


def _windows_disjoint(first, second):
    lo0, hi0 = first
    lo1, hi1 = second
    return bool(np.any(np.minimum(hi0, hi1) <= np.maximum(lo0, lo1) + 1e-14))


def _cell_mask(background, window):
    lo, hi = (np.asarray(window[0], float), np.asarray(window[1], float))
    centers = np.asarray(background.cell_centers, float)
    tol = 64.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(np.r_[lo, hi]))), 1.0
    )
    mask = np.all(centers >= lo[None, :] - tol, axis=1) & np.all(
        centers <= hi[None, :] + tol, axis=1
    )
    if not np.any(mask):
        raise RuntimeError("terminal dissipative energy window contains no cells")
    return mask


def _exact_refined_mass(background, context):
    conductivity_hodge, edge_loss = _build_conductivity_hodge(background, context)
    exact_eps, legacy_eps, _eps_meta = _build_permittivity_weights(background, context)
    sigma, *_ = background.cell_properties(context, None, em=True)
    legacy_loss = np.asarray(
        background.edge_cell_hodge @ np.asarray(sigma, float), float
    ).reshape(-1)
    return conductivity_hodge, edge_loss, exact_eps, legacy_eps, legacy_loss


def _balanced_state(
    module,
    parent,
    patch,
    geometry,
    port,
    parent_potential,
    window,
    *,
    phi=None,
    fine_step,
    certify_parent=False,
    exact_refined=False,
):
    patch._sdfmpneo_scalar_charge_target_only = True
    context = patch.geometry_context(geometry, assemble_thermal=False)
    p = int(port)
    if p < 0 or p >= len(context.geometry.coils):
        raise AssertionError("terminal dissipative source port is invalid")
    q_target, charge_meta = terminal_charge_target(patch, context.geometry.coils[p])
    if abs(float(np.sum(q_target))) > 5e-14:
        raise FloatingPointError("terminal dissipative scalar source is not globally balanced")

    G = gradient_operator(patch, gauge_fixed=False)
    diagonal, sigma, _hs = module._volume_edge_diagonal(patch, context)
    conductivity_hodge = patch.edge_cell_hodge @ sp.diags(np.asarray(sigma, float), format="csr")
    edge_loss = np.asarray(patch.edge_cell_hodge @ np.asarray(sigma, float), float).reshape(-1)
    material_semantics = "production_legacy_complex_mass"
    if exact_refined:
        conductivity_hodge, edge_loss, exact_eps, legacy_eps, legacy_loss = _exact_refined_mass(
            patch, context
        )
        diagonal = np.asarray(diagonal, complex) + (
            1j * float(patch.omega) * (edge_loss - legacy_loss)
            - (float(patch.omega) ** 2) * (exact_eps - legacy_eps)
        )
        material_semantics = "exact_edge_dual_sigma_epsilon"

    scalar = (G.T @ sp.diags(np.asarray(diagonal, complex), format="csr") @ G).tocsc()
    scalar.sum_duplicates()
    scalar.eliminate_zeros()
    scalar_rhs = (-1j * float(patch.omega)) * np.asarray(q_target, complex)

    boundary = module._boundary_node_mask(patch)
    interior = ~boundary
    restricted = _consistency._restricted_parent_potential(
        module, parent, parent_potential, patch
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
        phi_nodes = restricted
        solve_residual = consistency
    else:
        phi_nodes = np.zeros(G.shape[1], complex)
        phi_nodes[boundary] = module._boundary_values(
            parent, parent_potential, patch, boundary
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
    q_cells = np.asarray(0.5 * (conductivity_hodge.T @ abs2), float).reshape(-1)
    mask = _cell_mask(patch, window)
    local_q = np.where(mask, q_cells, 0.0)
    local_d = float(2.0 * np.sum(local_q))
    full_d = float(np.dot(edge_loss, abs2))
    modal = None
    if phi is not None:
        local_phi = module._interpolate_cell_basis(parent, patch, phi)
        modal = np.asarray(2.0 * (local_phi.T @ local_q), float)

    return {
        "local_d_vol": local_d,
        "full_d_vol": full_d,
        "modal_h": modal,
        "scalar_relative_residual": float(solve_residual),
        "parent_restriction_relative_residual": (
            None if consistency is None else float(consistency)
        ),
        "n_cells": int(patch.n_cells),
        "scalar_dofs": int(np.count_nonzero(interior)),
        "fine_step": float(fine_step),
        "source_port": p,
        "terminal_charge_support_nodes": int(
            charge_meta.get("terminal_charge_support_nodes", 0)
        ),
        "terminal_charge_contact_length": float(
            charge_meta.get("terminal_charge_contact_length", 0.0)
        ),
        "energy_window_lo": np.asarray(window[0], float).tolist(),
        "energy_window_hi": np.asarray(window[1], float).tolist(),
        "energy_window_cells": int(np.count_nonzero(mask)),
        "material_semantics": material_semantics,
        "balanced_terminal_charge": True,
    }


def _terminal_axes(module, background, geometry, port, terminal, coarse_axes, cells):
    boxes, contact, coil = _terminal_refinement._contact_boxes(
        module, background, geometry, int(port)
    )
    t = int(terminal)
    requested_upper = float(module._config(background)["fine_step"])
    steps = _terminal_refinement._directional_axis_steps(
        background, coil, float(cells), requested_upper
    )
    cfg = _config(background)
    halo = float(cfg["core_padding_factor"]) * max(
        float(coil.conductor_width), float(coil.conductor_thickness)
    )
    lo, hi = boxes[t]
    axes = tuple(
        _terminal_refinement._subdivide_axis(
            coarse_axes[axis],
            ((float(lo[axis] - halo), float(hi[axis] + halo)),),
            float(steps[axis]),
        )
        for axis in range(3)
    )
    return axes, np.asarray(steps, float), boxes, float(contact), float(halo)


def _select_coarse_patch(module, background, geometry, port, parent_potential, *, phi=None):
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
        # Use the first contact only to certify the parent operator; the scalar
        # equation/source is the same for both terminal energy windows.
        boxes, _contact, coil = _terminal_refinement._contact_boxes(
            module, background, full_geometry, int(port)
        )
        halo = float(_config(background)["core_padding_factor"]) * max(
            float(coil.conductor_width), float(coil.conductor_thickness)
        )
        window = _snap_window(coarse_axes, boxes[0], halo)
        state = _balanced_state(
            module,
            background,
            coarse,
            full_geometry,
            int(port),
            parent_potential,
            window,
            phi=phi,
            fine_step=module._background_step(background),
            certify_parent=True,
            exact_refined=False,
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
            f"port={int(port) + 1}, padding={float(padding):.6g}m, "
            f"residual={value:.3e}, accepted={'yes' if value <= tolerance else 'no'}",
            flush=True,
        )
        if value <= tolerance:
            state = dict(state)
            state["selected_boundary_padding"] = float(padding)
            state["boundary_selection_attempts"] = attempts.copy()
            return coarse_axes, full_geometry, coarse, state
    raise RuntimeError(
        "no balanced terminal dissipative coarse patch reproduces the parent scalar "
        f"restriction; port={int(port) + 1}, attempts={attempts!r}"
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
    prepared=None,
):
    if prepared is None:
        _context, potentials, _audit = module._global_scalar_potentials(background, geometry)
        parent_potential = np.asarray(potentials[:, int(port)], complex)
        coarse_axes, full_geometry, coarse_background, coarse_meta = _select_coarse_patch(
            module, background, geometry, int(port), parent_potential, phi=phi
        )
    else:
        parent_potential, coarse_axes, full_geometry, coarse_background, coarse_meta = prepared

    axes, steps, boxes, contact, halo = _terminal_axes(
        module,
        background,
        full_geometry,
        int(port),
        int(terminal),
        coarse_axes,
        float(cells_per_support),
    )
    windows = tuple(_snap_window(coarse_axes, box, halo) for box in boxes)
    if not _windows_disjoint(windows[0], windows[1]):
        raise RuntimeError(
            "feed/return terminal dissipative energy windows overlap on the parent coarse "
            f"grid for port={int(port) + 1}; refuse to double-count local Joule energy"
        )
    window = windows[int(terminal)]
    coarse = _balanced_state(
        module,
        background,
        coarse_background,
        full_geometry,
        int(port),
        parent_potential,
        window,
        phi=phi,
        fine_step=module._background_step(background),
        certify_parent=True,
        exact_refined=False,
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
    refined = _balanced_state(
        module,
        background,
        patch,
        full_geometry,
        int(port),
        parent_potential,
        window,
        phi=phi,
        fine_step=float(np.min(steps)),
        certify_parent=False,
        exact_refined=True,
    )
    delta_d = float(refined["local_d_vol"] - coarse["local_d_vol"])
    delta_modal = None
    if phi is not None:
        delta_modal = np.asarray(refined["modal_h"] - coarse["modal_h"], float)
    return {
        "port": int(port),
        "terminal": int(terminal),
        "terminal_name": _TERMINAL_NAMES[int(terminal)],
        "cells_per_support": float(cells_per_support),
        "axis_steps": steps.tolist(),
        "terminal_contact_length": float(contact),
        "energy_window": {
            "lo": np.asarray(window[0], float).tolist(),
            "hi": np.asarray(window[1], float).tolist(),
        },
        "coarse": coarse,
        "refined": refined,
        "delta_d_vol": delta_d,
        "delta_modal_h": delta_modal,
        "coarse_parent_metadata": coarse_meta,
        "balanced_full_port_source": True,
        "refined_material_semantics": "exact_edge_dual_sigma_epsilon",
        "prepared": (
            parent_potential,
            coarse_axes,
            full_geometry,
            coarse_background,
            coarse_meta,
        ),
    }


def _relative_defect(reference, validation):
    dr = float(reference["delta_d_vol"])
    dv = float(validation["delta_d_vol"])
    scale = max(
        abs(dv),
        abs(dr),
        abs(float(validation["refined"]["local_d_vol"])),
        abs(float(validation["coarse"]["local_d_vol"])),
        np.finfo(float).tiny,
    )
    return float(abs(dr - dv) / scale)


def install(module, implementation_module):
    del implementation_module  # retained in signature for a stable install surface.
    if bool(getattr(module, "_terminal_dissipative_defect_installed", False)):
        return module
    required = (
        "_config",
        "_background_step",
        "_global_scalar_potentials",
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
        own.setdefault("max_cells", int(long_cfg.get("terminal_patch_max_cells", 575000)))
        own.setdefault("relative_tolerance", float(mesh_cfg.get("relative_tolerance", 1e-1)))
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

    def _port_prepared(background, geometry, port, *, phi=None):
        _context, potentials, _audit = module._global_scalar_potentials(background, geometry)
        parent_potential = np.asarray(potentials[:, int(port)], complex)
        coarse_axes, full_geometry, coarse_background, coarse_meta = _select_coarse_patch(
            module,
            background,
            geometry,
            int(port),
            parent_potential,
            phi=phi,
        )
        return parent_potential, coarse_axes, full_geometry, coarse_background, coarse_meta

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
            prepared = _port_prepared(background, geometry, p, phi=phi)
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
                    prepared=prepared,
                )
                delta[p] += float(state["delta_d_vol"])
                if modal is not None:
                    modal[:, p] += np.asarray(state["delta_modal_h"], float)
                terminal_rows.append(state)
            rows.append(
                {"port": int(p), "delta_d_vol": float(delta[p]), "terminals": terminal_rows}
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
                "balanced_full_port_source_terminal_local_energy_defects"
            ),
        )
        raw["audit"] = audit
        return raw

    module._correction = correction

    def audit_reference_convergence(background, geometry):
        report = dict(original_audit(background, geometry))
        cfg = _config(background)
        if not cfg["enabled"] or not bool(report.get("converged", False)):
            return report
        reference_cells = float(cfg["reference_cells_per_support"])
        validation_cells = float(cfg["validation_cells_per_support"])
        rows = []
        worst = 0.0
        max_residual = 0.0
        max_consistency = 0.0
        for p in range(len(background.coil_materials)):
            prepared = _port_prepared(background, geometry, p, phi=None)
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
                    prepared=prepared,
                )
                validation = _terminal_reference(
                    module,
                    background,
                    geometry,
                    p,
                    terminal,
                    cells_per_support=validation_cells,
                    phi=None,
                    prepared=prepared,
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
            "semantics": "balanced_full_port_source_terminal_local_energy_defects",
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


__all__ = ["_config", "_snap_window", "_windows_disjoint", "install"]
