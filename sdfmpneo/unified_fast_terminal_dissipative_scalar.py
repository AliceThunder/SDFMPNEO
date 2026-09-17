"""Fast terminal-local dissipative scalar references.

The expensive geometry-resolved sigma/epsilon edge-dual experiment did not
improve the certified dissipative h-convergence, while production timing showed
that its OBB assembly dominated wall time.  The terminal dissipative Gate should
isolate source/Galerkin resolution, not simultaneously redefine material
geometry.  Refined patches therefore inherit the *production parent* EM
sigma/epsilon field piecewise-constantly on parent cells and refine only the
selected terminal charge/Galerkin space.

Large refined Dirichlet systems still use the true-residual-certified fast
scalar solver.  Stage timings remain visible in production logs.
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp

from . import unified_terminal_dissipative_defect as _terminal
from .unified_charge_regularized_source import terminal_charge_target
from .unified_gradient_block_maxwell import gradient_operator
from .unified_fast_scalar_solve import solve_refined


_MATERIAL_SEMANTICS = "production_parent_piecewise_constant_complex_mass_v1"


def _as_geometry(module, geometry):
    if hasattr(geometry, "coils") and hasattr(geometry, "packages"):
        return geometry
    return module.UnifiedUWPTGeometry.from_mapping(geometry)


def _parent_context(module, parent, geometry):
    cached = module._cached_context(parent, geometry)
    if cached is not None:
        return cached
    context = parent.geometry_context(geometry, assemble_thermal=False)
    module._remember_context(parent, geometry, context)
    return context


def _parent_cell_ids(parent, patch):
    """Map every refined patch cell to its containing production parent cell."""
    ix = np.searchsorted(np.asarray(parent.x, float), np.asarray(patch.cell_axes[0], float), side="right") - 1
    iy = np.searchsorted(np.asarray(parent.y, float), np.asarray(patch.cell_axes[1], float), side="right") - 1
    iz = np.searchsorted(np.asarray(parent.z, float), np.asarray(patch.cell_axes[2], float), side="right") - 1
    ix = np.clip(ix, 0, int(parent.nx) - 1).astype(np.int64)
    iy = np.clip(iy, 0, int(parent.ny) - 1).astype(np.int64)
    iz = np.clip(iz, 0, int(parent.nz) - 1).astype(np.int64)
    ids = (
        (ix[:, None, None] * int(parent.ny) + iy[None, :, None]) * int(parent.nz)
        + iz[None, None, :]
    )
    out = np.asarray(ids, np.int64).reshape(-1)
    if out.shape != (int(patch.n_cells),):
        raise AssertionError("refined scalar parent-cell map has the wrong size")
    return out


def _prolong_parent_cell_values(parent, patch, values):
    source = np.asarray(values).reshape(-1)
    if source.shape != (int(parent.n_cells),):
        raise ValueError("parent material field has incompatible cell count")
    return np.asarray(source[_parent_cell_ids(parent, patch)]).reshape(-1)


def install(module):
    if bool(getattr(_terminal, "_fast_terminal_dissipative_scalar_installed", False)):
        return module

    original = _terminal._balanced_state
    original_reference = _terminal._terminal_reference

    def balanced_state(
        module_arg,
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
        if bool(certify_parent) or not bool(exact_refined):
            return original(
                module_arg,
                parent,
                patch,
                geometry,
                port,
                parent_potential,
                window,
                phi=phi,
                fine_step=fine_step,
                certify_parent=certify_parent,
                exact_refined=exact_refined,
            )

        total_started = time.perf_counter()
        patch._sdfmpneo_scalar_charge_target_only = True
        p = int(port)
        g = _as_geometry(module_arg, geometry)
        if p < 0 or p >= len(g.coils):
            raise AssertionError("fast terminal dissipative source port is invalid")

        t0 = time.perf_counter()
        q_target, charge_meta = terminal_charge_target(patch, g.coils[p])
        charge_seconds = time.perf_counter() - t0
        if abs(float(np.sum(q_target))) > 5e-14:
            raise FloatingPointError("fast terminal dissipative source is not balanced")

        t0 = time.perf_counter()
        G = gradient_operator(patch, gauge_fixed=False)
        gradient_seconds = time.perf_counter() - t0

        # Freeze material geometry to the certified production parent operator.
        # Every refined patch cell inherits sigma/epsilon from the production
        # parent cell that contains its center.  This isolates terminal source
        # resolution and avoids rebuilding expensive OBB material intersections.
        t0 = time.perf_counter()
        parent_context = _parent_context(module_arg, parent, g)
        parent_context_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        parent_sigma, parent_eps, _mu_inv, _k, _cap, _temperature = parent.cell_properties(
            parent_context, None, em=True
        )
        sigma = np.asarray(_prolong_parent_cell_values(parent, patch, parent_sigma), float)
        eps = np.asarray(_prolong_parent_cell_values(parent, patch, parent_eps), float)
        edge_loss = np.asarray(patch.edge_cell_hodge @ sigma, float).reshape(-1)
        edge_eps = np.asarray(patch.edge_cell_hodge @ eps, float).reshape(-1)
        diagonal = (
            1j * float(patch.omega) * edge_loss.astype(complex)
            - (float(patch.omega) ** 2) * edge_eps
        )
        material_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        scalar = (G.T @ sp.diags(np.asarray(diagonal, complex), format="csr") @ G).tocsc()
        scalar.sum_duplicates()
        scalar.eliminate_zeros()
        scalar_rhs = (-1j * float(patch.omega)) * np.asarray(q_target, complex)
        matrix_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        boundary = module_arg._boundary_node_mask(patch)
        interior = ~boundary
        phi_nodes = np.zeros(G.shape[1], complex)
        phi_nodes[boundary] = module_arg._boundary_values(
            parent, parent_potential, patch, boundary
        )
        Sii = scalar[interior][:, interior].tocsr()
        Sib = scalar[interior][:, boundary].tocsr()
        rhs_i = np.asarray(
            scalar_rhs[interior] - Sib @ phi_nodes[boundary], complex
        ).reshape(-1)
        restricted = _terminal._consistency._restricted_parent_potential(
            module_arg, parent, parent_potential, patch
        )
        x0 = np.asarray(restricted[interior], complex).reshape(-1)
        boundary_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        solved, solve_residual, solver_label = solve_refined(
            Sii, rhs_i, parent=parent, patch=patch, x0=x0
        )
        solve_seconds = time.perf_counter() - t0
        phi_nodes[interior] = solved

        t0 = time.perf_counter()
        field = np.asarray(G @ phi_nodes, complex).reshape(-1)
        abs2 = np.abs(field) ** 2
        # Same production legacy loss operator as the scalar matrix above.
        q_cells = np.asarray(
            0.5 * sigma * np.asarray(patch.edge_cell_hodge.T @ abs2, float).reshape(-1),
            float,
        )
        mask = _terminal._cell_mask(patch, window)
        local_q = np.where(mask, q_cells, 0.0)
        local_d = float(2.0 * np.sum(local_q))
        full_d = float(np.dot(edge_loss, abs2))
        modal = None
        if phi is not None:
            local_phi = module_arg._interpolate_cell_basis(parent, patch, phi)
            modal = np.asarray(2.0 * (local_phi.T @ local_q), float)
        contraction_seconds = time.perf_counter() - t0
        total_seconds = time.perf_counter() - total_started

        print(
            "terminal scalar stages: "
            f"cells={patch.n_cells}, dofs={int(np.count_nonzero(interior))}, "
            f"parent_context={parent_context_seconds:.1f}s, charge={charge_seconds:.1f}s, "
            f"gradient={gradient_seconds:.1f}s, material_prolong={material_seconds:.1f}s, "
            f"matrix={matrix_seconds:.1f}s, boundary={boundary_seconds:.1f}s, "
            f"solve={solve_seconds:.1f}s, contraction={contraction_seconds:.1f}s, "
            f"total={total_seconds:.1f}s",
            flush=True,
        )

        return {
            "local_d_vol": local_d,
            "full_d_vol": full_d,
            "modal_h": modal,
            "scalar_relative_residual": float(solve_residual),
            "parent_restriction_relative_residual": None,
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
            "material_semantics": _MATERIAL_SEMANTICS,
            "balanced_terminal_charge": True,
            "scalar_solver": solver_label,
            "stage_seconds": {
                "parent_context": float(parent_context_seconds),
                "charge": float(charge_seconds),
                "gradient": float(gradient_seconds),
                "material_prolong": float(material_seconds),
                "matrix": float(matrix_seconds),
                "boundary": float(boundary_seconds),
                "solve": float(solve_seconds),
                "contraction": float(contraction_seconds),
                "total": float(total_seconds),
            },
        }

    def terminal_reference(*args, **kwargs):
        result = dict(original_reference(*args, **kwargs))
        refined = result.get("refined")
        if isinstance(refined, dict):
            result["refined_material_semantics"] = refined.get(
                "material_semantics", _MATERIAL_SEMANTICS
            )
        return result

    _terminal._balanced_state = balanced_state
    _terminal._terminal_reference = terminal_reference
    _terminal._fast_terminal_dissipative_scalar_installed = True
    return module


__all__ = [
    "_MATERIAL_SEMANTICS",
    "_parent_cell_ids",
    "_prolong_parent_cell_values",
    "install",
]
