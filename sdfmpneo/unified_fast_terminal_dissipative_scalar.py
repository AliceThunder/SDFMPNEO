"""Accelerate terminal dissipative scalar references without changing physics.

The v32 terminal dissipative Gate intentionally uses finer 3.2 -> 4.267
cells/support patches than the reactive certificate.  Reusing the coarser
reactive fields would silently weaken that Gate, so this adapter keeps the
existing terminal patch geometry, exact sigma/epsilon mass, balanced q_target,
energy windows, source quadrature and 10% convergence test unchanged.

Only the linear algebra changes.  Large refined scalar Dirichlet systems are
solved by the same certified two-level scalar routine used by the fast reactive
path.  The true residual of the original matrix must remain <= 1e-9; otherwise
the routine falls back to the historical sparse direct solve.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import unified_terminal_dissipative_defect as _terminal
from .unified_charge_regularized_source import terminal_charge_target
from .unified_gradient_block_maxwell import gradient_operator
from .unified_fast_scalar_solve import solve_refined


def install(module):
    if bool(getattr(_terminal, "_fast_terminal_dissipative_scalar_installed", False)):
        return module

    original = _terminal._balanced_state

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
        # Preserve the production coarse-parent certificate byte-for-byte.  The
        # expensive path is only the refined terminal reference/validation.
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

        patch._sdfmpneo_scalar_charge_target_only = True
        context = patch.geometry_context(geometry, assemble_thermal=False)
        p = int(port)
        if p < 0 or p >= len(context.geometry.coils):
            raise AssertionError("fast terminal dissipative source port is invalid")
        q_target, charge_meta = terminal_charge_target(patch, context.geometry.coils[p])
        if abs(float(np.sum(q_target))) > 5e-14:
            raise FloatingPointError("fast terminal dissipative source is not balanced")

        G = gradient_operator(patch, gauge_fixed=False)
        diagonal, sigma, _hs = module_arg._volume_edge_diagonal(patch, context)
        conductivity_hodge, edge_loss, exact_eps, legacy_eps, legacy_loss = (
            _terminal._exact_refined_mass(patch, context)
        )
        diagonal = np.asarray(diagonal, complex) + (
            1j * float(patch.omega) * (edge_loss - legacy_loss)
            - (float(patch.omega) ** 2) * (exact_eps - legacy_eps)
        )
        scalar = (G.T @ sp.diags(np.asarray(diagonal, complex), format="csr") @ G).tocsc()
        scalar.sum_duplicates()
        scalar.eliminate_zeros()
        scalar_rhs = (-1j * float(patch.omega)) * np.asarray(q_target, complex)

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
        solved, solve_residual, solver_label = solve_refined(
            Sii, rhs_i, parent=parent, patch=patch, x0=x0
        )
        phi_nodes[interior] = solved

        field = np.asarray(G @ phi_nodes, complex).reshape(-1)
        abs2 = np.abs(field) ** 2
        q_cells = np.asarray(0.5 * (conductivity_hodge.T @ abs2), float).reshape(-1)
        mask = _terminal._cell_mask(patch, window)
        local_q = np.where(mask, q_cells, 0.0)
        local_d = float(2.0 * np.sum(local_q))
        full_d = float(np.dot(edge_loss, abs2))
        modal = None
        if phi is not None:
            local_phi = module_arg._interpolate_cell_basis(parent, patch, phi)
            modal = np.asarray(2.0 * (local_phi.T @ local_q), float)

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
            "material_semantics": "exact_edge_dual_sigma_epsilon",
            "balanced_terminal_charge": True,
            "scalar_solver": solver_label,
        }

    _terminal._balanced_state = balanced_state
    _terminal._fast_terminal_dissipative_scalar_installed = True
    return module


__all__ = ["install"]
