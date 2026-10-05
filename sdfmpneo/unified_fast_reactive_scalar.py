"""Fast linear algebra for refined reactive longitudinal scalar patches.

Physics/Gate semantics are unchanged.  Stage timings make topology/material
assembly visible alongside the certified scalar solve.
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp

from . import unified_longitudinal_patch_consistency as _consistency
from .unified_charge_regularized_source import terminal_charge_target
from .unified_fast_scalar_solve import solve_refined


def install(module):
    if bool(getattr(module, "_fast_reactive_scalar_installed", False)):
        return module

    original = _consistency._full_patch_state

    def full_patch_state(
        module_arg,
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
        refined = bool(
            not certify_parent
            and float(fine_step) < module._background_step(parent) * (1.0 - 1e-12)
        )
        if not refined:
            return original(
                module_arg,
                parent,
                patch,
                geometry,
                global_potential,
                source_port=source_port,
                phi=phi,
                fine_step=fine_step,
                certify_parent=certify_parent,
            )

        total_started = time.perf_counter()
        patch._sdfmpneo_scalar_charge_target_only = True
        t0 = time.perf_counter()
        context = patch.geometry_context(geometry, assemble_thermal=False)
        context_seconds = time.perf_counter() - t0
        p = int(source_port)
        if p < 0 or p >= len(context.geometry.coils):
            raise AssertionError("fast reactive scalar source port is invalid")

        t0 = time.perf_counter()
        q_target, charge_meta = terminal_charge_target(patch, context.geometry.coils[p])
        charge_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        G = module.gradient_operator(patch, gauge_fixed=False)
        gradient_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        diagonal, sigma, _hs = module._volume_edge_diagonal(patch, context)
        mass_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
        scalar.sum_duplicates()
        scalar.eliminate_zeros()
        scalar_rhs = (-1j * float(patch.omega)) * np.asarray(q_target, complex)
        matrix_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        boundary = module._boundary_node_mask(patch)
        interior = ~boundary
        phi_nodes = np.zeros(G.shape[1], complex)
        phi_nodes[boundary] = module._boundary_values(
            parent, global_potential, patch, boundary
        )
        Sii = scalar[interior][:, interior].tocsr()
        Sib = scalar[interior][:, boundary].tocsr()
        rhs_i = np.asarray(
            scalar_rhs[interior] - Sib @ phi_nodes[boundary], complex
        ).reshape(-1)
        restricted = _consistency._restricted_parent_potential(
            module, parent, global_potential, patch
        )
        x0 = np.asarray(restricted[interior], complex).reshape(-1)
        boundary_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        solved, residual, solver_label = solve_refined(
            Sii, rhs_i, parent=parent, patch=patch, x0=x0
        )
        solve_seconds = time.perf_counter() - t0
        phi_nodes[interior] = solved

        t0 = time.perf_counter()
        field = np.asarray(G @ phi_nodes, complex).reshape(-1)
        abs2 = np.abs(field) ** 2
        z_reaction = complex(-np.asarray(q_target, float) @ phi_nodes)
        sigma = np.asarray(sigma, float)
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
        contraction_seconds = time.perf_counter() - t0
        total_seconds = time.perf_counter() - total_started
        print(
            "reactive scalar stages: "
            f"cells={patch.n_cells}, dofs={int(np.count_nonzero(interior))}, "
            f"context={context_seconds:.1f}s, charge={charge_seconds:.1f}s, "
            f"gradient={gradient_seconds:.1f}s, mass={mass_seconds:.1f}s, "
            f"matrix={matrix_seconds:.1f}s, boundary={boundary_seconds:.1f}s, "
            f"solve={solve_seconds:.1f}s, contraction={contraction_seconds:.1f}s, "
            f"total={total_seconds:.1f}s",
            flush=True,
        )

        return {
            "z_reaction": z_reaction,
            "d_vol": d_vol,
            "modal_h": modal,
            "scalar_relative_residual": float(residual),
            "n_cells": int(patch.n_cells),
            "scalar_dofs": int(np.count_nonzero(interior)),
            "fine_step": float(fine_step),
            "parent_restriction_relative_residual": None,
            "full_geometry_material_assembly": True,
            "source_port": p,
            "scalar_charge_rhs_direct": True,
            "scalar_charge_support_nodes": int(
                charge_meta.get("terminal_charge_support_nodes", 0)
            ),
            "scalar_charge_contact_length": float(
                charge_meta.get("terminal_charge_contact_length", 0.0)
            ),
            "scalar_solver": solver_label,
            "stage_seconds": {
                "context": float(context_seconds),
                "charge": float(charge_seconds),
                "gradient": float(gradient_seconds),
                "mass": float(mass_seconds),
                "matrix": float(matrix_seconds),
                "boundary": float(boundary_seconds),
                "solve": float(solve_seconds),
                "contraction": float(contraction_seconds),
                "total": float(total_seconds),
            },
        }

    _consistency._full_patch_state = full_patch_state
    module._fast_reactive_scalar_installed = True
    return module


__all__ = ["install"]
