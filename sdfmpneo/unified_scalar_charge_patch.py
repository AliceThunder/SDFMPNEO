"""Memory-bounded direct-charge solve for refined longitudinal scalar patches.

Production Maxwell truth uses the complete compatible edge source and certifies
``G.T S = q_target``.  A pure scalar reference needs only that nodal divergence:

    G.T B = -i omega q_target,
    -S.T (G phi) = -q_target.T phi.

Rebuilding the auxiliary graph-Laplacian edge lift on a 0.3--0.8 mm terminal
patch is therefore algebraically redundant and can nearly double peak sparse-LU
memory.  This adapter applies only to refined scalar-only patches.  Coarse parent
consistency and every real Maxwell solve keep the full compatible edge source.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import unified_longitudinal_patch_consistency as _consistency
from .unified_charge_regularized_source import terminal_charge_target


def install(module):
    if bool(getattr(module, "_scalar_direct_charge_patch_installed", False)):
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
        # Parent-consistency must continue to exercise the complete production
        # source realization.  Only a genuinely refined scalar reference uses
        # the direct nodal-charge identity.
        if bool(certify_parent) or float(fine_step) >= module._background_step(parent) * (1.0 - 1e-12):
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

        patch._sdfmpneo_scalar_charge_target_only = True
        context = patch.geometry_context(geometry, assemble_thermal=False)
        p = int(source_port)
        if p < 0 or p >= len(context.geometry.coils):
            raise AssertionError("direct-charge scalar patch source port is invalid")

        q_target, charge_meta = terminal_charge_target(
            patch, context.geometry.coils[p]
        )
        G = module.gradient_operator(patch, gauge_fixed=False)
        if q_target.shape != (G.shape[1],):
            raise AssertionError("direct-charge scalar target dimension mismatch")
        diagonal, sigma, _hs = module._volume_edge_diagonal(patch, context)
        scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
        scalar.sum_duplicates()
        scalar.eliminate_zeros()
        scalar_rhs = (-1j * float(patch.omega)) * np.asarray(q_target, complex)

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
        # Exact compatible identity: S.T G phi = (G.T S).T phi = q_target.T phi.
        z_reaction = complex(-np.asarray(q_target, float) @ phi_nodes)
        edge_loss = np.asarray(patch.edge_cell_hodge @ sigma, float).reshape(-1)
        d_vol = float(np.dot(edge_loss, abs2))
        q_cells = np.asarray(
            0.5
            * sigma
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
            "scalar_relative_residual": residual,
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
        }

    _consistency._full_patch_state = full_patch_state
    module._scalar_direct_charge_patch_installed = True
    return module


__all__ = ["install"]
