"""Use exact package/seawater edge-dual conductivity in scalar loss references.

This adapter deliberately does *not* replace the production full-Maxwell
operator.  The full field keeps the already-certified coarse global operator and
local transverse/cross correction.  Only the independently certified
longitudinal dissipative reference is upgraded:

* the current/background scalar state remains the legacy scalar component of the
  same operator used by full Maxwell;
* refined whole-domain reference/validation scalar states integrate seawater
  conductivity exactly over Cartesian edge-dual wedges cut by rotated insulating
  package OBBs.

Production therefore replaces the unresolved longitudinal self-loss part by a
common geometry-resolved reference while leaving mutual/transverse physics
untouched.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .unified_charge_regularized_source import terminal_charge_target
from .unified_gradient_block_maxwell import _edge_mass_diagonal, gradient_operator
from .unified_resolved_conductive_hodge import _MODEL as _HODGE_MODEL
from .unified_resolved_conductive_hodge import _build_conductivity_hodge


_MODEL = "global_longitudinal_dissipative_exact_edge_dual_reference_v2"


def _exact_scalar_state(module, parent, background, geometry, *, phi=None):
    context = background.geometry_context(geometry, assemble_thermal=False)
    conductivity_hodge, edge_loss = _build_conductivity_hodge(background, context)
    G = gradient_operator(background, gauge_fixed=True)

    # Keep every non-conductive term exactly as in the compatible scalar block;
    # only replace the arithmetic cell-mixed H_sigma by its exact dual-volume
    # integral.
    diagonal = np.asarray(_edge_mass_diagonal(background, context), complex).reshape(-1)
    sigma, *_ = background.cell_properties(context, None, em=True)
    legacy_edge_loss = np.asarray(
        background.edge_cell_hodge @ np.asarray(sigma, float), float
    ).reshape(-1)
    diagonal = diagonal + 1j * float(background.omega) * (edge_loss - legacy_edge_loss)
    scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
    scalar.sum_duplicates()
    scalar.eliminate_zeros()
    try:
        factor = spla.splu(
            scalar,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        factor = spla.splu(scalar)

    n = len(background.coil_materials)
    d = np.zeros(n, float)
    z = np.zeros(n, complex)
    modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], n), float)
    local_phi = None if phi is None else module._interpolate_basis(parent, background, phi)
    residuals = []
    supports = []

    for p, coil in enumerate(context.geometry.coils):
        q_full, meta = terminal_charge_target(background, coil)
        q = np.asarray(q_full[1:], complex)
        scalar_rhs = (-1j * float(background.omega)) * q
        potential = np.asarray(factor.solve(scalar_rhs), complex).reshape(-1)
        residuals.append(
            float(
                np.linalg.norm(scalar_rhs - scalar @ potential)
                / max(float(np.linalg.norm(scalar_rhs)), np.finfo(float).tiny)
            )
        )
        field = np.asarray(G @ potential, complex).reshape(-1)
        abs2 = np.abs(field) ** 2
        d[p] = float(np.dot(edge_loss, abs2))
        z[p] = complex(-q @ potential)
        supports.append(int(meta.get("terminal_charge_support_nodes", 0)))
        if modal is not None:
            q_cells = np.asarray(0.5 * (conductivity_hodge.T @ abs2), float).reshape(-1)
            modal[:, p] = np.asarray(2.0 * (local_phi.T @ q_cells), float)

    return {
        "fine_step": float(module._background_step(background)),
        "n_cells": int(background.n_cells),
        "scalar_dofs": int(G.shape[1]),
        "d_vol": d,
        "z_reaction": z,
        "modal_h": modal,
        "maximum_scalar_relative_residual": max(residuals, default=0.0),
        "terminal_charge_support_nodes": supports,
        "conductive_hodge_model": _HODGE_MODEL,
        "conductive_hodge_legacy_relative_difference": float(
            np.linalg.norm(edge_loss - legacy_edge_loss)
            / max(
                float(np.linalg.norm(edge_loss)),
                float(np.linalg.norm(legacy_edge_loss)),
                np.finfo(float).tiny,
            )
        ),
    }


def install(module):
    if bool(getattr(module, "_resolved_dissipative_reference_installed", False)):
        return module

    original_scalar_state = module._scalar_state

    def scalar_state(module_arg, parent, background, geometry, *, phi=None):
        # The current scalar state is intentionally the same legacy longitudinal
        # component as the full-Maxwell operator.  Refined reference backgrounds
        # use the geometry-resolved dual Hodge so the correction replaces, rather
        # than double-counts, the unresolved coarse longitudinal loss.
        if background is parent:
            return original_scalar_state(
                module_arg, parent, background, geometry, phi=phi
            )
        return _exact_scalar_state(
            module_arg, parent, background, geometry, phi=phi
        )

    module._scalar_state = scalar_state
    module._MODEL = _MODEL
    module._resolved_dissipative_reference_installed = True
    return module


__all__ = ["install"]
