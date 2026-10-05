"""Geometry-resolved dielectric edge Hodge for longitudinal scalar references.

The expensive package/dual OBB intersection geometry is shared with the resolved
conductivity builder.  Pure-seawater wedges are initialized vectorially; only
sparse edge-cell entries whose parent cells overlap a package receive geometric
corrections.
"""
from __future__ import annotations

import numpy as np

from .unified_background import EPS0
from .unified_resolved_conductive_hodge import _resolved_dual_package_volumes


_MODEL = "resolved_obb_edge_dual_permittivity_v1"


def _package_epsilons(background):
    values = []
    for material in background.package_materials:
        epsilon = EPS0 * float(background.materials[material].get("relative_permittivity", 1.0))
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("package permittivity must be finite and positive")
        values.append(float(epsilon))
    return np.asarray(values, float)


def _build_permittivity_weights(background, context):
    sea_epsilon = EPS0 * float(
        background.materials[background.seawater_material].get("relative_permittivity", 1.0)
    )
    if not np.isfinite(sea_epsilon) or sea_epsilon <= 0.0:
        raise ValueError("seawater permittivity must be finite and positive")

    legacy_geometry = background.edge_cell_hodge.tocsr()
    geometric_row_sum = np.asarray(legacy_geometry.sum(axis=1), float).reshape(-1)
    exact = sea_epsilon * geometric_row_sum

    positions, edges, dual_volumes, package_volumes = _resolved_dual_package_volumes(
        background, context
    )
    if positions.size:
        package_eps = _package_epsilons(background)
        if package_volumes.shape[1] != package_eps.size:
            raise AssertionError("resolved package-volume/permittivity count mismatch")
        delta_integral = package_volumes @ (package_eps - sea_epsilon)
        contribution = np.zeros_like(delta_integral)
        good = dual_volumes > np.finfo(float).tiny
        contribution[good] = (
            np.asarray(legacy_geometry.data[positions[good]], float)
            * delta_integral[good]
            / dual_volumes[good]
        )
        np.add.at(exact, np.asarray(edges, dtype=np.int64), contribution)

    # Restore the existing stranded-coil dielectric-volume semantics inside each
    # package as the same conservative cell-fraction correction used by the
    # production model.
    cell_correction = np.zeros(background.n_cells, float)
    for coil_material, package_material in zip(
        background.coil_materials, background.package_materials
    ):
        coil_epsilon = EPS0 * float(
            background.materials[coil_material].get("relative_permittivity", 1.0)
        )
        package_epsilon = EPS0 * float(
            background.materials[package_material].get("relative_permittivity", 1.0)
        )
        fraction = np.asarray(context.fractions[coil_material], float).reshape(-1)
        cell_correction += fraction * (coil_epsilon - package_epsilon)
    exact += np.asarray(legacy_geometry @ cell_correction, float).reshape(-1)

    _sigma, legacy_eps, *_rest = background.cell_properties(context, None, em=True)
    legacy = np.asarray(
        legacy_geometry @ np.asarray(legacy_eps, float), float
    ).reshape(-1)
    if exact.shape != (background.n_edges,) or np.any(~np.isfinite(exact)) or np.any(exact <= 0.0):
        raise FloatingPointError("resolved dielectric edge Hodge is invalid")
    scale = max(
        float(np.linalg.norm(exact)),
        float(np.linalg.norm(legacy)),
        np.finfo(float).tiny,
    )
    return exact, legacy, {
        "dielectric_hodge_model": _MODEL,
        "dielectric_hodge_legacy_relative_difference": float(
            np.linalg.norm(exact - legacy) / scale
        ),
    }


__all__ = ["_MODEL", "_build_permittivity_weights"]
