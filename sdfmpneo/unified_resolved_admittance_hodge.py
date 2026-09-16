"""Geometry-resolved dielectric edge Hodge for longitudinal scalar references.

The production scalar mass is

    D = -omega**2 H_epsilon + i omega H_sigma + D_open.

The package/seawater interface is an oriented box cut by a Cartesian edge-dual
mesh.  The dissipative reference already integrates H_sigma on each quarter-cell
dual wedge.  Doing only that still leaves H_epsilon as an arithmetic cut-cell
mixture on the same interface.  This module evaluates the dielectric part on the
same geometric dual wedges while preserving the existing coil-volume semantics.

For the package/seawater part every dual wedge starts as seawater and exact OBB
intersection volumes replace seawater epsilon by the corresponding package
epsilon.  The embedded stranded-coil volume then applies the existing
conservative cell-fraction correction from package epsilon to coil epsilon.  The
wire conductivity remains excluded from the Maxwell volume model exactly as in
the production constitutive law.
"""
from __future__ import annotations

import numpy as np

from .unified_background import EPS0
from .unified_resolved_package_fraction import _SIGNS, _intersection_volume
from .unified_resolved_conductive_hodge import _dual_wedge_bounds


_MODEL = "resolved_obb_edge_dual_permittivity_v1"


def _package_rows(background, context):
    rows = []
    for package, material in zip(context.geometry.packages, background.package_materials):
        half = np.asarray(package.half_extent, float).reshape(3)
        vertices = np.asarray(package.pose.apply(_SIGNS * half), float)
        epsilon = EPS0 * float(background.materials[material].get("relative_permittivity", 1.0))
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("package permittivity must be finite and positive")
        rows.append(
            (
                package,
                half,
                vertices,
                np.min(vertices, axis=0),
                np.max(vertices, axis=0),
                float(epsilon),
            )
        )
    return rows


def _build_permittivity_weights(background, context):
    sea_epsilon = EPS0 * float(
        background.materials[background.seawater_material].get("relative_permittivity", 1.0)
    )
    if not np.isfinite(sea_epsilon) or sea_epsilon <= 0.0:
        raise ValueError("seawater permittivity must be finite and positive")

    packages = _package_rows(background, context)
    legacy_geometry = background.edge_cell_hodge.tocsr()
    indices = legacy_geometry.indices
    indptr = legacy_geometry.indptr
    exact = np.zeros(background.n_edges, float)

    for edge in range(background.n_edges):
        length = float(background.edge_lengths[edge])
        l2 = length * length
        total = 0.0
        for pos in range(indptr[edge], indptr[edge + 1]):
            cell = int(indices[pos])
            dual_volume = float(legacy_geometry.data[pos]) * l2
            if dual_volume <= 0.0:
                continue
            lo, hi = _dual_wedge_bounds(background, edge, cell)
            integral = sea_epsilon * dual_volume
            occupied = 0.0
            for package, half, vertices, world_lo, world_hi, package_epsilon in packages:
                if np.any(hi <= world_lo) or np.any(lo >= world_hi):
                    continue
                volume = _intersection_volume(package, half, vertices, lo, hi)
                if volume <= 0.0:
                    continue
                # Packages are certified non-overlapping.  Clip only against
                # floating-point sliver overlap at shared/tangent boundaries.
                available = max(0.0, dual_volume - occupied)
                use = min(float(volume), available)
                integral += (package_epsilon - sea_epsilon) * use
                occupied += use
            total += integral / l2
        exact[edge] = total

    # The exact OBB replacement above treats each package as uniformly filled by
    # package dielectric.  Restore the existing stranded-coil dielectric volume
    # semantics as a conservative cell-fraction correction inside its package.
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
