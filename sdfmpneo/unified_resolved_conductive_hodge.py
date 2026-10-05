"""Geometry-resolved conductive edge Hodge for the open-domain Maxwell model.

Only dual wedges whose parent cells intersect a package require geometric work.
Pure-seawater wedges are initialized vectorially.  Package/dual intersection
volumes are cached on the geometry context and shared with the resolved
permittivity builder, so sigma and epsilon never repeat the same OBB clipping.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .unified_resolved_package_fraction import _SIGNS, _intersection_volume


_MODEL = "exact_obb_edge_dual_conductivity_v1"


def _cell_ijk(background, cell):
    c = int(cell)
    yz = int(background.ny * background.nz)
    i = c // yz
    rem = c - i * yz
    j = rem // int(background.nz)
    k = rem - j * int(background.nz)
    return int(i), int(j), int(k)


def _dual_wedge_bounds(background, edge, cell):
    axis, ei, ej, ek = background.edge_tuples[int(edge)]
    ci, cj, ck = _cell_ijk(background, cell)
    cell_indices = (ci, cj, ck)
    edge_indices = (int(ei), int(ej), int(ek))
    grids = (background.x, background.y, background.z)
    centers = background.cell_axes
    lo = np.empty(3, float)
    hi = np.empty(3, float)
    for dim in range(3):
        if dim == int(axis):
            idx = cell_indices[dim]
            lo[dim] = float(grids[dim][idx])
            hi[dim] = float(grids[dim][idx + 1])
        else:
            node = float(grids[dim][edge_indices[dim]])
            center = float(centers[dim][cell_indices[dim]])
            lo[dim] = min(node, center)
            hi[dim] = max(node, center)
    return lo, hi


def _package_data(background, context):
    rows = []
    active = np.zeros(background.n_cells, dtype=bool)
    for package, material in zip(context.geometry.packages, background.package_materials):
        sigma = float(background._temperature_material(material, background.ambient_temperature))
        if abs(sigma) > 1e-14:
            raise ValueError(
                "exact edge-dual conductivity currently requires electrically insulating packages"
            )
        half = np.asarray(package.half_extent, float).reshape(3)
        vertices = np.asarray(package.pose.apply(_SIGNS * half), float)
        world_lo = np.min(vertices, axis=0)
        world_hi = np.max(vertices, axis=0)
        rows.append((package, half, vertices, world_lo, world_hi))
        index_sets = []
        for grid, lower, upper in zip((background.x, background.y, background.z), world_lo, world_hi):
            values = np.asarray(grid, float)
            index_sets.append(
                np.flatnonzero((values[:-1] < float(upper)) & (values[1:] > float(lower)))
            )
        if all(len(values) for values in index_sets):
            ii, jj, kk = np.meshgrid(*index_sets, indexing="ij")
            cells = ((ii * background.ny + jj) * background.nz + kk).reshape(-1)
            active[np.asarray(cells, dtype=np.int64)] = True
    return rows, active


def _resolved_dual_package_volumes(background, context):
    """Return active CSR positions and exact package volumes in those dual wedges.

    This is the expensive OBB clipping stage.  It is performed once per spatial
    context and reused by both the conductivity and permittivity Hodge builders.
    """
    cached = getattr(context, "_sdfmpneo_resolved_dual_package_volumes", None)
    if cached is not None:
        return cached

    packages, active_cells = _package_data(background, context)
    legacy = background.edge_cell_hodge.tocsr()
    indices = legacy.indices
    indptr = legacy.indptr
    positions = np.flatnonzero(active_cells[indices]).astype(np.int64, copy=False)
    if positions.size:
        edges = np.searchsorted(indptr, positions, side="right") - 1
        edges = np.asarray(edges, dtype=np.int64)
    else:
        edges = np.empty(0, dtype=np.int64)
    dual_volumes = np.zeros(positions.size, float)
    package_volumes = np.zeros((positions.size, len(packages)), float)

    for n, (pos, edge) in enumerate(zip(positions, edges)):
        cell = int(indices[int(pos)])
        length = float(background.edge_lengths[int(edge)])
        l2 = length * length
        dual_volume = float(legacy.data[int(pos)]) * l2
        dual_volumes[n] = dual_volume
        if dual_volume <= 0.0:
            continue
        lo, hi = _dual_wedge_bounds(background, int(edge), cell)
        remaining = dual_volume
        for p, (package, half, vertices, world_lo, world_hi) in enumerate(packages):
            if remaining <= 0.0:
                break
            if np.any(hi <= world_lo) or np.any(lo >= world_hi):
                continue
            volume = float(_intersection_volume(package, half, vertices, lo, hi))
            use = min(max(volume, 0.0), remaining)
            package_volumes[n, p] = use
            remaining -= use

    cached = (positions, edges, dual_volumes, package_volumes)
    context._sdfmpneo_resolved_dual_package_volumes = cached
    return cached


def _build_conductivity_hodge(background, context):
    sea_sigma = float(
        background._temperature_material(background.seawater_material, background.ambient_temperature)
    )
    if not np.isfinite(sea_sigma) or sea_sigma < 0.0:
        raise ValueError("seawater conductivity must be finite and non-negative")

    legacy = background.edge_cell_hodge.tocsr()
    data = sea_sigma * np.asarray(legacy.data, float).copy()
    positions, _edges, dual_volumes, package_volumes = _resolved_dual_package_volumes(
        background, context
    )
    if positions.size:
        insulating = np.minimum(
            np.sum(package_volumes, axis=1), np.maximum(dual_volumes, 0.0)
        )
        ratio = np.ones_like(dual_volumes)
        good = dual_volumes > np.finfo(float).tiny
        ratio[good] = np.maximum(0.0, 1.0 - insulating[good] / dual_volumes[good])
        ratio[~good] = 0.0
        data[positions] = sea_sigma * np.asarray(legacy.data[positions], float) * ratio

    matrix = sp.csr_matrix(
        (data, legacy.indices.copy(), legacy.indptr.copy()), shape=legacy.shape, dtype=float
    )
    matrix.eliminate_zeros()
    weights = np.asarray(matrix.sum(axis=1), float).reshape(-1)
    if weights.shape != (background.n_edges,) or np.any(~np.isfinite(weights)) or np.any(weights < -1e-14):
        raise FloatingPointError("resolved conductive edge Hodge is invalid")
    return matrix, np.maximum(weights, 0.0)


def install(background_cls):
    if bool(getattr(background_cls, "_resolved_conductive_hodge_installed", False)):
        return background_cls

    original_spatial_context = background_cls._spatial_context
    original_em_operator = background_cls.em_operator

    def spatial_context(self, geometry):
        context = original_spatial_context(self, geometry)
        matrix, weights = _build_conductivity_hodge(self, context)
        context.em_conductivity_hodge = matrix
        context.em_edge_conductivity_weights = weights
        context.em_conductivity_hodge_model = _MODEL
        sigma, *_ = self.cell_properties(context, None, em=True)
        legacy = np.asarray(self.edge_cell_hodge @ np.asarray(sigma, float), float).reshape(-1)
        scale = max(float(np.linalg.norm(weights)), float(np.linalg.norm(legacy)), np.finfo(float).tiny)
        context.em_conductivity_hodge_legacy_relative_difference = float(
            np.linalg.norm(weights - legacy) / scale
        )
        return context

    def edge_conductivity_hodge(self, context, state=None):
        if state is not None:
            for name in self.package_materials + (self.seawater_material,):
                if float(self.materials[name].get("resistivity_temperature_coefficient", 0.0)) != 0.0:
                    raise ValueError(
                        "resolved conductive Hodge requires temperature-independent non-wire EM materials"
                    )
        matrix = getattr(context, "em_conductivity_hodge", None)
        if matrix is None:
            matrix, weights = _build_conductivity_hodge(self, context)
            context.em_conductivity_hodge = matrix
            context.em_edge_conductivity_weights = weights
            context.em_conductivity_hodge_model = _MODEL
        return matrix

    def edge_conductivity_weights(self, context, state=None):
        edge_conductivity_hodge(self, context, state)
        return np.asarray(context.em_edge_conductivity_weights, float).reshape(-1)

    def cell_joule_from_edge_energy(self, context, edge_abs2, state=None):
        matrix = edge_conductivity_hodge(self, context, state)
        values = np.asarray(edge_abs2, float).reshape(-1)
        if values.shape != (self.n_edges,):
            raise ValueError("edge energy must contain one value per Maxwell edge")
        return np.asarray(0.5 * (matrix.T @ values), float).reshape(-1)

    def modal_edge_conductivity_weights(self, context, cell_values, state=None):
        matrix = edge_conductivity_hodge(self, context, state)
        values = np.asarray(cell_values, float).reshape(-1)
        if values.shape != (self.n_cells,):
            raise ValueError("modal cell values must contain one value per cell")
        return np.asarray(matrix @ values, float).reshape(-1)

    def em_operator(self, context, state=None):
        operator = original_em_operator(self, context, state)
        sigma, *_ = self.cell_properties(context, state, em=True)
        legacy = np.asarray(self.edge_cell_hodge @ np.asarray(sigma, float), float).reshape(-1)
        resolved = edge_conductivity_weights(self, context, state)
        correction = 1j * float(self.omega) * sp.diags(resolved - legacy, format="csr")
        return (operator + correction).tocsr()

    background_cls._spatial_context = spatial_context
    background_cls.em_conductivity_hodge = edge_conductivity_hodge
    background_cls.em_edge_conductivity_weights = edge_conductivity_weights
    background_cls.em_cell_joule_from_edge_energy = cell_joule_from_edge_energy
    background_cls.em_modal_edge_conductivity_weights = modal_edge_conductivity_weights
    background_cls.em_operator = em_operator
    background_cls.em_conductivity_hodge_model = _MODEL
    background_cls._resolved_conductive_hodge_installed = True
    return background_cls


__all__ = [
    "_MODEL",
    "_build_conductivity_hodge",
    "_dual_wedge_bounds",
    "_resolved_dual_package_volumes",
    "install",
]
