"""Geometry-resolved conductive edge Hodge for the open-domain Maxwell model.

The production geometry contains conductive seawater surrounding electrically
insulating coil packages.  A cell-volume fraction is sufficient for volume
bookkeeping, but using the arithmetic cell mixture directly in the edge Hodge
smears a thin 0 S/m package / 5 S/m seawater interface into an artificial
conductive layer whose thickness changes with the Cartesian mesh.

For the diagonal Cartesian edge mass used by this discretization, every
edge--cell contribution is an axis-aligned quarter-cell dual wedge with volume
``cell_volume/4``.  The package is an oriented box.  We can therefore integrate
conductivity over each dual wedge geometrically, using the same exact OBB/AABB
intersection kernel as the resolved package-volume model:

    H_sigma[e,c] = (1 / edge_length[e]**2)
                   * integral_{dual(e,c)} sigma(x) dV.

In the current production material model all EM coil/package conductivity is
zero and only seawater conducts, so the integral is simply the seawater volume
of the dual wedge times sigma_seawater.  The sparse edge--cell matrix is retained
rather than only its row sum so Maxwell loss, cell Joule heat and thermal modal
heat all share exactly the same loss operator.
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
    active_cells = set()
    for package, material in zip(context.geometry.packages, background.package_materials):
        sigma = float(background._temperature_material(material, background.ambient_temperature))
        # The exact production specialization relies on the package and the wire
        # both being insulating in the Maxwell volume model.  If a future model
        # makes the package conductive, the embedded zero-sigma wire geometry
        # must also be integrated explicitly before this shortcut is valid.
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
            for i in index_sets[0]:
                for j in index_sets[1]:
                    base = (int(i) * background.ny + int(j)) * background.nz
                    for k in index_sets[2]:
                        active_cells.add(int(base + int(k)))
    return rows, active_cells


def _build_conductivity_hodge(background, context):
    sea_sigma = float(
        background._temperature_material(background.seawater_material, background.ambient_temperature)
    )
    if not np.isfinite(sea_sigma) or sea_sigma < 0.0:
        raise ValueError("seawater conductivity must be finite and non-negative")
    packages, active_cells = _package_data(background, context)
    legacy = background.edge_cell_hodge.tocsr()
    indices = legacy.indices
    indptr = legacy.indptr
    # Pure-seawater dual wedges need no geometric work.
    data = sea_sigma * np.asarray(legacy.data, float).copy()

    if active_cells:
        active_cells = frozenset(active_cells)
        for edge in range(background.n_edges):
            length = float(background.edge_lengths[edge])
            l2 = length * length
            for pos in range(indptr[edge], indptr[edge + 1]):
                cell = int(indices[pos])
                if cell not in active_cells:
                    continue
                dual_volume = float(legacy.data[pos]) * l2
                if dual_volume <= 0.0:
                    data[pos] = 0.0
                    continue
                lo, hi = _dual_wedge_bounds(background, edge, cell)
                insulating = 0.0
                for package, half, vertices, world_lo, world_hi in packages:
                    if np.any(hi <= world_lo) or np.any(lo >= world_hi):
                        continue
                    insulating += _intersection_volume(package, half, vertices, lo, hi)
                insulating = float(np.clip(insulating, 0.0, dual_volume))
                data[pos] = sea_sigma * max(0.0, dual_volume - insulating) / l2

    matrix = sp.csr_matrix(
        (data, indices.copy(), indptr.copy()), shape=legacy.shape, dtype=float
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


__all__ = ["install"]
