"""Open-boundary production Maxwell background with geometry-aware thermal ROM."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import scipy.sparse as sp

from .unified_background import EPS0, MU0, FixedMultiscaleBackground


class OpenBoundaryBackground(FixedMultiscaleBackground):
    """Fixed Cartesian EM background; thermal basis is generated per geometry."""

    boundary_model = "silver_muller_impedance"

    def __init__(self, *args, thermal_library=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.thermal_library = None
        if thermal_library is not None:
            self.set_thermal_library(thermal_library)

    @property
    def thermal_rank(self):
        if self.thermal_library is not None:
            return int(self.thermal_library.rank)
        return super().thermal_rank

    def set_thermal_library(self, library):
        if library is None or int(getattr(library, "rank", 0)) < 1:
            raise ValueError("geometry-aware thermal library must have positive rank")
        if len(library.local_modes) != len(self.coil_materials):
            raise ValueError("thermal library port count differs from background")
        self.thermal_library = library
        # Production must not silently fall back to one fixed basis.
        self.thermal_basis = None
        return library

    def geometry_context(self, geometry, *, assemble_thermal=True):
        context = super().geometry_context(geometry, assemble_thermal=False)
        if not assemble_thermal:
            return context
        if self.thermal_library is None:
            raise ValueError("geometry-aware thermal library has not been constructed")
        phi = self.thermal_library.basis_for_geometry(self, context.geometry)
        M, K = self.thermal_operator_full(context.fractions)
        Mr = phi.T @ (M @ phi)
        Kr = phi.T @ (K @ phi)
        limit = float(getattr(self.thermal_library, "conditioning_limit", 1e10))
        conditions = {}
        for name, matrix in (("M_r", Mr), ("K_r", Kr)):
            symmetric = 0.5 * (matrix + matrix.T)
            eig = np.linalg.eigvalsh(symmetric)
            if eig.size == 0 or not np.all(np.isfinite(eig)) or eig[0] <= 0.0:
                raise RuntimeError(f"{name}(g) is not positive definite")
            condition = float(eig[-1] / eig[0])
            if not np.isfinite(condition) or condition > limit:
                raise RuntimeError(
                    f"{name}(g) conditioning failed: cond={condition:.3e}, limit={limit:.3e}"
                )
            conditions[name] = condition
        context.thermal_basis = phi
        context.thermal_mass_full = M
        context.thermal_stiffness_full = K
        context.thermal_mass_reduced = Mr
        context.thermal_stiffness_reduced = Kr
        context.thermal_mass_condition = conditions["M_r"]
        context.thermal_stiffness_condition = conditions["K_r"]
        return context

    def _state_temperature(self, context, state):
        material_rise = None
        if state is None:
            rise = np.zeros(self.n_cells)
        elif isinstance(state, Mapping):
            material_rise = {str(name): float(value) for name, value in state.items()}
            if any(not np.isfinite(value) for value in material_rise.values()):
                raise ValueError("material temperature rises must be finite")
            rise = np.zeros(self.n_cells)
            for name, fraction in context.fractions.items():
                rise += fraction * material_rise.get(name, 0.0)
        else:
            value = np.asarray(state, float)
            if value.ndim == 0:
                rise = np.full(self.n_cells, float(value))
            else:
                value = value.reshape(-1)
                if value.shape == (self.n_cells,):
                    rise = value
                else:
                    phi = getattr(context, "thermal_basis", None)
                    if phi is None or value.shape != (phi.shape[1],):
                        raise ValueError("thermal state dimension mismatch")
                    rise = np.asarray(phi, float) @ value
            if np.any(~np.isfinite(rise)):
                raise ValueError("thermal state must be finite")
        return self.ambient_temperature + rise, material_rise

    def _build_edges(self):
        # Keep every edge DOF; the absorbing boundary closes tangential E.
        full, lengths, maps = [], [], []
        for axis in range(3):
            amap = {}
            if axis == 0:
                ranges = (range(self.nx), range(self.ny + 1), range(self.nz + 1))
            elif axis == 1:
                ranges = (range(self.nx + 1), range(self.ny), range(self.nz + 1))
            else:
                ranges = (range(self.nx + 1), range(self.ny + 1), range(self.nz))
            for i in ranges[0]:
                for j in ranges[1]:
                    for k in ranges[2]:
                        amap[(i, j, k)] = len(full)
                        full.append((axis, i, j, k))
                        if axis == 0:
                            lengths.append(self.dx[i])
                        elif axis == 1:
                            lengths.append(self.dy[j])
                        else:
                            lengths.append(self.dz[k])
            maps.append(amap)
        self.edge_tuples = tuple(full)
        self.edge_lengths = np.asarray(lengths, float)
        self.edge_maps = maps

        rows, cols, data = [], [], []
        for e, (axis, i, j, k) in enumerate(self.edge_tuples):
            cells = []
            if axis == 0:
                for jj in (j - 1, j):
                    for kk in (k - 1, k):
                        if 0 <= jj < self.ny and 0 <= kk < self.nz:
                            cells.append(self._cell_id(i, jj, kk))
            elif axis == 1:
                for ii in (i - 1, i):
                    for kk in (k - 1, k):
                        if 0 <= ii < self.nx and 0 <= kk < self.nz:
                            cells.append(self._cell_id(ii, j, kk))
            else:
                for ii in (i - 1, i):
                    for jj in (j - 1, j):
                        if 0 <= ii < self.nx and 0 <= jj < self.ny:
                            cells.append(self._cell_id(ii, jj, k))
            for c in cells:
                rows.append(e)
                cols.append(c)
                data.append(self.cell_volumes[c] / (4.0 * self.edge_lengths[e] ** 2))
        self.edge_cell_hodge = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.n_edges, self.n_cells)
        )
        self._build_boundary_edge_hodge()

    def _build_boundary_edge_hodge(self):
        weights = np.zeros(self.n_edges, float)

        def add(axis, key, area):
            e = self.edge_maps[axis].get(key)
            if e is not None:
                weights[e] += float(area) / (2.0 * self.edge_lengths[e] ** 2)

        for i_face in (0, self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    area = self.dy[j] * self.dz[k]
                    add(1, (i_face, j, k), area); add(1, (i_face, j, k + 1), area)
                    add(2, (i_face, j, k), area); add(2, (i_face, j + 1, k), area)
        for j_face in (0, self.ny):
            for i in range(self.nx):
                for k in range(self.nz):
                    area = self.dx[i] * self.dz[k]
                    add(0, (i, j_face, k), area); add(0, (i, j_face, k + 1), area)
                    add(2, (i, j_face, k), area); add(2, (i + 1, j_face, k), area)
        for k_face in (0, self.nz):
            for i in range(self.nx):
                for j in range(self.ny):
                    area = self.dx[i] * self.dy[j]
                    add(0, (i, j, k_face), area); add(0, (i, j + 1, k_face), area)
                    add(1, (i, j, k_face), area); add(1, (i + 1, j, k_face), area)
        if not np.any(weights > 0.0):
            raise RuntimeError("open boundary mass matrix is empty")
        self.boundary_edge_hodge = weights

    def boundary_admittance(self):
        material = self.materials[self.seawater_material]
        sigma = float(self._temperature_material(self.seawater_material, self.ambient_temperature))
        epsilon = EPS0 * float(material.get("relative_permittivity", 1.0))
        mu = MU0 * float(material.get("relative_permeability", 1.0))
        if mu <= 0.0 or epsilon <= 0.0 or sigma < 0.0:
            raise ValueError("open-boundary medium must be passive with positive epsilon/mu")
        admittance = np.sqrt((epsilon - 1j * sigma / self.omega) / mu)
        if admittance.real < 0.0 or (
            abs(admittance.real) <= np.finfo(float).eps and admittance.imag > 0.0
        ):
            admittance = -admittance
        if admittance.real < -1e-14:
            raise ValueError("open-boundary admittance selected a non-passive branch")
        return complex(admittance)

    def outward_loss_weights(self):
        return float(self.boundary_admittance().real) * self.boundary_edge_hodge

    def em_operator(self, context, state=None):
        volume = super().em_operator(context, state)
        boundary = 1j * self.omega * self.boundary_admittance() * sp.diags(
            self.boundary_edge_hodge
        )
        return (volume + boundary).tocsr()


__all__ = ["OpenBoundaryBackground"]
