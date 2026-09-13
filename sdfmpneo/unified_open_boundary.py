"""Open electromagnetic boundary for the production Maxwell truth solve.

The background keeps all tangential boundary edge degrees of freedom and adds a
first-order Silver--Mueller/Sommerfeld impedance condition matched to seawater.
The same boundary mass matrix provides an independent outward-power quadratic
form for the Physics Gate.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .unified_background import EPS0, MU0, FixedMultiscaleBackground


class OpenBoundaryBackground(FixedMultiscaleBackground):
    """Fixed Cartesian background with a passive first-order open EM boundary."""

    boundary_model = "silver_muller_impedance"

    def _build_edges(self):
        # Unlike the legacy finite-PEC background, keep every edge DOF.  The
        # tangential boundary field is closed by a Robin/impedance term in
        # em_operator rather than by deleting those unknowns.
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
                        lengths.append((self.dx[i], self.dy[j], self.dz[k])[axis])
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
        """Mass-lumped integral of |E_t|^2 over the six outer faces."""
        weights = np.zeros(self.n_edges, float)

        def add(axis, key, area):
            e = self.edge_maps[axis].get(key)
            if e is not None:
                # A rectangular face has two parallel edge basis functions per
                # tangential component, so constant tangential fields integrate
                # exactly with area/(2*l^2) on each of those two edges.
                weights[e] += float(area) / (2.0 * self.edge_lengths[e] ** 2)

        # x-normal faces: tangential y/z edges.
        for i_face in (0, self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    area = self.dy[j] * self.dz[k]
                    add(1, (i_face, j, k), area)
                    add(1, (i_face, j, k + 1), area)
                    add(2, (i_face, j, k), area)
                    add(2, (i_face, j + 1, k), area)

        # y-normal faces: tangential x/z edges.
        for j_face in (0, self.ny):
            for i in range(self.nx):
                for k in range(self.nz):
                    area = self.dx[i] * self.dz[k]
                    add(0, (i, j_face, k), area)
                    add(0, (i, j_face, k + 1), area)
                    add(2, (i, j_face, k), area)
                    add(2, (i + 1, j_face, k), area)

        # z-normal faces: tangential x/y edges.
        for k_face in (0, self.nz):
            for i in range(self.nx):
                for j in range(self.ny):
                    area = self.dx[i] * self.dy[j]
                    add(0, (i, j, k_face), area)
                    add(0, (i, j + 1, k_face), area)
                    add(1, (i, j, k_face), area)
                    add(1, (i + 1, j, k_face), area)

        if not np.any(weights > 0.0):
            raise RuntimeError("open boundary mass matrix is empty")
        self.boundary_edge_hodge = weights

    def boundary_admittance(self):
        """Passive wave admittance of the seawater surrounding medium.

        With the package convention exp(+i omega t), the outgoing-wave branch is

            Y = sqrt((epsilon - i sigma/omega) / mu).

        Its real part is non-negative and produces outward time-average power.
        """
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
        """Diagonal quadratic-form weights for independent outward Poynting power."""
        return float(self.boundary_admittance().real) * self.boundary_edge_hodge

    def em_operator(self, context, state=None):
        volume = super().em_operator(context, state)
        boundary = 1j * self.omega * self.boundary_admittance() * sp.diags(
            self.boundary_edge_hodge
        )
        return (volume + boundary).tocsr()


__all__ = ["OpenBoundaryBackground"]
