from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.linalg
import scipy.sparse as sp


@dataclass(frozen=True)
class RectilinearComplex3D:
    """Oriented orthogonal 3-D cell complex on a rectilinear grid."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    grad: sp.csr_matrix
    curl: sp.csr_matrix
    edge_axis: np.ndarray
    edge_ijk: np.ndarray
    face_axis: np.ndarray
    face_ijk: np.ndarray

    @classmethod
    def build(cls, x: Iterable[float], y: Iterable[float], z: Iterable[float]) -> "RectilinearComplex3D":
        x = np.asarray(tuple(x), dtype=float)
        y = np.asarray(tuple(y), dtype=float)
        z = np.asarray(tuple(z), dtype=float)
        if min(x.size, y.size, z.size) < 2:
            raise ValueError("Each coordinate axis requires at least two nodes")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0) or np.any(np.diff(z) <= 0):
            raise ValueError("Coordinates must be strictly increasing")

        nx, ny, nz = x.size, y.size, z.size

        def nidx(i: int, j: int, k: int) -> int:
            return (i * ny + j) * nz + k

        edge_map: dict[tuple[int, int, int, int], int] = {}
        edge_axis: list[int] = []
        edge_ijk: list[tuple[int, int, int]] = []
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        e = 0

        for i in range(nx - 1):
            for j in range(ny):
                for k in range(nz):
                    edge_map[(0, i, j, k)] = e
                    edge_axis.append(0)
                    edge_ijk.append((i, j, k))
                    rows += [e, e]
                    cols += [nidx(i, j, k), nidx(i + 1, j, k)]
                    data += [-1, 1]
                    e += 1
        for i in range(nx):
            for j in range(ny - 1):
                for k in range(nz):
                    edge_map[(1, i, j, k)] = e
                    edge_axis.append(1)
                    edge_ijk.append((i, j, k))
                    rows += [e, e]
                    cols += [nidx(i, j, k), nidx(i, j + 1, k)]
                    data += [-1, 1]
                    e += 1
        for i in range(nx):
            for j in range(ny):
                for k in range(nz - 1):
                    edge_map[(2, i, j, k)] = e
                    edge_axis.append(2)
                    edge_ijk.append((i, j, k))
                    rows += [e, e]
                    cols += [nidx(i, j, k), nidx(i, j, k + 1)]
                    data += [-1, 1]
                    e += 1

        grad = sp.coo_matrix((data, (rows, cols)), shape=(e, nx * ny * nz), dtype=float).tocsr()

        frows: list[int] = []
        fcols: list[int] = []
        fdata: list[int] = []
        face_axis: list[int] = []
        face_ijk: list[tuple[int, int, int]] = []
        f = 0

        def add_face(entries, axis: int, ijk: tuple[int, int, int]) -> None:
            nonlocal f
            for edge, sign in entries:
                frows.append(f)
                fcols.append(edge)
                fdata.append(sign)
            face_axis.append(axis)
            face_ijk.append(ijk)
            f += 1

        for i in range(nx):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    add_face([
                        (edge_map[(1, i, j, k)], 1),
                        (edge_map[(2, i, j + 1, k)], 1),
                        (edge_map[(1, i, j, k + 1)], -1),
                        (edge_map[(2, i, j, k)], -1),
                    ], 0, (i, j, k))

        for i in range(nx - 1):
            for j in range(ny):
                for k in range(nz - 1):
                    add_face([
                        (edge_map[(2, i, j, k)], 1),
                        (edge_map[(0, i, j, k + 1)], 1),
                        (edge_map[(2, i + 1, j, k)], -1),
                        (edge_map[(0, i, j, k)], -1),
                    ], 1, (i, j, k))

        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz):
                    add_face([
                        (edge_map[(0, i, j, k)], 1),
                        (edge_map[(1, i + 1, j, k)], 1),
                        (edge_map[(0, i, j + 1, k)], -1),
                        (edge_map[(1, i, j, k)], -1),
                    ], 2, (i, j, k))

        curl = sp.coo_matrix((fdata, (frows, fcols)), shape=(f, e), dtype=float).tocsr()
        return cls(
            x=x,
            y=y,
            z=z,
            grad=grad,
            curl=curl,
            edge_axis=np.asarray(edge_axis, dtype=int),
            edge_ijk=np.asarray(edge_ijk, dtype=int),
            face_axis=np.asarray(face_axis, dtype=int),
            face_ijk=np.asarray(face_ijk, dtype=int),
        )

    @property
    def shape_cells(self) -> tuple[int, int, int]:
        return (self.x.size - 1, self.y.size - 1, self.z.size - 1)

    @property
    def n_nodes(self) -> int:
        return self.x.size * self.y.size * self.z.size

    @property
    def n_edges(self) -> int:
        return self.grad.shape[0]

    @property
    def n_faces(self) -> int:
        return self.curl.shape[0]

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape_cells))

    def _cell_array(self, values) -> np.ndarray:
        a = np.asarray(values, dtype=float)
        if a.shape != self.shape_cells:
            raise ValueError(f"cell field must have shape {self.shape_cells}")
        return a

    def edge_hodge(self, cell_values) -> sp.csr_matrix:
        q = self._cell_array(cell_values)
        dx, dy, dz = np.diff(self.x), np.diff(self.y), np.diff(self.z)
        nx, ny, nz = self.shape_cells
        weights = np.zeros(self.n_edges, dtype=float)

        for e, (axis, ijk) in enumerate(zip(self.edge_axis, self.edge_ijk)):
            i, j, k = map(int, ijk)
            value = 0.0
            if axis == 0:
                for jj in (j - 1, j):
                    if 0 <= jj < ny:
                        for kk in (k - 1, k):
                            if 0 <= kk < nz:
                                value += q[i, jj, kk] * 0.5 * dy[jj] * 0.5 * dz[kk]
                weights[e] = value / dx[i]
            elif axis == 1:
                for ii in (i - 1, i):
                    if 0 <= ii < nx:
                        for kk in (k - 1, k):
                            if 0 <= kk < nz:
                                value += q[ii, j, kk] * 0.5 * dx[ii] * 0.5 * dz[kk]
                weights[e] = value / dy[j]
            else:
                for ii in (i - 1, i):
                    if 0 <= ii < nx:
                        for jj in (j - 1, j):
                            if 0 <= jj < ny:
                                value += q[ii, jj, k] * 0.5 * dx[ii] * 0.5 * dy[jj]
                weights[e] = value / dz[k]
        return sp.diags(weights, format="csr")

    def face_hodge(self, cell_values) -> sp.csr_matrix:
        q = self._cell_array(cell_values)
        dx, dy, dz = np.diff(self.x), np.diff(self.y), np.diff(self.z)
        nx, ny, nz = self.shape_cells
        weights = np.zeros(self.n_faces, dtype=float)

        for f, (axis, ijk) in enumerate(zip(self.face_axis, self.face_ijk)):
            i, j, k = map(int, ijk)
            value = 0.0
            if axis == 0:
                if i > 0:
                    value += q[i - 1, j, k] * 0.5 * dx[i - 1]
                if i < nx:
                    value += q[i, j, k] * 0.5 * dx[i]
                weights[f] = value / (dy[j] * dz[k])
            elif axis == 1:
                if j > 0:
                    value += q[i, j - 1, k] * 0.5 * dy[j - 1]
                if j < ny:
                    value += q[i, j, k] * 0.5 * dy[j]
                weights[f] = value / (dx[i] * dz[k])
            else:
                if k > 0:
                    value += q[i, j, k - 1] * 0.5 * dz[k - 1]
                if k < nz:
                    value += q[i, j, k] * 0.5 * dz[k]
                weights[f] = value / (dx[i] * dy[j])
        return sp.diags(weights, format="csr")

    def tree_cotree_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic spanning-tree and cotree edge indices."""

        parent = np.arange(self.n_nodes, dtype=int)
        rank = np.zeros(self.n_nodes, dtype=int)

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = int(parent[node])
            return node

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        tree: list[int] = []
        G = self.grad.tocsr()
        for edge in range(self.n_edges):
            nodes = G.indices[G.indptr[edge] : G.indptr[edge + 1]]
            if len(nodes) != 2:
                raise RuntimeError("edge incidence row must contain exactly two nodes")
            if union(int(nodes[0]), int(nodes[1])):
                tree.append(edge)

        tree_set = set(tree)
        cotree = [edge for edge in range(self.n_edges) if edge not in tree_set]
        return np.asarray(tree, dtype=int), np.asarray(cotree, dtype=int)

    def gauge_basis(self) -> np.ndarray:
        """Deterministic tree-cotree gauge selector.

        Tree-edge vector-potential coordinates are fixed to zero. Cotree edges
        are the independent A coordinates. This gives a unique representative
        of each gauge class on the connected rectilinear graph without SVD,
        penalty parameters, or numerical rank thresholds.
        """

        _, cotree = self.tree_cotree_edges()
        R = np.zeros((self.n_edges, cotree.size), dtype=float)
        R[cotree, np.arange(cotree.size)] = 1.0
        return R

    def conductive_gradient(self, conductivity_support_cells) -> np.ndarray:
        support = self._cell_array(conductivity_support_cells).astype(bool)
        edge_active = np.asarray(self.edge_hodge(support.astype(float)).diagonal() > 0)
        if not np.any(edge_active):
            return np.zeros((self.n_edges, 0), dtype=float)

        G = self.grad.tocsr()
        adjacency = [set() for _ in range(self.n_nodes)]
        active_nodes: set[int] = set()
        for edge in np.flatnonzero(edge_active):
            nodes = G.indices[G.indptr[edge] : G.indptr[edge + 1]]
            a, b = map(int, nodes)
            adjacency[a].add(b)
            adjacency[b].add(a)
            active_nodes.update((a, b))

        seen: set[int] = set()
        kept_nodes: list[int] = []
        for root in sorted(active_nodes):
            if root in seen:
                continue
            stack = [root]
            component: list[int] = []
            seen.add(root)
            while stack:
                u = stack.pop()
                component.append(u)
                for v in adjacency[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            reference = min(component)
            kept_nodes.extend(v for v in sorted(component) if v != reference)
        return G[:, kept_nodes].toarray()

    def topology_defect(self) -> float:
        product = self.curl @ self.grad
        return 0.0 if product.nnz == 0 else float(np.max(np.abs(product.data)))


@dataclass(frozen=True)
class SpatialAphiAssembly:
    complex3d: RectilinearComplex3D
    discretization: object


def build_compatible_aphi_from_cells(
    complex3d: RectilinearComplex3D,
    *,
    omega: float,
    reluctivity_cell: np.ndarray,
    conductivity0_cell: np.ndarray,
    conductivity_state_cell: np.ndarray,
    thermal_test_cell: np.ndarray,
    source_current: np.ndarray,
) -> SpatialAphiAssembly:
    from .compatible import CompatibleAphiDiscretization

    grid = complex3d
    nu = np.asarray(reluctivity_cell, dtype=float)
    sigma0 = np.asarray(conductivity0_cell, dtype=float)
    sigma_state = np.asarray(conductivity_state_cell, dtype=float)
    thermal_test = np.asarray(thermal_test_cell, dtype=float)

    if omega <= 0:
        raise ValueError("omega must be positive")
    if nu.shape != grid.shape_cells or sigma0.shape != grid.shape_cells:
        raise ValueError("cell material field shape mismatch")
    if sigma_state.ndim != 4 or sigma_state.shape[1:] != grid.shape_cells:
        raise ValueError("conductivity_state_cell must have shape (n_thermal,*shape_cells)")
    if thermal_test.shape != sigma_state.shape:
        raise ValueError("thermal_test_cell must match conductivity_state_cell shape")
    if np.any(sigma0 < 0):
        raise ValueError("reference conductivity must be non-negative")
    if np.any((np.abs(sigma_state) > 0) & (sigma0[None, ...] <= 0)):
        raise ValueError("thermal coefficients may not create new conductivity support")

    n_thermal = sigma_state.shape[0]
    if np.asarray(source_current).shape != (grid.n_edges,):
        raise ValueError("source_current shape mismatch")

    reluctivity_hodge = grid.face_hodge(nu).toarray()
    conductivity0 = grid.edge_hodge(sigma0).toarray()
    conductivity_state = np.stack(
        [grid.edge_hodge(sigma_state[k]).toarray() for k in range(n_thermal)],
        axis=0,
    )

    grad_c = grid.conductive_gradient(sigma0 > 0)
    a_basis = grid.gauge_basis()
    C = grid.curl.toarray().astype(complex)
    R = np.asarray(a_basis, dtype=complex)
    G = np.asarray(grad_c, dtype=complex)
    Nu = np.asarray(reluctivity_hodge, dtype=complex)
    S0 = np.asarray(conductivity0, dtype=complex)

    K_A = R.conj().T @ (C.conj().T @ Nu @ C) @ R
    L_E = np.hstack([-1j * omega * R, -G])
    n_total = R.shape[1] + G.shape[1]
    H_mag = np.zeros((n_total, n_total), dtype=complex)
    H_mag[: R.shape[1], : R.shape[1]] = 0.5 * K_A
    H_metric = H_mag + (0.5 / omega) * (L_E.conj().T @ S0 @ L_E)
    H_metric = 0.5 * (H_metric + H_metric.conj().T)
    try:
        scipy.linalg.cholesky(H_metric, lower=True, check_finite=True)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "physical electromagnetic Riesz metric is not positive definite after gauge elimination"
        ) from exc

    thermal_loss_hodge0 = np.stack(
        [grid.edge_hodge(sigma0 * thermal_test[j]).toarray() for j in range(n_thermal)],
        axis=0,
    )
    thermal_loss_hodge_state = np.empty(
        (n_thermal, n_thermal, grid.n_edges, grid.n_edges), dtype=float
    )
    for j in range(n_thermal):
        for k in range(n_thermal):
            thermal_loss_hodge_state[j, k] = grid.edge_hodge(
                sigma_state[k] * thermal_test[j]
            ).toarray()

    discretization = CompatibleAphiDiscretization(
        curl=C,
        grad_c=G,
        a_basis=R,
        reluctivity_hodge=reluctivity_hodge,
        conductivity0=conductivity0,
        conductivity_state=conductivity_state,
        source_current=np.asarray(source_current, dtype=complex),
        omega=omega,
        riesz_metric=H_metric,
        thermal_loss_hodge0=thermal_loss_hodge0,
        thermal_loss_hodge_state=thermal_loss_hodge_state,
    )
    return SpatialAphiAssembly(grid, discretization)


def face_loop_source(
    complex3d: RectilinearComplex3D,
    face_index: int,
    amplitude: complex = 1.0,
) -> np.ndarray:
    if not 0 <= face_index < complex3d.n_faces:
        raise ValueError("face_index out of range")
    return amplitude * complex3d.curl.getrow(face_index).toarray().ravel().astype(complex)
