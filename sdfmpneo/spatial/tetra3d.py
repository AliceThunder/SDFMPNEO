from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.sparse as sp


_LOCAL_FACE_COMBINATIONS = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _oriented_tetrahedron(vertices: np.ndarray, tet: np.ndarray) -> tuple[np.ndarray, float]:
    out = np.asarray(tet, dtype=int).copy()
    x = vertices[out]
    det = float(np.linalg.det(np.column_stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]])))
    if det == 0.0:
        raise ValueError("degenerate tetrahedron with zero signed volume")
    if det < 0.0:
        out[[0, 1]] = out[[1, 0]]
        det = -det
    return out, det / 6.0


def _barycentric_gradients(vertices: np.ndarray, tet: np.ndarray) -> np.ndarray:
    x = vertices[tet]
    B = np.column_stack([np.ones(4), x])
    inverse = np.linalg.inv(B)
    # Column i contains coefficients of lambda_i in [1,x,y,z].
    return inverse[1:, :].T


@dataclass(frozen=True)
class TetrahedralThermalAssembly:
    M: sp.csr_matrix
    K: sp.csr_matrix
    free_nodes: np.ndarray
    boundary_nodes: np.ndarray
    n_full_nodes: int
    tetrahedra: np.ndarray

    def expand_free(self, values: np.ndarray, boundary_value: float = 0.0) -> np.ndarray:
        state = np.asarray(values, dtype=float)
        if state.shape != (self.free_nodes.size,):
            raise ValueError("free thermal state dimension mismatch")
        full = np.full(self.n_full_nodes, float(boundary_value), dtype=float)
        full[self.free_nodes] = state
        return full

    def cell_average(self, values: np.ndarray, boundary_value: float = 0.0) -> np.ndarray:
        full = self.expand_free(values, boundary_value=boundary_value)
        return np.mean(full[self.tetrahedra], axis=1)


@dataclass(frozen=True)
class TetrahedralComplex3D:
    """First-order compatible tetrahedral complex.

    Global edges are oriented from the lower to the higher vertex index. Global
    triangular faces use the sorted vertex orientation `(a,b,c)` and boundary
    `+[a,b] + [b,c] - [a,c]`. The resulting incidence matrices satisfy

        C @ G = 0

    exactly in integer sparse arithmetic.

    Electromagnetic material matrices use first-order Whitney/Nedelec edge
    functions. Thermal matrices use first-order nodal P1 finite elements on the
    same tetrahedra.
    """

    vertices: np.ndarray
    tetrahedra: np.ndarray
    volumes: np.ndarray
    edge_vertices: np.ndarray
    face_vertices: np.ndarray
    tet_edge_indices: np.ndarray
    grad: sp.csr_matrix
    curl: sp.csr_matrix
    boundary_face_indices: np.ndarray

    @classmethod
    def build(cls, vertices: np.ndarray, tetrahedra: np.ndarray) -> "TetrahedralComplex3D":
        xyz = np.asarray(vertices, dtype=float)
        tets_in = np.asarray(tetrahedra, dtype=int)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("vertices must have shape (n_vertices,3)")
        if tets_in.ndim != 2 or tets_in.shape[1] != 4:
            raise ValueError("tetrahedra must have shape (n_tetra,4)")
        if xyz.shape[0] == 0 or tets_in.shape[0] == 0:
            raise ValueError("mesh cannot be empty")
        if np.any(~np.isfinite(xyz)):
            raise ValueError("vertices must be finite")
        if np.any(tets_in < 0) or np.any(tets_in >= xyz.shape[0]):
            raise ValueError("tetrahedron vertex index out of range")
        if any(len(set(map(int, tet))) != 4 for tet in tets_in):
            raise ValueError("tetrahedron must contain four distinct vertices")

        oriented = np.empty_like(tets_in)
        volumes = np.empty(tets_in.shape[0], dtype=float)
        for q, tet in enumerate(tets_in):
            oriented[q], volumes[q] = _oriented_tetrahedron(xyz, tet)

        edge_set: set[tuple[int, int]] = set()
        face_count: dict[tuple[int, int, int], int] = {}
        for tet in oriented:
            for u, v in combinations(map(int, tet), 2):
                edge_set.add((min(u, v), max(u, v)))
            for local_face in _LOCAL_FACE_COMBINATIONS:
                face = tuple(sorted(int(tet[i]) for i in local_face))
                face_count[face] = face_count.get(face, 0) + 1

        edges = np.asarray(sorted(edge_set), dtype=int)
        faces = np.asarray(sorted(face_count), dtype=int)
        edge_index = {tuple(edge): i for i, edge in enumerate(edges)}
        face_index = {tuple(face): i for i, face in enumerate(faces)}

        tet_edges = np.empty((oriented.shape[0], 6), dtype=int)
        for q, tet in enumerate(oriented):
            local = []
            for u, v in combinations(map(int, tet), 2):
                local.append(edge_index[(min(u, v), max(u, v))])
            tet_edges[q] = local

        # Gradient incidence: edge integral of grad(phi) = phi(head)-phi(tail).
        rows = np.repeat(np.arange(edges.shape[0]), 2)
        cols = edges.ravel()
        data = np.tile(np.array([-1, 1], dtype=int), edges.shape[0])
        G = sp.csr_matrix((data, (rows, cols)), shape=(edges.shape[0], xyz.shape[0]), dtype=int)

        # Curl incidence from globally oriented triangular face boundaries.
        c_rows: list[int] = []
        c_cols: list[int] = []
        c_data: list[int] = []
        for fidx, (a, b, c) in enumerate(faces):
            boundary = [
                ((int(a), int(b)), +1),
                ((int(b), int(c)), +1),
                ((int(a), int(c)), -1),
            ]
            for edge, sign in boundary:
                c_rows.append(fidx)
                c_cols.append(edge_index[edge])
                c_data.append(sign)
        C = sp.csr_matrix(
            (c_data, (c_rows, c_cols)),
            shape=(faces.shape[0], edges.shape[0]),
            dtype=int,
        )

        product = C @ G
        if product.nnz != 0:
            raise RuntimeError("internal topology error: curl @ grad is not exactly zero")

        boundary_faces = np.asarray(
            [face_index[face] for face, count in face_count.items() if count == 1],
            dtype=int,
        )
        boundary_faces.sort()

        return cls(
            vertices=xyz,
            tetrahedra=oriented,
            volumes=volumes,
            edge_vertices=edges,
            face_vertices=faces,
            tet_edge_indices=tet_edges,
            grad=G,
            curl=C,
            boundary_face_indices=boundary_faces,
        )

    @property
    def n_nodes(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.face_vertices.shape[0]

    @property
    def n_tetrahedra(self) -> int:
        return self.tetrahedra.shape[0]

    def topology_defect(self) -> float:
        product = self.curl @ self.grad
        return 0.0 if product.nnz == 0 else float(np.max(np.abs(product.data)))

    def boundary_nodes(self) -> np.ndarray:
        if self.boundary_face_indices.size == 0:
            return np.empty(0, dtype=int)
        return np.unique(self.face_vertices[self.boundary_face_indices].ravel())

    def tree_cotree_edges(self) -> tuple[np.ndarray, np.ndarray]:
        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(self.n_nodes)]
        for eidx, (u, v) in enumerate(self.edge_vertices):
            adjacency[int(u)].append((int(v), eidx))
            adjacency[int(v)].append((int(u), eidx))
        for entries in adjacency:
            entries.sort()

        visited = np.zeros(self.n_nodes, dtype=bool)
        tree: list[int] = []
        for root in range(self.n_nodes):
            if visited[root]:
                continue
            visited[root] = True
            queue = [root]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v, eidx in adjacency[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
                        tree.append(eidx)

        tree_array = np.asarray(sorted(set(tree)), dtype=int)
        mask = np.ones(self.n_edges, dtype=bool)
        mask[tree_array] = False
        cotree_array = np.nonzero(mask)[0]
        return tree_array, cotree_array

    def gauge_basis(self) -> sp.csr_matrix:
        _, cotree = self.tree_cotree_edges()
        rows = cotree
        cols = np.arange(cotree.size)
        data = np.ones(cotree.size)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.n_edges, cotree.size))

    def conductive_gradient(self, active_tetrahedra: np.ndarray) -> sp.csr_matrix:
        active = np.asarray(active_tetrahedra, dtype=bool)
        if active.shape != (self.n_tetrahedra,):
            raise ValueError("active_tetrahedra must have shape (n_tetrahedra,)")
        if not np.any(active):
            return sp.csr_matrix((self.n_edges, 0), dtype=float)

        active_nodes = set(map(int, self.tetrahedra[active].ravel()))
        adjacency = {node: set() for node in active_nodes}
        for tet in self.tetrahedra[active]:
            for u, v in combinations(map(int, tet), 2):
                adjacency[u].add(v)
                adjacency[v].add(u)

        kept: list[int] = []
        seen: set[int] = set()
        for root in sorted(active_nodes):
            if root in seen:
                continue
            stack = [root]
            component: list[int] = []
            seen.add(root)
            while stack:
                u = stack.pop()
                component.append(u)
                for v in sorted(adjacency[u]):
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            reference = min(component)
            kept.extend(v for v in sorted(component) if v != reference)

        return self.grad[:, kept].astype(float).tocsr()

    def _local_edge_vertex_indices(self, tet: np.ndarray) -> list[tuple[int, int]]:
        local_of_global = {int(vertex): i for i, vertex in enumerate(tet)}
        pairs = []
        for u, v in combinations(map(int, tet), 2):
            a, b = (u, v) if u < v else (v, u)
            pairs.append((local_of_global[a], local_of_global[b]))
        return pairs

    def assemble_nedelec_edge_matrices(
        self,
        reluctivity_tetra: np.ndarray,
        conductivity_tetra: np.ndarray,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Assemble first-order Nedelec curl-curl and conductivity mass matrices."""

        nu = np.asarray(reluctivity_tetra, dtype=float)
        sigma = np.asarray(conductivity_tetra, dtype=float)
        if nu.shape != (self.n_tetrahedra,) or sigma.shape != (self.n_tetrahedra,):
            raise ValueError("tetra material arrays must have shape (n_tetrahedra,)")
        if np.any(nu <= 0) or np.any(sigma < 0):
            raise ValueError("reluctivity must be positive and conductivity non-negative")

        k_rows: list[int] = []
        k_cols: list[int] = []
        k_data: list[float] = []
        m_rows: list[int] = []
        m_cols: list[int] = []
        m_data: list[float] = []

        for q, tet in enumerate(self.tetrahedra):
            V = float(self.volumes[q])
            gradients = _barycentric_gradients(self.vertices, tet)
            local_pairs = self._local_edge_vertex_indices(tet)
            integrals = np.full((4, 4), V / 20.0)
            np.fill_diagonal(integrals, V / 10.0)

            local_K = np.zeros((6, 6), dtype=float)
            local_M = np.zeros((6, 6), dtype=float)
            curls = []
            for i, j in local_pairs:
                curls.append(2.0 * np.cross(gradients[i], gradients[j]))

            for p, (i, j) in enumerate(local_pairs):
                gi, gj = gradients[i], gradients[j]
                for r, (k, l) in enumerate(local_pairs):
                    gk, gl = gradients[k], gradients[l]
                    mass = (
                        np.dot(gj, gl) * integrals[i, k]
                        - np.dot(gj, gk) * integrals[i, l]
                        - np.dot(gi, gl) * integrals[j, k]
                        + np.dot(gi, gk) * integrals[j, l]
                    )
                    local_M[p, r] = sigma[q] * mass
                    local_K[p, r] = nu[q] * V * np.dot(curls[p], curls[r])

            global_edges = self.tet_edge_indices[q]
            for p, ep in enumerate(global_edges):
                for r, er in enumerate(global_edges):
                    if local_K[p, r] != 0.0:
                        k_rows.append(int(ep))
                        k_cols.append(int(er))
                        k_data.append(float(local_K[p, r]))
                    if local_M[p, r] != 0.0:
                        m_rows.append(int(ep))
                        m_cols.append(int(er))
                        m_data.append(float(local_M[p, r]))

        K = sp.coo_matrix((k_data, (k_rows, k_cols)), shape=(self.n_edges, self.n_edges)).tocsr()
        M = sp.coo_matrix((m_data, (m_rows, m_cols)), shape=(self.n_edges, self.n_edges)).tocsr()
        K.sum_duplicates()
        M.sum_duplicates()
        return K, M

    def assemble_edge_mass(self, coefficient_tetra: np.ndarray) -> sp.csr_matrix:
        """Assemble only the first-order Nedelec edge mass for a cell coefficient."""

        coefficient = np.asarray(coefficient_tetra, dtype=float)
        if coefficient.shape != (self.n_tetrahedra,):
            raise ValueError("coefficient_tetra shape mismatch")
        _, mass = self.assemble_nedelec_edge_matrices(
            np.ones(self.n_tetrahedra),
            coefficient,
        )
        return mass

    def assemble_p1_thermal(
        self,
        rho_cp_tetra: np.ndarray,
        conductivity_tetra: np.ndarray,
        *,
        homogeneous_dirichlet_boundary: bool = True,
    ) -> TetrahedralThermalAssembly:
        rho_cp = np.asarray(rho_cp_tetra, dtype=float)
        kappa = np.asarray(conductivity_tetra, dtype=float)
        if rho_cp.shape != (self.n_tetrahedra,) or kappa.shape != (self.n_tetrahedra,):
            raise ValueError("thermal tetra arrays must have shape (n_tetrahedra,)")
        if np.any(rho_cp <= 0) or np.any(kappa <= 0):
            raise ValueError("rho_cp and thermal conductivity must be positive")

        rows: list[int] = []
        cols: list[int] = []
        mass_data: list[float] = []
        stiff_data: list[float] = []

        for q, tet in enumerate(self.tetrahedra):
            V = float(self.volumes[q])
            gradients = _barycentric_gradients(self.vertices, tet)
            local_M = np.full((4, 4), rho_cp[q] * V / 20.0)
            np.fill_diagonal(local_M, rho_cp[q] * V / 10.0)
            local_K = kappa[q] * V * (gradients @ gradients.T)

            for i, vi in enumerate(tet):
                for j, vj in enumerate(tet):
                    rows.append(int(vi))
                    cols.append(int(vj))
                    mass_data.append(float(local_M[i, j]))
                    stiff_data.append(float(local_K[i, j]))

        M_full = sp.coo_matrix(
            (mass_data, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()
        K_full = sp.coo_matrix(
            (stiff_data, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()
        M_full.sum_duplicates()
        K_full.sum_duplicates()

        boundary = self.boundary_nodes()
        if homogeneous_dirichlet_boundary:
            mask = np.ones(self.n_nodes, dtype=bool)
            mask[boundary] = False
            free = np.nonzero(mask)[0]
            if free.size == 0:
                raise ValueError("homogeneous Dirichlet boundary leaves no thermal free nodes")
            M = M_full[free][:, free].tocsr()
            K = K_full[free][:, free].tocsr()
        else:
            free = np.arange(self.n_nodes, dtype=int)
            M = M_full
            K = K_full

        return TetrahedralThermalAssembly(
            M=M,
            K=K,
            free_nodes=free,
            boundary_nodes=boundary,
            n_full_nodes=self.n_nodes,
            tetrahedra=self.tetrahedra.copy(),
        )
