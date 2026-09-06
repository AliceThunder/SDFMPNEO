from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .certified_riesz import CertifiedEnergyPreconditioner


def _deterministic_structural_row_matching(S: sp.csr_matrix) -> np.ndarray:
    matrix = sp.csc_matrix(S)
    n_rows, n_cols = matrix.shape
    matched_column_for_row = np.full(n_rows, -1, dtype=int)

    def augment(column: int, seen_rows: np.ndarray) -> bool:
        start, stop = matrix.indptr[column], matrix.indptr[column + 1]
        for row in sorted(int(r) for r in matrix.indices[start:stop]):
            if seen_rows[row]:
                continue
            seen_rows[row] = True
            previous = int(matched_column_for_row[row])
            if previous < 0 or augment(previous, seen_rows):
                matched_column_for_row[row] = column
                return True
        return False

    for column in range(n_cols):
        seen = np.zeros(n_rows, dtype=bool)
        if not augment(column, seen):
            raise ValueError("face-curl incidence has no full structural row matching")

    row_for_column = np.full(n_cols, -1, dtype=int)
    for row, column in enumerate(matched_column_for_row):
        if column >= 0:
            row_for_column[column] = row
    if np.any(row_for_column < 0):
        raise RuntimeError("internal face-curl matching is incomplete")
    return row_for_column


def _scc_topological_permutation(S: sp.csr_matrix) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    adjacency = S.copy().tocsr()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.int8)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    count, labels = connected_components(
        adjacency,
        directed=True,
        connection="strong",
        return_labels=True,
    )
    components = [np.flatnonzero(labels == k) for k in range(count)]
    outgoing = [set() for _ in range(count)]
    indegree = np.zeros(count, dtype=int)
    coo = adjacency.tocoo()
    for i, j in zip(coo.row, coo.col):
        p, q = int(labels[i]), int(labels[j])
        if p != q and q not in outgoing[p]:
            outgoing[p].add(q)
            indegree[q] += 1
    ready = sorted(k for k in range(count) if indegree[k] == 0)
    order: list[int] = []
    while ready:
        p = ready.pop(0)
        order.append(p)
        for q in sorted(outgoing[p]):
            indegree[q] -= 1
            if indegree[q] == 0:
                ready.append(q)
                ready.sort()
    if len(order) != count:
        raise RuntimeError("face-curl SCC condensation graph must be acyclic")
    ordered = tuple(np.asarray(components[k], dtype=int) for k in order)
    permutation = np.concatenate(ordered) if ordered else np.empty(0, dtype=int)
    return permutation, ordered


def _face_area_vector(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[int(i)] for i in face)
    return 0.5 * np.cross(b - a, c - a)


def _incident_tetrahedra(mesh) -> tuple[tuple[int, ...], ...]:
    face_map = {tuple(map(int, face)): i for i, face in enumerate(mesh.face_vertices)}
    incident: list[list[int]] = [[] for _ in range(mesh.n_faces)]
    for q, tet in enumerate(mesh.tetrahedra):
        for local in combinations(map(int, tet), 3):
            f = face_map[tuple(sorted(local))]
            incident[f].append(q)
    return tuple(tuple(sorted(items)) for items in incident)


def _assign_selected_faces_to_tetrahedra(mesh, selected_faces: np.ndarray) -> np.ndarray:
    """Find the lexicographically first physically independent face assignment.

    Each selected global face is assigned to one incident tetrahedron.  A
    tetrahedron may receive at most three faces, and their area vectors must have
    full row rank.  There is no capacity parameter: three is the physical curl
    dimension in 3-D.  Backtracking is exhaustive over the finite incidence
    choices and therefore reports failure rather than tuning an assignment rule.
    """

    incident = _incident_tetrahedra(mesh)
    selected = [int(f) for f in selected_faces]
    assignment = np.full(len(selected), -1, dtype=int)
    by_tet: list[list[int]] = [[] for _ in range(mesh.n_tetrahedra)]

    def admissible(tet: int, face: int) -> bool:
        trial = by_tet[tet] + [face]
        if len(trial) > 3:
            return False
        normals = np.vstack([_face_area_vector(mesh.vertices, mesh.face_vertices[f]) for f in trial])
        return np.linalg.matrix_rank(normals) == len(trial)

    def search(position: int) -> bool:
        if position == len(selected):
            return True
        face = selected[position]
        for tet in incident[face]:
            if not admissible(tet, face):
                continue
            by_tet[tet].append(face)
            assignment[position] = tet
            if search(position + 1):
                return True
            by_tet[tet].pop()
            assignment[position] = -1
        return False

    if not search(0):
        raise ValueError("selected face constraints cannot be assigned to independent tetrahedral curl components")
    return assignment


@dataclass(frozen=True)
class _LocalSCCLU:
    positions: np.ndarray
    factor: np.ndarray
    pivots: np.ndarray


@dataclass(frozen=True)
class MagneticFaceCirculationEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """Topology-exact face-circulation auxiliary magnetic preconditioner.

    With alpha denoting cotree magnetic coordinates and

        S = (C R_A)[selected faces,:],
        y = S alpha,

    Stokes' theorem gives each selected face circulation directly from the
    constant tetrahedral curl.  For the selected faces assigned to tetrahedron q,
    let B_q contain their oriented area vectors as rows.  Then

        nu_q V_q ||curl A||^2
        >= y_q^T W_q y_q,
        W_q = nu_q V_q (B_q B_q^T)^-1.

    Summing over tetrahedra yields the exact operator inequality

        K_A >= P_A = S^T W S.

    Hence m_K=1.  S is an integer topology matrix; its SCC decomposition contains
    no floating near-zero edges.  P_A^-1 is applied as

        S^-1 W^-1 S^-T,

    where W_q^-1=(B_q B_q^T)/(nu_q V_q) is explicit and only SCC diagonal blocks
    of S are locally factorized.
    """

    dimension: int
    lower_spectral_equivalence_bound: float
    selected_faces: np.ndarray
    face_tetra_assignment: np.ndarray
    permutation: np.ndarray
    component_sizes: tuple[int, ...]
    maximum_scc_size: int
    _S_permuted: np.ndarray
    _local_factors: tuple[_LocalSCCLU, ...]
    _Winv: np.ndarray

    @classmethod
    def build_from_problem(cls, problem) -> "MagneticFaceCirculationEnergyPreconditioner":
        required = ("mesh", "reluctivity_tetra", "a_basis", "n_A")
        if any(not hasattr(problem, name) for name in required):
            raise TypeError("problem does not expose tetrahedral face-curl data")
        mesh = problem.mesh
        R = sp.csr_matrix(problem.a_basis, dtype=int)
        C_R = (mesh.curl @ R).tocsr()
        C_R.sum_duplicates()
        C_R.eliminate_zeros()
        n_A = int(problem.n_A)
        if C_R.shape[1] != n_A or n_A == 0:
            raise ValueError("face-curl gauge coordinates are inconsistent")

        selected_faces = _deterministic_structural_row_matching(C_R)
        S = C_R[selected_faces, :].tocsr()
        if S.shape != (n_A, n_A):
            raise RuntimeError("selected face-curl topology matrix is not square")
        if np.any(np.asarray(S.diagonal()) == 0):
            raise RuntimeError("matched face-curl topology matrix must have nonzero diagonal")

        assignment = _assign_selected_faces_to_tetrahedra(mesh, selected_faces)
        W_inv = np.zeros((n_A, n_A), dtype=float)
        nu = np.asarray(problem.reluctivity_tetra, dtype=float)
        for tet in range(mesh.n_tetrahedra):
            positions = np.flatnonzero(assignment == tet)
            if positions.size == 0:
                continue
            normals = np.vstack(
                [
                    _face_area_vector(mesh.vertices, mesh.face_vertices[int(selected_faces[p])])
                    for p in positions
                ]
            )
            gram = normals @ normals.T
            W_inv[np.ix_(positions, positions)] = gram / float(nu[tet] * mesh.volumes[tet])

        permutation, components = _scc_topological_permutation(S)
        Sperm = S[permutation, :][:, permutation].toarray().astype(complex)
        factors: list[_LocalSCCLU] = []
        offset = 0
        for component in components:
            size = int(component.size)
            positions = np.arange(offset, offset + size, dtype=int)
            block = Sperm[np.ix_(positions, positions)]
            factor, pivots = scipy.linalg.lu_factor(block, check_finite=False)
            if np.any(np.diag(factor) == 0):
                raise ValueError("selected face-curl SCC block is singular")
            factors.append(_LocalSCCLU(positions, factor, pivots))
            offset += size

        return cls(
            dimension=n_A,
            lower_spectral_equivalence_bound=1.0,
            selected_faces=selected_faces.copy(),
            face_tetra_assignment=assignment.copy(),
            permutation=permutation.copy(),
            component_sizes=tuple(int(component.size) for component in components),
            maximum_scc_size=max(int(component.size) for component in components),
            _S_permuted=Sperm,
            _local_factors=tuple(factors),
            _Winv=W_inv,
        )

    def _solve_S(self, rhs: np.ndarray, *, adjoint: bool) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        if vector.shape != (self.dimension,):
            raise ValueError("face-circulation auxiliary rhs mismatch")
        b = vector[self.permutation].copy()
        x = np.zeros_like(b)
        if not adjoint:
            for local in reversed(self._local_factors):
                pos = local.positions
                end = int(pos[-1]) + 1
                local_rhs = b[pos].copy()
                if end < self.dimension:
                    local_rhs -= self._S_permuted[np.ix_(pos, np.arange(end, self.dimension))] @ x[end:]
                x[pos] = scipy.linalg.lu_solve(
                    (local.factor, local.pivots), local_rhs, trans=0, check_finite=False
                )
        else:
            for local in self._local_factors:
                pos = local.positions
                start = int(pos[0])
                local_rhs = b[pos].copy()
                if start > 0:
                    local_rhs -= (
                        self._S_permuted[np.ix_(np.arange(start), pos)].conj().T @ x[:start]
                    )
                x[pos] = scipy.linalg.lu_solve(
                    (local.factor, local.pivots), local_rhs, trans=2, check_finite=False
                )
        out = np.empty_like(x)
        out[self.permutation] = x
        return out

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        z = self._solve_S(rhs, adjoint=True)
        w = self._Winv @ z
        return self._solve_S(w, adjoint=False)

    def auxiliary_matrix(self, problem) -> sp.csr_matrix:
        C_R = (problem.mesh.curl @ sp.csr_matrix(problem.a_basis, dtype=int)).tocsr()
        S = C_R[self.selected_faces, :].toarray().astype(float)
        W = np.linalg.inv(self._Winv)
        return sp.csr_matrix(S.T @ W @ S)
