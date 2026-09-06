from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .certified_riesz import CertifiedEnergyPreconditioner


def _gamma(operation_count: int) -> float:
    eps = np.finfo(float).eps
    count = max(1, int(operation_count))
    if count * eps >= 1.0:
        raise FloatingPointError("operation count is too large for the gamma_k bound")
    return count * eps / (1.0 - count * eps)


def _barycentric_gradients(vertices: np.ndarray, tet: np.ndarray) -> np.ndarray:
    x = vertices[tet]
    matrix = np.column_stack([np.ones(4), x])
    inverse = np.linalg.inv(matrix)
    return inverse[1:, :].T


def build_gauge_restricted_magnetic_curl_factor(problem) -> sp.csr_matrix:
    """Build F_A such that the physical magnetic energy is K_A=F_A^H F_A.

    For tetrahedron q and Cartesian component c, the row contribution is

        sqrt(nu_q |T_q|) curl(N_e)_c.

    The six local first-order Nedelec curls are constant on a tetrahedron, hence
    stacking the three physical curl components for every tetrahedron gives an
    exact Gram factor of the assembled curl-curl operator in exact arithmetic.
    The tree-cotree gauge restriction is then applied on the right.
    """

    required = ("mesh", "reluctivity_tetra", "a_basis", "magnetic_stiffness")
    if any(not hasattr(problem, name) for name in required):
        raise TypeError("problem does not expose tetrahedral magnetic factor data")

    mesh = problem.mesh
    nu = np.asarray(problem.reluctivity_tetra, dtype=float)
    if nu.shape != (mesh.n_tetrahedra,) or np.any(nu <= 0.0):
        raise ValueError("reluctivity_tetra must be positive with shape (n_tetrahedra,)")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for q, tet in enumerate(mesh.tetrahedra):
        gradients = _barycentric_gradients(mesh.vertices, tet)
        local_of_global = {int(vertex): i for i, vertex in enumerate(tet)}
        local_pairs: list[tuple[int, int]] = []
        for u, v in combinations(map(int, tet), 2):
            a, b = (u, v) if u < v else (v, u)
            local_pairs.append((local_of_global[a], local_of_global[b]))
        scale = math.sqrt(float(nu[q] * mesh.volumes[q]))
        for p, (i, j) in enumerate(local_pairs):
            curl = 2.0 * np.cross(gradients[i], gradients[j])
            edge = int(mesh.tet_edge_indices[q, p])
            for component in range(3):
                value = scale * float(curl[component])
                if value != 0.0:
                    rows.append(3 * q + component)
                    cols.append(edge)
                    data.append(value)

    F_edge = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(3 * mesh.n_tetrahedra, mesh.n_edges),
        dtype=float,
    ).tocsr()
    F_edge.sum_duplicates()
    F_edge.eliminate_zeros()
    R = sp.csr_matrix(problem.a_basis, dtype=complex)
    F_A = (F_edge.astype(complex) @ R).tocsr()
    F_A.sum_duplicates()
    F_A.eliminate_zeros()

    K_A = (R.conj().T @ problem.magnetic_stiffness.astype(complex) @ R).tocsr()
    gram = (F_A.conj().T @ F_A).tocsr()
    defect = (K_A - gram).tocsr()
    defect_norm = float(np.sqrt(np.sum(np.abs(defect.data) ** 2))) if defect.nnz else 0.0
    K_norm = float(np.sqrt(np.sum(np.abs(K_A.data) ** 2))) if K_A.nnz else 0.0
    # Both matrices are assembled from the same O(n_tet) local arithmetic.  The
    # comparison is only an implementation consistency guard; it does not define
    # a physical approximation tolerance.
    operation_count = max(1, 256 * mesh.n_tetrahedra)
    backward = _gamma(operation_count) * max(K_norm, 1.0)
    if defect_norm > backward:
        raise RuntimeError(
            "reconstructed magnetic curl factor does not reproduce the assembled magnetic stiffness to backward-error scale"
        )
    return F_A


def _maximum_structural_row_matching(F: sp.csr_matrix) -> np.ndarray:
    """Deterministic maximum matching from columns to distinct nonzero rows."""

    matrix = sp.csc_matrix(F)
    n_rows, n_cols = matrix.shape
    matched_column_for_row = np.full(n_rows, -1, dtype=int)

    def augment(column: int, seen_rows: np.ndarray) -> bool:
        start, stop = matrix.indptr[column], matrix.indptr[column + 1]
        candidate_rows = sorted(int(r) for r in matrix.indices[start:stop])
        for row in candidate_rows:
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
            raise ValueError("magnetic curl factor has no full structural row matching")

    row_for_column = np.full(n_cols, -1, dtype=int)
    for row, column in enumerate(matched_column_for_row):
        if column >= 0:
            row_for_column[column] = row
    if np.any(row_for_column < 0):
        raise RuntimeError("internal magnetic row matching is incomplete")
    return row_for_column


def _strong_components_of_selected_factor(S: sp.csr_matrix) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return a topological SCC ordering of the square selected row factor.

    Rows are ordered by their matched column, so the diagonal pattern is nonzero.
    A directed edge i->j is created for every off-diagonal S[i,j].  SCCs are the
    irreducible diagonal blocks of a block-triangular permutation of S.
    """

    n = S.shape[0]
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

    # Condensation graph edge p->q whenever S has a nonzero from component p to q.
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
        raise RuntimeError("SCC condensation graph must be acyclic")

    ordered_components = tuple(np.asarray(components[p], dtype=int) for p in order)
    permutation = np.concatenate(ordered_components) if ordered_components else np.empty(0, dtype=int)
    return permutation, ordered_components


@dataclass(frozen=True)
class _LocalLU:
    positions: np.ndarray
    factor: np.ndarray
    pivots: np.ndarray


@dataclass(frozen=True)
class MagneticCurlSubsetEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """Topological auxiliary preconditioner P_A=S^H S for the magnetic block.

    A structural matching selects one physical curl row per gauge-reduced magnetic
    coordinate from F_A.  With S denoting the selected square row submatrix,

        K_A = F_A^H F_A >= S^H S = P_A,

    so the certified lower spectral-equivalence constant is exactly one at the
    operator level.  No eigenvalue estimate, K_A inverse, damping, or strength
    threshold is needed.

    S is permuted into SCC block-triangular form.  Only the irreducible diagonal
    SCC blocks are LU-factorized; inter-block coupling is handled by exact block
    substitution.  The exposed ``maximum_scc_size`` prevents a globally
    irreducible factor from being mislabeled as a local scalable action.
    """

    dimension: int
    lower_spectral_equivalence_bound: float
    selected_rows: np.ndarray
    permutation: np.ndarray
    component_sizes: tuple[int, ...]
    maximum_scc_size: int
    _S_permuted: np.ndarray
    _local_factors: tuple[_LocalLU, ...]

    @classmethod
    def build_from_problem(cls, problem) -> "MagneticCurlSubsetEnergyPreconditioner":
        F_A = build_gauge_restricted_magnetic_curl_factor(problem)
        n = F_A.shape[1]
        if n == 0:
            raise ValueError("magnetic gauge space is empty")
        selected_rows = _maximum_structural_row_matching(F_A)
        S = F_A[selected_rows, :].tocsr()
        if S.shape != (n, n):
            raise RuntimeError("selected magnetic curl factor is not square")
        diagonal = np.asarray(S.diagonal(), dtype=complex)
        if np.any(diagonal == 0.0):
            raise RuntimeError("matched magnetic curl factor must have a nonzero diagonal")

        permutation, components = _strong_components_of_selected_factor(S)
        Sperm = S[permutation, :][:, permutation].toarray().astype(complex)

        # In this ordering the SCC condensation graph is upper block triangular.
        factors: list[_LocalLU] = []
        offset = 0
        for component in components:
            size = int(component.size)
            positions = np.arange(offset, offset + size, dtype=int)
            block = Sperm[np.ix_(positions, positions)]
            factor, pivots = scipy.linalg.lu_factor(block, check_finite=False)
            if np.any(np.diag(factor) == 0.0):
                raise ValueError("selected magnetic SCC block is singular")
            factors.append(_LocalLU(positions, factor, pivots))
            offset += size

        return cls(
            dimension=n,
            lower_spectral_equivalence_bound=1.0,
            selected_rows=selected_rows.copy(),
            permutation=permutation.copy(),
            component_sizes=tuple(int(component.size) for component in components),
            maximum_scc_size=max(int(component.size) for component in components),
            _S_permuted=Sperm,
            _local_factors=tuple(factors),
        )

    def _solve_S(self, rhs: np.ndarray, *, adjoint: bool) -> np.ndarray:
        b_original = np.asarray(rhs, dtype=complex)
        if b_original.shape != (self.dimension,):
            raise ValueError("magnetic auxiliary rhs dimension mismatch")
        b = b_original[self.permutation].copy()
        x = np.zeros_like(b)

        if not adjoint:
            # Upper block triangular: backward substitution.
            iterable = reversed(self._local_factors)
            for local in iterable:
                pos = local.positions
                end = int(pos[-1]) + 1
                rhs_local = b[pos].copy()
                if end < self.dimension:
                    rhs_local -= self._S_permuted[np.ix_(pos, np.arange(end, self.dimension))] @ x[end:]
                x[pos] = scipy.linalg.lu_solve(
                    (local.factor, local.pivots),
                    rhs_local,
                    trans=0,
                    check_finite=False,
                )
        else:
            # S^H is lower block triangular: forward substitution.
            for local in self._local_factors:
                pos = local.positions
                start = int(pos[0])
                rhs_local = b[pos].copy()
                if start > 0:
                    rhs_local -= (
                        self._S_permuted[np.ix_(np.arange(0, start), pos)].conj().T
                        @ x[:start]
                    )
                x[pos] = scipy.linalg.lu_solve(
                    (local.factor, local.pivots),
                    rhs_local,
                    trans=2,
                    check_finite=False,
                )

        out = np.empty_like(x)
        out[self.permutation] = x
        return out

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        z = self._solve_S(rhs, adjoint=True)
        return self._solve_S(z, adjoint=False)
