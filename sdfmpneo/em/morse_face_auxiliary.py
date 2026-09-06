from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from .block_riesz import _certified_inverse_inf_upper
from .certified_riesz import CertifiedEnergyPreconditioner
from .magnetic_auxiliary import _maximum_structural_row_matching
from .magnetic_face_auxiliary import (
    _assign_selected_faces_to_tetrahedra,
    _dual_tree_complement_faces,
    _face_area_vector,
)


def _pattern_adjacency(S: sp.csr_matrix, indices: np.ndarray) -> sp.csr_matrix:
    A = S[indices, :][:, indices].copy().tocsr()
    A.data = np.ones(A.nnz, dtype=np.int8)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _cycle_generated_coarse_indices(S: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Split matched coordinates into acyclic fine and topology-generated coarse sets.

    Rows of S are ordered by their matched column, so every diagonal pattern
    entry is nonzero.  A directed edge i->j represents an off-diagonal dependency
    S[i,j].  Repeatedly, every nontrivial SCC contributes exactly one deterministic
    coarse variable: the vertex of largest internal incident degree, with the
    smallest global index breaking ties.  Removing those vertices continues until
    the induced fine graph is acyclic.

    No SCC-size threshold or requested coarse dimension is supplied.  The coarse
    set is generated solely by the directed topology required to break all
    algebraic cycles in the matched face-curl system.
    """

    n = S.shape[0]
    active = np.ones(n, dtype=bool)
    coarse: list[int] = []

    while True:
        indices = np.flatnonzero(active)
        if indices.size <= 1:
            break
        adjacency = _pattern_adjacency(S, indices)
        count, labels = connected_components(
            adjacency,
            directed=True,
            connection="strong",
            return_labels=True,
        )
        components = [np.flatnonzero(labels == k) for k in range(count)]
        cyclic = [component for component in components if component.size > 1]
        if not cyclic:
            break

        removals: list[int] = []
        coo = adjacency.tocoo()
        for component in sorted(cyclic, key=lambda c: int(indices[int(np.min(c))])):
            local_set = set(map(int, component))
            degree = {int(v): 0 for v in component}
            for i, j in zip(coo.row, coo.col):
                ii, jj = int(i), int(j)
                if ii in local_set and jj in local_set:
                    degree[ii] += 1
                    degree[jj] += 1
            chosen_local = min(
                (int(v) for v in component),
                key=lambda v: (-degree[v], int(indices[v])),
            )
            removals.append(int(indices[chosen_local]))

        for index in sorted(set(removals)):
            if active[index]:
                active[index] = False
                coarse.append(index)

    fine = np.flatnonzero(active)
    coarse_array = np.asarray(sorted(coarse), dtype=int)
    return fine.astype(int), coarse_array


def _topological_fine_order(S: sp.csr_matrix, fine: np.ndarray) -> np.ndarray:
    if fine.size <= 1:
        return fine.copy()
    adjacency = _pattern_adjacency(S, fine)
    indegree = np.asarray(adjacency.astype(bool).sum(axis=0)).ravel().astype(int)
    ready = sorted(int(i) for i in np.flatnonzero(indegree == 0))
    order_local: list[int] = []
    while ready:
        i = ready.pop(0)
        order_local.append(i)
        start, stop = adjacency.indptr[i], adjacency.indptr[i + 1]
        for j in sorted(int(v) for v in adjacency.indices[start:stop]):
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)
                ready.sort()
    if len(order_local) != fine.size:
        raise RuntimeError("cycle-generated fine face-curl graph must be acyclic")
    return fine[np.asarray(order_local, dtype=int)]


def _build_matched_face_system(problem):
    mesh = problem.mesh
    n_A = int(problem.n_A)
    R = sp.csr_matrix(problem.a_basis, dtype=int)
    C_R = (mesh.curl @ R).tocsr()
    C_R.sum_duplicates()
    C_R.eliminate_zeros()

    basis_faces = _dual_tree_complement_faces(mesh, n_A)
    S0 = C_R[basis_faces, :].tocsr()
    row_for_column = _maximum_structural_row_matching(S0)
    selected_faces = basis_faces[row_for_column]
    S = S0[row_for_column, :].astype(complex).tocsr()
    diagonal = np.asarray(S.diagonal(), dtype=complex)
    if np.any(diagonal == 0.0):
        raise RuntimeError("matched topology-exact face-curl system must have nonzero diagonal")

    assignment = _assign_selected_faces_to_tetrahedra(mesh, selected_faces)
    return selected_faces, assignment, S


def _build_local_face_weights(problem, selected_faces: np.ndarray, assignment: np.ndarray):
    mesh = problem.mesh
    n = selected_faces.size
    nu = np.asarray(problem.reluctivity_tetra, dtype=float)
    if nu.shape != (mesh.n_tetrahedra,) or np.any(nu <= 0.0):
        raise ValueError("reluctivity_tetra must be positive")

    w_rows: list[int] = []
    w_cols: list[int] = []
    w_data: list[complex] = []
    winv_blocks: list[tuple[np.ndarray, np.ndarray]] = []

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
        gram = 0.5 * ((normals @ normals.T) + (normals @ normals.T).T)
        try:
            factor = scipy.linalg.cho_factor(gram, lower=True, check_finite=False)
            gram_inverse = scipy.linalg.cho_solve(
                factor,
                np.eye(positions.size),
                check_finite=False,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError("assigned face normals are not locally independent") from exc

        scale = float(nu[tet] * mesh.volumes[tet])
        local_W = scale * gram_inverse
        local_Winv = gram / scale
        winv_blocks.append((positions.astype(int), np.asarray(local_Winv, dtype=complex)))
        for i, pi in enumerate(positions):
            for j, pj in enumerate(positions):
                value = complex(local_W[i, j])
                if value != 0.0:
                    w_rows.append(int(pi))
                    w_cols.append(int(pj))
                    w_data.append(value)

    W = sp.coo_matrix(
        (w_data, (w_rows, w_cols)),
        shape=(n, n),
        dtype=complex,
    ).tocsr()
    W.sum_duplicates()
    W.eliminate_zeros()
    return W, tuple(winv_blocks)


@dataclass(frozen=True)
class MorseFaceCirculationEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """Topology-generated two-level exact action for the face-circulation lower energy.

    A topology-exact dual-tree face basis gives a nonsingular square face-curl
    matrix S.  Structural row matching makes the diagonal pattern explicit.
    Directed cycles in that matched system are not hidden inside dense SCC
    factors: cycle-breaking variables become a coarse space generated entirely by
    topology.  The remaining fine matrix is acyclic and therefore sparse upper
    triangular after topological ordering.

    For

        P_A = S^H W S,

    Stokes/local magnetic-energy inequalities give

        K_A >= P_A,

    so the magnetic spectral-equivalence constant is exactly m_K=1.  Applying
    P_A^{-1} uses

        S^{-1} W^{-1} S^{-H},

    with S^{-1} realized by sparse fine triangular substitution plus a coarse
    Schur-complement solve.  No full magnetic factorization, strength threshold,
    configured block size, or requested coarse dimension appears.
    """

    dimension: int
    lower_spectral_equivalence_bound: float
    selected_faces: np.ndarray
    face_tetra_assignment: np.ndarray
    fine_dimension: int
    coarse_dimension: int
    coarse_fraction: float
    inverse_inf_upper_bound: float
    _S: sp.csr_matrix
    _W: sp.csr_matrix
    _Winv_blocks: tuple[tuple[np.ndarray, np.ndarray], ...]
    _fine: np.ndarray
    _coarse: np.ndarray
    _A: sp.csr_matrix
    _B: sp.csr_matrix
    _C: sp.csr_matrix
    _D: np.ndarray
    _schur_factor: tuple[np.ndarray, np.ndarray] | None
    _P: sp.csr_matrix

    @classmethod
    def build_from_problem(cls, problem) -> "MorseFaceCirculationEnergyPreconditioner":
        selected_faces, assignment, S = _build_matched_face_system(problem)
        W, Winv_blocks = _build_local_face_weights(problem, selected_faces, assignment)
        P = (S.conj().T @ W @ S).tocsr()
        P = (0.5 * (P + P.conj().T)).tocsr()
        P.sum_duplicates()
        P.eliminate_zeros()

        fine_raw, coarse = _cycle_generated_coarse_indices(S)
        fine = _topological_fine_order(S, fine_raw)
        A = S[fine, :][:, fine].tocsr()
        B = S[fine, :][:, coarse].tocsr()
        C = S[coarse, :][:, fine].tocsr()
        D = S[coarse, :][:, coarse].toarray().astype(complex)

        if fine.size:
            diagonal = np.asarray(A.diagonal(), dtype=complex)
            if np.any(diagonal == 0.0):
                raise RuntimeError("acyclic fine face-curl block lost a matched pivot")
            lower_pattern = sp.tril(A, k=-1).tocsr()
            if lower_pattern.nnz:
                raise RuntimeError("fine face-curl block is not upper triangular after topological ordering")

        schur_factor = None
        if coarse.size:
            if fine.size:
                X = spla.spsolve_triangular(A, B.toarray(), lower=False)
                Schur = D - np.asarray(C @ X, dtype=complex)
            else:
                Schur = D.copy()
            factor, pivots = scipy.linalg.lu_factor(Schur, check_finite=False)
            if np.any(np.diag(factor) == 0.0):
                raise ValueError("topology-generated coarse face-curl Schur complement is singular")
            schur_factor = (factor, pivots)

        provisional = cls(
            dimension=int(S.shape[0]),
            lower_spectral_equivalence_bound=1.0,
            selected_faces=selected_faces.copy(),
            face_tetra_assignment=assignment.copy(),
            fine_dimension=int(fine.size),
            coarse_dimension=int(coarse.size),
            coarse_fraction=float(coarse.size / S.shape[0]),
            inverse_inf_upper_bound=float("nan"),
            _S=S,
            _W=W,
            _Winv_blocks=Winv_blocks,
            _fine=fine,
            _coarse=coarse,
            _A=A,
            _B=B,
            _C=C,
            _D=D,
            _schur_factor=schur_factor,
            _P=P,
        )
        inverse_upper = _certified_inverse_inf_upper(P, provisional)
        return replace(provisional, inverse_inf_upper_bound=float(inverse_upper))

    def auxiliary_matrix(self) -> sp.csr_matrix:
        return self._P.copy()

    def _solve_fine(self, rhs: np.ndarray, *, adjoint: bool) -> np.ndarray:
        if self.fine_dimension == 0:
            return np.empty(0, dtype=complex)
        vector = np.asarray(rhs, dtype=complex)
        if vector.shape[0] != self.fine_dimension:
            raise ValueError("fine face-curl rhs mismatch")
        if vector.ndim == 1:
            if adjoint:
                return np.asarray(
                    spla.spsolve_triangular(self._A.conj().T.tocsr(), vector, lower=True),
                    dtype=complex,
                )
            return np.asarray(spla.spsolve_triangular(self._A, vector, lower=False), dtype=complex)
        columns = [self._solve_fine(vector[:, j], adjoint=adjoint) for j in range(vector.shape[1])]
        return np.column_stack(columns)

    def _solve_S(self, rhs: np.ndarray, *, adjoint: bool) -> np.ndarray:
        b = np.asarray(rhs, dtype=complex)
        if b.shape != (self.dimension,):
            raise ValueError("face-curl rhs dimension mismatch")
        f = self._fine
        c = self._coarse
        out = np.zeros(self.dimension, dtype=complex)

        if not adjoint:
            bf = b[f]
            bc = b[c]
            if self.fine_dimension:
                u = self._solve_fine(bf, adjoint=False)
            else:
                u = np.empty(0, dtype=complex)
            if self.coarse_dimension:
                rhs_c = bc - np.asarray(self._C @ u, dtype=complex)
                xc = scipy.linalg.lu_solve(self._schur_factor, rhs_c, trans=0, check_finite=False)
                xf = self._solve_fine(bf - np.asarray(self._B @ xc, dtype=complex), adjoint=False)
                out[c] = xc
            else:
                xf = u
            out[f] = xf
            return out

        bf = b[f]
        bc = b[c]
        if self.fine_dimension:
            u = self._solve_fine(bf, adjoint=True)
        else:
            u = np.empty(0, dtype=complex)
        if self.coarse_dimension:
            rhs_c = bc - np.asarray(self._B.conj().T @ u, dtype=complex)
            yc = scipy.linalg.lu_solve(self._schur_factor, rhs_c, trans=2, check_finite=False)
            yf = self._solve_fine(
                bf - np.asarray(self._C.conj().T @ yc, dtype=complex),
                adjoint=True,
            )
            out[c] = yc
        else:
            yf = u
        out[f] = yf
        return out

    def _apply_Winv(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=complex)
        if v.shape != (self.dimension,):
            raise ValueError("face-weight rhs dimension mismatch")
        out = np.zeros_like(v)
        for positions, block in self._Winv_blocks:
            out[positions] = block @ v[positions]
        return out

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        z = self._solve_S(rhs, adjoint=True)
        w = self._apply_Winv(z)
        return self._solve_S(w, adjoint=False)
