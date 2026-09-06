from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from .adaptive_block import AdaptiveAggregateEnergyPreconditioner
from .certified_riesz import CertifiedEnergyPreconditioner


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
    out = tuple(tuple(sorted(items)) for items in incident)
    if any(len(items) not in (1, 2) for items in out):
        raise ValueError("tetrahedral chart must be a conforming 3-D manifold with one or two cells per face")
    return out


def _dual_tree_complement_faces(mesh, expected_dimension: int) -> np.ndarray:
    """Return the complement of a deterministic augmented-dual spanning tree.

    The augmented dual graph contains one node per tetrahedron and one exterior
    node.  Interior faces join two tetrahedra; boundary faces join one tetrahedron
    to the exterior.  For a connected contractible tetrahedral chart the
    complement contains exactly the magnetic cotree dimension.  A count mismatch
    exposes an unhandled topological/harmonic chart rather than invoking a
    floating numerical rank decision.
    """

    incident = _incident_tetrahedra(mesh)
    exterior = int(mesh.n_tetrahedra)
    n_dual_nodes = exterior + 1
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_dual_nodes)]
    for face, cells in enumerate(incident):
        if len(cells) == 2:
            u, v = int(cells[0]), int(cells[1])
        else:
            u, v = int(cells[0]), exterior
        adjacency[u].append((v, face))
        adjacency[v].append((u, face))
    for entries in adjacency:
        entries.sort(key=lambda item: (item[1], item[0]))

    visited = np.zeros(n_dual_nodes, dtype=bool)
    visited[exterior] = True
    queue = [exterior]
    head = 0
    tree_faces: list[int] = []
    while head < len(queue):
        u = queue[head]
        head += 1
        for v, face in adjacency[u]:
            if visited[v]:
                continue
            visited[v] = True
            queue.append(v)
            tree_faces.append(int(face))

    if not np.all(visited):
        raise ValueError("augmented tetrahedral dual graph is disconnected")
    if len(tree_faces) != n_dual_nodes - 1:
        raise RuntimeError("internal dual spanning-tree construction failed")

    mask = np.ones(mesh.n_faces, dtype=bool)
    mask[np.asarray(tree_faces, dtype=int)] = False
    selected = np.flatnonzero(mask)
    if selected.size != int(expected_dimension):
        raise ValueError(
            "primal/dual tree counts do not match the magnetic cotree dimension; "
            "the chart requires explicit harmonic/topological modes"
        )
    return selected


def _assign_selected_faces_to_tetrahedra(mesh, selected_faces: np.ndarray) -> np.ndarray:
    """Assign selected faces to cells using only the exact 3-D curl dimension.

    Any at most three distinct faces of a non-degenerate tetrahedron have
    linearly independent normals.  Therefore no numerical rank tolerance is
    required: a tetrahedron may receive at most three selected faces.
    """

    incident = _incident_tetrahedra(mesh)
    selected = [int(f) for f in selected_faces]
    assignment = np.full(len(selected), -1, dtype=int)
    by_tet: list[list[int]] = [[] for _ in range(mesh.n_tetrahedra)]

    def search(position: int) -> bool:
        if position == len(selected):
            return True
        face = selected[position]
        for tet in incident[face]:
            if len(by_tet[tet]) >= 3:
                continue
            by_tet[tet].append(face)
            assignment[position] = tet
            if search(position + 1):
                return True
            by_tet[tet].pop()
            assignment[position] = -1
        return False

    if not search(0):
        raise ValueError(
            "dual-tree face basis cannot be assigned within the exact three-component tetrahedral curl dimension"
        )
    return assignment


def _build_face_energy_lower_matrix(problem, selected_faces: np.ndarray, assignment: np.ndarray) -> sp.csr_matrix:
    """Build P_A=S^T W S with P_A <= K_A from local Stokes energy bounds."""

    mesh = problem.mesh
    R = sp.csr_matrix(problem.a_basis, dtype=int)
    C_R = (mesh.curl @ R).tocsr()
    S = C_R[selected_faces, :].astype(complex).tocsr()
    n_A = int(problem.n_A)
    if S.shape != (n_A, n_A):
        raise RuntimeError("dual-tree face-curl topology matrix is not square")

    nu = np.asarray(problem.reluctivity_tetra, dtype=float)
    w_rows: list[int] = []
    w_cols: list[int] = []
    w_data: list[complex] = []
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
        gram = 0.5 * (gram + gram.T)
        try:
            factor = scipy.linalg.cho_factor(gram, lower=True, check_finite=False)
            gram_inverse = scipy.linalg.cho_solve(
                factor,
                np.eye(positions.size),
                check_finite=False,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError("assigned tetrahedral face normals are not positive definite") from exc
        local_W = float(nu[tet] * mesh.volumes[tet]) * gram_inverse
        for i, pi in enumerate(positions):
            for j, pj in enumerate(positions):
                value = complex(local_W[i, j])
                if value != 0.0:
                    w_rows.append(int(pi))
                    w_cols.append(int(pj))
                    w_data.append(value)

    W = sp.coo_matrix(
        (w_data, (w_rows, w_cols)),
        shape=(n_A, n_A),
        dtype=complex,
    ).tocsr()
    W.sum_duplicates()
    W.eliminate_zeros()
    P_face = (S.conj().T @ W @ S).tocsr()
    P_face = (0.5 * (P_face + P_face.conj().T)).tocsr()
    P_face.sum_duplicates()
    P_face.eliminate_zeros()
    return P_face


@dataclass(frozen=True)
class MagneticFaceCirculationEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """Local certified magnetic action derived from face-circulation energy.

    The topology/geometry construction first proves

        K_A >= P_face = S^T W S.

    No inverse of S is used in the production action.  Instead a
    certificate-driven aggregate preconditioner constructs a block-diagonal
    matrix Q_A from principal blocks of P_face and proves

        P_face >= m_face Q_A.

    Hence

        K_A >= m_face Q_A.

    ``solve(rhs)`` applies Q_A^-1 using only the final local aggregate Cholesky
    factors.  The block size is selected by the block-Gershgorin certificate and
    is exposed explicitly.  If it collapses to the whole magnetic dimension the
    result is correctness-only and cannot be labelled scalable.
    """

    dimension: int
    lower_spectral_equivalence_bound: float
    selected_faces: np.ndarray
    face_tetra_assignment: np.ndarray
    maximum_block_size: int
    aggregation_steps: int
    final_block_count: int
    _face_energy_matrix: sp.csr_matrix
    _local_action: AdaptiveAggregateEnergyPreconditioner

    @classmethod
    def build_from_problem(cls, problem) -> "MagneticFaceCirculationEnergyPreconditioner":
        required = ("mesh", "reluctivity_tetra", "a_basis", "n_A")
        if any(not hasattr(problem, name) for name in required):
            raise TypeError("problem does not expose tetrahedral face-curl data")
        n_A = int(problem.n_A)
        if n_A <= 0:
            raise ValueError("magnetic gauge space is empty")

        selected_faces = _dual_tree_complement_faces(problem.mesh, n_A)
        assignment = _assign_selected_faces_to_tetrahedra(problem.mesh, selected_faces)
        P_face = _build_face_energy_lower_matrix(problem, selected_faces, assignment)
        local_action = AdaptiveAggregateEnergyPreconditioner.build(P_face)
        lower = float(local_action.lower_spectral_equivalence_bound)
        if lower <= 0.0:
            raise ValueError("face-circulation local action has no positive certified lower bound")

        return cls(
            dimension=n_A,
            lower_spectral_equivalence_bound=lower,
            selected_faces=selected_faces.copy(),
            face_tetra_assignment=assignment.copy(),
            maximum_block_size=int(local_action.maximum_block_size),
            aggregation_steps=int(local_action.aggregation_steps),
            final_block_count=int(local_action.final_block_count),
            _face_energy_matrix=P_face,
            _local_action=local_action,
        )

    @property
    def inverse_inf_upper_bound(self) -> float:
        """Certified upper bound for ||Q_A^-1||_inf."""

        return self._local_action.inverse_inf_upper_bound

    def face_energy_matrix(self) -> sp.csr_matrix:
        """Return P_face in the chain K_A >= P_face >= m_face Q_A."""

        return self._face_energy_matrix.copy()

    def preconditioner_matrix(self) -> sp.csr_matrix:
        """Return the local block matrix Q_A actually inverted by ``solve``."""

        return self._local_action.preconditioner_matrix(self._face_energy_matrix)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return self._local_action.solve(rhs)
