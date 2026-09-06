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
    """Return the complement of a deterministic augmented-dual spanning tree."""

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
    """Deterministic capacity-three bipartite matching of faces to cells.

    Every tetrahedron contributes exactly three physical curl components, so it
    exposes three matching slots.  A selected face connects to all three slots of
    each incident tetrahedron.  Standard augmenting-path matching then finds a
    complete assignment in polynomial time.  Capacity three is fixed by 3-D
    vector physics, not a tunable algorithmic parameter.
    """

    incident = _incident_tetrahedra(mesh)
    selected = [int(f) for f in selected_faces]
    n_slots = 3 * mesh.n_tetrahedra
    matched_face_for_slot = np.full(n_slots, -1, dtype=int)

    def candidate_slots(face: int) -> list[int]:
        return [3 * int(tet) + slot for tet in incident[face] for slot in range(3)]

    def augment(face_position: int, seen_slots: np.ndarray) -> bool:
        face = selected[face_position]
        for slot in candidate_slots(face):
            if seen_slots[slot]:
                continue
            seen_slots[slot] = True
            previous = int(matched_face_for_slot[slot])
            if previous < 0 or augment(previous, seen_slots):
                matched_face_for_slot[slot] = face_position
                return True
        return False

    for position in range(len(selected)):
        seen = np.zeros(n_slots, dtype=bool)
        if not augment(position, seen):
            raise ValueError(
                "dual-tree face basis cannot be assigned within the exact three-component tetrahedral curl dimension"
            )

    slot_for_face = np.full(len(selected), -1, dtype=int)
    for slot, position in enumerate(matched_face_for_slot):
        if position >= 0:
            slot_for_face[position] = slot
    if np.any(slot_for_face < 0):
        raise RuntimeError("internal face-to-tetrahedral-slot matching is incomplete")
    return (slot_for_face // 3).astype(int)


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

    The topology/geometry construction proves

        K_A >= P_face=S^T W S.

    A certificate-driven aggregate action then constructs Q_A and proves

        P_face >= m_face Q_A,

    giving K_A >= m_face Q_A.  The production action therefore never inverts S
    and only solves the certificate-selected local aggregate blocks of Q_A.
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
        return self._local_action.inverse_inf_upper_bound

    def face_energy_matrix(self) -> sp.csr_matrix:
        return self._face_energy_matrix.copy()

    def preconditioner_matrix(self) -> sp.csr_matrix:
        return self._local_action.preconditioner_matrix(self._face_energy_matrix)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return self._local_action.solve(rhs)
