from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

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
    """Return the deterministic complement of an augmented dual spanning tree.

    The augmented dual graph contains one vertex for each tetrahedron and one
    exterior vertex.  Each interior triangular face is a dual edge between its
    two tetrahedra; each boundary face connects its tetrahedron to the exterior.

    For a connected contractible tetrahedral chart the complement of a dual
    spanning tree contains

        n_faces - n_tetrahedra = n_edges - n_nodes + 1

    faces, exactly the number of cotree magnetic coordinates.  Together with the
    primal spanning-tree gauge this is the standard topological tree--cotree
    basis of the discrete curl complex.  A count mismatch is therefore treated
    as a topology/chart failure (for example an unhandled harmonic subspace),
    never repaired by a numerical rank threshold.
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


def _assign_selected_faces_to_tetrahedra(mesh, selected_faces: np.ndarray) -> np.ndarray:
    """Assign selected faces to incident tetrahedra without numerical rank tests.

    A non-degenerate tetrahedron has four distinct face planes.  Any set of at
    most three distinct face normals is linearly independent: the corresponding
    three planes meet at one tetrahedron vertex, and degeneracy would contradict
    positive tetrahedral volume.  Therefore the only local physical constraint
    is the exact curl dimension, namely at most three selected faces per cell.

    Backtracking is exhaustive over the finite face--cell incidence choices and
    contains no floating rank threshold or user capacity parameter.
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
            "dual-tree face basis cannot be assigned to tetrahedra within the exact three-component curl dimension"
        )
    return assignment


@dataclass(frozen=True)
class _LocalSCCLU:
    positions: np.ndarray
    factor: np.ndarray
    pivots: np.ndarray


@dataclass(frozen=True)
class MagneticFaceCirculationEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """Topology-exact face-circulation auxiliary magnetic preconditioner.

    Let alpha denote cotree magnetic coordinates and select the complement of an
    augmented dual spanning tree.  With

        S=(C R_A)[selected faces,:],
        y=S alpha,

    Stokes' theorem gives each selected face circulation from the constant
    tetrahedral curl.  For selected faces assigned to tetrahedron q, let B_q
    contain the globally oriented face-area vectors.  Then

        nu_q V_q ||curl A||^2 >= y_q^T W_q y_q,
        W_q=nu_q V_q (B_q B_q^T)^-1.

    Summing the disjoint assigned local contributions yields

        K_A >= P_A=S^T W S,

    so the magnetic spectral-equivalence constant is exactly m_K=1.

    The selected S is integer topology, not a floating geometric factor.  Its
    current inverse implementation uses exact block substitution after an SCC
    decomposition.  ``maximum_scc_size`` is intentionally exposed: a chart for
    which it equals the full magnetic dimension is a correctness implementation,
    not yet a scalable local solve.
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

        selected_faces = _dual_tree_complement_faces(mesh, n_A)
        S = C_R[selected_faces, :].tocsr()
        if S.shape != (n_A, n_A):
            raise RuntimeError("dual-tree face-curl topology matrix is not square")

        # Under the contractible-chart tree--cotree theorem S is nonsingular.
        # The local block factorizations below are implementation guards; no
        # magnitude/rank threshold is used to choose the topological basis.
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
                raise ValueError(
                    "dual-tree face-curl basis is singular; the geometry chart violates the assumed contractible tree--cotree topology"
                )
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
