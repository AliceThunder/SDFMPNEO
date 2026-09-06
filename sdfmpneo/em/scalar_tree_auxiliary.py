from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .block_riesz import _certified_inverse_inf_upper
from .certified_riesz import CertifiedEnergyPreconditioner
from .constitutive import ReciprocalLinearResistivity


def _conductive_support(problem) -> np.ndarray:
    mesh = problem.mesh
    support = np.zeros(mesh.n_tetrahedra, dtype=bool)
    T0 = np.asarray(problem.temperature_reference_local, dtype=float)
    for region in problem.conductivity_regions:
        mask = np.asarray(region.mask, dtype=bool)
        if not np.any(mask):
            continue
        values = np.asarray(region.law.evaluate(T0[mask]), dtype=float)
        derivatives = np.asarray(region.law.derivative(T0[mask]), dtype=float)
        dynamic = np.any(derivatives != 0.0, axis=1)
        nonzero = np.any(values > 0.0, axis=1)
        local = np.flatnonzero(mask)
        support[local[nonzero | dynamic]] = True
    return support


def _conductive_components_and_scalar_nodes(problem, support: np.ndarray):
    mesh = problem.mesh
    active_nodes = set(map(int, mesh.tetrahedra[support].ravel()))
    adjacency = {node: set() for node in active_nodes}
    edge_index = {tuple(map(int, edge)): i for i, edge in enumerate(mesh.edge_vertices)}
    for tet in mesh.tetrahedra[support]:
        for u, v in combinations(map(int, tet), 2):
            adjacency[u].add(v)
            adjacency[v].add(u)

    components: list[tuple[int, tuple[int, ...]]] = []
    scalar_nodes: list[int] = []
    seen: set[int] = set()
    for root_candidate in sorted(active_nodes):
        if root_candidate in seen:
            continue
        stack = [root_candidate]
        component: list[int] = []
        seen.add(root_candidate)
        while stack:
            u = stack.pop()
            component.append(u)
            for v in sorted(adjacency[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        component_sorted = tuple(sorted(component))
        reference = int(component_sorted[0])
        components.append((reference, component_sorted))
        scalar_nodes.extend(v for v in component_sorted if v != reference)

    expected = mesh.grad[:, scalar_nodes].astype(complex).tocsr()
    actual = sp.csr_matrix(problem.grad_c, dtype=complex)
    defect = (actual - expected).tocsr()
    defect.eliminate_zeros()
    if defect.nnz:
        raise RuntimeError("conductive scalar coordinates do not match the physical nodal gauge")
    return tuple(components), tuple(scalar_nodes), adjacency, edge_index


def _deterministic_tree_edges(problem, support: np.ndarray):
    components, scalar_nodes, adjacency, edge_index = _conductive_components_and_scalar_nodes(problem, support)
    scalar_position = {node: k for k, node in enumerate(scalar_nodes)}
    selected_edges: list[int] = []
    child_columns: list[int] = []

    for reference, component in components:
        component_set = set(component)
        visited = {int(reference)}
        queue = [int(reference)]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            candidates = []
            for v in adjacency[u]:
                if v not in component_set or v in visited:
                    continue
                edge = (min(u, v), max(u, v))
                candidates.append((int(v), int(edge_index[edge])))
            for v, eidx in sorted(candidates, key=lambda item: (item[0], item[1])):
                if v in visited:
                    continue
                visited.add(v)
                queue.append(v)
                selected_edges.append(eidx)
                child_columns.append(int(scalar_position[v]))
        if visited != component_set:
            raise ValueError("conductive component graph is disconnected")

    if len(selected_edges) != problem.n_scalar:
        raise RuntimeError("conductive spanning forest does not match scalar gauge dimension")
    G = sp.csr_matrix(problem.grad_c, dtype=complex)
    S = G[np.asarray(selected_edges, dtype=int), :].tocsr()
    order = np.asarray(child_columns, dtype=int)
    Sperm = S[:, order].tocsr()
    if Sperm.shape != (problem.n_scalar, problem.n_scalar):
        raise RuntimeError("conductive tree incidence is not square")
    if sp.triu(Sperm, k=1).nnz:
        raise RuntimeError("rooted conductive tree incidence is not lower triangular")
    if np.any(np.asarray(Sperm.diagonal()) == 0.0):
        raise RuntimeError("conductive tree incidence lost a child pivot")
    return np.asarray(selected_edges, dtype=int), order, S, Sperm


def _edge_incident_support_tetrahedra(problem, support: np.ndarray):
    mesh = problem.mesh
    incident: list[list[int]] = [[] for _ in range(mesh.n_edges)]
    for q in np.flatnonzero(support):
        for edge in mesh.tet_edge_indices[int(q)]:
            incident[int(edge)].append(int(q))
    return tuple(tuple(sorted(set(items))) for items in incident)


def _assign_tree_edges_capacity_two(problem, support: np.ndarray, selected_edges: np.ndarray) -> np.ndarray:
    """Assign each tree-edge constraint to a conductive tetrahedron, at most two per tetrahedron.

    Any two distinct edges of a non-degenerate tetrahedron are linearly
    independent as 3-D vectors, so capacity two gives a topology-only sufficient
    condition for the local gradient-energy Gram to be positive definite.
    """

    incident = _edge_incident_support_tetrahedra(problem, support)
    n_tet = problem.mesh.n_tetrahedra
    matched_edge_for_slot = np.full(2 * n_tet, -1, dtype=int)

    def augment(position: int, seen_slots: np.ndarray) -> bool:
        edge = int(selected_edges[position])
        slots = []
        for tet in incident[edge]:
            slots.extend((2 * tet, 2 * tet + 1))
        for slot in sorted(slots):
            if seen_slots[slot]:
                continue
            seen_slots[slot] = True
            previous = int(matched_edge_for_slot[slot])
            if previous < 0 or augment(previous, seen_slots):
                matched_edge_for_slot[slot] = position
                return True
        return False

    for position in range(selected_edges.size):
        seen = np.zeros(2 * n_tet, dtype=bool)
        if not augment(position, seen):
            raise ValueError(
                "conductive spanning-tree constraints cannot be assigned with the exact two-edge local independence rule"
            )

    assignment = np.full(selected_edges.size, -1, dtype=int)
    for slot, position in enumerate(matched_edge_for_slot):
        if position >= 0:
            assignment[position] = slot // 2
    if np.any(assignment < 0):
        raise RuntimeError("conductive tree-edge assignment is incomplete")
    return assignment


def _conductivity_lower_tetra(problem, state: np.ndarray, support: np.ndarray) -> np.ndarray:
    T = np.asarray(problem.temperature_local(state), dtype=float)
    lower = np.zeros(problem.mesh.n_tetrahedra, dtype=float)
    constitutive = problem.constitutive_certificate(state)
    reciprocal_relative = float(constitutive.maximum_inverse_relative_bound)
    if not 0.0 <= reciprocal_relative < 1.0:
        raise ValueError("reciprocal constitutive certificate does not provide a positive lower enclosure")

    for region in problem.conductivity_regions:
        mask = np.asarray(region.mask, dtype=bool) & support
        for q in np.flatnonzero(mask):
            values = np.asarray(region.law.evaluate(T[int(q)]), dtype=float)
            sigma_min = float(np.min(values))
            if isinstance(region.law, ReciprocalLinearResistivity):
                sigma_min *= 1.0 - reciprocal_relative
            if sigma_min <= 0.0:
                raise ValueError("conductive scalar auxiliary requires strictly positive conductivity on its support")
            lower[int(q)] = float(np.nextafter(sigma_min, 0.0))
    if np.any(lower[support] <= 0.0):
        raise ValueError("conductive support contains a tetrahedron without a positive certified conductivity lower bound")
    return lower


@dataclass(frozen=True)
class ConductiveScalarTreeEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """State-aware topology-exact scalar-potential energy preconditioner.

    For the gauge-fixed P1 scalar potential,

        E_psi = omega integral sigma(T) |grad psi|^2 dV.

    One voltage-difference constraint is selected for every edge of a
    deterministic conductive spanning forest.  These constraints form a square
    rooted incidence matrix S.  Each selected edge is assigned to a conductive
    tetrahedron with at most two constraints per tetrahedron.  For local edge
    vectors collected in B_q,

        omega sigma_q^- V_q |grad psi|^2
        >= y_q^T [omega sigma_q^- V_q (B_q B_q^T)^-1] y_q.

    Therefore

        E_psi >= P_psi=S^T W S,

    so m_E=1.  S is a rooted-tree incidence minor: after the topology-generated
    child ordering it is lower triangular. P_psi^-1 is therefore applied by two
    sparse triangular tree solves and explicit 1x1/2x2 local W^-1 blocks. No
    global scalar factorization or spectral threshold is used.
    """

    dimension: int
    lower_spectral_equivalence_bound: float
    selected_edges: np.ndarray
    coordinate_order: np.ndarray
    edge_tetra_assignment: np.ndarray
    maximum_local_constraint_count: int
    inverse_inf_upper_bound: float
    _S: sp.csr_matrix
    _Sperm: sp.csr_matrix
    _Winv_blocks: tuple[tuple[np.ndarray, np.ndarray], ...]
    _P: sp.csr_matrix

    @classmethod
    def build_from_problem(
        cls,
        problem,
        state: np.ndarray,
    ) -> "ConductiveScalarTreeEnergyPreconditioner":
        if problem.n_scalar == 0:
            raise ValueError("conductive scalar gauge space is empty")
        a = np.asarray(state, dtype=float)
        if a.shape != (problem.n_thermal,):
            raise ValueError("thermal state dimension mismatch")

        support = _conductive_support(problem)
        selected_edges, order, S, Sperm = _deterministic_tree_edges(problem, support)
        assignment = _assign_tree_edges_capacity_two(problem, support, selected_edges)
        sigma_lower = _conductivity_lower_tetra(problem, a, support)

        rows: list[int] = []
        cols: list[int] = []
        data: list[complex] = []
        winv_blocks: list[tuple[np.ndarray, np.ndarray]] = []
        counts = np.bincount(assignment, minlength=problem.mesh.n_tetrahedra)
        for tet in range(problem.mesh.n_tetrahedra):
            positions = np.flatnonzero(assignment == tet)
            if positions.size == 0:
                continue
            edge_vectors = []
            for p in positions:
                edge = problem.mesh.edge_vertices[int(selected_edges[int(p)])]
                u, v = map(int, edge)
                edge_vectors.append(problem.mesh.vertices[v] - problem.mesh.vertices[u])
            B = np.vstack(edge_vectors)
            gram = 0.5 * ((B @ B.T) + (B @ B.T).T)
            try:
                factor = scipy.linalg.cho_factor(gram, lower=True, check_finite=False)
                gram_inverse = scipy.linalg.cho_solve(
                    factor,
                    np.eye(positions.size),
                    check_finite=False,
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError("assigned conductive tree edges are not locally independent") from exc

            scale = float(problem.omega * sigma_lower[tet] * problem.mesh.volumes[tet])
            local_W = scale * gram_inverse
            local_Winv = gram / scale
            winv_blocks.append((positions.astype(int), np.asarray(local_Winv, dtype=complex)))
            for i, pi in enumerate(positions):
                for j, pj in enumerate(positions):
                    value = complex(local_W[i, j])
                    if value != 0.0:
                        rows.append(int(pi))
                        cols.append(int(pj))
                        data.append(value)

        W = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(problem.n_scalar, problem.n_scalar),
            dtype=complex,
        ).tocsr()
        W.sum_duplicates()
        W.eliminate_zeros()
        P = (S.conj().T @ W @ S).tocsr()
        P = (0.5 * (P + P.conj().T)).tocsr()
        P.sum_duplicates()
        P.eliminate_zeros()

        provisional = cls(
            dimension=int(problem.n_scalar),
            lower_spectral_equivalence_bound=1.0,
            selected_edges=selected_edges.copy(),
            coordinate_order=order.copy(),
            edge_tetra_assignment=assignment.copy(),
            maximum_local_constraint_count=int(np.max(counts)),
            inverse_inf_upper_bound=float("nan"),
            _S=S,
            _Sperm=Sperm,
            _Winv_blocks=tuple(winv_blocks),
            _P=P,
        )
        inverse_upper = _certified_inverse_inf_upper(P, provisional)
        return replace(provisional, inverse_inf_upper_bound=float(inverse_upper))

    def auxiliary_matrix(self) -> sp.csr_matrix:
        return self._P.copy()

    def _solve_S(self, rhs: np.ndarray) -> np.ndarray:
        """Apply S^{-1}; row order stays fixed while scalar columns are permuted."""

        b = np.asarray(rhs, dtype=complex)
        if b.shape != (self.dimension,):
            raise ValueError("scalar tree rhs dimension mismatch")
        xperm = spla.spsolve_triangular(self._Sperm, b, lower=True)
        out = np.empty_like(np.asarray(xperm, dtype=complex))
        out[self.coordinate_order] = np.asarray(xperm, dtype=complex)
        return out

    def _solve_SH(self, rhs: np.ndarray) -> np.ndarray:
        """Apply S^{-H}; the coordinate RHS is permuted, row-space output is not."""

        b = np.asarray(rhs, dtype=complex)
        if b.shape != (self.dimension,):
            raise ValueError("scalar tree rhs dimension mismatch")
        bperm = b[self.coordinate_order]
        return np.asarray(
            spla.spsolve_triangular(self._Sperm.conj().T.tocsr(), bperm, lower=False),
            dtype=complex,
        )

    def _apply_Winv(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=complex)
        out = np.zeros_like(v)
        for positions, block in self._Winv_blocks:
            out[positions] = block @ v[positions]
        return out

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        z = self._solve_SH(rhs)
        w = self._apply_Winv(z)
        return self._solve_S(w)
