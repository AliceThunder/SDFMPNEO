from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from sdfmpneo.spatial.barycentric_polynomial import (
    _unit_simplex_moment,
    polynomial_multiply,
    polynomial_p1,
)
from sdfmpneo.spatial.tetra3d import _barycentric_gradients

from .sparse_reduced import (
    SparseEnergyReducedEMModel as _BaseSparseEnergyReducedEMModel,
    SparseEnergyResidualGreedyEMReducer as _BaseSparseEnergyResidualGreedyEMReducer,
)


@lru_cache(maxsize=8192)
def _unit_pair_moment_matrix(powers: tuple[int, int, int, int]) -> np.ndarray:
    """Unit-volume matrix of integral lambda_i lambda_j lambda^powers."""
    result = np.empty((4, 4), dtype=float)
    for i in range(4):
        for j in range(i, 4):
            augmented = list(powers)
            augmented[i] += 1
            augmented[j] += 1
            value = _unit_simplex_moment(tuple(int(v) for v in augmented))
            result[i, j] = value
            result[j, i] = value
    result.setflags(write=False)
    return result


class _ReducedNedelecAssembler:
    """Exact direct-reduced Nedelec assembly with vectorized element chunks."""

    def __init__(self, mesh, edge_fields: np.ndarray) -> None:
        fields = np.asarray(edge_fields, dtype=complex)
        if fields.ndim != 2 or fields.shape[0] != mesh.n_edges:
            raise ValueError("edge_fields must have shape (n_edges,n_reduced)")
        self.mesh = mesh
        self.fields = fields
        coefficients = np.zeros((mesh.n_tetrahedra, 6, 4, 3), dtype=float)
        for q, tet in enumerate(mesh.tetrahedra):
            gradients = _barycentric_gradients(mesh.vertices, tet)
            for p, (i, j) in enumerate(mesh._local_edge_vertex_indices(tet)):
                coefficients[q, p, i] = gradients[j]
                coefficients[q, p, j] = -gradients[i]
        self._coefficients = np.ascontiguousarray(coefficients)
        self._volumes = np.ascontiguousarray(mesh.volumes, dtype=float)
        self._local_fields = np.ascontiguousarray(
            fields[np.asarray(mesh.tet_edge_indices, dtype=int)], dtype=complex
        )

    @staticmethod
    def _moments(volume, poly):
        # Geometry contributes only the scalar volume.  Every polynomial monomial
        # reuses a cached unit-simplex 4x4 pair-moment kernel instead of performing
        # ten Python-level integrations independently.
        moments = np.zeros((4, 4), dtype=float)
        for powers, coefficient in poly.items():
            moments += float(coefficient) * _unit_pair_moment_matrix(
                tuple(int(v) for v in powers)
            )
        return float(volume) * moments

    @staticmethod
    def _reduced_local(coefficients, fields, moments):
        local = np.einsum("pik,ij,qjk->pq", coefficients, moments, coefficients)
        return fields.conj().T @ (local @ fields)

    def _assemble_family(self, polynomials) -> np.ndarray:
        if len(polynomials) != self.mesh.n_tetrahedra:
            raise ValueError("one polynomial is required for every tetrahedron")
        n = self.fields.shape[1]
        reduced = np.zeros((n, n), dtype=complex)
        # Keep temporary arrays bounded even for large UWPT meshes.  The expensive
        # FE contractions run as dense NumPy kernels while constitutive polynomial
        # generation remains exact and state dependent.
        chunk = 128
        for start in range(0, self.mesh.n_tetrahedra, chunk):
            stop = min(start + chunk, self.mesh.n_tetrahedra)
            active = [q for q in range(start, stop) if polynomials[q]]
            if not active:
                continue
            index = np.asarray(active, dtype=int)
            moments = np.stack([
                self._moments(self._volumes[q], polynomials[q]) for q in active
            ])
            coefficients = self._coefficients[index]
            local = np.einsum(
                "qpik,qij,qrjk->qpr",
                coefficients,
                moments,
                coefficients,
                optimize=True,
            )
            fields = self._local_fields[index]
            weighted = np.einsum("qpr,qrb->qpb", local, fields, optimize=True)
            reduced += np.einsum(
                "qpa,qpb->ab", np.conj(fields), weighted, optimize=True
            )
        return reduced

    def assemble_many(self, tetra_polynomial_families) -> np.ndarray:
        """Assemble exact reduced weights; each family uses vectorized cell chunks."""
        families = tuple(tuple(values) for values in tetra_polynomial_families)
        if not families:
            n = self.fields.shape[1]
            return np.empty((0, n, n), dtype=complex)
        return np.stack([self._assemble_family(values) for values in families])

    def assemble_products(self, tetra_polynomial_families, multiplier_families) -> np.ndarray:
        """Assemble products of state weights with fixed test polynomials."""
        families = tuple(tuple(values) for values in tetra_polynomial_families)
        multipliers = tuple(tuple(values) for values in multiplier_families)
        for values in families + multipliers:
            if len(values) != self.mesh.n_tetrahedra:
                raise ValueError("one polynomial is required for every tetrahedron")
        n = self.fields.shape[1]
        reduced = np.zeros((len(multipliers), len(families), n, n), dtype=complex)
        for output, tests in enumerate(multipliers):
            for family, polynomials in enumerate(families):
                weighted = [
                    polynomial_multiply(poly, test) if poly and test else {}
                    for poly, test in zip(polynomials, tests)
                ]
                reduced[output, family] = self._assemble_family(weighted)
        return reduced

    def assemble(self, tetra_polynomials) -> np.ndarray:
        return self._assemble_family(tuple(tetra_polynomials))


class SparseEnergyReducedEMModel(_BaseSparseEnergyReducedEMModel):
    """Sparse reduced EM model with an exact direct-reduced tetrahedral fast path.

    For the nonlinear tetrahedral UWPT problem, the training path needs only
    V^H A V and V^H H_j V. Those matrices are assembled element-by-element
    directly in the reduced coordinates instead of first materializing the
    full global conductivity/loss matrices and projecting them afterwards.
    Generic sparse EM problems transparently fall back to the base class.
    """

    def __init__(self, problem, basis, **kwargs) -> None:
        super().__init__(problem, basis, **kwargs)
        required = (
            "mesh", "a_basis", "grad_c", "magnetic_stiffness", "omega",
            "n_A", "_weighted_polynomials",
        )
        self._direct_reduced = all(hasattr(problem, name) for name in required)
        self._fused_reduced = self._direct_reduced and hasattr(problem, "thermal_test_local")
        if not self._direct_reduced:
            self._conductivity_reduced = None
            self._loss_reduced_assembler = None
            self._magnetic_reduced = None
            self._thermal_test_polynomials = None
            return

        coordinate_map = sp.hstack(
            [problem.a_basis, problem.grad_c], format="csr", dtype=complex
        )
        edge_coordinate_fields = np.asarray(coordinate_map @ self.V)
        electric_fields = (-1j * float(problem.omega)) * edge_coordinate_fields
        magnetic_fields = np.asarray(problem.a_basis @ self.V[: problem.n_A])

        self._conductivity_reduced = _ReducedNedelecAssembler(
            problem.mesh, edge_coordinate_fields
        )
        self._loss_reduced_assembler = _ReducedNedelecAssembler(
            problem.mesh, electric_fields
        )
        self._magnetic_reduced = magnetic_fields.conj().T @ (
            problem.magnetic_stiffness @ magnetic_fields
        )
        if self._fused_reduced:
            self._thermal_test_polynomials = tuple(
                tuple(polynomial_p1(problem.thermal_test_local[j, q])
                      for q in range(problem.mesh.n_tetrahedra))
                for j in range(problem.n_thermal)
            )
        else:
            self._thermal_test_polynomials = None

    def _state_polynomial_family(self, state, *, derivatives: bool):
        base, _ = self.problem._weighted_polynomials(state)
        if not derivatives:
            return (tuple(base),)
        family = [tuple(base)]
        for mode in range(self.problem.n_thermal):
            values, _ = self.problem._weighted_polynomials(
                state, derivative_mode=mode
            )
            family.append(tuple(values))
        return tuple(family)

    def _assembled_state_family(self, state, *, derivatives: bool):
        polynomials = self._state_polynomial_family(state, derivatives=derivatives)
        conductivity = self._conductivity_reduced.assemble_many(polynomials)
        losses = 0.5 * self._loss_reduced_assembler.assemble_products(
            polynomials, self._thermal_test_polynomials
        )
        return conductivity, losses

    def operator_reduced(self, a: np.ndarray) -> np.ndarray:
        if not self._direct_reduced:
            return super().operator_reduced(a)
        state = np.asarray(a, dtype=float)
        polynomials, _ = self.problem._weighted_polynomials(state)
        conductivity = self._conductivity_reduced.assemble(polynomials)
        return self._magnetic_reduced + 1j * self.problem.omega * conductivity

    def _operator_derivative_reduced(self, a: np.ndarray, mode: int) -> np.ndarray:
        state = np.asarray(a, dtype=float)
        polynomials, _ = self.problem._weighted_polynomials(
            state, derivative_mode=mode
        )
        return 1j * self.problem.omega * self._conductivity_reduced.assemble(polynomials)

    def _loss_operator_reduced(
        self,
        output_mode: int,
        a: np.ndarray,
        derivative_mode: int | None = None,
    ) -> np.ndarray:
        state = np.asarray(a, dtype=float)
        polynomials, _ = self.problem._weighted_polynomials(
            state,
            derivative_mode=derivative_mode,
            test_mode=output_mode,
        )
        return 0.5 * self._loss_reduced_assembler.assemble(polynomials)

    def heat_source_for_rhs(self, a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if not self._direct_reduced:
            return super().heat_source_for_rhs(a, rhs)
        state = np.asarray(a, dtype=float)
        source = np.asarray(rhs, dtype=complex)
        if state.shape != (self.problem.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        if source.shape != (self.problem.n_em,):
            raise ValueError("rhs dimension mismatch")

        if not self._fused_reduced:
            c = scipy.linalg.solve(
                self.operator_reduced(state), self.rhs_reduced(source), assume_a="gen"
            )
            return np.array(
                [
                    np.real(np.vdot(c, self._loss_operator_reduced(j, state) @ c))
                    for j in range(self.problem.n_thermal)
                ],
                dtype=float,
            )

        conductivity, loss_family = self._assembled_state_family(
            state, derivatives=False
        )
        Ar = self._magnetic_reduced + 1j * self.problem.omega * conductivity[0]
        c = scipy.linalg.solve(Ar, self.rhs_reduced(source), assume_a="gen")
        losses = loss_family[:, 0]
        return np.array(
            [np.real(np.vdot(c, H @ c)) for H in losses],
            dtype=float,
        )

    def heat_source_and_jacobian_for_rhs(self, a: np.ndarray, rhs: np.ndarray):
        if not self._direct_reduced:
            return super().heat_source_and_jacobian_for_rhs(a, rhs)
        state = np.asarray(a, dtype=float)
        source = np.asarray(rhs, dtype=complex)
        if state.shape != (self.problem.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        if source.shape != (self.problem.n_em,):
            raise ValueError("rhs dimension mismatch")

        if not self._fused_reduced:
            Ar = self.operator_reduced(state)
            factor = scipy.linalg.lu_factor(Ar)
            c = scipy.linalg.lu_solve(factor, self.rhs_reduced(source))
            losses = [
                self._loss_operator_reduced(j, state)
                for j in range(self.problem.n_thermal)
            ]
            q = np.array([np.real(np.vdot(c, H @ c)) for H in losses], dtype=float)
            J = np.zeros((self.problem.n_thermal, self.problem.n_thermal), dtype=float)
            for k in range(self.problem.n_thermal):
                Akr = self._operator_derivative_reduced(state, k)
                dc = scipy.linalg.lu_solve(factor, -(Akr @ c))
                for j, Hj in enumerate(losses):
                    dH = self._loss_operator_reduced(j, state, derivative_mode=k)
                    J[j, k] = (
                        2.0 * np.real(np.vdot(dc, Hj @ c))
                        + np.real(np.vdot(c, dH @ c))
                    )
            return q, J

        conductivity, loss_family = self._assembled_state_family(
            state, derivatives=True
        )
        Ar = self._magnetic_reduced + 1j * self.problem.omega * conductivity[0]
        factor = scipy.linalg.lu_factor(Ar)
        c = scipy.linalg.lu_solve(factor, self.rhs_reduced(source))
        losses = loss_family[:, 0]
        q = np.array([np.real(np.vdot(c, H @ c)) for H in losses], dtype=float)
        J = np.zeros((self.problem.n_thermal, self.problem.n_thermal), dtype=float)

        for k in range(self.problem.n_thermal):
            Akr = 1j * self.problem.omega * conductivity[k + 1]
            dc = scipy.linalg.lu_solve(factor, -(Akr @ c))
            for j, Hj in enumerate(losses):
                dH = loss_family[j, k + 1]
                J[j, k] = (
                    2.0 * np.real(np.vdot(dc, Hj @ c))
                    + np.real(np.vdot(c, dH @ c))
                )
        return q, J


class SparseEnergyResidualGreedyEMReducer(_BaseSparseEnergyResidualGreedyEMReducer):
    """Certified reducer that returns the fast model without changing basis construction."""

    def build_multi_rhs(self, candidate_states, rhs_matrix, *, requested_energy_state_error):
        model = super().build_multi_rhs(
            candidate_states,
            rhs_matrix,
            requested_energy_state_error=requested_energy_state_error,
        )
        return SparseEnergyReducedEMModel(
            model.problem,
            model.V,
            reference_energy_metric=model.reference_energy_metric,
            reduction_certificate=model.reduction_certificate,
            riesz_action_factory=model.riesz_action_factory,
        )
