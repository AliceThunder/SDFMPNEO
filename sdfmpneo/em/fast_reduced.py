from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from sdfmpneo.spatial.barycentric_polynomial import (
    integrate_polynomial_times_lambda_pair,
    polynomial_multiply,
    polynomial_p1,
)
from sdfmpneo.spatial.tetra3d import _barycentric_gradients

from .sparse_reduced import (
    SparseEnergyReducedEMModel as _BaseSparseEnergyReducedEMModel,
    SparseEnergyResidualGreedyEMReducer as _BaseSparseEnergyResidualGreedyEMReducer,
)


class _ReducedNedelecAssembler:
    """Exact element assembly of W^H M(sigma) W without building global M."""

    def __init__(self, mesh, edge_fields: np.ndarray) -> None:
        fields = np.asarray(edge_fields, dtype=complex)
        if fields.ndim != 2 or fields.shape[0] != mesh.n_edges:
            raise ValueError("edge_fields must have shape (n_edges,n_reduced)")
        self.mesh = mesh
        self.fields = fields
        self._local = []
        for q, tet in enumerate(mesh.tetrahedra):
            gradients = _barycentric_gradients(mesh.vertices, tet)
            local_pairs = mesh._local_edge_vertex_indices(tet)
            coefficients = np.zeros((6, 4, 3))
            for p, (i, j) in enumerate(local_pairs):
                coefficients[p, i] = gradients[j]
                coefficients[p, j] = -gradients[i]
            edges = mesh.tet_edge_indices[q]
            self._local.append((float(mesh.volumes[q]), coefficients, fields[edges]))

    @staticmethod
    def _moments(volume, poly):
        moments = np.empty((4, 4))
        for i in range(4):
            for j in range(i, 4):
                moments[i, j] = moments[j, i] = integrate_polynomial_times_lambda_pair(
                    volume, poly, i, j
                )
        return moments

    @staticmethod
    def _reduced_local(coefficients, fields, moments):
        local = np.einsum("pik,ij,qjk->pq", coefficients, moments, coefficients)
        return fields.conj().T @ (local @ fields)

    def assemble_many(self, tetra_polynomial_families) -> np.ndarray:
        """Assemble several exact polynomial weights in one tetrahedral sweep."""
        families = tuple(tuple(values) for values in tetra_polynomial_families)
        for polynomials in families:
            if len(polynomials) != self.mesh.n_tetrahedra:
                raise ValueError("one polynomial is required for every tetrahedron")
        n = self.fields.shape[1]
        reduced = np.zeros((len(families), n, n), dtype=complex)
        for q, (volume, coefficients, fields) in enumerate(self._local):
            for family, polynomials in enumerate(families):
                poly = polynomials[q]
                if not poly:
                    continue
                moments = self._moments(volume, poly)
                reduced[family] += self._reduced_local(coefficients, fields, moments)
        return reduced

    def assemble_products(self, tetra_polynomial_families, multiplier_families) -> np.ndarray:
        """Assemble products of base/derivative weights with fixed test polynomials.

        The output has shape (n_multiplier,n_family,n_reduced,n_reduced).  This
        keeps the exact barycentric multiplication/integration order while
        avoiding repeated full mesh sweeps for every (output, derivative) pair.
        """
        families = tuple(tuple(values) for values in tetra_polynomial_families)
        multipliers = tuple(tuple(values) for values in multiplier_families)
        for values in families + multipliers:
            if len(values) != self.mesh.n_tetrahedra:
                raise ValueError("one polynomial is required for every tetrahedron")
        n = self.fields.shape[1]
        reduced = np.zeros((len(multipliers), len(families), n, n), dtype=complex)
        for q, (volume, coefficients, fields) in enumerate(self._local):
            for output, tests in enumerate(multipliers):
                test = tests[q]
                if not test:
                    continue
                for family, polynomials in enumerate(families):
                    poly = polynomials[q]
                    if not poly:
                        continue
                    weighted = polynomial_multiply(poly, test)
                    if not weighted:
                        continue
                    moments = self._moments(volume, weighted)
                    reduced[output, family] += self._reduced_local(
                        coefficients, fields, moments
                    )
        return reduced

    def assemble(self, tetra_polynomials) -> np.ndarray:
        return self.assemble_many([tetra_polynomials])[0]


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
            "n_A", "_weighted_polynomials", "thermal_test_local",
        )
        self._direct_reduced = all(hasattr(problem, name) for name in required)
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
        # Thermal test functions are geometry/context fixed. Build their P1
        # polynomial representation once instead of once per state/Jacobian.
        self._thermal_test_polynomials = tuple(
            tuple(polynomial_p1(problem.thermal_test_local[j, q])
                  for q in range(problem.mesh.n_tetrahedra))
            for j in range(problem.n_thermal)
        )

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
        """Prepare all reduced conductivity/loss matrices for one thermal state."""
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

        conductivity, loss_family = self._assembled_state_family(
            state, derivatives=True
        )
        Ar = self._magnetic_reduced + 1j * self.problem.omega * conductivity[0]
        # One LU serves both the state solve and every thermal sensitivity solve.
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
