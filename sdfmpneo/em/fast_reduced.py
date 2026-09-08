from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from sdfmpneo.spatial.barycentric_polynomial import integrate_polynomial_times_lambda_pair
from sdfmpneo.spatial.tetra3d import _barycentric_gradients

from .sparse_reduced import SparseEnergyReducedEMModel as _BaseSparseEnergyReducedEMModel


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

    def assemble(self, tetra_polynomials) -> np.ndarray:
        polynomials = tuple(tetra_polynomials)
        if len(polynomials) != self.mesh.n_tetrahedra:
            raise ValueError("one polynomial is required for every tetrahedron")
        n = self.fields.shape[1]
        reduced = np.zeros((n, n), dtype=complex)
        for poly, (volume, coefficients, fields) in zip(polynomials, self._local):
            if not poly:
                continue
            moments = np.empty((4, 4))
            for i in range(4):
                for j in range(i, 4):
                    moments[i, j] = moments[j, i] = integrate_polynomial_times_lambda_pair(
                        volume, poly, i, j
                    )
            local = np.einsum("pik,ij,qjk->pq", coefficients, moments, coefficients)
            reduced += fields.conj().T @ (local @ fields)
        return reduced


class SparseEnergyReducedEMModel(_BaseSparseEnergyReducedEMModel):
    """Sparse reduced EM model with an exact direct-reduced tetrahedral fast path.

    For the nonlinear tetrahedral UWPT problem, the training path needs only
    V^H A V and V^H H_j V.  Those matrices are assembled element-by-element
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
        if not self._direct_reduced:
            self._conductivity_reduced = None
            self._loss_reduced_assembler = None
            self._magnetic_reduced = None
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

    def heat_source_and_jacobian_for_rhs(self, a: np.ndarray, rhs: np.ndarray):
        if not self._direct_reduced:
            return super().heat_source_and_jacobian_for_rhs(a, rhs)
        state = np.asarray(a, dtype=float)
        source = np.asarray(rhs, dtype=complex)
        if state.shape != (self.problem.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        if source.shape != (self.problem.n_em,):
            raise ValueError("rhs dimension mismatch")

        Ar = self.operator_reduced(state)
        c = scipy.linalg.solve(Ar, self.rhs_reduced(source), assume_a="gen")
        factor = scipy.linalg.lu_factor(Ar)
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
