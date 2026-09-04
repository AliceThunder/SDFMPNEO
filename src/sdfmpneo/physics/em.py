from __future__ import annotations

import torch
from torch import Tensor

from ..contracts import (
    EMOperators,
    EMReducedSystem,
    EMSolution,
    MatrixFreeEMActions,
)


def _as_column_frequency(omega: Tensor) -> Tensor:
    if omega.ndim == 1:
        return omega[:, None, None]
    if omega.ndim == 2 and omega.shape[-1] == 1:
        return omega[..., None]
    if omega.ndim == 3:
        return omega
    raise ValueError("omega must have shape [B], [B,1], or [B,1,1]")


class DenseEMActionAdapter:
    """Reference adapter from full dense EM matrices to block operator actions.

    This exists for tests and small reference problems only. Production backends
    should implement `MatrixFreeEMActions` directly.
    """

    def __init__(self, operators: EMOperators) -> None:
        self.operators = operators

    @property
    def z_background(self) -> Tensor:
        return self.operators.z_background

    def _complex_dtype(self) -> torch.dtype:
        return (
            torch.complex128
            if self.operators.resistance.dtype == torch.float64
            else torch.complex64
        )

    def rhs_fields(self) -> Tensor:
        omega = _as_column_frequency(self.operators.omega)
        coupling = self.operators.coupling.to(self._complex_dtype())
        return -1j * omega * coupling

    def apply_system(self, vectors: Tensor) -> Tensor:
        omega = _as_column_frequency(self.operators.omega)
        dtype = self._complex_dtype()
        resistance = self.operators.resistance.to(dtype)
        inductance = self.operators.inductance.to(dtype)
        v = vectors.to(dtype)
        return resistance @ v + 1j * omega * (inductance @ v)

    def port_feedback(self, vectors: Tensor) -> Tensor:
        omega = _as_column_frequency(self.operators.omega)
        coupling = self.operators.coupling.to(self._complex_dtype())
        v = vectors.to(self._complex_dtype())
        return 1j * omega * coupling.conj().transpose(-2, -1) @ v

    def apply_dissipation(self, vectors: Tensor) -> Tensor:
        resistance = self.operators.resistance.to(self._complex_dtype())
        return resistance @ vectors.to(self._complex_dtype())


def project_matrix_free_em_system(
    basis: Tensor,
    actions: MatrixFreeEMActions,
) -> EMReducedSystem:
    """Project deterministic EM block actions without forming full operators.

    The only full-space objects touched are the active basis block `[B,Nj,r]`
    and the small number of multiport source fields `[B,Nj,P]`. The environment
    operator itself remains matrix-free.
    """
    if basis.ndim != 3:
        raise ValueError("basis must have shape [B,Nj,r]")

    system_on_basis = actions.apply_system(basis)
    dissipation_on_basis = actions.apply_dissipation(basis)
    source_fields = actions.rhs_fields()
    feedback = actions.port_feedback(basis)

    if system_on_basis.shape != basis.shape:
        raise ValueError("apply_system must preserve the trial-vector block shape")
    if dissipation_on_basis.shape != basis.shape:
        raise ValueError("apply_dissipation must preserve the trial-vector block shape")
    if source_fields.ndim != 3 or source_fields.shape[:2] != basis.shape[:2]:
        raise ValueError("rhs_fields must have shape [B,Nj,P]")
    if feedback.ndim != 3 or feedback.shape[0] != basis.shape[0]:
        raise ValueError("port_feedback must have shape [B,P,r]")
    if feedback.shape[-1] != basis.shape[-1]:
        raise ValueError("port_feedback reduced-column dimension must equal basis rank")
    if feedback.shape[1] != source_fields.shape[-1]:
        raise ValueError("port_feedback/source port counts must match")

    complex_dtype = (
        torch.complex128
        if basis.dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )
    b = basis.to(complex_dtype)
    bh = b.conj().transpose(-2, -1)

    system_matrix = bh @ system_on_basis.to(complex_dtype)
    rhs_matrix = bh @ source_fields.to(complex_dtype)
    dissipation_matrix = bh @ dissipation_on_basis.to(complex_dtype)

    return EMReducedSystem(
        system_matrix=system_matrix,
        rhs_matrix=rhs_matrix,
        feedback_matrix=feedback.to(complex_dtype),
        dissipation_matrix=dissipation_matrix,
        z_background=actions.z_background.to(complex_dtype),
    )


def solve_reduced_em_system(system: EMReducedSystem) -> EMSolution:
    """Solve all ports in one shared reduced electromagnetic trial space."""
    coefficients = torch.linalg.solve(system.system_matrix, system.rhs_matrix)
    impedance_sea = system.feedback_matrix @ coefficients
    impedance_total = system.z_background + impedance_sea
    seawater_loss_matrix = (
        coefficients.conj().transpose(-2, -1)
        @ system.dissipation_matrix
        @ coefficients
    )
    return EMSolution(
        reduced_coefficients=coefficients,
        impedance_sea=impedance_sea,
        impedance_total=impedance_total,
        seawater_loss_matrix=seawater_loss_matrix,
        current_dofs=None,
    )


def solve_em_matrix_free(
    basis: Tensor,
    actions: MatrixFreeEMActions,
) -> EMSolution:
    return solve_reduced_em_system(project_matrix_free_em_system(basis, actions))


def solve_em_schur(basis: Tensor, operators: EMOperators) -> EMSolution:
    """Dense reference wrapper exactly matching the original Schur formulation."""
    solution = solve_em_matrix_free(basis, DenseEMActionAdapter(operators))
    solution.current_dofs = basis.to(solution.reduced_coefficients.dtype) @ solution.reduced_coefficients
    return solution


def reciprocity_defect(z: Tensor) -> Tensor:
    """Relative Frobenius symmetry defect for each batch item."""
    numerator = torch.linalg.matrix_norm(z - z.transpose(-2, -1), ord="fro")
    denominator = torch.linalg.matrix_norm(z, ord="fro").clamp_min(1.0e-30)
    return numerator / denominator


def minimum_dissipation_eigenvalue(z: Tensor) -> Tensor:
    """Smallest eigenvalue of the Hermitian dissipative part Re_H(Z)."""
    hermitian_real = 0.5 * (z + z.conj().transpose(-2, -1)).real
    return torch.linalg.eigvalsh(hermitian_real).amin(dim=-1)


def sea_loss_consistency(solution: EMSolution) -> Tensor:
    """Compare port dissipation with the independently projected Joule form."""
    z = solution.impedance_sea
    port_dissipation = 0.5 * (z + z.conj().transpose(-2, -1)).real
    joule = solution.seawater_loss_matrix.real
    numerator = torch.linalg.matrix_norm(port_dissipation - joule, ord="fro")
    denominator = torch.linalg.matrix_norm(joule, ord="fro").clamp_min(1.0e-30)
    return numerator / denominator
