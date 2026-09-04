from __future__ import annotations

import torch
from torch import Tensor

from ..contracts import EMOperators, EMSolution


def _as_column_frequency(omega: Tensor) -> Tensor:
    if omega.ndim == 1:
        return omega[:, None, None]
    if omega.ndim == 2 and omega.shape[-1] == 1:
        return omega[..., None]
    if omega.ndim == 3:
        return omega
    raise ValueError("omega must have shape [B], [B,1], or [B,1,1]")


def solve_em_schur(basis: Tensor, operators: EMOperators) -> EMSolution:
    """Solve all port excitations in the shared reduced EM trial space.

    Governing equation, using RMS phasors and exp(+j omega t):
        (R_s + j omega L_s) q = -j omega C I

    The port feedback is recovered by Schur elimination:
        Z_sea = omega^2 C_r^H A_r^{-1} C_r

    A single excitation-independent basis is used for all ports, which preserves
    the symmetric multiport structure when the physical operators are symmetric.
    """
    if basis.ndim != 3:
        raise ValueError("basis must have shape [B,Nj,r]")

    omega = _as_column_frequency(operators.omega)
    complex_dtype = torch.complex128 if basis.dtype == torch.float64 else torch.complex64

    b = basis.to(complex_dtype)
    r = operators.resistance.to(complex_dtype)
    l = operators.inductance.to(complex_dtype)
    c = operators.coupling.to(complex_dtype)
    z0 = operators.z_background.to(complex_dtype)

    a_full = r + 1j * omega * l
    a_reduced = b.conj().transpose(-2, -1) @ a_full @ b
    c_reduced = b.conj().transpose(-2, -1) @ c

    response = torch.linalg.solve(a_reduced, c_reduced)
    reduced_coefficients = -1j * omega * response
    current_dofs = b @ reduced_coefficients

    impedance_sea = (omega**2) * c_reduced.conj().transpose(-2, -1) @ response
    impedance_total = z0 + impedance_sea

    r_reduced = b.conj().transpose(-2, -1) @ r @ b
    seawater_loss_matrix = (
        reduced_coefficients.conj().transpose(-2, -1)
        @ r_reduced
        @ reduced_coefficients
    )

    return EMSolution(
        reduced_coefficients=reduced_coefficients,
        current_dofs=current_dofs,
        impedance_sea=impedance_sea,
        impedance_total=impedance_total,
        seawater_loss_matrix=seawater_loss_matrix,
    )


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
    """Compare Schur-port dissipation with the explicitly reduced Joule form."""
    z = solution.impedance_sea
    port_dissipation = 0.5 * (z + z.conj().transpose(-2, -1)).real
    joule = solution.seawater_loss_matrix.real
    numerator = torch.linalg.matrix_norm(port_dissipation - joule, ord="fro")
    denominator = torch.linalg.matrix_norm(joule, ord="fro").clamp_min(1.0e-30)
    return numerator / denominator
