from __future__ import annotations

import torch
from torch import Tensor


def conservative_mortar_matrix(
    package_to_conductor: Tensor,
    conductance: Tensor,
) -> Tensor:
    """Build the 1D-conductor/3D-domain conservative exchange operator.

    Let P map the 3D thermal DOFs to the conductor exchange locations. With
    theta=[T_c, T_3d], define the temperature jump

        e = T_c - P T_3d = E theta,  E=[I, -P].

    The exchange operator K_cp = E^T W E is symmetric positive semidefinite.
    The same interface heat flux therefore leaves one side and enters the other
    with exactly opposite sign in the assembled thermal equations.

    Args:
        package_to_conductor: [Nc, N3] interpolation/mortar map P.
        conductance: [Nc] positive weights or [Nc, Nc] symmetric PSD W.

    Returns:
        K_cp: [Nc+N3, Nc+N3].
    """
    if package_to_conductor.ndim != 2:
        raise ValueError("package_to_conductor must have shape [Nc,N3]")
    nc, _ = package_to_conductor.shape
    eye = torch.eye(
        nc,
        dtype=package_to_conductor.dtype,
        device=package_to_conductor.device,
    )
    e = torch.cat((eye, -package_to_conductor), dim=-1)

    if conductance.ndim == 1:
        if conductance.shape[0] != nc:
            raise ValueError("conductance vector length must equal Nc")
        if torch.any(conductance < 0):
            raise ValueError("conductance weights must be non-negative")
        weighted_e = conductance[:, None] * e
        return e.transpose(-2, -1) @ weighted_e

    if conductance.ndim == 2:
        if conductance.shape != (nc, nc):
            raise ValueError("conductance matrix must have shape [Nc,Nc]")
        return e.transpose(-2, -1) @ conductance @ e

    raise ValueError("conductance must have shape [Nc] or [Nc,Nc]")


def thermal_jump(
    conductor_temperature: Tensor,
    domain_temperature: Tensor,
    package_to_conductor: Tensor,
) -> Tensor:
    """Return T_c - P T_3d at conductor/package exchange locations."""
    return conductor_temperature - package_to_conductor @ domain_temperature


def rms_joule_power(resistance: Tensor, current: Tensor) -> Tensor:
    """RMS-phasor Joule power I^H R I.

    `resistance` may be a scalar, diagonal vector, or square matrix. This helper
    exists mainly to keep the RMS convention explicit throughout the thermal
    backend.
    """
    if resistance.ndim == 0:
        return resistance * current.abs().square().sum()
    if resistance.ndim == 1:
        if resistance.shape[0] != current.shape[-1]:
            raise ValueError("diagonal resistance length must match current")
        return torch.sum(resistance * current.abs().square())
    if resistance.ndim == 2:
        return (current.conj() @ resistance.to(current.dtype) @ current).real
    raise ValueError("resistance must be scalar, vector, or matrix")
