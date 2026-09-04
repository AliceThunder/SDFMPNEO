from __future__ import annotations

import torch
from torch import Tensor


def positive_sst_state(
    log_state: Tensor,
    reference_value: float | Tensor,
) -> Tensor:
    """Recover a strictly positive SST state from its logarithmic variable."""
    reference = torch.as_tensor(
        reference_value, dtype=log_state.dtype, device=log_state.device
    )
    if torch.any(reference <= 0):
        raise ValueError("reference_value must be strictly positive")
    return reference * torch.exp(log_state)


def turbulent_thermal_conductivity(
    density: Tensor,
    heat_capacity: Tensor,
    turbulent_kinematic_viscosity: Tensor,
    turbulent_prandtl: float | Tensor,
) -> Tensor:
    """Return rho * cp * nu_t / Pr_t for the RANS energy equation."""
    pr_t = torch.as_tensor(
        turbulent_prandtl,
        dtype=density.dtype,
        device=density.device,
    )
    if torch.any(pr_t <= 0):
        raise ValueError("turbulent Prandtl number must be positive")
    if torch.any(turbulent_kinematic_viscosity < 0):
        raise ValueError("turbulent kinematic viscosity must be non-negative")
    return density * heat_capacity * turbulent_kinematic_viscosity / pr_t


def magnetic_reynolds_number(
    permeability: Tensor | float,
    conductivity: Tensor | float,
    velocity_scale: Tensor | float,
    length_scale: Tensor | float,
) -> Tensor:
    """Rm = mu sigma U L, used to decide whether v x B feedback is required."""
    values = [permeability, conductivity, velocity_scale, length_scale]
    tensors = [torch.as_tensor(value) for value in values]
    if any(torch.any(value < 0) for value in tensors):
        raise ValueError("Rm inputs must be non-negative")
    mu, sigma, velocity, length = tensors
    return mu * sigma * velocity * length


def motional_emf_required(rm: Tensor | float, threshold: float = 1.0e-3) -> Tensor:
    """Return a boolean mask for cases where direct flow-to-EM coupling is retained.

    The threshold is a model configuration criterion, not a universal physical
    constant. The final backend may use a stricter value after validation.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return torch.as_tensor(rm) >= threshold
