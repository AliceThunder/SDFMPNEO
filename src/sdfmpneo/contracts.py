from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch import Tensor


@dataclass
class GeometryEncoding:
    """Geometry/environment inputs consumed by the shared encoder.

    Shapes:
        coil_tokens:    [B, Nc, Dc]
        package_tokens: [B, Np, Dp]
        global_features:[B, Dg]
    """

    coil_tokens: Tensor
    package_tokens: Tensor
    global_features: Tensor


@dataclass
class TopologyOperators:
    """Reference-domain exact-sequence maps.

    curl_current maps electromagnetic potential coefficients into seawater
    H(div) current DOFs. curl_velocity does the same for incompressible
    velocity DOFs. Harmonic columns complete ker(div) on multiply connected
    domains.

    Shapes:
        curl_current:     [Nj, Naj]
        harmonic_current: [Nj, Nhj]
        curl_velocity:    [Nv, Nav]
        harmonic_velocity:[Nv, Nhv]
    """

    curl_current: Tensor
    harmonic_current: Tensor
    curl_velocity: Tensor
    harmonic_velocity: Tensor


@dataclass
class RawBasisBundle:
    """Coordinate-decoder outputs before hard physical maps.

    Shapes use the maximum nested rank configured for the model.
    """

    current_potential: Tensor  # [B, Naj+Nhj, rJmax]
    thermal: Tensor  # [B, Nt, rTmax]
    velocity_potential: Tensor  # [B, Nav+Nhv, rVmax]
    log_tke: Tensor  # [B, Nk, rKmax]
    log_omega: Tensor  # [B, Nw, rWmax]


@dataclass
class BasisBundle:
    current: Tensor  # [B, Nj, rJmax]
    thermal: Tensor  # [B, Nt, rTmax]
    velocity: Tensor  # [B, Nv, rVmax]
    log_tke: Tensor  # [B, Nk, rKmax]
    log_omega: Tensor  # [B, Nw, rWmax]


@dataclass
class EMOperators:
    """Full-space electromagnetic operators for one batch.

    RMS phasor convention is used throughout.

    resistance: [B, Nj, Nj], real symmetric positive definite/semi-definite
    inductance: [B, Nj, Nj], real symmetric
    coupling:   [B, Nj, P],  real coil-to-seawater coupling
    z_background:[B, P, P], complex copper + air contribution
    omega:      [B] or [B, 1]
    """

    resistance: Tensor
    inductance: Tensor
    coupling: Tensor
    z_background: Tensor
    omega: Tensor


@dataclass
class EMSolution:
    reduced_coefficients: Tensor  # [B, rJ, P], complex
    current_dofs: Tensor  # [B, Nj, P], complex
    impedance_sea: Tensor  # [B, P, P], complex
    impedance_total: Tensor  # [B, P, P], complex
    seawater_loss_matrix: Tensor  # [B, P, P], complex Hermitian quadratic form


@dataclass
class ReducedState:
    thermal: Tensor
    velocity: Tensor
    log_tke: Tensor
    log_omega: Tensor

    def flatten(self) -> Tensor:
        return torch.cat(
            [self.thermal, self.velocity, self.log_tke, self.log_omega], dim=-1
        )


class ThermoFluidResidual(Protocol):
    """Reduced coupled slow-physics residual F_r(z; xi)."""

    def __call__(self, z: Tensor) -> Tensor: ...


class ResidualAssembler(Protocol):
    """Independent full-space or certification-space residual action."""

    def __call__(self, z: Tensor) -> Tensor: ...


ResidualFactory = Callable[[BasisBundle, EMSolution], ThermoFluidResidual]
