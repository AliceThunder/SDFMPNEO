from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch import Tensor


@dataclass
class BasisQueryFeatures:
    """Local query features for coordinate-conditioned basis decoding.

    The first dimension is batch. Query feature dimensions may differ between
    physical heads; each tensor contains the local coordinates/metric/boundary
    descriptors needed by the corresponding decoder.

    Shapes:
        current_potential:  [B, Naj + Nhj, DqJ]
        thermal:           [B, Nt, DqT]
        velocity_potential:[B, Nav + Nhv, DqV]
        log_tke:           [B, Nk, DqK]
        log_omega:         [B, Nw, DqW]
    """

    current_potential: Tensor
    thermal: Tensor
    velocity_potential: Tensor
    log_tke: Tensor
    log_omega: Tensor


@dataclass
class GeometryEncoding:
    """Geometry/environment inputs consumed by the neural basis generator.

    `global_features` is shared by all heads and must not contain actual port
    excitation amplitudes. `slow_features` is visible only to thermal/flow
    heads and may contain excitation invariants such as |I|^2 or operating-load
    descriptors. This separation prevents excitation leakage into the shared EM
    trial space.

    Shapes:
        coil_tokens:    [B, Nc, Dc]
        package_tokens: [B, Np, Dp]
        global_features:[B, Dg]
        slow_features:  [B, Ds]
    """

    coil_tokens: Tensor
    package_tokens: Tensor
    global_features: Tensor
    slow_features: Tensor
    basis_queries: BasisQueryFeatures


@dataclass
class TopologyOperators:
    """Gauge-reduced reference-domain exact-sequence generators.

    `curl_current` and `curl_velocity` are not arbitrary raw incidence matrices:
    the backend must remove gauge-null directions so their columns are linearly
    independent exact-space generators. Harmonic columns complete the
    solenoidal space on multiply connected domains. Divergence maps are carried
    explicitly so the exact-sequence identities can be verified at runtime and
    in CI.

    Shapes:
        curl_current:      [Nj, Naj]
        harmonic_current:  [Nj, Nhj]
        divergence_current:[Ndj, Nj]
        curl_velocity:     [Nv, Nav]
        harmonic_velocity: [Nv, Nhv]
        divergence_velocity:[Ndv, Nv]
    """

    curl_current: Tensor
    harmonic_current: Tensor
    divergence_current: Tensor
    curl_velocity: Tensor
    harmonic_velocity: Tensor
    divergence_velocity: Tensor


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
