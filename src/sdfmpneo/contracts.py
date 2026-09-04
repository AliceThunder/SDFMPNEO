from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch import Tensor


@dataclass
class BasisQueryFeatures:
    """Local query features for coordinate-conditioned basis decoding."""

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
    heads. This separation prevents excitation leakage into the shared EM trial
    space.
    """

    coil_tokens: Tensor
    package_tokens: Tensor
    global_features: Tensor
    slow_features: Tensor
    basis_queries: BasisQueryFeatures


@dataclass
class TopologyOperators:
    """Gauge-reduced reference-domain exact-sequence generators."""

    curl_current: Tensor
    harmonic_current: Tensor
    curl_velocity: Tensor
    harmonic_velocity: Tensor
    divergence_current: Tensor | None = None
    divergence_velocity: Tensor | None = None


@dataclass
class RawBasisBundle:
    """Coordinate-decoder outputs before hard physical maps."""

    current_potential: Tensor
    thermal: Tensor
    velocity_potential: Tensor
    log_tke: Tensor
    log_omega: Tensor


@dataclass
class PhysicsSeedBundle:
    """Operator-constructed seed spaces in physical DOF coordinates.

    These columns are produced without solution labels. Typical constructions:
      * EM: source/operator compression and rational Krylov modes;
      * thermal: low diffusion/generalised-energy eigenmodes;
      * velocity: divergence-free Stokes modes;
      * SST log states: constant/wall-distance/operator modes.

    Shapes are `[B,Np,sp]`; zero-column tensors are valid. Seed ranks must not
    exceed the first nested rank level because every online rank must retain the
    complete physical backbone.
    """

    current: Tensor
    thermal: Tensor
    velocity: Tensor
    log_tke: Tensor
    log_omega: Tensor


@dataclass
class BasisBundle:
    current: Tensor
    thermal: Tensor
    velocity: Tensor
    log_tke: Tensor
    log_omega: Tensor


@dataclass
class EMOperators:
    """Dense reference EM operators used only by adapters/tests.

    Production multiphysics backends should expose `MatrixFreeEMActions`
    instead of materialising full-space resistance/inductance matrices.
    """

    resistance: Tensor
    inductance: Tensor
    coupling: Tensor
    z_background: Tensor
    omega: Tensor


class MatrixFreeEMActions(Protocol):
    """Matrix-free electromagnetic actions on blocks of trial vectors.

    All vector arguments have shape `[B,Nj,K]`. Implementations may use FFT
    Green convolution, FMM, H/H2 matrices, DSE kernels, sparse PDE actions, or
    another deterministic backend. No full `[Nj,Nj]` operator is required.

    `project_rhs(B)` returns the multiport source projection `[B,r,P]` directly.
    This allows structured coil/source kernels to accumulate reduced moments
    without materialising a full `[B,Nj,P]` source field.
    """

    @property
    def z_background(self) -> Tensor: ...

    def apply_system(self, vectors: Tensor) -> Tensor: ...

    def project_rhs(self, basis: Tensor) -> Tensor: ...

    def port_feedback(self, vectors: Tensor) -> Tensor: ...

    def apply_dissipation(self, vectors: Tensor) -> Tensor: ...


@dataclass
class EMReducedSystem:
    """Small system obtained by matrix-free projection onto one shared EM basis."""

    system_matrix: Tensor  # [B,r,r], complex
    rhs_matrix: Tensor  # [B,r,P], complex
    feedback_matrix: Tensor  # [B,P,r], complex
    dissipation_matrix: Tensor  # [B,r,r], Hermitian positive semidefinite
    z_background: Tensor  # [B,P,P], complex


@dataclass
class EMSolution:
    reduced_coefficients: Tensor
    impedance_sea: Tensor
    impedance_total: Tensor
    seawater_loss_matrix: Tensor
    current_dofs: Tensor | None = None


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
    def __call__(self, z: Tensor) -> Tensor: ...


class ResidualAssembler(Protocol):
    def __call__(self, z: Tensor) -> Tensor: ...


ResidualFactory = Callable[[BasisBundle, EMSolution], ThermoFluidResidual]
