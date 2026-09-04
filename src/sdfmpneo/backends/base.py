from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from torch import Tensor

from ..contracts import (
    BasisBundle,
    EMOperators,
    EMSolution,
    GeometryEncoding,
    PhysicsSeedBundle,
    TopologyOperators,
)


@dataclass
class CertificationIndicators:
    impedance: float
    thermal: float
    flow: float

    def accepted(self, z_tol: float, t_tol: float, v_tol: float) -> bool:
        return (
            self.impedance <= z_tol
            and self.thermal <= t_tol
            and self.flow <= v_tol
        )


@dataclass
class ReducedAssembly:
    em_operators: EMOperators
    slow_residual_from_em: Callable[[EMSolution], Tensor]


@dataclass
class FinalFields:
    temperature: Tensor | None = None
    conductor_temperature: Tensor | None = None
    velocity: Tensor | None = None
    turbulent_kinetic_energy: Tensor | None = None
    specific_dissipation_rate: Tensor | None = None


class MultiphysicsBackend(Protocol):
    """Bridge from neural trial spaces to deterministic physics kernels."""

    def topology(self, geometry: GeometryEncoding) -> TopologyOperators: ...

    def basis_metrics(
        self, geometry: GeometryEncoding, topology: TopologyOperators
    ) -> dict[str, Tensor]: ...

    def physics_seed_bases(
        self,
        geometry: GeometryEncoding,
        topology: TopologyOperators,
    ) -> PhysicsSeedBundle:
        """Return operator-constructed seed spaces with zero solution labels.

        The seed compiler may use deterministic source/operator compression,
        eigenproblems, or physics solves such as rational Krylov compilation,
        but it must not fit to FEM/CFD/full-order solution snapshots or measured
        target fields. These columns form the mandatory prefix of every nested
        online trial space.
        """
        ...

    def initial_slow_state(
        self,
        geometry: GeometryEncoding,
        bases: BasisBundle,
        previous_state: Tensor | None,
    ) -> Tensor: ...

    def assemble_reduced(
        self,
        geometry: GeometryEncoding,
        bases: BasisBundle,
        slow_state: Tensor,
    ) -> ReducedAssembly: ...

    def pseudo_mass(
        self,
        geometry: GeometryEncoding,
        bases: BasisBundle,
        slow_state: Tensor,
    ) -> Tensor | None: ...

    def training_residual(
        self,
        geometry: GeometryEncoding,
        bases: BasisBundle,
        slow_state: Tensor,
        em_solution: EMSolution,
    ) -> Tensor:
        """Return a nondimensionalised/Riesz-whitened independent residual."""
        ...

    def certify(
        self,
        geometry: GeometryEncoding,
        bases: BasisBundle,
        slow_state: Tensor,
        em_solution: EMSolution,
    ) -> CertificationIndicators: ...

    def reconstruct_fields(
        self,
        geometry: GeometryEncoding,
        bases: BasisBundle,
        slow_state: Tensor,
    ) -> FinalFields: ...
