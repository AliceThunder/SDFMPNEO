from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from torch import Tensor

from ..contracts import (
    BasisBundle,
    EMSolution,
    GeometryEncoding,
    MatrixFreeEMActions,
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
    """One strongly coupled reduced assembly at a candidate slow state.

    The backend exposes matrix-free EM actions rather than full-space dense
    matrices. The core projects those actions onto the active shared EM basis,
    solves the small port system, then passes the resulting coefficients and
    impedance into the slow thermo-fluid residual closure.
    """

    em_actions: MatrixFreeEMActions
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
        """Return operator-constructed seed spaces with zero solution labels."""
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
    ) -> ReducedAssembly:
        """Assemble state-dependent reduced physics through operator actions.

        A production EM implementation must not construct dense `[Nj,Nj]`
        resistance/inductance/Green matrices merely to project them. It should
        implement `MatrixFreeEMActions` using deterministic operator kernels and
        evaluate those kernels only on the active trial-vector block.
        """
        ...

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
