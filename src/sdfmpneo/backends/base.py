from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from torch import Tensor

from ..contracts import BasisBundle, EMOperators, EMSolution, GeometryEncoding, TopologyOperators


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
    """One reduced multiphysics assembly at a candidate slow state.

    The backend reconstructs only the quantities required at the current
    cubature/operator points, updates material coefficients, and returns the EM
    operators plus a closure mapping the resulting EM solution into the coupled
    thermo-fluid residual.
    """

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
    """Bridge from neural trial spaces to deterministic physics kernels.

    The high-performance implementation is expected to combine:
      * BFZI/DSE matrix-free electromagnetic kernels,
      * mixed-dimensional 1D-3D conjugate heat transfer,
      * incompressible RANS-SST operators,
      * basis-operator cubature for local nonlinear terms,
      * an independent certification set.
    """

    def topology(self, geometry: GeometryEncoding) -> TopologyOperators: ...

    def basis_metrics(
        self, geometry: GeometryEncoding, topology: TopologyOperators
    ) -> dict[str, Tensor]: ...

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
