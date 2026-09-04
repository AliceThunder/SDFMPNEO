from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .backends.base import CertificationIndicators, FinalFields, MultiphysicsBackend
from .config import ModelConfig, RankLevel
from .contracts import BasisBundle, EMSolution, GeometryEncoding
from .networks.base import NeuralBasisGenerator
from .physics.em import solve_em_schur
from .reduction import build_physical_bases, truncate_bases
from .seeds import validate_seed_prefix_ranks
from .solvers.equilibrium import EquilibriumResult, solve_pseudo_transient_newton
from .topology import validate_topology


@dataclass
class SDFMPNEOOutput:
    em: EMSolution
    slow_state: Tensor
    fields: FinalFields
    indicators: CertificationIndicators
    rank: RankLevel
    equilibrium: EquilibriumResult
    requires_fallback: bool


class SDFMPNEO(nn.Module):
    """Adaptive-rank strongly coupled SDF-MPNEO forward operator."""

    def __init__(
        self,
        basis_generator: NeuralBasisGenerator,
        backend: MultiphysicsBackend,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.basis_generator = basis_generator
        self.backend = backend
        self.config = config or ModelConfig()

    @staticmethod
    def _rank_tuple(level: RankLevel) -> tuple[int, int, int, int, int]:
        return level.em, level.thermal, level.velocity, level.tke, level.omega

    def _residual(
        self,
        geometry: GeometryEncoding,
        bases: BasisBundle,
        slow_state: Tensor,
    ) -> Tensor:
        assembly = self.backend.assemble_reduced(geometry, bases, slow_state)
        em = solve_em_schur(bases.current, assembly.em_operators)
        return assembly.slow_residual_from_em(em)

    def _evaluate_final(
        self,
        geometry: GeometryEncoding,
        bases: BasisBundle,
        slow_state: Tensor,
    ) -> tuple[EMSolution, CertificationIndicators, FinalFields]:
        assembly = self.backend.assemble_reduced(geometry, bases, slow_state)
        em = solve_em_schur(bases.current, assembly.em_operators)
        indicators = self.backend.certify(geometry, bases, slow_state, em)
        fields = self.backend.reconstruct_fields(geometry, bases, slow_state)
        return em, indicators, fields

    def forward(self, geometry: GeometryEncoding) -> SDFMPNEOOutput:
        topology = self.backend.topology(geometry)
        validate_topology(topology)
        metrics = self.backend.basis_metrics(geometry, topology)
        seeds = self.backend.physics_seed_bases(geometry, topology)
        validate_seed_prefix_ranks(seeds, self.config.ranks.levels[0])
        raw_bases = self.basis_generator(geometry, topology)
        full_bases = build_physical_bases(
            raw_bases,
            topology,
            metrics,
            seeds,
            jitter=self.config.metric_jitter,
            rank_rtol=self.config.basis_rank_rtol,
        )

        previous_state: Tensor | None = None
        final_output: SDFMPNEOOutput | None = None

        for level in self.config.ranks.levels:
            bases = truncate_bases(full_bases, self._rank_tuple(level))
            initial = self.backend.initial_slow_state(
                geometry, bases, previous_state=previous_state
            )
            if initial.ndim != 1:
                raise ValueError(
                    "The foundation solver currently handles one parameter sample "
                    "at a time; initial_slow_state must return shape [n]."
                )

            def residual(z: Tensor) -> Tensor:
                return self._residual(geometry, bases, z)

            mass = self.backend.pseudo_mass(geometry, bases, initial)
            equilibrium = solve_pseudo_transient_newton(
                residual,
                initial,
                self.config.solver,
                pseudo_mass=mass,
            )
            em, indicators, fields = self._evaluate_final(
                geometry, bases, equilibrium.state
            )

            accepted = equilibrium.converged and indicators.accepted(
                self.config.impedance_tolerance,
                self.config.thermal_indicator_tolerance,
                self.config.flow_indicator_tolerance,
            )
            final_output = SDFMPNEOOutput(
                em=em,
                slow_state=equilibrium.state,
                fields=fields,
                indicators=indicators,
                rank=level,
                equilibrium=equilibrium,
                requires_fallback=not accepted,
            )
            if accepted:
                return final_output

            previous_state = equilibrium.state

        assert final_output is not None
        return final_output
