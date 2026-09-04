from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import RankSchedule
from ..contracts import GeometryEncoding, RawBasisBundle, TopologyOperators
from .base import NeuralBasisGenerator


class _MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )


class CoordinateModeDecoder(nn.Module):
    """Evaluate ordered neural basis modes only at requested coordinates."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        hidden_dim: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.query_encoder = _MLP(query_dim, hidden_dim, hidden_dim)
        self.context_encoder = _MLP(context_dim, hidden_dim, hidden_dim)
        self.mode_embedding = nn.Parameter(torch.empty(rank, hidden_dim))
        self.mode_scale = nn.Parameter(torch.ones(rank))
        nn.init.orthogonal_(self.mode_embedding)

    @property
    def rank(self) -> int:
        return self.mode_embedding.shape[0]

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        if queries.ndim != 3 or context.ndim != 2:
            raise ValueError("queries must be [B,N,Dq] and context must be [B,Dc]")
        if queries.shape[0] != context.shape[0]:
            raise ValueError("query/context batch dimensions must match")
        local = self.query_encoder(queries)
        global_context = self.context_encoder(context)[:, None, :]
        feature = torch.tanh(local + global_context)
        values = torch.einsum("bnh,rh->bnr", feature, self.mode_embedding)
        values = values / math.sqrt(feature.shape[-1])
        return values * self.mode_scale[None, None, :]


class ReferenceCoordinateBasisGenerator(NeuralBasisGenerator):
    """Executable foundation network matching the final SDF-MPNEO data flow.

    This is intentionally a reference implementation, not the final SE(3)-
    equivariant encoder. It already enforces the important information-flow
    rule: the electromagnetic head only receives the shared geometry/environment
    context, while thermal/flow heads may additionally receive slow operating
    features. Replacing the shared encoder later does not change any physical
    layer or backend contract.
    """

    def __init__(
        self,
        coil_token_dim: int,
        package_token_dim: int,
        global_dim: int,
        slow_dim: int,
        query_dims: tuple[int, int, int, int, int],
        ranks: RankSchedule | None = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        ranks = ranks or RankSchedule()
        maximum = ranks.maximum

        self.coil_encoder = _MLP(coil_token_dim, hidden_dim, hidden_dim)
        self.package_encoder = _MLP(package_token_dim, hidden_dim, hidden_dim)
        self.global_encoder = _MLP(global_dim, hidden_dim, hidden_dim)
        self.shared_fusion = _MLP(3 * hidden_dim, hidden_dim, hidden_dim)

        self.slow_encoder = _MLP(slow_dim, hidden_dim, hidden_dim)
        self.slow_fusion = _MLP(2 * hidden_dim, hidden_dim, hidden_dim)

        qj, qt, qv, qk, qw = query_dims
        self.current_head = CoordinateModeDecoder(qj, hidden_dim, hidden_dim, maximum.em)
        self.thermal_head = CoordinateModeDecoder(qt, hidden_dim, hidden_dim, maximum.thermal)
        self.velocity_head = CoordinateModeDecoder(qv, hidden_dim, hidden_dim, maximum.velocity)
        self.tke_head = CoordinateModeDecoder(qk, hidden_dim, hidden_dim, maximum.tke)
        self.omega_head = CoordinateModeDecoder(qw, hidden_dim, hidden_dim, maximum.omega)

    @staticmethod
    def _mean_tokens(encoded: Tensor) -> Tensor:
        if encoded.ndim != 3:
            raise ValueError("encoded tokens must have shape [B,N,H]")
        if encoded.shape[1] == 0:
            raise ValueError("token sets must not be empty in the reference encoder")
        return encoded.mean(dim=1)

    def _contexts(self, geometry: GeometryEncoding) -> tuple[Tensor, Tensor]:
        coil = self._mean_tokens(self.coil_encoder(geometry.coil_tokens))
        package = self._mean_tokens(self.package_encoder(geometry.package_tokens))
        global_latent = self.global_encoder(geometry.global_features)
        shared = self.shared_fusion(torch.cat((coil, package, global_latent), dim=-1))

        slow = self.slow_encoder(geometry.slow_features)
        slow_context = self.slow_fusion(torch.cat((shared, slow), dim=-1))
        return shared, slow_context

    @staticmethod
    def _validate_query_counts(
        geometry: GeometryEncoding, topology: TopologyOperators
    ) -> None:
        q = geometry.basis_queries
        expected_current = topology.curl_current.shape[-1] + topology.harmonic_current.shape[-1]
        expected_velocity = topology.curl_velocity.shape[-1] + topology.harmonic_velocity.shape[-1]
        if q.current_potential.shape[1] != expected_current:
            raise ValueError(
                f"current query count {q.current_potential.shape[1]} != {expected_current}"
            )
        if q.velocity_potential.shape[1] != expected_velocity:
            raise ValueError(
                f"velocity query count {q.velocity_potential.shape[1]} != {expected_velocity}"
            )

    def forward(
        self,
        geometry: GeometryEncoding,
        topology: TopologyOperators,
    ) -> RawBasisBundle:
        self._validate_query_counts(geometry, topology)
        shared, slow = self._contexts(geometry)
        q = geometry.basis_queries

        return RawBasisBundle(
            current_potential=self.current_head(q.current_potential, shared),
            thermal=self.thermal_head(q.thermal, slow),
            velocity_potential=self.velocity_head(q.velocity_potential, slow),
            log_tke=self.tke_head(q.log_tke, slow),
            log_omega=self.omega_head(q.log_omega, slow),
        )
