from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from ..contracts import GeometryEncoding, RawBasisBundle, TopologyOperators


class NeuralBasisGenerator(nn.Module, ABC):
    """Shared-encoder, physics-head interface.

    Implementations must generate *trial-space coordinates*, not physical
    solution labels. Hard de Rham maps and metric orthogonalisation are applied
    after this module.
    """

    @abstractmethod
    def forward(
        self,
        geometry: GeometryEncoding,
        topology: TopologyOperators,
    ) -> RawBasisBundle:
        raise NotImplementedError
