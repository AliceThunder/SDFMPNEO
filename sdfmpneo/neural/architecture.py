from __future__ import annotations

import torch
from torch import nn


class GeometryConditionedEncoder(nn.Module):
    """Encode physical geometry/material parameters into a latent state.

    This encoder does not replace the physical operators. It only provides a
    compact condition vector for the evolution correction network.
    """

    def __init__(self, n_geometry: int, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_geometry, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, geometry):
        return self.net(geometry)
