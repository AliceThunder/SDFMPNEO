from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NeuralEvolutionState:
    state: torch.Tensor
    derivative: torch.Tensor


class PhysicsNeuralEvolutionOperator(nn.Module):
    """Physics-guided correction of analytical thermal evolution.

    The network learns only the unresolved correction:

        a(t)=a0*exp(-lambda*t)+delta_a_theta

    rather than learning the complete physical trajectory.
    """

    def __init__(self, n_modes: int, geometry_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.n_modes = n_modes
        self.geometry_dim = geometry_dim
        self.net = nn.Sequential(
            nn.Linear(2*n_modes + 1 + geometry_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_modes),
        )

    def forward(self, a0, t, lambdas, geometry=None):
        if t.ndim == 1:
            t = t[:, None]
        analytic = a0 * torch.exp(-lambdas * t)
        inputs = [a0, t]
        if geometry is not None:
            inputs.append(geometry)
        correction = self.net(torch.cat(inputs, dim=-1))
        return analytic + t * correction

    def evaluate_state(self, a0, t, lambdas, geometry=None):
        t = t.requires_grad_(True)
        state = self.forward(a0, t, lambdas, geometry)
        derivative = torch.autograd.grad(
            state.sum(), t, create_graph=True
        )[0]
        return NeuralEvolutionState(state, derivative)
