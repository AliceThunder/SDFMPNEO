"""Minimal demo for the physics neural evolution extension."""

import torch

from sdfmpneo.neural import PhysicsNeuralEvolutionOperator, GeometryConditionedEncoder


if __name__ == "__main__":
    modes = 4
    operator = PhysicsNeuralEvolutionOperator(n_modes=modes, geometry_dim=8)
    encoder = GeometryConditionedEncoder(8)

    a0 = torch.zeros((1, modes))
    t = torch.ones((1, 1))
    lambdas = torch.ones((1, modes))
    geometry = encoder(torch.zeros((1, 8)))

    state = operator(a0, t, lambdas, geometry)
    print(state.shape)
