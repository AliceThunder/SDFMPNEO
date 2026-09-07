import torch

from sdfmpneo.neural import PhysicsNeuralEvolutionOperator


def test_operator_forward():
    model = PhysicsNeuralEvolutionOperator(3)
    a0 = torch.zeros((2, 3))
    t = torch.ones((2, 1))
    lambdas = torch.ones((2, 3))
    out = model(a0, t, lambdas)
    assert out.shape == (2, 3)
