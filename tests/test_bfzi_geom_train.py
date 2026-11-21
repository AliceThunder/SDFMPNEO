import numpy as np
import torch

from sdfmpneo_bfzi.geom_train import shared_step


def test_shared_step_matches_deployment_block_solve():
    rng = np.random.default_rng(12)
    rank = 48
    a = rng.normal(size=(rank, rank))
    k = a.T @ a / rank + np.eye(rank) * 0.4
    prev = rng.normal(size=(3, rank))
    guess = rng.normal(size=(3, rank))
    q = rng.normal(size=(3, rank))
    dt = 1.875
    mix = 0.5

    expected = []
    for p, g, src in zip(prev, guess, q):
        mat_s = np.eye(rank - 5) / dt + k[5:, 5:]
        phys = np.linalg.solve(
            mat_s, p[5:] / dt + src[5:] - k[5:, :5] @ g[:5])
        shape = (1. - mix) * g[5:] + mix * phys
        mat_m = np.eye(5) / dt + k[:5, :5]
        mean = np.linalg.solve(
            mat_m, p[:5] / dt + src[:5] - k[:5, 5:] @ shape)
        expected.append(np.r_[mean, shape])

    out = shared_step(
        torch.tensor(prev, dtype=torch.float64),
        torch.tensor(guess, dtype=torch.float64),
        dt,
        torch.tensor(k, dtype=torch.float64),
        torch.tensor(q, dtype=torch.float64),
        mix)
    assert np.allclose(out.detach().numpy(), expected, rtol=1e-11, atol=1e-11)


def test_shared_step_has_gradients_through_neural_guess():
    rank = 48
    k = torch.eye(rank, dtype=torch.float64)
    prev = torch.zeros((2, rank), dtype=torch.float64)
    guess = torch.randn((2, rank), dtype=torch.float64, requires_grad=True)
    q = torch.randn((2, rank), dtype=torch.float64)
    out = shared_step(prev, guess, 1.875, k, q, 0.5)
    out.square().mean().backward()
    assert guess.grad is not None
    assert torch.isfinite(guess.grad).all()
    assert float(guess.grad.norm()) > 0.
