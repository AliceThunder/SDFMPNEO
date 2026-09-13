import numpy as np
import pytest
import scipy.sparse as sp

from sdfmpneo.unified_dataset import operator_encoding
from sdfmpneo.unified_maxwell import NeuralMaxwellAccelerator


def random_problem(seed=5):
    rng = np.random.default_rng(seed)
    n, r, p = 8, 4, 2
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = raw.conj().T @ raw + (0.5 + 0.3j) * np.eye(n)
    B = rng.normal(size=(n, p)) + 1j * rng.normal(size=(n, p))
    q, _ = np.linalg.qr(rng.normal(size=(n, r)) + 1j * rng.normal(size=(n, r)))
    return sp.csr_matrix(A), B, q


def test_residual_quadratic_form_equals_explicit_full_background_residual():
    A, B, V = random_problem()
    _, baseline, scale, gram, linear, norm2 = operator_encoding(A, B, V)
    rng = np.random.default_rng(9)
    y = rng.normal(size=baseline.shape)
    real_coeff = baseline + scale[:, None] * y
    r = V.shape[1]
    C = (real_coeff[:, :r] + 1j * real_coeff[:, r:]).T

    explicit = np.sum(np.abs(B - A @ (V @ C)) ** 2, axis=0)
    quadratic = (
        norm2
        - 2.0 * np.sum(real_coeff * linear, axis=1)
        + np.einsum("pi,ij,pj->p", real_coeff, gram, real_coeff)
    )
    assert np.allclose(quadratic, explicit, rtol=1e-11, atol=1e-11)


def _network(torch, output_dimension, *, nan=False):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

        def forward(self, x):
            value = torch.full(
                (x.shape[0], output_dimension),
                float("nan") if nan else 0.0,
                dtype=x.dtype,
                device=x.device,
            )
            return value + 0.0 * self.anchor

    return Network().double()


def test_neural_guess_is_only_an_initial_guess_and_true_residual_controls_answer():
    torch = pytest.importorskip("torch")
    A, B, _ = random_problem(seed=11)
    V = np.eye(A.shape[0], dtype=complex)
    feature, _, _, _, _, _ = operator_encoding(A, B, V)
    output_dimension = 2 * V.shape[1] * B.shape[1]

    accelerator = NeuralMaxwellAccelerator(
        _network(torch, output_dimension),
        V,
        residual_tolerance=1e-10,
        max_iterations=100,
    )
    X, report = accelerator.solve(A, B)
    relative = np.linalg.norm(B - A @ X, axis=0) / np.linalg.norm(B, axis=0)

    assert feature.ndim == 1
    assert max(report.final_relative_residual) <= 5e-10
    assert np.max(relative) <= 5e-10
    assert np.allclose(A @ X, B, rtol=1e-9, atol=1e-9)


def test_nonfinite_neural_guess_falls_back_to_physical_maxwell_correction():
    torch = pytest.importorskip("torch")
    A, B, _ = random_problem(seed=17)
    V = np.eye(A.shape[0], dtype=complex)
    output_dimension = 2 * V.shape[1] * B.shape[1]
    accelerator = NeuralMaxwellAccelerator(
        _network(torch, output_dimension, nan=True),
        V,
        residual_tolerance=1e-10,
        max_iterations=100,
    )

    X, report = accelerator.solve(A, B)
    relative = np.linalg.norm(B - A @ X, axis=0) / np.linalg.norm(B, axis=0)

    assert np.allclose(report.initial_relative_residual, (1.0, 1.0), rtol=1e-12, atol=1e-12)
    assert max(report.final_relative_residual) <= 5e-10
    assert np.max(relative) <= 5e-10
