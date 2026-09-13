import numpy as np
import pytest
import scipy.sparse as sp

from sdfmpneo.unified_maxwell import NeuralMaxwellAccelerator
from sdfmpneo.unified_neural_operator import build_edge_residual_operator, residual_features


class TinyTopology:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([0.0, 1.0, 2.0, 3.0])
    edge_tuples = (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1),
        (1, 1, 0, 1), (1, 1, 1, 1), (1, 1, 2, 1),
        (2, 1, 1, 0), (2, 1, 1, 1),
    )
    n_edges = len(edge_tuples)


def random_problem(seed=5):
    rng = np.random.default_rng(seed)
    n, p = TinyTopology.n_edges, 2
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = raw.conj().T @ raw + (0.5 + 0.3j) * np.eye(n)
    B = rng.normal(size=(n, p)) + 1j * rng.normal(size=(n, p))
    return sp.csr_matrix(A), B


def test_residual_features_are_full_edge_local_and_have_jacobi_baseline():
    A, B = random_problem()
    features, jacobi, scale = residual_features(A, B)
    assert features.shape == (B.shape[1], A.shape[0], 9)
    assert jacobi.shape == B.shape
    assert scale.shape == (B.shape[1],)
    assert np.allclose(A.diagonal()[:, None] * jacobi, B)
    assert np.all(np.isfinite(features))


def test_residual_features_see_offdiagonal_sparse_coupling_not_only_the_diagonal():
    n = TinyTopology.n_edges
    diagonal = (2.0 + 0.4j) * np.ones(n)
    A0 = sp.diags(diagonal, format="csr")
    A1 = A0 + sp.diags([0.7 * np.ones(n - 1), 0.7 * np.ones(n - 1)], [-1, 1], format="csr")
    residual = np.ones((n, 1), complex)

    f0, _, _ = residual_features(A0, residual)
    f1, _, _ = residual_features(A1, residual)

    # Columns 7/8 are coupling ratio and normalized sparse row degree.
    assert np.allclose(f0[..., :7], f1[..., :7])
    assert np.max(np.abs(f1[..., 7:] - f0[..., 7:])) > 0.0


def test_zero_initialized_network_reduces_to_physical_jacobi_then_fgmres_closes_true_residual():
    pytest.importorskip("torch")
    A, B = random_problem(seed=11)
    network = build_edge_residual_operator(
        TinyTopology(),
        {"width": 8, "levels": 1, "blocks_per_level": 1, "activation": "silu"},
    ).double()
    accelerator = NeuralMaxwellAccelerator(network, residual_tolerance=1e-10,
                                            max_iterations=80, restart=8)
    X, report = accelerator.solve(A, B)
    relative = np.linalg.norm(B - A @ X, axis=0) / np.linalg.norm(B, axis=0)
    assert max(report.final_relative_residual) <= 1e-10
    assert np.max(relative) <= 1e-10
    assert np.allclose(A @ X, B, rtol=1e-9, atol=1e-9)
    assert max(report.correction_iterations) > 0


def test_nonfinite_neural_output_falls_back_to_jacobi_not_to_a_surrogate_answer():
    torch = pytest.importorskip("torch")
    A, B = random_problem(seed=17)

    class NanNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
            self.n_edges = A.shape[0]

        def forward(self, x):
            return torch.full((x.shape[0], x.shape[1], 2), float("nan"),
                              dtype=x.dtype, device=x.device) + 0.0 * self.anchor

    accelerator = NeuralMaxwellAccelerator(NanNetwork().double(), residual_tolerance=1e-10,
                                            max_iterations=80, restart=8)
    X, report = accelerator.solve(A, B)
    relative = np.linalg.norm(B - A @ X, axis=0) / np.linalg.norm(B, axis=0)
    assert np.max(relative) <= 1e-10
    assert max(report.final_relative_residual) <= 1e-10
