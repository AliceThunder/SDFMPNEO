import numpy as np
import pytest
import scipy.sparse as sp

from sdfmpneo.unified_maxwell import NeuralMaxwellAccelerator
from sdfmpneo.unified_neural_operator import (
    build_edge_residual_operator,
    edge_multiscale_group_ids,
    neural_correction,
    operator_feature_statistics,
    residual_features,
)
from sdfmpneo.unified_trainer import _jacobi_arnoldi_seed_bank, _step_loss_weights


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


def network_config(**overrides):
    value = {
        "width": 8,
        "fine_message_steps": 1,
        "coarse_levels": 2,
        "coarse_message_steps": 1,
        "fusion_message_steps": 1,
        "solver_steps": 2,
        "activation": "silu",
    }
    value.update(overrides)
    return value


def test_residual_features_are_full_edge_and_use_operator_statistics():
    A, B = random_problem()
    groups = edge_multiscale_group_ids(TinyTopology(), 2)
    graph = operator_feature_statistics(A, group_ids=groups)
    features, jacobi, scale = residual_features(B, graph)
    assert features.shape == (B.shape[1], A.shape[0], 9)
    assert jacobi.shape == B.shape
    assert scale.shape == (B.shape[1],)
    assert graph.normalized_coupling.shape == A.shape
    assert len(graph.coarse_couplings) == 2
    assert graph.coarse_couplings[0].shape[0] < A.shape[0]
    assert graph.coarse_couplings[1].shape[0] <= graph.coarse_couplings[0].shape[0]
    assert np.all(np.isfinite(features))


def test_sparse_hierarchy_retains_signed_complex_coupling():
    n = TinyTopology.n_edges
    diagonal = (2.0 + 0.4j) * np.ones(n)
    A = sp.diags(diagonal, format="csr") + sp.diags(
        [0.7 * np.ones(n - 1), -0.45j * np.ones(n - 1)], [-1, 1], format="csr"
    )
    groups = edge_multiscale_group_ids(TinyTopology(), 2)
    graph = operator_feature_statistics(A, group_ids=groups)
    values = graph.normalized_coupling.data
    assert np.any(np.abs(values.real) > 0)
    assert np.any(np.abs(values.imag) > 0)
    assert np.max(np.asarray(np.abs(graph.normalized_coupling).sum(axis=1)).ravel()) <= 1.0 + 1e-12
    for coarse in graph.coarse_couplings:
        assert np.max(np.asarray(np.abs(coarse).sum(axis=1)).ravel(), initial=0.0) <= 1.0 + 1e-12


def test_network_uses_multiscale_hierarchy_and_outputs_full_edge_correction():
    torch = pytest.importorskip("torch")
    A, B = random_problem(seed=9)
    network = build_edge_residual_operator(TinyTopology(), network_config()).double()
    graph = operator_feature_statistics(A, group_ids=network.multiscale_group_ids)
    features, _, _ = residual_features(B, graph)
    hierarchy = graph.torch_hierarchy(torch, "cpu", torch.float64)
    output = network(torch.as_tensor(features, dtype=torch.float64), hierarchy)
    assert len(hierarchy) == 3
    assert tuple(output.shape) == (B.shape[1], A.shape[0], 2)
    assert torch.max(torch.abs(output)).item() == 0.0


def test_training_seed_bank_matches_fgmres_arnoldi_distribution_and_preserves_port_weight():
    A, B = random_problem(seed=13)
    graph = operator_feature_statistics(A)
    bank, weights = _jacobi_arnoldi_seed_bank(
        A,
        B,
        graph.diagonal,
        vectors_per_port=2,
        port_weight=0.6,
        rng=np.random.default_rng(19),
    )
    assert bank.shape == (TinyTopology.n_edges, 6)
    assert weights.shape == (6,)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.isclose(np.sum(weights[:2]), 0.6)
    assert np.isclose(np.sum(weights[2:]), 0.4)
    assert np.linalg.matrix_rank(bank) > np.linalg.matrix_rank(B)
    q, _ = np.linalg.qr(B)
    outside = []
    for column in range(B.shape[1], bank.shape[1]):
        vector = bank[:, column]
        projected = q @ (q.conj().T @ vector)
        outside.append(np.linalg.norm(vector - projected) / np.linalg.norm(vector))
    assert min(outside) > 1e-3


def test_three_step_objective_focuses_on_final_residual():
    weights = _step_loss_weights(3, 0.7)
    assert np.allclose(weights, [0.1, 0.2, 0.7])
    assert np.isclose(sum(weights), 1.0)


def test_zero_initialized_network_fallback_then_fgmres_closes_true_residual():
    pytest.importorskip("torch")
    A, B = random_problem(seed=11)
    network = build_edge_residual_operator(TinyTopology(), network_config()).double()
    correction = neural_correction(network, A, B)
    assert np.all(np.isfinite(correction))
    assert np.linalg.norm(correction) > 0

    accelerator = NeuralMaxwellAccelerator(
        network, residual_tolerance=1e-10, max_iterations=80, restart=8, neural_steps=2
    )
    X, report = accelerator.solve(A, B)
    relative = np.linalg.norm(B - A @ X, axis=0) / np.linalg.norm(B, axis=0)
    assert max(report.final_relative_residual) <= 1e-10
    assert np.max(relative) <= 1e-10
    assert np.allclose(A @ X, B, rtol=1e-9, atol=1e-9)


def test_nonfinite_neural_output_falls_back_without_surrogate_answer():
    torch = pytest.importorskip("torch")
    A, B = random_problem(seed=17)

    class NanNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
            self.n_edges = A.shape[0]

        def forward(self, x, coupling):
            return torch.full(
                (x.shape[0], x.shape[1], 2), float("nan"), dtype=x.dtype, device=x.device
            ) + 0.0 * self.anchor

    accelerator = NeuralMaxwellAccelerator(
        NanNetwork().double(), residual_tolerance=1e-10, max_iterations=80, restart=8, neural_steps=2
    )
    X, report = accelerator.solve(A, B)
    relative = np.linalg.norm(B - A @ X, axis=0) / np.linalg.norm(B, axis=0)
    assert np.max(relative) <= 1e-10
    assert max(report.final_relative_residual) <= 1e-10
