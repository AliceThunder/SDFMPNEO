import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_basis import build_residual_basis, reduced_solution


def test_nested_minimum_residual_spaces_are_monotone():
    rng = np.random.default_rng(123)
    n = 14
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    # Deliberately non-normal complex operator.
    A = np.triu(raw) + 0.3 * np.tril(raw, -1) + (2.0 + 0.7j) * np.eye(n)
    B = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    q, _ = np.linalg.qr(rng.normal(size=(n, 8)) + 1j * rng.normal(size=(n, 8)))
    denom = np.linalg.norm(B, axis=0)

    previous = np.full(B.shape[1], np.inf)
    for rank in range(1, q.shape[1] + 1):
        X = reduced_solution(sp.csr_matrix(A), B, q[:, :rank])
        residual = np.linalg.norm(B - A @ X, axis=0) / denom
        assert np.all(residual <= previous + 1e-11)
        previous = residual


class ToyBackground:
    def __init__(self):
        self.n_edges = 3
        self._operators = {
            0: sp.eye(3, dtype=complex, format="csr"),
            1: sp.csr_matrix(np.array([[2.0, 1.0, 0.0], [0.0, 1.0j, 1.0], [0.0, 0.0, 0.5]], complex)),
        }
        self._rhs = {
            0: np.array([[1.0], [0.0], [0.0]], complex),
            1: np.array([[0.0], [1.0], [1.0]], complex),
        }

    def geometry_context(self, geometry, assemble_thermal=False):
        return int(geometry)

    def em_operator(self, context, state):
        return self._operators[context]

    def rhs_matrix(self, context):
        return self._rhs[context]


def test_automatic_rank_is_set_by_anchor_residual_not_a_fixed_budget():
    background = ToyBackground()
    V, report = build_residual_basis(
        background,
        geometry_samples=[0, 1],
        state_samples=[np.zeros(1), np.zeros(1)],
        target_relative_residual=1e-10,
    )
    assert report.converged
    assert report.stop_reason == "target_reached"
    assert report.basis_dimension == V.shape[1]
    assert 1 <= report.basis_dimension <= background.n_edges
    assert report.maximum_anchor_relative_residual <= 1e-10
