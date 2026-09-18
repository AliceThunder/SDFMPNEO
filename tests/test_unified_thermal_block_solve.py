import types

import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_thermal import _maxwell_port_fields, _solve_block, _trajectory_metric


def test_thermal_block_solve_matches_dense_multiple_rhs():
    A = sp.csr_matrix(
        np.array(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ]
        )
    )
    B = np.array([[1.0, 0.0], [2.0, 1.0], [0.5, -1.0]])
    got = _solve_block(A, B)
    expected = np.linalg.solve(A.toarray(), B)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_wire_trajectory_metric_uses_observable_vector_scale():
    context = types.SimpleNamespace(
        line_heat_weights=(
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        )
    )
    truth = np.array([1.0, 0.0])
    approx = np.array([0.99, 1.0e-6])
    metrics = _trajectory_metric(
        None,
        context,
        np.ones(2),
        truth,
        approx,
    )
    assert metrics["maximum_wire_average_relative_error"] < 0.02
    assert metrics["composite_relative_error"] < 0.02


def test_thermal_anchor_maxwell_uses_installed_truth_solver(monkeypatch):
    import sdfmpneo.unified_tensor_surrogate as tensor_truth

    expected = np.array([[1.0 + 2.0j], [3.0 - 1.0j]])
    called = {}

    def fake(background, context):
        called["background"] = background
        called["context"] = context
        return expected.copy(), 1.0e-12

    monkeypatch.setattr(tensor_truth, "_solve_port_fields", fake)
    background = types.SimpleNamespace(
        background_config={"linear_solver": {"relative_residual_tolerance": 1e-9}}
    )
    context = object()
    got = _maxwell_port_fields(background, context)
    np.testing.assert_allclose(got, expected)
    assert called == {"background": background, "context": context}
