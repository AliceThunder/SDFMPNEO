from __future__ import annotations

import numpy as np
import pytest

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph, evaluate_parametric_stable
from sdfmpneo.cpp_dag_backend import backend_info, gn_linearize
from sdfmpneo.training.cpp_dag_runtime import (
    _native_value_jacobian_batch,
    _native_values,
    _plan,
)
from sdfmpneo.training.observation_runtime import sparse_weight_value_jacobian_observed


def _require_backend():
    info = backend_info(auto_build=True)
    if not info["available"]:
        pytest.skip(f"native DAG compiler/backend unavailable: {info['error']}")
    return info


def _graph():
    graph = ParametricAnalyticEvolutionGraph([0.7, 1.3], ["u0", "u1"])
    graph.add_product_response("r0", 0, ("u0", "u0"), 0.7)
    graph.add_product_response("r1", 1, ("a0_0", "r0", "u1"), -0.3)
    graph.add_product_response("r2", 0, ("a0_1", "r1"), 0.2)
    graph.add_product_response("r3", 1, (), -0.11)
    return graph


def _points():
    # [a0_0, a0_1, u0, u1, t]
    return np.array([
        [0.2, -0.1, 1.1, 0.4, 0.0],
        [-0.3, 0.15, 0.7, 1.2, 0.03],
        [0.05, 0.2, 1.4, 0.8, 2.0],
        [0.1, -0.05, 0.9, 1.1, 1000.0],
        [0.1, -0.05, 0.9, 1.1, np.inf],
    ], dtype=float)


def test_native_dag_values_and_sparse_weight_jacobian_match_python():
    info = _require_backend()
    assert info["native_threads"] >= 1
    graph = _graph()
    points = _points()
    assert _plan(graph) is not None

    values = _native_values(graph, points)
    jacobian_values = _native_value_jacobian_batch(graph, points)
    assert values is not None and jacobian_values is not None
    a_native, da_native = values
    a_j, da_j, ja_native, jda_native = jacobian_values
    assert np.allclose(a_native, a_j, rtol=2e-12, atol=2e-13)
    assert np.allclose(da_native, da_j, rtol=2e-12, atol=2e-13)

    for index, point in enumerate(points):
        a_ref, da_ref = evaluate_parametric_stable(
            graph, float(point[-1]), a0=point[:2], operating=point[2:4]
        )
        a_py, da_py, ja_py, jda_py = sparse_weight_value_jacobian_observed(graph, point)
        assert np.allclose(a_native[index], a_ref, rtol=2e-12, atol=2e-13)
        assert np.allclose(da_native[index], da_ref, rtol=2e-12, atol=2e-13)
        assert np.allclose(a_j[index], a_py, rtol=2e-12, atol=2e-13)
        assert np.allclose(da_j[index], da_py, rtol=2e-12, atol=2e-13)
        assert np.allclose(ja_native[index], ja_py, rtol=3e-12, atol=3e-13)
        assert np.allclose(jda_native[index], jda_py, rtol=3e-12, atol=3e-13)


def test_native_gauss_newton_linearization_matches_numpy():
    _require_backend()
    rng = np.random.default_rng(44)
    n_points, n_modes, n_weights = 11, 2, 7
    da = rng.normal(size=(n_points, n_modes))
    F = rng.normal(size=(n_points, n_modes))
    ja = rng.normal(size=(n_points, n_modes, n_weights))
    jda = rng.normal(size=ja.shape)
    JF = rng.normal(size=(n_points, n_modes, n_modes))

    actual = gn_linearize(da, F, ja, jda, JF)
    assert actual is not None
    residual, J = actual
    expected_r = da - F
    expected_J = jda - np.einsum("pij,pjw->piw", JF, ja)
    assert np.allclose(residual, expected_r, rtol=2e-14, atol=2e-14)
    assert np.allclose(J, expected_J, rtol=2e-14, atol=2e-14)


def test_native_plan_fails_closed_for_multiple_response_parents():
    graph = ParametricAnalyticEvolutionGraph([0.8, 1.1], ["u0"])
    graph.add_product_response("r0", 0, ("u0",), 0.4)
    graph.add_product_response("r1", 1, ("u0",), 0.5)
    graph.add_product_response("legacy", 0, ("r0", "r1"), 0.2)
    assert _plan(graph) is None
