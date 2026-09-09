from __future__ import annotations

import pytest

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.cpp_dag_backend import backend_info
from sdfmpneo.training.cpp_dag_runtime import _plan
from sdfmpneo.training.native_thread_probe import observed_native_threads


def test_native_openmp_probe_reports_real_team_size():
    info = backend_info(auto_build=True)
    if not info["available"]:
        pytest.skip(f"native DAG compiler/backend unavailable: {info['error']}")
    observed = observed_native_threads(min(4, info["native_threads"]))
    assert observed >= 1
    if info["openmp"] and info["native_threads"] >= 2:
        assert observed >= 2


def test_complex_weight_graph_falls_back_instead_of_dropping_imaginary_part():
    graph = ParametricAnalyticEvolutionGraph([0.7], ["u0"])
    graph.add_product_response("complex_response", 0, ("u0",), 0.4 + 0.2j)
    assert _plan(graph) is None
