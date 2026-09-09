from itertools import combinations_with_replacement
from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.long_time import (
    _PROPAGATOR_CACHES,
    clear_realization_propagator_cache,
    realization_action,
)
from sdfmpneo.analytic.parametric_realization import (
    compile_parametric_realization,
    evaluate_parametric_stable_with_jacobians,
)
from sdfmpneo.training import research as training_research
from sdfmpneo.training import late_stage_runtime as late
from sdfmpneo.training import late_stage_batch as batch


def _graph():
    graph = ParametricAnalyticEvolutionGraph([0.2, 0.35], ["u", "g"])
    graph.add_product_response("r0", 0, ["u"], 0.4)
    graph.add_product_response("r1", 1, ["a0_0", "g"], -0.2)
    graph.add_product_response("r2", 0, ["g", "r0"], 0.15)
    graph.add_product_response("r3", 1, ["r2", "a0_1"], 0.08)
    return graph


def test_admissible_parent_generator_matches_historical_filter_order():
    base = tuple(f"b{i}" for i in range(14))
    response = tuple(f"r{i}" for i in range(32))
    names = base + response
    response_set = set(response)
    for degree in range(4):
        expected = [
            parents for parents in combinations_with_replacement(names, degree)
            if sum(parent in response_set for parent in parents) <= 1
        ]
        actual = list(batch._iter_admissible_parent_tuples(names, response_set, degree, 1))
        assert actual == expected

    large_names = base + tuple(f"r{i}" for i in range(256))
    large_response = set(large_names[len(base):])
    count = sum(
        len(list(batch._iter_admissible_parent_tuples(large_names, large_response, degree, 1)))
        for degree in range(4)
    )
    assert count == 31400


def test_sparse_ancestry_weight_jacobian_matches_dense_reference():
    graph = _graph()
    a0 = np.array([0.17, -0.09])
    operating = np.array([1.7, -0.35])
    for time in (0.0, 0.3, 4.0, 1e4, np.inf):
        dense = evaluate_parametric_stable_with_jacobians(
            graph, time, a0=a0, operating=operating, weight_derivatives=True
        )
        point = np.concatenate([a0, operating, [time]])
        sparse = late._sparse_weight_value_jacobian(graph, point)
        for new, old in zip(sparse, dense):
            assert np.allclose(new, old, rtol=2e-11, atol=2e-12)


def test_structured_candidate_scores_match_generic_kronecker_path(monkeypatch):
    monkeypatch.setenv("SDFMPNEO_POINT_WORKERS", "1")
    graph = _graph()
    point_data = [
        (np.array([0.1, -0.04]), np.array([1.2, 0.3]), 0.15),
        (np.array([-0.08, 0.07]), np.array([0.6, -0.2]), 2.5),
        (np.array([0.03, 0.02]), np.array([1.8, 0.1]), np.inf),
    ]
    records = []
    compiled = []
    for index, (a0, operating, time) in enumerate(point_data):
        records.append(SimpleNamespace(
            initial=a0,
            u=operating,
            time=time,
            residual=np.array([0.13 - 0.02 * index, -0.08 + 0.01 * index]),
            J=np.array([[-0.21, 0.025], [0.04, -0.31]]) + 0.003 * index,
        ))
        compiled.append(compile_parametric_realization(graph, a0=a0, operating=operating))

    parents = [(), ("u",), ("a0_0", "g"), ("g", "r0"), ("a0_1", "r2")]
    plans = [batch._parent_plan(graph, value) for value in parents]
    accelerated = batch._score_parent_batch(graph, records, compiled, plans)
    for parent, (inner_new, norm_new) in zip(parents, accelerated):
        inner_old, norm_old = training_research._candidate_tangent_scores(
            graph, records, compiled, parent
        )
        assert np.allclose(inner_new, inner_old, rtol=2e-10, atol=2e-12)
        assert np.allclose(norm_new, norm_old, rtol=2e-10, atol=2e-12)


def test_candidate_propagators_do_not_evict_main_cache():
    clear_realization_propagator_cache()
    A = np.diag([-0.2, -0.7]).astype(complex)
    B = np.array([1.0, -0.3], dtype=complex)
    realization_action(A, B, 3.0)
    main_before = len(_PROPAGATOR_CACHES["main"])
    candidate_before = len(_PROPAGATOR_CACHES["candidate"])
    assert main_before == 1
    assert candidate_before == 0

    realization_action(A, B, 3.0, cache_namespace="candidate")
    assert len(_PROPAGATOR_CACHES["main"]) == main_before
    assert len(_PROPAGATOR_CACHES["candidate"]) == 1


def test_geometry_working_set_cache_reserves_unique_geometries(monkeypatch):
    monkeypatch.setenv("SDFMPNEO_GEOMETRY_CACHE_MAX", "512")

    class FakeGeometryModel:
        def __init__(self):
            self.cache_size = 1
            self.thermal_model = SimpleNamespace(rank=1)
            self.geometry_names = ("gx", "gy")
            self.calls = []

        def denormalize(self, z):
            return np.asarray(z, dtype=float)

        def context(self, geometry):
            key = tuple(np.asarray(geometry, dtype=float))
            self.calls.append(key)
            return key

    late.install_geometry_working_set_cache(FakeGeometryModel)
    model = FakeGeometryModel()
    # Layout: one initial coordinate, two geometry coordinates, one current, time.
    points = np.array([
        [0.0, -1.0, -1.0, 0.2, 0.0],
        [0.1, 0.0, 0.5, 0.4, 1.0],
        [0.2, 1.0, 1.0, 0.6, np.inf],
        [0.3, 0.0, 0.5, 0.8, 4.0],
    ])
    required = model.prepare_training_contexts(points)
    assert required == 3
    assert model.cache_size == 3
    assert len(model.calls) == 3
