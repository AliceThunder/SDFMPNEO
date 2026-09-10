from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sdfmpneo import FixedAnalyticResponseNetwork
from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph, evaluate_parametric_stable
from sdfmpneo.training import research as training_research
from sdfmpneo.training.fixed_network_runtime import (
    _make_network,
    train_fixed_analytic_response_network,
)


def _network():
    return FixedAnalyticResponseNetwork(
        [0.7, 1.1], ["g", "u"],
        input_center=np.zeros(4), input_scale=np.ones(4),
        depth=3, channels_per_mode=2, quadratic_rank=3,
        cross_rank=2, state_rank=2,
    )


def test_fixed_network_parameter_jacobian_matches_finite_difference():
    network = _network()
    a0 = np.array([0.2, -0.1])
    operating = np.array([0.3, 0.6])
    time = 0.4
    a, da, ja, jda = network.evaluate_parameter_jacobian(
        time, a0=a0, operating=operating)
    for index in (0, network.parameter_count // 3, network.parameter_count - 1):
        epsilon = 1.0e-7
        parameters = network.parameters.copy()
        parameters[index] += epsilon
        trial = network.with_parameters(parameters)
        a1, da1 = trial.evaluate(time, a0=a0, operating=operating)
        assert np.allclose((a1 - a) / epsilon, ja[:, index], rtol=2e-5, atol=2e-8)
        assert np.allclose((da1 - da) / epsilon, jda[:, index], rtol=2e-5, atol=2e-8)


def test_fixed_network_operating_jacobian_matches_finite_difference():
    network = _network()
    a0 = np.array([0.2, -0.1])
    operating = np.array([0.3, 0.6])
    time = 0.4
    a, da, ja, jda = network.evaluate_operating_jacobian(
        time, a0=a0, operating=operating)
    for index in range(2):
        epsilon = 1.0e-7
        trial_u = operating.copy()
        trial_u[index] += epsilon
        a1, da1 = network.evaluate(time, a0=a0, operating=trial_u)
        assert np.allclose((a1 - a) / epsilon, ja[:, index], rtol=2e-5, atol=2e-8)
        assert np.allclose((da1 - da) / epsilon, jda[:, index], rtol=2e-5, atol=2e-8)


def test_fixed_network_stationary_query_is_direct_and_has_zero_slope():
    network = _network()
    a, da = network.evaluate(
        np.inf, a0=np.array([0.2, -0.1]), operating=np.array([0.3, 0.6]))
    assert np.all(np.isfinite(a))
    assert np.array_equal(da, np.zeros(2))


def test_fixed_network_metadata_roundtrip_preserves_response():
    network = _network()
    restored = FixedAnalyticResponseNetwork.from_metadata(
        network.to_metadata(), network.parameters.copy())
    for time in (0.0, 1.0e-4, 0.4, 30.0, np.inf):
        expected = network.evaluate(
            time, a0=np.array([0.2, -0.1]), operating=np.array([0.3, 0.6]))
        actual = restored.evaluate(
            time, a0=np.array([0.2, -0.1]), operating=np.array([0.3, 0.6]))
        assert np.allclose(actual[0], expected[0], rtol=2e-13, atol=2e-13)
        assert np.allclose(actual[1], expected[1], rtol=2e-13, atol=2e-13)


def test_public_stable_evaluator_dispatches_to_fixed_network():
    network = _network()
    direct = network.evaluate(
        0.7, a0=np.array([0.1, 0.2]), operating=np.array([-0.3, 0.4]))
    public = evaluate_parametric_stable(
        network, 0.7, a0=np.array([0.1, 0.2]), operating=np.array([-0.3, 0.4]))
    assert np.allclose(public[0], direct[0])
    assert np.allclose(public[1], direct[1])


def test_fresh_empty_graph_becomes_fixed_network_but_trained_legacy_graph_does_not():
    config = SimpleNamespace(
        initial_lower=(-1.0, -1.0), initial_upper=(1.0, 1.0),
        operating_lower=(-1.0, -1.0), operating_upper=(1.0, 1.0),
    )
    field = SimpleNamespace(thermal_model=SimpleNamespace(lambdas=np.array([0.7, 1.1])))
    empty = ParametricAnalyticEvolutionGraph([0.7, 1.1], ["g", "u"])
    assert isinstance(_make_network(field, config, empty), FixedAnalyticResponseNetwork)
    empty.add_product_response("legacy", 0, ("u",), 0.1)
    assert _make_network(field, config, empty) is None


def test_fixed_research_checkpoint_roundtrip_preserves_network(tmp_path):
    from sdfmpneo import ResearchElectroThermalModel, demo_research_model

    model = demo_research_model()
    rank = model.core.thermal_model.rank
    n_operating = model.rhs_map.n_operating
    network = FixedAnalyticResponseNetwork(
        model.core.thermal_model.lambdas,
        [f"u{i}" for i in range(n_operating)],
        input_center=np.zeros(rank + n_operating),
        input_scale=np.ones(rank + n_operating),
        depth=2, channels_per_mode=1, quadratic_rank=2,
        cross_rank=1, state_rank=1,
    )
    model.graph = network
    model.training_config = None
    model.training_report = None
    path = model.save(tmp_path / "fixed_network.npz")
    loaded = ResearchElectroThermalModel.load(path)
    assert isinstance(loaded.graph, FixedAnalyticResponseNetwork)
    assert loaded.graph.to_metadata() == network.to_metadata()
    assert np.array_equal(loaded.graph.parameters, network.parameters)

    a0 = np.full(rank, 0.1)
    operating = np.full(n_operating, 0.2)
    for time in (0.0, 0.2, 1000.0, np.inf):
        expected = model.predict(time, a0=a0, operating=operating, diagnostics=False)
        actual = loaded.predict(time, a0=a0, operating=operating, diagnostics=False)
        assert np.allclose(actual["thermal_coordinates"], expected["thermal_coordinates"])
        assert np.allclose(actual["thermal_derivative"], expected["thermal_derivative"])


def test_package_fresh_training_binding_has_no_candidate_search_driver():
    assert training_research.train_research_graph is train_fixed_analytic_response_network
