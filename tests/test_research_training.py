import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from sdfmpneo.training.fixed_layer_cache_runtime import (
    clear_layer_program_cache,
    evaluate_layer_physics as evaluate_cached_layer_physics,
    layer_program,
    prewarm_layer_basis,
)
from sdfmpneo.training.fixed_physics_runtime import evaluate_physics_batch
from sdfmpneo.training.fixed_trial_runtime import _candidate_amplitude_layer
from sdfmpneo.training.monitor import TrainingStopped
from sdfmpneo.training.research import (
    ResearchTrainingConfig,
    _reopen_soft_iteration_budget,
    train_research_network,
)


class _LinearField:
    thermal_model = SimpleNamespace(lambdas=np.array([2.0]))

    def vector_field(self, a, operating):
        return -2.0 * np.asarray(a) + np.asarray([4.0])

    def evaluate(self, a, operating):
        return SimpleNamespace(
            vector_field=self.vector_field(a, operating),
            vector_field_jacobian=np.array([[-2.0]]),
        )


class _NonlinearTwoModeField:
    thermal_model = SimpleNamespace(lambdas=np.array([1.0, 2.0]))

    def vector_field(self, a, operating):
        a = np.asarray(a, dtype=float)
        return np.asarray([
            -a[0] + 0.4 + 0.2 * a[0] * a[0],
            -2.0 * a[1] - 0.3 + 0.15 * a[0] * a[1],
        ])

    def evaluate(self, a, operating):
        a = np.asarray(a, dtype=float)
        return SimpleNamespace(
            vector_field=self.vector_field(a, operating),
            vector_field_jacobian=np.asarray([
                [-1.0 + 0.4 * a[0], 0.0],
                [0.15 * a[1], -2.0 + 0.15 * a[0]],
            ]),
        )


class _StopAfterFirstAccepted:
    def __init__(self):
        self.best_network = None
        self.records = 0

    def retain(self, network):
        self.best_network = network

    def record(self, network, objective, maximum, count, **kwargs):
        self.best_network = network
        self.records += 1
        if self.records == 2:
            raise TrainingStopped("test stop after accepted update")

    def phase(self, *args, **kwargs):
        return None

    def checkpoint(self):
        return None

    def activity(self, *args, **kwargs):
        return None

    def validation(self, *args, **kwargs):
        return None


def _config():
    return ResearchTrainingConfig(
        initial_lower=(-0.2,), initial_upper=(2.2,),
        operating_lower=(), operating_upper=(),
        max_response_time=2.0, residual_tolerance=2e-7,
        sample_count=8, validation_count=8,
        semigroup_sample_count=4, semigroup_validation_count=4,
        max_network_depth=1, max_channels_per_mode=1,
        max_quadratic_rank=1, max_cross_rank=1, max_state_rank=1,
        max_iterations=16, max_iterations_per_layer=16,
        max_validation_epochs=1, prune_rounds=0,
    )


def _network():
    return FixedAnalyticResponseNetwork(
        [2.0], [], max_response_time=2.0,
        input_center=[1.0], input_scale=[1.2],
        depth=1, channels_per_mode=1,
        quadratic_rank=1, cross_rank=1, state_rank=1,
    )


def _deep_network():
    return FixedAnalyticResponseNetwork(
        [1.0, 2.0], [], max_response_time=2.0,
        input_center=[0.0, 0.0], input_scale=[1.0, 1.0],
        depth=2, channels_per_mode=1,
        linear_rank=1, hidden_rank=1, quadratic_rank=1, square_rank=1,
        cross_rank=1, state_rank=1,
        layer_widths=(2, 1),
        layer_hidden_ranks=(1,), layer_cross_ranks=(1,), layer_state_ranks=(1,),
    )


def test_current_trainer_reaches_physics_and_restart_consistency():
    config = _config()
    trained, report = train_research_network(
        _LinearField(), config, network=_network()
    )
    assert isinstance(trained, FixedAnalyticResponseNetwork)
    assert report.numerical_tolerance_met
    assert report.maximum_training_physics_residual <= config.residual_tolerance
    assert report.maximum_validation_physics_residual <= config.residual_tolerance
    assert report.maximum_training_semigroup_rate_defect <= config.residual_tolerance
    assert report.maximum_validation_semigroup_rate_defect <= config.residual_tolerance


def test_deep_candidate_compiled_program_matches_full_exact_residual():
    base = _deep_network()
    theta = base.parameters.copy()
    theta[base._indices("bias_1")[0]] = 0.12
    theta[base._indices("hidden_linear_out_1")[0, 0]] = -0.08
    parent = base.with_parameters(theta)

    ids = np.asarray(parent.layer_amplitude_parameter_indices(1), dtype=int)
    trial_theta = parent.parameters.copy()
    trial_theta[ids] += np.linspace(-2e-3, 3e-3, len(ids))
    trial = parent.with_parameters(trial_theta)
    layer, detected_ids = _candidate_amplitude_layer(parent, trial)
    assert layer == 1
    np.testing.assert_array_equal(detected_ids, ids)

    field = _NonlinearTwoModeField()
    point = np.asarray([[0.25, -0.15, 0.7]])
    clear_layer_program_cache()
    program = layer_program(parent, point, layer)
    assert program is not None
    state, derivative = program.reconstruct(trial, point)
    fast_residual = derivative[0] - field.vector_field(state[0], ())
    exact = evaluate_physics_batch(trial, field, point)[0]
    np.testing.assert_allclose(
        fast_residual, exact.residual, rtol=2e-12, atol=2e-13
    )

    next_theta = trial.parameters.copy()
    next_theta[ids] += np.linspace(1e-3, -1.5e-3, len(ids))
    next_trial = trial.with_parameters(next_theta)
    reused = layer_program(next_trial, point, layer)
    assert reused is program
    state_next, derivative_next = reused.reconstruct(next_trial, point)
    fast_next = derivative_next[0] - field.vector_field(state_next[0], ())
    exact_next = evaluate_physics_batch(next_trial, field, point)[0]
    np.testing.assert_allclose(
        fast_next, exact_next.residual, rtol=2e-12, atol=2e-13
    )


def test_deep_layer_jacobian_reuses_compiled_program_after_amplitude_update():
    base = _deep_network()
    theta = base.parameters.copy()
    theta[base._indices("bias_1")[0]] = 0.09
    theta[base._indices("hidden_linear_out_1")[0, 0]] = -0.06
    network = base.with_parameters(theta)
    points = np.asarray([
        [0.25, -0.15, 0.7],
        [-0.10, 0.35, 1.2],
        [0.40, 0.05, 1.7],
    ])
    field = _NonlinearTwoModeField()

    clear_layer_program_cache()
    assert prewarm_layer_basis(network, points, 1)
    program = layer_program(network, points, 1)
    assert program is not None
    assert program.n_points == len(points)

    records, ids = evaluate_cached_layer_physics(network, field, points, 1)
    assert layer_program(network, points, 1) is program
    for point, record in zip(points, records):
        a, da, ja, jda, direct_ids = network.evaluate_layer_amplitude_jacobian(
            float(point[-1]), a0=point[:2], operating=(), layer=1
        )
        exact = field.evaluate(a, ())
        np.testing.assert_array_equal(ids, direct_ids)
        np.testing.assert_allclose(
            record.residual,
            da - exact.vector_field,
            rtol=2e-12,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            record.parameter_jacobian,
            jda - exact.vector_field_jacobian @ ja,
            rtol=0.0,
            atol=5e-12,
        )

    trial_theta = network.parameters.copy()
    trial_theta[ids] += np.linspace(-1e-3, 2e-3, len(ids))
    trial = network.with_parameters(trial_theta)
    next_records, next_ids = evaluate_cached_layer_physics(trial, field, points, 1)
    np.testing.assert_array_equal(next_ids, ids)
    assert layer_program(trial, points, 1) is program
    assert any(
        not np.allclose(old.residual, new.residual)
        for old, new in zip(records, next_records)
    )


def test_iteration_limit_is_a_soft_block_not_a_stall_condition():
    config = replace(_config(), max_iterations_per_layer=1)
    trained, report = train_research_network(
        _LinearField(), config, network=_network()
    )
    assert report.numerical_tolerance_met
    assert trained.training_state["phase"] == "completed"
    assert len(report.objective_history) > 2


def test_soft_iteration_block_reopens_only_after_an_accepted_block_tail():
    config = replace(_config(), max_iterations_per_layer=1)
    network = _network()
    network.training_state = {
        "phase": "stalled",
        "layer": 0,
        "total_accepted": 7,
        "optimizer": {
            "phase": "physics", "layer": 0, "iteration": 1,
            "damping": 0.25, "retry": 0, "backtrack": 0,
        },
    }
    assert _reopen_soft_iteration_budget(network, config)
    assert network.training_state["phase"] == "physics"
    assert network.training_state["optimizer"]["iteration"] == 0
    assert network.training_state["optimizer"]["damping"] == 0.25

    network.training_state["phase"] = "stalled"
    network.training_state["optimizer"]["iteration"] = 0
    assert not _reopen_soft_iteration_budget(network, config)


def test_training_state_roundtrip_in_network_metadata():
    network = _network()
    network.training_state = {
        "version": 1,
        "phase": "physics",
        "layer": 0,
        "optimizer": {
            "phase": "physics", "layer": 0, "iteration": 3,
            "damping": 1e-4, "retry": 1, "backtrack": 2,
        },
        "points": [[0.0, 0.25]],
        "objective_history": [1.0, 0.25],
    }
    metadata = json.loads(json.dumps(network.to_metadata()))
    restored = FixedAnalyticResponseNetwork.from_metadata(
        metadata, network.parameters
    )
    assert restored.training_state == network.training_state


def test_resume_after_accepted_update_matches_uninterrupted_training():
    config = _config()
    baseline, baseline_report = train_research_network(
        _LinearField(), config, network=_network()
    )

    monitor = _StopAfterFirstAccepted()
    try:
        train_research_network(
            _LinearField(), config, network=_network(), monitor=monitor
        )
    except TrainingStopped:
        pass
    else:
        raise AssertionError("test monitor did not interrupt after accepted update")

    checkpoint = monitor.best_network
    assert checkpoint is not None
    assert checkpoint.training_state["phase"] in {"physics", "restart", "restart_pending"}
    restored = FixedAnalyticResponseNetwork.from_metadata(
        json.loads(json.dumps(checkpoint.to_metadata())),
        checkpoint.parameters,
    )
    resumed, resumed_report = train_research_network(
        _LinearField(), config, network=restored
    )

    np.testing.assert_allclose(
        resumed.parameters, baseline.parameters, rtol=0.0, atol=2e-12
    )
    np.testing.assert_allclose(
        resumed_report.objective_history,
        baseline_report.objective_history,
        rtol=0.0,
        atol=2e-14,
    )
    assert resumed_report.status == baseline_report.status
    assert resumed_report.numerical_tolerance_met == baseline_report.numerical_tolerance_met


def test_trainer_rejects_non_fixed_checkpoint():
    config = ResearchTrainingConfig(
        initial_lower=(0.0,), initial_upper=(0.0,), operating_lower=(), operating_upper=(),
        max_response_time=1.0, residual_tolerance=1e-5,
        sample_count=1, validation_count=1,
        semigroup_sample_count=0, semigroup_validation_count=0,
    )
    try:
        train_research_network(_LinearField(), config, network=object())
    except TypeError as exc:
        assert "FixedAnalyticResponseNetwork" in str(exc)
    else:
        raise AssertionError("non-fixed checkpoint was accepted")
