import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
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
