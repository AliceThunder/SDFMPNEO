from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from sdfmpneo.training.research import ResearchTrainingConfig, train_research_network


class _LinearField:
    thermal_model = SimpleNamespace(lambdas=np.array([2.0]))

    def vector_field(self, a, operating):
        return -2.0 * np.asarray(a) + np.asarray([4.0])

    def evaluate(self, a, operating):
        return SimpleNamespace(
            vector_field=self.vector_field(a, operating),
            vector_field_jacobian=np.array([[-2.0]]),
        )


def test_current_trainer_uses_only_fixed_network_and_reaches_simple_exact_solution():
    config = ResearchTrainingConfig(
        initial_lower=(-0.2,), initial_upper=(0.2,),
        operating_lower=(), operating_upper=(),
        time_horizon=2.0, residual_tolerance=2e-7,
        sample_count=8, validation_count=8,
        max_network_depth=1, max_channels_per_mode=1,
        max_quadratic_rank=1, max_cross_rank=1, max_state_rank=1,
        max_iterations=12, max_validation_epochs=1, prune_rounds=0,
    )
    network = FixedAnalyticResponseNetwork(
        [2.0], [], input_center=[0.0], input_scale=[0.2],
        depth=1, channels_per_mode=1, quadratic_rank=1, cross_rank=1, state_rank=1,
    )
    trained, report = train_research_network(_LinearField(), config, network=network)
    assert isinstance(trained, FixedAnalyticResponseNetwork)
    assert report.numerical_tolerance_met
    assert report.maximum_training_residual <= config.residual_tolerance
    assert report.maximum_validation_residual <= config.residual_tolerance


def test_trainer_rejects_non_fixed_checkpoint():
    config = ResearchTrainingConfig(
        initial_lower=(0.0,), initial_upper=(0.0,), operating_lower=(), operating_upper=(),
        time_horizon=1.0, residual_tolerance=1e-5, sample_count=1, validation_count=1,
    )
    try:
        train_research_network(_LinearField(), config, network=object())
    except TypeError as exc:
        assert "FixedAnalyticResponseNetwork" in str(exc)
    else:
        raise AssertionError("non-fixed checkpoint was accepted")
