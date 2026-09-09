import numpy as np

from sdfmpneo import ResearchTrainingConfig
from sdfmpneo.training import adaptive_runtime


def _high_dimensional_config():
    return ResearchTrainingConfig(
        initial_lower=(-0.1, -0.1),
        initial_upper=(0.1, 0.1),
        operating_lower=(-1.0,) * 10 + (0.0, 0.0),
        operating_upper=(1.0,) * 10 + (10.0, 10.0),
        time_horizon=1e5,
        residual_tolerance=1e-5,
        sample_count=8,
        validation_count=8,
        max_nodes=16,
        max_degree=3,
        time_sampling="mixed_log",
        time_min=1e-6,
        include_steady_state=True,
        max_parent_responses=1,
        max_realization_dimension=64,
    )


def test_high_dimensional_policy_doubles_qmc_and_adds_axis_anchors():
    config = _high_dimensional_config()
    training = config.points()
    validation = config.points(validation=True)

    # Expanded QMC: 16 finite + 3 historical special points + 16 steady.
    # Axis coverage: 14 parameters * 2 bounds * {t=0, t=inf} = 56.
    assert training.shape == (91, 15)
    # Validation remains independent QMC: 16 finite + 16 steady, no training anchors.
    assert validation.shape == (32, 15)

    lower = np.asarray(config.initial_lower + config.operating_lower, dtype=float)
    upper = np.asarray(config.initial_upper + config.operating_upper, dtype=float)
    centre = 0.5 * (lower + upper)
    for parameter in range(14):
        for value in (lower[parameter], upper[parameter]):
            target = centre.copy()
            target[parameter] = value
            assert np.any(np.all(training[:, :-1] == target, axis=1) & (training[:, -1] == 0.0))
            assert np.any(np.all(training[:, :-1] == target, axis=1) & np.isposinf(training[:, -1]))


def test_low_dimensional_sampling_semantics_are_unchanged():
    config = ResearchTrainingConfig(
        (0.0,), (1.0,), (0.0,), (1.0,), 100.0, 1e-5,
        sample_count=8, validation_count=8, max_nodes=8,
        time_sampling="mixed_log", include_steady_state=True,
    )
    training = config.points()
    validation = config.points(validation=True)
    assert training.shape == (19, 3)  # 8 + centre/lower/upper + 8 steady
    assert validation.shape == (16, 3)


def test_high_dimensional_coverage_changes_continuation_signature():
    high = _high_dimensional_config()
    low = ResearchTrainingConfig(
        (0.0,), (1.0,), (0.0,), (1.0,), 100.0, 1e-5,
        sample_count=8, validation_count=8, max_nodes=8,
    )
    high_signature = adaptive_runtime._config_signature(high)
    low_signature = adaptive_runtime._config_signature(low)
    assert isinstance(high_signature, str) and len(high_signature) == 64
    assert isinstance(low_signature, str) and len(low_signature) == 64
    assert high_signature != low_signature
