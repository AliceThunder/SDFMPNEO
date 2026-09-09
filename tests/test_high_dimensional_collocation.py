from types import SimpleNamespace

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


def test_high_dimensional_policy_keeps_training_compact_and_broadens_search():
    config = _high_dimensional_config()
    training = config.points()
    search = config.points(validation=True)

    # Persistent set: historical 8 finite + 3 special + 8 steady = 19,
    # plus 14 parameters * 2 bounds * {t=0, t=inf} = 56 anchors.
    assert training.shape == (75, 15)
    # Search pool: 8x validation QMC => 64 finite + 64 steady = 128,
    # plus 14 * 2 bounds * {0, time_min, geometric-middle, horizon, inf}=140.
    assert search.shape == (268, 15)

    lower = np.asarray(config.initial_lower + config.operating_lower, dtype=float)
    upper = np.asarray(config.initial_upper + config.operating_upper, dtype=float)
    centre = 0.5 * (lower + upper)
    for parameter in range(14):
        for value in (lower[parameter], upper[parameter]):
            target = centre.copy()
            target[parameter] = value
            assert np.any(np.all(training[:, :-1] == target, axis=1) & (training[:, -1] == 0.0))
            assert np.any(np.all(training[:, :-1] == target, axis=1) & np.isposinf(training[:, -1]))
            assert np.any(np.all(search[:, :-1] == target, axis=1) & (search[:, -1] == config.time_horizon))


def test_high_dimensional_exchange_keeps_only_four_worst_failed_points():
    config = _high_dimensional_config()
    points = config.points(validation=True)[:10]
    norms = np.array([2e-6, 8e-5, 3e-5, 4e-6, 1.2e-4, 7e-5, 9e-6, 5e-5, 2e-5, 1e-7])
    records = [SimpleNamespace(residual=np.array([value, 0.0])) for value in norms]

    selected, guards, measured = adaptive_runtime._select_adaptive_validation_points(
        records, points, tolerance=1e-5
    )
    expected_indices = [4, 1, 5, 7]  # descending residual among failed points
    assert np.array_equal(selected, points[expected_indices])
    assert guards.shape == (0, points.shape[1])
    assert np.array_equal(measured, norms)


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

    points = validation[:4]
    norms = [2e-5, 5e-6, 3e-5, 1e-6]
    records = [SimpleNamespace(residual=np.array([value])) for value in norms]
    failed, passed, _ = adaptive_runtime._select_adaptive_validation_points(
        records, points, tolerance=1e-5
    )
    assert np.array_equal(failed, points[[0, 2]])
    assert np.array_equal(passed, points[[1, 3]])


def test_high_dimensional_exchange_changes_continuation_signature():
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
