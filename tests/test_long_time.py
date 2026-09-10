import numpy as np

from sdfmpneo.training.research import ResearchTrainingConfig


def test_finite_horizon_sampling_keeps_early_and_endpoint_checks():
    config = ResearchTrainingConfig(
        initial_lower=(0.0,), initial_upper=(1.0,),
        operating_lower=(0.0,), operating_upper=(1.0,),
        max_response_time=100.0, residual_tolerance=1e-5,
        sample_count=64, validation_count=64,
        semigroup_sample_count=32, semigroup_validation_count=32,
        time_sampling="mixed_log", time_min=1e-6,
    )
    train = config.points()
    check1 = config.points(True)
    check2 = config.points(True, seed=3)
    assert np.any(train[:, -1] == 0.0)
    assert np.any(train[:, -1] == config.max_response_time)
    assert np.count_nonzero((train[:, -1] > 0) & (train[:, -1] < 0.01)) > 5
    assert np.all(np.isfinite(train[:, -1]))
    assert np.all(train[:, -1] <= config.max_response_time)
    assert not np.array_equal(check1, check2)


def test_semigroup_samples_are_finite_and_fit_inside_one_segment():
    config = ResearchTrainingConfig(
        initial_lower=(-1.0,), initial_upper=(1.0,),
        operating_lower=(), operating_upper=(),
        max_response_time=100.0, residual_tolerance=1e-5,
        sample_count=4, validation_count=4,
        semigroup_sample_count=64, semigroup_validation_count=64,
    )
    rows = config.semigroup_points()
    assert np.all(np.isfinite(rows))
    assert np.all(rows[:, -2] >= 0.0)
    assert np.all(rows[:, -1] >= 0.0)
    assert np.all(rows[:, -2] + rows[:, -1] <= config.max_response_time + 1e-12)
