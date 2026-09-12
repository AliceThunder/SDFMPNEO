import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.sampling import (
    StateDomainInsufficientError,
    hybrid_reachable_state_geometry_samples,
)


def test_hybrid_sampler_mixes_box_and_reachable_decay_states():
    def vector_field(state, geometry, operating):
        del geometry, operating
        return -np.asarray(state, dtype=float)

    result = hybrid_reachable_state_geometry_samples(
        vector_field,
        state_lower=np.array([-1.0]),
        state_upper=np.array([1.0]),
        geometry_lower=np.array([-1.0]),
        geometry_upper=np.array([1.0]),
        operating_lower=np.array([0.0]),
        operating_upper=np.array([1.0]),
        n_samples=20,
        initial_lower=np.array([-0.5]),
        initial_upper=np.array([0.5]),
        box_fraction=0.25,
        trajectory_count=4,
        samples_per_trajectory=6,
        time_horizon=2.0,
        time_min=1e-3,
        seed=4,
    )
    assert result.states.shape == (20, 1)
    assert result.geometries.shape == (20, 1)
    assert result.report.box_sample_count == 5
    assert result.report.reachable_sample_count == 15
    assert result.report.trajectory_count == 4
    assert np.all(result.states >= -1.0)
    assert np.all(result.states <= 1.0)
    assert np.all(result.geometries >= -1.0)
    assert np.all(result.geometries <= 1.0)


def test_hybrid_sampler_refuses_to_hide_an_insufficient_state_box():
    def vector_field(state, geometry, operating):
        del geometry, operating
        return np.ones_like(state)

    with pytest.raises(StateDomainInsufficientError) as caught:
        hybrid_reachable_state_geometry_samples(
            vector_field,
            state_lower=np.array([0.0]),
            state_upper=np.array([0.6]),
            geometry_lower=np.array([-1.0]),
            geometry_upper=np.array([1.0]),
            operating_lower=np.array([0.0]),
            operating_upper=np.array([1.0]),
            n_samples=12,
            initial_lower=np.array([0.4]),
            initial_upper=np.array([0.5]),
            box_fraction=0.25,
            trajectory_count=2,
            samples_per_trajectory=4,
            time_horizon=1.0,
            time_min=1e-3,
            seed=2,
        )
    assert caught.value.observed_upper[0] > 0.6
