from types import SimpleNamespace

import numpy as np
import pytest

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from sdfmpneo.rollout import rollout_fixed_network, solve_physical_steady_state


def _exact_linear_flow():
    network = FixedAnalyticResponseNetwork(
        [2.0], [], max_response_time=1.0,
        depth=1, channels_per_mode=1, quadratic_rank=1, cross_rank=1, state_rank=1,
    )
    theta = np.zeros(network.parameter_count)
    theta[network._indices("bias_0")[0]] = 4.0
    theta[network._indices("channel_gate")[0, 0]] = 1.0
    return network.with_parameters(theta)


def test_segmented_rollout_composes_exact_finite_horizon_flow():
    network = _exact_linear_flow()
    result = rollout_fixed_network(
        network, 2.5, a0=[0.0], operating=[],
        state_lower=[-0.1], state_upper=[2.1],
    )
    assert result.segment_count == 3
    assert result.segment_durations == (1.0, 1.0, 0.5)
    expected = 2.0 * (1.0 - np.exp(-5.0))
    assert np.allclose(result.state, [expected], rtol=1e-12, atol=1e-12)


def test_rollout_rejects_restart_that_leaves_trained_state_box():
    network = _exact_linear_flow()
    with pytest.raises(ValueError, match="left the trained restart-state box"):
        rollout_fixed_network(
            network, 2.0, a0=[0.0], operating=[],
            state_lower=[-0.1], state_upper=[0.1],
        )


class _LinearField:
    def evaluate(self, state, operating):
        state = np.asarray(state, float)
        return SimpleNamespace(
            vector_field=-2.0 * state + np.array([4.0]),
            vector_field_jacobian=np.array([[-2.0]]),
        )


def test_physical_steady_state_is_solved_without_network_infinity():
    result = solve_physical_steady_state(
        _LinearField(), [], initial_guess=[0.0], tolerance=1e-12
    )
    assert result.converged
    assert result.residual_norm <= 1e-12
    assert np.allclose(result.state, [2.0])
