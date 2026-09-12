from types import SimpleNamespace

import numpy as np

from sdfmpneo.training.research_helpers import _gn_apply_field_jacobian
from sdfmpneo.training.research_multilayer import predicted_layer_metrics


class _QuadraticHeatEM:
    def __init__(self, n):
        self.problem = SimpleNamespace(n_thermal=n)

    def heat_source_for_rhs(self, state, rhs):
        a = np.asarray(state, dtype=float)
        return 0.1 * a * a


class _QuadraticField:
    def __init__(self, lambdas):
        self.thermal_model = SimpleNamespace(lambdas=np.asarray(lambdas, dtype=float))
        self.em_model = _QuadraticHeatEM(len(lambdas))
        self.rhs_map = SimpleNamespace()
        self.thermal_forcing = np.zeros(len(lambdas))

    def rhs(self, operating):
        return np.ones(1, dtype=complex)


def test_low_rank_feedback_action_recovers_rank_one_state_sensitivity():
    lambdas = np.array([0.5, 0.8, 1.1, 1.4])
    field = _QuadraticField(lambdas)
    state = np.array([0.2, -0.3, 0.4, 0.1])
    direction = np.array([0.3, -0.2, 0.4, 0.1])
    direction /= np.linalg.norm(direction)
    weights = np.array([1.0, -0.5, 0.25])
    Ja = direction[:, None] * weights[None, :]

    observed = _gn_apply_field_jacobian(
        field,
        state,
        np.array([0.0]),
        Ja,
        feedback_rank=1,
    )
    exact = (-np.diag(lambdas) + np.diag(0.2 * state)) @ Ja
    np.testing.assert_allclose(observed, exact, rtol=2e-5, atol=2e-7)


def test_predicted_layer_metrics_respect_actual_backtrack_factor():
    record = SimpleNamespace(
        residual=np.array([1.0, -0.5]),
        parameter_jacobian=np.array([[1.0, 0.2], [0.1, 0.8]]),
    )
    ids = np.array([2, 5])
    delta = np.zeros(7)
    delta[ids] = np.array([-0.4, 0.1])
    full = predicted_layer_metrics([record], np.ones(1), delta, ids, 1.0)
    half = predicted_layer_metrics([record], np.ones(1), delta, ids, 0.5)
    expected_full = np.linalg.norm(record.residual + record.parameter_jacobian @ delta[ids])
    expected_half = np.linalg.norm(record.residual + 0.5 * record.parameter_jacobian @ delta[ids])
    assert np.isclose(full[0], expected_full)
    assert np.isclose(half[0], expected_half)
    assert full[0] != half[0]
