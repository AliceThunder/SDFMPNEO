from types import SimpleNamespace

import numpy as np

import sdfmpneo.training.research_trainer as trainer
from sdfmpneo.training.research_helpers import _gn_apply_field_jacobian


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


def test_trial_prescreen_caps_full_candidates(monkeypatch):
    record = SimpleNamespace(
        residual=np.array([1.0, -0.5]),
        parameter_jacobian=np.array([[1.0, 0.2], [0.1, 0.8]]),
    )
    linearized = SimpleNamespace(records=[record])

    def fake_direction(records, weights, damping, network, parameter_indices=None):
        return np.array([-0.4, 0.1]) / (1.0 + float(damping))

    monkeypatch.setattr(trainer, "_solve_direction", fake_direction)
    candidates = trainer._rank_trial_candidates(
        linearized,
        np.ones(1),
        1e-3,
        SimpleNamespace(),
        parameter_ids=None,
    )
    assert len(candidates) == trainer._MAX_EXACT_TRIALS_PER_ITERATION == 3
    assert all(np.isfinite(candidate[0][0]) for candidate in candidates)
