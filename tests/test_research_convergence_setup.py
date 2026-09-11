from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from sdfmpneo.training.research_config import (
    ResearchTrainingConfig,
    _default_capacity,
    make_fixed_network,
)
from sdfmpneo.training.research_linearization import _balanced_physics_subset
from sdfmpneo.training.research_trainer import (
    _accept_progressive,
    _accept_progressive_reason,
    _initialize_center_source,
    train_research_network,
)


class _LinearField:
    def __init__(self):
        self.thermal_model = SimpleNamespace(lambdas=np.array([0.5, 1.25]))

    def vector_field(self, state, operating):
        a = np.asarray(state, dtype=float)
        u = np.asarray(operating, dtype=float)
        source = np.array([2.0 + 0.25 * u[0], 3.0 - 0.5 * u[0]])
        return -self.thermal_model.lambdas * a + source


def _config():
    return ResearchTrainingConfig(
        initial_lower=(-0.2, -0.1),
        initial_upper=(0.2, 0.1),
        operating_lower=(-1.0,),
        operating_upper=(1.0,),
        max_response_time=10.0,
        residual_tolerance=1e-5,
        sample_count=8,
        validation_count=8,
        max_network_depth=1,
    )


def test_high_rank_capacity_scales_with_static_dimension():
    depth, channels, linear, hidden, quadratic, square, cross, state = _default_capacity(198, 12)
    assert depth == 1
    assert channels == 1
    assert linear == 6
    assert hidden == 1
    assert quadratic == 14
    assert square == 6
    assert cross == 1
    assert state == 1


def test_default_gn_budget_is_twelve_points():
    assert _config().jacobian_point_budget == 12


def test_center_source_initialization_matches_exact_t0_vector_field():
    field = _LinearField()
    config = _config()
    network = make_fixed_network(field, config, operating_names=("u",))
    initialized = _initialize_center_source(network, field, config)
    a0 = np.array([0.0, 0.0])
    operating = np.array([0.0])
    state, derivative = initialized.evaluate(0.0, a0=a0, operating=operating)
    np.testing.assert_allclose(state, a0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(
        derivative,
        field.vector_field(a0, operating),
        atol=1e-12,
        rtol=1e-12,
    )


def test_progressive_acceptance_allows_global_fit_far_from_tolerance():
    old = np.array([0.0400, 0.0300, 0.0280])
    new = np.array([0.0404, 0.0240, 0.0230])
    accepted, reason = _accept_progressive_reason(old, new, 1e-5, np.ones_like(old))
    assert accepted
    assert reason == "global_merit_reduced"
    assert _accept_progressive(old, new, 1e-5, np.ones_like(old))


def test_progressive_acceptance_is_strict_near_tolerance():
    old = np.array([1.50e-5, 1.20e-5])
    new = np.array([1.51e-5, 1.00e-5])
    accepted, reason = _accept_progressive_reason(old, new, 1e-5, np.ones_like(old))
    assert not accepted
    assert reason == "near_target_max_regressed"


def test_balanced_linearization_keeps_worst_and_adds_representatives():
    config = _config()
    points = np.array([
        [-0.2, -0.1, -1.0, 0.0],
        [0.2, 0.1, 1.0, 10.0],
        [-0.2, 0.1, 1.0, 1e-6],
        [0.2, -0.1, -1.0, 1e-3],
        [0.0, 0.0, 0.0, 5.0],
        [-0.1, 0.05, -0.5, 0.1],
        [0.1, -0.05, 0.5, 8.0],
        [0.0, 0.1, -1.0, 0.01],
    ])
    norms = np.array([8.0, 7.0, 6.0, 1.0, 1.1, 1.2, 1.3, 1.4])
    subset = _balanced_physics_subset(points, norms, 6, config)
    assert subset.shape == (6, 4)
    np.testing.assert_allclose(subset[:3], points[:3])
    assert len({tuple(row) for row in subset}) == 6


def test_stalled_training_skips_independent_validation():
    field = _LinearField()
    config = replace(
        _config(),
        residual_tolerance=1e-12,
        max_iterations=0,
        semigroup_sample_count=0,
        semigroup_validation_count=0,
    )
    _, report = train_research_network(field, config, operating_names=("u",))
    assert report.status == "stalled"
    assert not report.validation_performed
    assert not report.numerical_tolerance_met
