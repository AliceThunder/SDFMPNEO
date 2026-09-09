from __future__ import annotations

import numpy as np

from sdfmpneo.training.max_residual_runtime import (
    _MAX_STAGNATION_REL,
    hard_point_weights_from_norms,
    max_aligned_train_research_graph,
    max_first_accept,
    max_residual_refine_weights,
    relative_max_improvement,
    weighted_linear_system,
    weighted_objective_from_norms,
)


def test_hard_point_weights_only_boost_tolerance_violations():
    tol = 1.0e-5
    norms = np.array([0.0, 0.5e-5, 1.0e-5, 2.0e-5, 6.0e-5])
    weights = hard_point_weights_from_norms(norms, tol)
    assert np.all(weights[:3] == 1.0)
    assert weights[3] > 1.0
    assert weights[4] == 25.0
    assert np.all(np.diff(weights[2:]) >= 0.0)


def test_unit_weights_recover_the_unweighted_gauss_newton_rows():
    rng = np.random.default_rng(12)
    residual = rng.normal(size=(7, 2))
    jacobian = rng.normal(size=(7, 2, 5))
    rw, jw = weighted_linear_system(residual, jacobian, np.ones(7))
    assert np.array_equal(rw, residual)
    assert np.array_equal(jw, jacobian)


def test_max_first_accept_protects_already_passed_points():
    tol = 1.0e-5
    old = np.array([0.8e-5, 6.0e-5, 2.0e-5])
    weights = hard_point_weights_from_norms(old, tol)
    bad = np.array([1.1e-5, 4.0e-5, 1.8e-5])
    assert not max_first_accept(old, bad, tol, weights)

    good = np.array([0.9e-5, 4.0e-5, 1.8e-5])
    assert max_first_accept(old, good, tol, weights)


def test_max_first_accept_allows_secondary_weighted_progress_without_max_regression():
    tol = 1.0e-5
    old = np.array([5.0e-5, 4.0e-5, 0.9e-5])
    weights = hard_point_weights_from_norms(old, tol)
    new = np.array([5.0e-5, 3.0e-5, 0.9e-5])
    assert weighted_objective_from_norms(new, weights) < weighted_objective_from_norms(old, weights)
    assert max_first_accept(old, new, tol, weights)


def test_stagnation_threshold_is_tied_to_max_residual_improvement():
    tol = 1.0e-5
    old = 6.0e-5
    clearly_useful = old * (1.0 - 2.0 * _MAX_STAGNATION_REL)
    negligible = old * (1.0 - 0.25 * _MAX_STAGNATION_REL)
    assert relative_max_improvement(old, clearly_useful, tol) > _MAX_STAGNATION_REL
    assert relative_max_improvement(old, negligible, tol) < _MAX_STAGNATION_REL


def test_package_installation_routes_training_to_max_aligned_runtime():
    from sdfmpneo.training import research as training_research

    assert training_research._refine_weights is max_residual_refine_weights
    assert training_research.train_research_graph is max_aligned_train_research_graph
