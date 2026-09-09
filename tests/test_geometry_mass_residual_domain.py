from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.certification.em_domain import ParameterBox
from sdfmpneo.certification.geometry_mass_residual_domain import (
    GeometryJouleDerivativeBounds,
    GeometryThermalOperatorBounds,
    bound_geometry_mass_residual_on_box,
    certify_geometry_mass_residual_domain,
    compose_geometry_mass_residual_physical_proof,
)


class _ZeroHeatEM:
    def heat_source_for_rhs(self, state, rhs):
        state = np.asarray(state, dtype=float)
        return np.zeros_like(state)


class _RHS:
    def evaluate(self, operating):
        return np.asarray(operating, dtype=complex)


class _ExactMassModel:
    """One-mode family whose analytic decay solves every geometry exactly."""

    def __init__(self):
        self.graph = ParametricAnalyticEvolutionGraph([0.4], ["g", "current_0"])
        self.geometry_names = ("g",)
        self.lower = np.array([-1.0])
        self.upper = np.array([1.0])
        self.current_matrix = np.zeros((1, 1), dtype=complex)

    def normalize(self, geometry):
        geometry = np.asarray(geometry, dtype=float)
        return geometry.copy()

    def context(self, geometry):
        g = float(np.asarray(geometry, dtype=float)[0])
        mass = 2.0 + 0.2 * g
        stiffness = 0.4 * mass
        return SimpleNamespace(
            M=np.array([[mass]], dtype=float),
            K=np.array([[stiffness]], dtype=float),
            rhs=_RHS(),
            em=_ZeroHeatEM(),
        )


def _proof(*, certified=True):
    thermal = GeometryThermalOperatorBounds(
        minimum_mass_eigenvalue=1.8,
        mass_matrix_norm_bound=2.2,
        stiffness_matrix_norm_bound=0.88,
        mass_geometry_derivative_norm_bounds=np.array([0.2]),
        stiffness_geometry_derivative_norm_bounds=np.array([0.08]),
        certified=certified,
        provenance="exact affine one-mode mass/stiffness family" if certified else "test evidence only",
    )
    joule = GeometryJouleDerivativeBounds(
        state_jacobian_norm_bound=0.0,
        geometry_jacobian_column_norm_bounds=np.array([0.0]),
        operating_jacobian_column_norm_bounds=np.array([0.0]),
        certified=certified,
        provenance="identically zero heat source" if certified else "test evidence only",
    )
    return compose_geometry_mass_residual_physical_proof(
        thermal,
        joule,
        provenance="closed-form manufactured proof" if certified else "",
    )


def test_mass_residual_box_bound_contains_exact_family_and_uses_physical_geometry_coordinates():
    model = _ExactMassModel()
    # Public coordinates are [a0, physical geometry, current, time].
    box = ParameterBox(
        np.array([0.09, -0.2, -0.1, 0.4]),
        np.array([0.11, 0.2, 0.1, 0.6]),
    )
    bound = bound_geometry_mass_residual_on_box(model, box, _proof())
    assert bound.certified_physics
    assert bound.center_mass_residual_norm < 1e-14
    assert bound.center_vector_residual_norm < 1e-14
    assert bound.mass_residual_upper_bound >= bound.center_mass_residual_norm
    assert bound.vector_residual_upper_bound >= bound.center_vector_residual_norm
    assert bound.directional_contributions.shape == (4,)
    assert np.all(bound.directional_contributions >= 0.0)

    # The manufactured family has exactly zero residual everywhere, so the
    # rigorous upper bound must dominate direct interior evaluations.
    for a0, g, current, time in (
        (0.091, -0.19, -0.09, 0.41),
        (0.1, 0.0, 0.0, 0.5),
        (0.109, 0.19, 0.09, 0.59),
    ):
        static = np.array([g, current])
        state, derivative = model.graph.evaluate(time, a0=np.array([a0]), operating=static)
        context = model.context(np.array([g]))
        heat = context.em.heat_source_for_rhs(state, context.rhs.evaluate([current]))
        residual = context.M @ derivative + context.K @ state - heat
        assert np.linalg.norm(residual) <= bound.mass_residual_upper_bound


def test_domain_certificate_is_fail_closed_when_physics_is_not_certified():
    model = _ExactMassModel()

    def evidence_only(_box):
        return _proof(certified=False)

    report = certify_geometry_mass_residual_domain(
        model,
        initial_lower=np.array([0.0999]),
        initial_upper=np.array([0.1001]),
        geometry_lower=np.array([-0.001]),
        geometry_upper=np.array([0.001]),
        operating_lower=np.array([-0.001]),
        operating_upper=np.array([0.001]),
        time_lower=0.499,
        time_upper=0.501,
        tolerance=1e-2,
        work_budget=1,
        physical_proof_factory=evidence_only,
    )
    assert report.status == "indeterminate"
    assert not report.certified
    assert report.unresolved_boxes > 0


def test_domain_certificate_can_close_a_small_continuous_box_without_sampling():
    model = _ExactMassModel()

    report = certify_geometry_mass_residual_domain(
        model,
        initial_lower=np.array([0.0999]),
        initial_upper=np.array([0.1001]),
        geometry_lower=np.array([-0.001]),
        geometry_upper=np.array([0.001]),
        operating_lower=np.array([-0.001]),
        operating_upper=np.array([0.001]),
        time_lower=0.499,
        time_upper=0.501,
        tolerance=1e-2,
        work_budget=8,
        physical_proof_factory=lambda _box: _proof(certified=True),
    )
    assert report.status == "certified"
    assert report.certified
    assert report.unresolved_boxes == 0
    assert report.maximum_vector_residual_bound <= report.tolerance
    assert report.finite_time_only


def test_infinite_time_is_an_explicit_separate_obligation():
    model = _ExactMassModel()
    try:
        certify_geometry_mass_residual_domain(
            model,
            initial_lower=np.array([0.1]),
            initial_upper=np.array([0.1]),
            geometry_lower=np.array([0.0]),
            geometry_upper=np.array([0.0]),
            operating_lower=np.array([0.0]),
            operating_upper=np.array([0.0]),
            time_lower=0.0,
            time_upper=np.inf,
            tolerance=1e-5,
            work_budget=1,
            physical_proof_factory=lambda _box: _proof(),
        )
    except ValueError as exc:
        assert "finite time" in str(exc)
    else:
        raise AssertionError("infinite time must not be silently claimed by the finite-time certificate")
