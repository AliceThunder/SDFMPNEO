from types import SimpleNamespace

import numpy as np

from sdfmpneo.certification.geometry_dynamics import (
    certify_geometry_mass_residual_equivalence,
    local_geometry_dynamics_diagnostic,
)


class _EM:
    def heat_source_and_jacobian_for_rhs(self, state, rhs):
        state = np.asarray(state, float)
        return np.zeros_like(state), np.zeros((state.size, state.size))


class _RHS:
    def evaluate(self, operating):
        return np.asarray(operating, float)


class _Model:
    lower = np.array([-1.0])
    upper = np.array([1.0])
    certificate = SimpleNamespace(
        p1_mass_ratio=(0.8, 1.2),
        certified_nondegenerate=True,
    )
    thermal_model = SimpleNamespace(lambdas=np.array([2.0, 3.0]))
    current_matrix = np.zeros((2, 2))

    def geometry_vector(self, geometry):
        g = np.asarray(geometry, float)
        if g.shape != (1,):
            raise ValueError
        return g

    def context(self, geometry):
        return SimpleNamespace(
            M=np.diag([2.0, 3.0]),
            K=np.diag([4.0, 9.0]),
            em=_EM(),
            rhs=_RHS(),
        )


def test_continuous_geometry_mass_residual_equivalence_uses_chart_form_bounds():
    cert = certify_geometry_mass_residual_equivalence(_Model())
    assert cert.continuous_geometry_box
    assert np.isclose(cert.reference_minimum_mass_eigenvalue, 2.0)
    assert np.isclose(cert.reference_maximum_mass_eigenvalue, 3.0)
    assert cert.minimum_mass_eigenvalue <= 1.6
    assert cert.maximum_mass_eigenvalue >= 3.6


def test_local_mass_residual_and_contractivity_are_consistent():
    model = _Model()
    state = np.array([0.1, -0.2])
    derivative = np.array([0.5, -0.25])
    diagnostic = local_geometry_dynamics_diagnostic(
        model,
        geometry=[0.0],
        a=state,
        operating=[0.0, 0.0],
        da=derivative,
    )

    # F = M^-1(-K a) = [-0.2, 0.6].
    expected_vector = np.array([0.7, -0.85])
    expected_mass = np.diag([2.0, 3.0]) @ expected_vector
    assert np.allclose(diagnostic.vector_residual, expected_vector)
    assert np.allclose(diagnostic.mass_residual, expected_mass)
    assert diagnostic.vector_norm_lower_from_mass <= diagnostic.vector_residual_norm
    assert diagnostic.vector_residual_norm <= diagnostic.vector_norm_upper_from_mass

    # J = diag(-2,-3); in the M norm its logarithmic abscissa is -2.
    assert np.isclose(diagnostic.local_mass_contractivity_margin, 2.0)
    assert diagnostic.locally_contractive
