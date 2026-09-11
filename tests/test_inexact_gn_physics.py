from types import SimpleNamespace

import numpy as np

from sdfmpneo.training.research_helpers import (
    _gn_field_jacobian,
    _physics_vector_field,
)


class _FallbackEM:
    def __init__(self, n):
        self.problem = SimpleNamespace(n_thermal=n)
        self.calls = 0

    def heat_source_for_rhs(self, state, rhs):
        self.calls += 1
        state = np.asarray(state, dtype=float)
        return 0.1 + 0.02 * state


class _Field:
    def __init__(self, n):
        self.thermal_model = SimpleNamespace(lambdas=np.linspace(0.5, 1.5, n))
        self.em_model = _FallbackEM(n)
        self.rhs_map = SimpleNamespace()
        self.thermal_forcing = np.zeros(n)

    def rhs(self, operating):
        return np.asarray([1.0 + float(np.sum(operating))], dtype=complex)

    def evaluate(self, state, operating):
        raise AssertionError("high-rank GN must not request the exact heat-source Jacobian")


def test_inexact_gn_uses_exact_residual_but_frozen_heat_jacobian():
    field = _Field(6)
    state = np.linspace(-0.2, 0.3, 6)
    operating = np.array([0.4])

    observed = _physics_vector_field(field, state, operating)
    expected_heat = 0.1 + 0.02 * state
    expected = -field.thermal_model.lambdas * state + expected_heat
    np.testing.assert_allclose(observed, expected)
    assert field.em_model.calls == 1

    jac = _gn_field_jacobian(field, state, operating)
    np.testing.assert_allclose(jac, -np.diag(field.thermal_model.lambdas))
