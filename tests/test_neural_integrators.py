from types import SimpleNamespace

import numpy as np
import scipy.linalg

from sdfmpneo.electrothermal_tensor.integrators import integrate_etd2, integrate_imex_euler
from sdfmpneo.electrothermal_tensor.validation import energy_logarithmic_norm
from sdfmpneo.electrothermal_tensor.vector_field import (
    FixedThermalOperatorFamily,
    NeuralElectroThermalVectorField,
)


class _ConstantHeatSurrogate:
    state_dimension = 2
    geometry_dimension = 0
    pod = SimpleNamespace(current_dimension=1)

    def heat_source_numpy(self, state, geometry, operating):
        u = float(np.asarray(operating)[0])
        return np.array([1.0 + 0.2 * u * u, 0.5 - 0.1 * u])


def test_generalized_etd2_is_exact_for_constant_source_with_nonidentity_mass():
    M = np.array([[2.0, 0.25], [0.25, 1.4]])
    K = np.array([[3.0, 0.2], [0.2, 1.8]])
    operators = FixedThermalOperatorFamily(M, K)
    field = NeuralElectroThermalVectorField(_ConstantHeatSurrogate(), operators)
    a0 = np.array([0.4, -0.2])
    u = np.array([0.7])
    t = 1.3
    q = field.heat_source(a0, np.empty(0), u)
    steady = np.linalg.solve(K, q)
    L = np.linalg.solve(M, K)
    exact = steady + scipy.linalg.expm(-L * t) @ (a0 - steady)
    result = integrate_etd2(
        field,
        t,
        initial_state=a0,
        geometry=np.empty(0),
        operating=u,
        max_step=t,
    )
    np.testing.assert_allclose(result.state, exact, rtol=2e-12, atol=2e-12)
    assert result.steps == 1


def test_imex_converges_when_step_is_refined():
    M = np.array([[1.6, 0.1], [0.1, 1.2]])
    K = np.array([[2.2, 0.15], [0.15, 1.7]])
    field = NeuralElectroThermalVectorField(
        _ConstantHeatSurrogate(),
        FixedThermalOperatorFamily(M, K),
    )
    a0 = np.array([0.3, 0.1])
    u = np.array([0.4])
    t = 2.0
    reference = integrate_etd2(
        field, t, initial_state=a0, geometry=np.empty(0), operating=u, max_step=t
    ).state
    coarse = integrate_imex_euler(
        field, t, initial_state=a0, geometry=np.empty(0), operating=u, max_step=0.25
    ).state
    fine = integrate_imex_euler(
        field, t, initial_state=a0, geometry=np.empty(0), operating=u, max_step=0.125
    ).state
    assert np.linalg.norm(fine - reference) < np.linalg.norm(coarse - reference)


def test_energy_logarithmic_norm_recovers_dissipative_linear_system():
    M = np.array([[2.0, 0.2], [0.2, 1.0]])
    K = np.array([[3.0, 0.1], [0.1, 2.0]])
    J = -np.linalg.solve(M, K)
    mu = energy_logarithmic_norm(J, M)
    expected = -np.min(scipy.linalg.eigh(K, M, eigvals_only=True))
    np.testing.assert_allclose(mu, expected, rtol=2e-13, atol=2e-13)
    assert mu < 0.0
