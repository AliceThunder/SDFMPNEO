import numpy as np

from sdfmpneo.electrothermal import CertifiedElectroThermalVectorField
from sdfmpneo.thermal import ThermalSpectralModel


class _Problem:
    n_thermal = 2
    n_em = 1
    b = np.array([1.0 + 0.0j])


class _EM:
    problem = _Problem()

    def heat_source_and_jacobian_for_rhs(self, a, rhs):
        scale = float(np.abs(rhs[0]) ** 2)
        q = scale * np.array([1.0 + 0.2 * a[0], 0.5 + 0.1 * a[1]])
        J = scale * np.diag([0.2, 0.1])
        return q, J


def test_vector_field_and_jacobian_are_single_closed_electrothermal_callable():
    thermal = ThermalSpectralModel(
        M=np.eye(2), K=np.diag([2.0, 3.0]), Phi=np.eye(2), lambdas=np.array([2.0, 3.0])
    )
    field = CertifiedElectroThermalVectorField(thermal, _EM())
    a = np.array([0.3, -0.2])
    e = field.evaluate(a)
    expected_q = np.array([1.06, 0.48])
    assert np.allclose(e.heat_source, expected_q)
    assert np.allclose(e.vector_field, -thermal.lambdas * a + expected_q)
    assert np.allclose(e.vector_field_jacobian, np.diag([-1.8, -2.9]))
    assert np.allclose(field.residual(a, e.vector_field), 0.0)
    assert field.contraction_margin(a) > 0.0
