import numpy as np

from sdfmpneo.thermal import ThermalSpectralModel


def test_mass_orthonormal_thermal_basis():
    M = np.diag([2.0, 1.0, 1.5])
    K = np.array([[4.0, -1.0, 0.0], [-1.0, 3.0, -0.5], [0.0, -0.5, 2.0]])
    model = ThermalSpectralModel.build(M, K)
    Mr, Kr = model.reduced_matrices()
    assert np.allclose(Mr, np.eye(3), atol=1e-12)
    assert np.allclose(Kr, np.diag(model.lambdas), atol=1e-12)
