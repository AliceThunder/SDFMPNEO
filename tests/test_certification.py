import numpy as np

from sdfmpneo.certification import contraction_certificate, state_error_certificate


def test_contraction_and_error_bound():
    lam = np.array([2.0, 3.0])
    J = np.array([[0.2, 0.1], [0.1, 0.3]])
    cert = contraction_certificate(lam, J)
    assert cert.kappa > 0
    err = state_error_certificate(0.01, 0.02, 0.03, cert.kappa)
    assert np.isclose(err.state_error_bound, 0.06 / cert.kappa)
