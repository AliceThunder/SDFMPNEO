import numpy as np
import scipy.sparse as sp

from sdfmpneo.thermal import build_partial_thermal_spectrum


def test_partial_thermal_spectrum_uses_only_low_modes_and_explicit_omitted_bound():
    M = sp.eye(5, format="csr")
    K = sp.diags([1.0, 2.0, 4.0, 7.0, 11.0], format="csr")
    partial = build_partial_thermal_spectrum(
        M, K, rank=2, first_omitted_lambda_lower_bound=3.9
    )
    assert partial.model.rank == 2
    assert partial.model.full_dimension == 5
    assert np.allclose(partial.computed_eigenvalues, [1.0, 2.0], rtol=1e-10, atol=1e-12)
    cert = partial.tail_certificate(
        initial_field=np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        source_dual_bound=0.39,
        requested_state_tolerance=1.1,
    )
    assert cert.first_omitted_lambda == 3.9
    assert cert.steady_forcing_tail_bound == 0.1
    assert cert.certified


def test_partial_thermal_spectrum_rejects_unproved_omitted_lower_bound():
    M = sp.eye(4, format="csr")
    K = sp.diags([1.0, 2.0, 3.0, 4.0], format="csr")
    try:
        build_partial_thermal_spectrum(M, K, rank=2, first_omitted_lambda_lower_bound=0.0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("missing omitted-eigenvalue certificate was accepted")
