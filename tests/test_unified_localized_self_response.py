import numpy as np
import scipy.sparse as sp

import sdfmpneo.unified_self_correction as correction


class _Gradient:
    def __init__(self, longitudinal):
        self.longitudinal = np.asarray(longitudinal, complex)

    def solve(self, rhs):
        del rhs
        return self.longitudinal.copy()


class _Local:
    omega = 2.0
    n_edges = 2
    edge_cell_hodge = sp.eye(2, format="csr")


def test_localized_self_response_removes_only_pure_longitudinal_energy(monkeypatch):
    root2 = np.sqrt(2.0)
    g = np.array([1.0, 1.0]) / root2
    h = np.array([1.0, -1.0]) / root2
    longitudinal = 1j * g
    transverse = 0.4j * h
    field = longitudinal + transverse

    # A=I and rhs=field make the compatible gradient Galerkin projection equal
    # longitudinal.  With rhs=-i*omega*S this also gives a real open-port source.
    A = sp.eye(2, format="csr", dtype=complex)
    rhs = field.copy()
    source = np.real(1j * rhs / _Local.omega)
    assert np.allclose(rhs, -1j * _Local.omega * source)

    monkeypatch.setattr(
        correction,
        "build_gradient_block",
        lambda local, context, check_topology=True: _Gradient(longitudinal),
    )

    # Nonuniform loss weights deliberately make the L/T cross term nonzero.
    edge_loss = np.array([1.0, 2.0])
    outward = np.array([0.2, 0.5])
    sigma = np.ones(2)
    local_phi = np.eye(2)

    result = correction._localized_self_response(
        _Local(),
        object(),
        A,
        rhs,
        field,
        source,
        sigma,
        edge_loss,
        outward,
        local_phi=local_phi,
    )

    full_abs2 = np.abs(field) ** 2
    long_abs2 = np.abs(longitudinal) ** 2
    expected_refinable_abs2 = full_abs2 - long_abs2
    pure_transverse_abs2 = np.abs(transverse) ** 2

    expected_full_z = complex(-source @ field)
    grad_energy = complex(np.vdot(longitudinal, A @ longitudinal))
    expected_longitudinal_z = complex(
        np.imag(grad_energy) / _Local.omega,
        np.real(grad_energy) / _Local.omega,
    )

    assert np.allclose(result["longitudinal_z"], expected_longitudinal_z)
    assert np.allclose(
        result["localized_z"],
        expected_full_z - expected_longitudinal_z,
    )
    assert np.isclose(
        result["localized_d_vol"],
        edge_loss @ expected_refinable_abs2,
    )
    assert np.isclose(
        result["localized_d_out"],
        outward @ expected_refinable_abs2,
    )
    # The whole point of v2 is to remove only pure longitudinal self energy.
    # Because the weights are nonuniform, the remaining cross term is nonzero.
    assert not np.isclose(
        result["localized_d_vol"],
        edge_loss @ pure_transverse_abs2,
    )

    expected_q = 0.5 * expected_refinable_abs2
    assert np.allclose(result["localized_modal_h"], 2.0 * expected_q)
