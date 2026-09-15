import numpy as np
import scipy.sparse as sp

import sdfmpneo.unified_self_correction as correction
import sdfmpneo.unified_stable_localized_self as stable
from sdfmpneo.unified_compensated_field import CompensatedComplexField


class _Gradient:
    pass


class _Local:
    omega = 2.0
    n_edges = 2
    edge_cell_hodge = sp.eye(2, format="csr")


def _install_projection(monkeypatch, longitudinal):
    monkeypatch.setattr(
        stable,
        "build_gradient_block",
        lambda local, context, check_topology=True: _Gradient(),
    )
    monkeypatch.setattr(
        stable,
        "refined_gradient_projection",
        lambda local, gradient, rhs, relative_tolerance, maximum_refinements: (
            longitudinal,
            {
                "initial_relative_residual": 1e-10,
                "relative_residual": 1e-15,
                "refinements": 1,
                "edge_low_relative_norm": 0.0,
            },
        ),
    )


def test_localized_self_response_removes_only_pure_longitudinal_energy(monkeypatch):
    root2 = np.sqrt(2.0)
    g = np.array([1.0, 1.0]) / root2
    h = np.array([1.0, -1.0]) / root2
    longitudinal = 1j * g
    transverse = 0.4j * h
    field = longitudinal + transverse

    A = sp.eye(2, format="csr", dtype=complex)
    rhs = field.copy()
    source = np.real(1j * rhs / _Local.omega)
    assert np.allclose(rhs, -1j * _Local.omega * source)
    _install_projection(monkeypatch, longitudinal)

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

    assert np.allclose(result["longitudinal_z"], complex(-source @ longitudinal))
    assert np.allclose(result["localized_z"], complex(-source @ transverse))
    assert np.isclose(result["localized_d_vol"], edge_loss @ expected_refinable_abs2)
    assert np.isclose(result["localized_d_out"], outward @ expected_refinable_abs2)
    assert not np.isclose(result["localized_d_vol"], edge_loss @ pure_transverse_abs2)

    expected_q = 0.5 * expected_refinable_abs2
    assert np.allclose(result["localized_modal_h"], 2.0 * expected_q)
    assert result["localized_contraction"] == "certified_gradient_compensated_transverse_v2"
    assert result["localized_gradient_projection_relative_residual"] == 1e-15
    assert result["localized_gradient_projection_refinements"] == 1


def test_localized_impedance_survives_extreme_longitudinal_cancellation(monkeypatch):
    longitudinal = np.array([1.0e20j, -2.0e20j], dtype=complex)
    transverse = np.array([3.0j, -5.0j], dtype=complex)
    field = CompensatedComplexField(longitudinal.copy(), transverse.copy())
    source = np.array([1.0, 0.25])
    rhs = np.zeros(2, dtype=complex)
    _install_projection(monkeypatch, longitudinal)

    result = correction._localized_self_response(
        _Local(),
        object(),
        sp.eye(2, format="csr", dtype=complex),
        rhs,
        field,
        source,
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
    )

    expected = complex(-source @ transverse)
    assert expected != 0.0
    assert result["localized_z"] == expected
    assert np.isfinite(result["transverse_field_relative_norm"])
    assert result["localized_contraction"] == "certified_gradient_compensated_transverse_v2"
