"""Small but physical checks for the independent D2 edge-element solver.

These verify a genuine 3D A-phi assembly, NOT TEAM-7 certification.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sdfmpneo_bfzi.direct3d_aphi import (
    box_tetra_mesh, solve_aphi, closed_polyline_potential,
)

_B0 = 1e-3


def _uniform_B_z(p):
    return np.column_stack((-_B0*p[:, 1]/2, _B0*p[:, 0]/2,
                            np.zeros(len(p))))


def _mesh(n, sigma):
    return box_tetra_mesh([[-.2, .2]]*3, [n]*3,
                          lambda p: sigma if max(abs(p)) < .075 else 0.)


def test_3d_curl_charge_and_joule_power():
    result = solve_aphi(_mesh(3, 5.), 100000., [_uniform_B_z])
    assert min(result.n_edge_dofs, result.n_potential_dofs,
               result.n_gauge_dofs) > 0
    assert result.power_matrix[0, 0].real > 0
    assert_allclose(result.power_matrix, result.power_from_fields,
                    rtol=5e-11, atol=1e-11)
    assert np.all(np.isfinite(result.current_at_cells))


def test_3d_reciprocity_and_arbitrary_phase():
    def second_source(p):
        return np.column_stack((np.zeros(len(p)), -_B0*p[:, 2]/2,
                                _B0*p[:, 1]/2))
    result = solve_aphi(_mesh(3, 5.), 100000.,
                        [_uniform_B_z, second_source])
    assert_allclose(result.delta_z, result.delta_z.T,
                    rtol=3e-11, atol=3e-10)
    assert_allclose(result.power_matrix, result.power_from_fields,
                    rtol=3e-11, atol=1e-11)
    for currents in ((1, 0), (0, 1), (1, 1j), (1, -1), (1, 1)):
        i = np.asarray(currents, complex)
        assert .5 * np.vdot(i, result.power_matrix @ i).real >= -1e-12


def test_3d_vacuum_limit():
    mesh = box_tetra_mesh([[-.2, .2]]*3, [2, 2, 2], lambda p: 0.)
    result = solve_aphi(mesh, 100000., [_uniform_B_z])
    assert_allclose(result.delta_z, 0., atol=1e-12)
    assert_allclose(result.power_from_fields, 0., atol=1e-12)


def test_true_closed_polygonal_port_source():
    square = np.array([[-.05, -.05, 0], [.05, -.05, 0],
                       [.05, .05, 0], [-.05, .05, 0],
                       [-.05, -.05, 0]])
    source = closed_polyline_potential(square)
    assert np.isfinite(source(np.array([[0., 0., .1], [.1, .1, .1]]))).all()
    with pytest.raises(ValueError):
        closed_polyline_potential(square[:-1])
    mesh = box_tetra_mesh([[-.2, .2]]*3, [3, 3, 3],
                          lambda p: 5. if max(abs(p)) > .08 else 0.)
    result = solve_aphi(mesh, 10000., [source])
    assert result.power_from_fields[0, 0].real > 0


def test_fixed_conductor_refinement_remains_finite():
    coarse = solve_aphi(_mesh(3, 5.), 25000., [_uniform_B_z])
    fine = solve_aphi(_mesh(6, 5.), 25000., [_uniform_B_z])
    assert np.isfinite(coarse.delta_z[0, 0])
    assert np.isfinite(fine.delta_z[0, 0])
    assert_allclose(fine.power_matrix, fine.power_from_fields,
                    rtol=1e-9, atol=1e-11)
