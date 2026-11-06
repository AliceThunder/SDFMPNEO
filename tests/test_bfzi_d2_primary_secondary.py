"""Direct 3D analytic-primary A-phi PDE tests, not paper-value fitting."""
import numpy as np
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_aphi import box_tetra_mesh, closed_polyline_potential
from sdfmpneo_bfzi.direct3d_probe import closed_polyline_magnetic
from sdfmpneo_bfzi.direct3d_primary_secondary import (
    solve_closed_primary_secondary,
)


def _square(z=0.):
    return np.array([
        [-.05,-.05,z],[.05,-.05,z],
        [.05,.05,z],[-.05,.05,z],[-.05,-.05,z],
    ])


def test_analytic_closed_primary_vacuum_returns_exact_spatial_fields():
    frequency = 1500.
    path = _square()
    mesh = box_tetra_mesh(
        [[-.2,.2]]*3,[3,3,3],lambda p:0.)
    response = solve_closed_primary_secondary(
        mesh,frequency,[[(1.,path)]])
    points = np.array([[.01,.01,.13],[-.02,.03,.14]])
    got = response.sample_fields(points)
    expected_b = closed_polyline_magnetic(path)(points)
    expected_e = -1j*2*np.pi*frequency*closed_polyline_potential(path)(points)
    assert_allclose(got["B_total_T"][0],expected_b,rtol=0,atol=1e-15)
    assert_allclose(got["E_total_V_m"][0],expected_e,rtol=0,atol=1e-14)
    assert_allclose(got["q_W_m3"],0.,rtol=0,atol=0)
    assert_allclose(response.volume_overlap_z,0.,atol=1e-13)
    assert_allclose(response.closed_source_reaction_z,0.,atol=1e-13)


def test_analytic_primary_seawater_charge_and_joule_is_field_derived():
    path = _square()
    mesh = box_tetra_mesh(
        [[-.2,.2]]*3,[4,4,4],
        lambda p: 4. if p[2]>.1 else 0.)
    response = solve_closed_primary_secondary(
        mesh,1000.,[[(1.,path)]])
    P = response.volume_power(np.array([1.+0j]))
    assert P>0.
    # This is the primary-secondary weak-form energy identity; it is
    # NOT the same assertion as a physical closed-source port identity.
    assert_allclose(.5*response.volume_overlap_z[0,0].real,
                    P,rtol=1e-8,atol=1e-14)
    samples = response.sample_fields(
        np.array([[.01,.01,.15]]),prefer_conductor=True)
    sigma = samples["sigma_S_m"]
    E = samples["E_total_V_m"]
    assert sigma[0]==4.
    assert_allclose(samples["J_A_m2"],sigma[None,:,None]*E,
                    rtol=0,atol=1e-14)
    assert_allclose(samples["q_W_m3"],
                    .5*sigma[None,:]*np.sum(abs(E)**2,axis=-1),
                    rtol=0,atol=1e-14)
    assert np.linalg.norm(samples["B_secondary_T"])>0.


def test_analytic_primary_two_ports_and_complex_current_power():
    left = _square(z=-.04)
    right = _square(z=.04)
    mesh = box_tetra_mesh(
        [[-.2,.2]]*3,[4,4,4],
        lambda p:4. if p[2]>.1 else 0.)
    response = solve_closed_primary_secondary(
        mesh,5000.,[[(1.,left)],[(1.,right)]])
    for current in ([1.,0.],[0.,1.],[1.,1.],[1.,1j]):
        i=np.asarray(current,complex)
        assert response.volume_power(i)>=0.
    assert_allclose(response.volume_overlap_z,
                    response.volume_overlap_z.T,rtol=1e-8,atol=1e-12)
