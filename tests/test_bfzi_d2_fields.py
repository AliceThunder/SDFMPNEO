"""Physical 3D field probes: analytic primary + Nedelec material reaction.

Checks interpolation, material-field separation and limiting behavior only.
These tests do NOT certify TEAM-7 experimental accuracy.
"""
import numpy as np
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_aphi import (
    box_tetra_mesh, closed_polyline_potential,
)
from sdfmpneo_bfzi.direct3d_probe import closed_polyline_magnetic
from sdfmpneo_bfzi.direct3d_galerkin import solve_closed_ports
from sdfmpneo_bfzi.direct3d_fields import sample_primary_secondary_fields


def _source():
    return np.array([
        [-.05,-.05,0.],[.05,-.05,0.],
        [.05,.05,0.],[-.05,.05,0.],[-.05,-.05,0.],
    ])


def test_vacuum_field_is_exact_known_closed_source():
    loop=_source()
    frequency=1500.
    mesh=box_tetra_mesh([[-.2,.2]]*3,[3,3,3],lambda x:0.)
    result=solve_closed_ports(mesh,frequency,[[(1.,loop)]])
    points=np.array([[.02,.01,.08],[0.,0.,.12],[-.06,.04,.14]])
    field=sample_primary_secondary_fields(
        result,[[(1.,loop)]],frequency,points)
    B=closed_polyline_magnetic(loop)(points)
    E=-1j*2*np.pi*frequency*closed_polyline_potential(loop)(points)
    assert_allclose(field['B_total_corrected_T'][0],B,rtol=0,atol=1e-15)
    assert_allclose(field['E_total_primary_secondary_V_m'][0],E,
                    rtol=0,atol=1e-13)
    assert_allclose(field['B_material_reaction_fem_T'],0.,atol=1e-14)
    assert_allclose(field['q_primary_secondary_W_m3'],0.,atol=1e-14)


def test_conductor_probe_joule_from_full_corrected_field():
    loop=_source()
    frequency=80000.
    mesh=box_tetra_mesh([[-.2,.2]]*3,[4,4,4],
                        lambda p:5. if p[2] < -.05 else 0.)
    result=solve_closed_ports(mesh,frequency,[[(1.,loop)]])
    points=np.array([[.01,.02,-.12],[.03,-.02,-.11]])
    field=sample_primary_secondary_fields(
        result,[[(1.,loop)]],frequency,points,
        prefer_conductor=True)
    E=field['E_total_primary_secondary_V_m']
    sigma=field['sigma_S_m'][None,:]
    assert (sigma>0).all()
    assert_allclose(field['J_total_primary_secondary_A_m2'],
                    sigma[:,:,None]*E,atol=1e-12)
    assert_allclose(field['q_primary_secondary_W_m3'],
                    .5*sigma*np.sum(abs(E)**2,axis=-1),atol=1e-12)
    assert np.linalg.norm(field['B_material_reaction_fem_T'])>0.
