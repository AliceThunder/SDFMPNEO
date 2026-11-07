"""Independent finite-frequency conducting-sphere Maxwell diffusion checks.

These test 3D PDE fields against an exact Bessel solution, not publication
table fits and not TEAM-7 high-fidelity certification.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_aphi import solve_aphi, box_tetra_mesh
from sdfmpneo_bfzi.direct3d_sphere_limit import (
    HarmonicConductingSphere, inscribed_sphere_tetra_mesh,
    sphere_d2_benchmark,
)


def _case():
    return HarmonicConductingSphere(
        .05, 3.526e7, 50., (0., 0., .001))


def test_independent_sphere_analytic_maxwell_ampere_and_interfaces():
    reference = _case()
    p = np.array([.018, .011, .013])
    h = 1e-6
    dB = []
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = h
        bplus = reference.fields(p+offset)["B_T"]
        bminus = reference.fields(p-offset)["B_T"]
        dB.append((bplus-bminus)/(2*h))
    jac = np.array(dB)
    curl_b = np.array([jac[1,2]-jac[2,1],
                       jac[2,0]-jac[0,2],
                       jac[0,1]-jac[1,0]])
    ampere = 4e-7*np.pi*reference.fields(p)["J_A_m2"]
    assert_allclose(curl_b,ampere,rtol=5e-6,atol=2e-8)
    a=reference.radius_m
    near=np.array([[a*(1-1e-8),0,0],[a*(1+1e-8),0,0]])
    f=reference.fields(near)
    assert_allclose(f["A_T_m"][0],f["A_T_m"][1],rtol=1e-6,atol=1e-13)
    assert_allclose(f["B_T"][0],f["B_T"][1],rtol=1e-6,atol=1e-9)


def test_low_induction_limit_and_positive_physical_joule():
    weak=HarmonicConductingSphere(.05,.001,10.,(.0004,.0002,.001))
    at_center=weak.fields(np.array([0.,0.,0.]))["B_T"]
    assert_allclose(at_center,weak.imposed_B_T,rtol=2e-7,atol=1e-10)
    assert weak.sphere_joule_power()>0
    assert _case().sphere_joule_power()>0


def test_inscribed_geometry_has_physical_positive_tetra_volume():
    exact=4*np.pi*.05**3/3
    prev=0.
    for n in (2,4,6):
        mesh=inscribed_sphere_tetra_mesh(.05,3.526e7,n)
        assert len(mesh.tetrahedra)==6*n**3
        xyz=mesh.nodes[mesh.tetrahedra]
        jac=(xyz[:,1:]-xyz[:,0,None,:]).transpose(0,2,1)
        volume=float(abs(np.linalg.det(jac)).sum()/6)
        assert prev<volume<exact
        prev=volume


def test_3d_aphi_spherical_skin_depth_E_B_convergence():
    reference=_case()
    coarse=sphere_d2_benchmark(reference,2)
    medium=sphere_d2_benchmark(reference,4)
    fine=sphere_d2_benchmark(reference,6)
    assert (coarse["complex_E_volume_relative_L2"]
            > medium["complex_E_volume_relative_L2"]
            > fine["complex_E_volume_relative_L2"])
    assert (coarse["complex_B_volume_relative_L2"]
            > medium["complex_B_volume_relative_L2"]
            > fine["complex_B_volume_relative_L2"])
    for record in (coarse,medium,fine):
        assert record["joule_fem_W"]>0
        assert record["joule_exact_same_polyhedron_W"]>0
        assert 0<record["polyhedron_to_sphere_volume_ratio"]<1


def test_prescribed_total_boundary_rejects_missing_data():
    mesh=box_tetra_mesh([[-.2,.2]]*3,[2,2,2],lambda p:5.)
    zero=lambda p:np.zeros_like(p)
    with pytest.raises(ValueError,match="prescribed_total"):
        solve_aphi(mesh,50.,[zero],outer_bc="prescribed_total")
    with pytest.raises(ValueError,match="only with prescribed_total"):
        solve_aphi(mesh,50.,[zero],boundary_total_potentials=[zero])
