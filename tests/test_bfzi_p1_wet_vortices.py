"""P1-B: two real, non-overlapping seawater vortices and exact Coulomb L."""
import numpy as np
from examples.bfzi_mqs_demo import build_demo_scene
from examples.bfzi_p1_wet_vortices import (
    support_vortices, vortex_fields, support_rule,
    support_geometry_moments, exact_vortex_matrix, wet_vortex_trial,
)
from sdfmpneo_bfzi.source import MU0


def test_physical_two_box_vortices_have_exact_nonzero_mutual_energy():
    scene=build_demo_scene()
    supports=support_vortices(scene)
    for c,axis,r in supports:
        assert scene.in_seawater(c)
        assert np.isclose(np.linalg.norm(axis),1)
        for a in scene.assemblies:
            distance,_=a.enclosure.squared_exterior_distance(
                c[None,:],a.pose,
            )
            assert np.sqrt(distance[0])>r
    matrix=exact_vortex_matrix(supports)
    assert np.all(np.diag(matrix)>0)
    assert abs(matrix[0,1])>0
    np.testing.assert_allclose(matrix,matrix.T,atol=0,rtol=0)

    # Independent self check: analytic h(r) for the compact polynomial
    # toroidal current, integrated by an ordinary positive volume rule.
    for c,axis,r in supports:
        q,w=support_rule(((c,axis,r),),7,12,24)
        d=q-c
        rr2=np.einsum("ni,ni->n",d,d)
        f=(1-rr2/(r*r))**2
        swirl=np.cross(axis[None,:],d)
        field=f[:,None]*swirl
        q2=rr2/(r*r)
        h=4*np.pi*r*r*(
            1/18-q2/10+q2*q2/14-q2*q2*q2/54
        )
        self_l=MU0/(4*np.pi)*np.einsum(
            "n,nc,nc->",w,field,h[:,None]*swirl,
        )
        np.testing.assert_allclose(
            self_l,matrix[0,0],rtol=5e-7,atol=0,
        )


def test_two_compact_vortex_self_and_mutual_original_coulomb_converge():
    # Both modes are identically zero near ALL seawater boundaries.
    # The exact off-diagonal dipole coupling is nonzero, so this tests
    # actual TWO-CURRENT mutual L, not merely another diagonal ball.
    supports=support_vortices(build_demo_scene())
    coarse=wet_vortex_trial(
        supports,(3,5,8),(4,6,10),
    )
    # Eight x-radial Gauss nodes exactly integrate polynomials up to
    # degree 15, including the analytic degree-14 self-energy moment.
    fine=wet_vortex_trial(
        supports,(8,10,16),(8,11,18),
    )
    assert np.isfinite(fine["frob_rel"])
    assert np.isfinite(fine["mutual_rel"])
    assert fine["frob_rel"]<coarse["frob_rel"]
    assert fine["diag_rel"]<0.001
    assert fine["mutual_rel"]<0.001
