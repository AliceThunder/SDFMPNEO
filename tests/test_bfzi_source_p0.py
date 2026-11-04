"""P0 physical-source tests; no seawater Galerkin or Green L computation."""
from dataclasses import replace
import numpy as np
import pytest

from sdfmpneo_vnext.scene import ConductorMaterial

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.geometry import BFZICoil, RigidPose
from sdfmpneo_bfzi.source import LineSource, MU0, build_line_source, build_bfzi_source
from sdfmpneo_bfzi.source_reference import (
    straight_segment_field, continuous_port_field,
)


def test_exact_straight_segment_matches_independent_high_order_line_integral():
    start=np.array([0.03,-0.07,0.06])
    stop=np.array([0.18,0.04,-0.02])
    points=np.array([
        [0.26,0.09,0.11],
        [0.01,-0.13,0.18],
        [0.03,0.03,-0.11],
    ])
    exact_c,exact_b=straight_segment_field(start,stop,points)
    u,w=np.polynomial.legendre.leggauss(64)
    t=0.5*(u+1.0)
    segment=stop-start
    node_position=start+t[:,None]*segment
    node_dl=0.5*w[:,None]*segment
    numeric=LineSource(node_position,node_dl)
    np.testing.assert_allclose(
        exact_c,numeric.potential_kernel(points),rtol=1e-12,atol=1e-14,
    )
    np.testing.assert_allclose(
        exact_b,numeric.magnetic_field(points).real,rtol=1e-12,atol=1e-16,
    )


def test_continuous_bfzi_reference_includes_circular_limit_and_closed_return():
    scene=build_demo_scene()
    for assembly in scene.assemblies:
        positions=np.array([
            assembly.pose.position([0.15,0.03,0.05]),
            assembly.pose.position([-0.13,0.08,0.09]),
        ])
        c16,b16=continuous_port_field(assembly,positions,corner_order=16)
        c24,b24=continuous_port_field(assembly,positions,corner_order=24)
        np.testing.assert_allclose(c16,c24,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(b16,b24,rtol=1e-9,atol=1e-15)
        for order in (3,6,12):
            line=build_line_source(assembly,order=order)
            length=np.linalg.norm(line.line_elements,axis=1).sum()
            assert np.linalg.norm(line.line_elements.sum(axis=0))/length < 1e-4
    # Circular BFZI has zero-length straight pieces: the reference must
    # integrate only its physical curved segments and the complete return.
    circle=BFZICoil(
        turns=2,outer_half_size=0.075,corner_radius=0.075,
        pitch=0.005,alpha=1.0,
    )
    original=scene.transmitter
    assembly=replace(original,coil=circle)
    xyz=np.array([
        assembly.pose.position([0.16,0.02,0.04]),
        assembly.pose.position([-0.12,0.13,0.07]),
    ])
    c,b=continuous_port_field(assembly,xyz,corner_order=20)
    assert np.isfinite(c).all() and np.isfinite(b).all()
    line=build_line_source(assembly,order=16)
    np.testing.assert_allclose(
        line.potential_kernel(xyz),c,rtol=1e-7,atol=1e-11,
    )
    np.testing.assert_allclose(
        line.magnetic_field(xyz).real,b,rtol=1e-7,atol=1e-14,
    )


def test_source_reference_is_pose_covariant_and_b_is_curl_of_c():
    assembly=build_demo_scene().transmitter
    local=np.array([[0.16,0.02,0.05],[-0.15,0.11,0.06]])
    world=assembly.pose.position(local)
    c,b=continuous_port_field(assembly,world,corner_order=24)
    theta=0.39
    ct,st=np.cos(theta),np.sin(theta)
    rot=np.array([
        [ct,0.0,st],
        [0.0,1.0,0.0],
        [-st,0.0,ct],
    ])
    moved=replace(
        assembly,
        pose=RigidPose(rot,[0.025,-0.03,0.08]),
    )
    moved_xyz=moved.pose.position(local)
    cm,bm=continuous_port_field(moved,moved_xyz,corner_order=24)
    np.testing.assert_allclose(cm,c@rot.T,rtol=2e-12,atol=1e-13)
    np.testing.assert_allclose(bm,b@rot.T,rtol=2e-12,atol=1e-17)
    p=world[0:1]
    step=2e-5
    gradient=np.zeros((3,3))
    for axis in range(3):
        offset=step*np.eye(3)[axis]
        up,_=continuous_port_field(assembly,p+offset,corner_order=24)
        dn,_=continuous_port_field(assembly,p-offset,corner_order=24)
        gradient[:,axis]=(up[0]-dn[0])/(2.0*step)
    curl=np.array([
        gradient[2,1]-gradient[1,2],
        gradient[0,2]-gradient[2,0],
        gradient[1,0]-gradient[0,1],
    ])
    np.testing.assert_allclose(
        MU0*curl,b[0],rtol=3e-6,atol=2e-14,
    )



def test_production_source_integrates_exact_straights_and_only_curved_corners():
    # Reconstruct the complete physical line with an independent analytic
    # straight-segment + curved-corner reference. Keep the original
    # all-Gauss nodes for copper heat and air Neumann integration.
    scene=build_demo_scene()
    for assembly in scene.assemblies:
        points=np.array([
            assembly.pose.position([0.13,0.04,0.045]),
            assembly.pose.position([-0.14,0.09,0.06]),
        ])
        order=4
        hybrid=build_line_source(assembly,order=order)
        assert hybrid.field_mask is not None
        assert hybrid.straight_segments is not None
        assert np.count_nonzero(hybrid.field_mask)==4*assembly.coil.turns*order
        assert hybrid.points.shape[0]==len(hybrid.line_elements)
        direct_c,direct_b=continuous_port_field(
            assembly,points,corner_order=order,
        )
        np.testing.assert_allclose(
            hybrid.potential_kernel(points),direct_c,
            rtol=3e-12,atol=2e-14,
        )
        np.testing.assert_allclose(
            hybrid.magnetic_field(points).real,direct_b,
            rtol=3e-12,atol=2e-16,
        )


def test_circular_production_field_keeps_only_arcs_and_return():
    scene=build_demo_scene()
    original=scene.transmitter
    circular=BFZICoil(
        turns=2,outer_half_size=0.075,corner_radius=0.075,
        pitch=0.005,alpha=1.0,
    )
    assembly=replace(original,coil=circular)
    x=np.array([
        assembly.pose.position([0.16,0.02,0.04]),
        assembly.pose.position([-0.12,0.13,0.07]),
    ])
    hybrid=build_line_source(assembly,order=6)
    # Only the return has finite straight segments in the circular limit.
    assert len(hybrid.straight_segments)==len(assembly.return_waypoints)+1
    c,b=continuous_port_field(assembly,x,corner_order=6)
    np.testing.assert_allclose(
        hybrid.potential_kernel(x),c,rtol=1e-11,atol=1e-13,
    )
    np.testing.assert_allclose(
        hybrid.magnetic_field(x).real,b,rtol=1e-11,atol=1e-16,
    )



def test_magnetic_wire_cannot_silently_use_uniform_mu0_sea_source():
    scene=build_demo_scene()
    magnetic=replace(
        scene.transmitter,
        material=ConductorMaterial(
            conductivity=5.8e7,relative_permeability=1.2,
        ),
    )
    with pytest.raises(ValueError,match="nonmagnetic wire"):
        build_bfzi_source(
            replace(scene,transmitter=magnetic),
            frequency_hz=200000.0,order=4,
        )


def test_source_coulomb_gauge_divergence_uses_exact_segment_endpoints():
    # Analytic source continuity identity is separate from B=curl C:
    # div ∫_a^b dl/|x-r| = 1/|x-a|-1/|x-b|.
    aa=np.array([0.03,-0.07,0.06])
    bb=np.array([0.18,0.04,-0.02])
    x=np.array([[0.26,0.09,0.11],[0.01,-0.13,0.18]])
    u,w=np.polynomial.legendre.leggauss(64)
    pts=aa+0.5*(u[:,None]+1.0)*(bb-aa)
    dl=0.5*w[:,None]*(bb-aa)
    sample=LineSource(pts,dl)
    expected=(
        1.0/np.linalg.norm(x-aa,axis=1)
        -1.0/np.linalg.norm(x-bb,axis=1)
    )/(4*np.pi)
    np.testing.assert_allclose(
        sample.divergence_kernel(x),expected,rtol=2e-13,atol=2e-14,
    )


def test_complete_port_analytic_straights_restore_coulomb_gauge():
    # A physically closed continuous filament is solenoidal. Remaining
    # finite div C is entirely the curved-corner quadrature residual.
    scene=build_demo_scene()
    for assembly in scene.assemblies:
        local=np.array([
            [0.13,0.04,0.045],[-0.14,0.09,0.06],
            [0.0,0.0,0.034],[0.15,-0.02,0.06],
        ])
        x=assembly.pose.position(local)
        coarse=build_line_source(assembly,order=4)
        fine=build_line_source(assembly,order=12)
        res_coarse=np.linalg.norm(coarse.divergence_kernel(x))
        res_fine=np.linalg.norm(fine.divergence_kernel(x))
        assert res_fine < 1e-7
        assert res_fine < res_coarse
