"""P1 geometry moments: analytic double-box seawater vs independent Gauss."""
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.coulomb_geometry_moments import (
    box_coulomb_moments, ball_coulomb_moments, sea_coulomb_moments,
)
from sdfmpneo_bfzi.geometry import RigidPose


def test_analytic_box_face_moments_match_independent_3d_volume_integration():
    # Distinct representations: exact rectangle surface corner primitives
    # versus an independent positive Gaussian VOLUME cubature.
    half=np.array([0.09,0.07,0.025])
    theta=0.49
    rot=np.array([
        [np.cos(theta),0,np.sin(theta)],
        [0,1,0],
        [-np.sin(theta),0,np.cos(theta)],
    ])
    pose=RigidPose(rot,[0.12,-0.04,0.17])
    local=np.array([[0.32,-0.2,0.25],[-0.23,0.29,-0.18]])
    x=pose.position(local)
    analytic_s,analytic_m=box_coulomb_moments(x,pose.center,pose.rotation,half)
    q,w=np.polynomial.legendre.leggauss(14)
    grid=np.stack(np.meshgrid(*[q*half[i] for i in range(3)],indexing="ij"),axis=-1).reshape(-1,3)
    weights=np.prod(np.stack(np.meshgrid(*[w*half[i] for i in range(3)],indexing="ij"),axis=-1),axis=-1).reshape(-1)
    points=pose.position(grid)
    d=points[None,:,:]-x[:,None,:]
    r=np.linalg.norm(d,axis=-1)
    exact_s=np.einsum("j,ij->i",weights,1/r)
    exact_m=np.einsum("j,ijc,ij->ic",weights,d,1/r)
    np.testing.assert_allclose(analytic_s,exact_s,rtol=3e-11,atol=3e-14)
    np.testing.assert_allclose(analytic_m,exact_m,rtol=3e-11,atol=3e-14)


def test_real_seawater_geometry_is_ball_minus_two_rotated_boxes():
    scene=build_demo_scene()
    points=np.array([
        [0.18,0.03,-0.08],
        [-0.21,0.14,0.02],
    ])
    assert np.all(scene.in_seawater(points))
    s,m=sea_coulomb_moments(scene,points)
    ball_s,ball_m=ball_coulomb_moments(
        points,scene.outer_center,scene.outer_radius,
    )
    for assembly in scene.assemblies:
        sb,mb=box_coulomb_moments(
            points,assembly.pose.center,assembly.pose.rotation,
            assembly.enclosure.half_extents,
        )
        ball_s-=sb
        ball_m-=mb
    np.testing.assert_allclose(s,ball_s,atol=1e-13,rtol=0)
    np.testing.assert_allclose(m,ball_m,atol=1e-13,rtol=0)
    assert np.all(s>0)


def test_box_moments_are_equivariant_under_rigid_pose():
    half=(0.08,0.05,0.017)
    x=np.array([[0.12,0.04,0.07],[-0.19,0.09,-0.11]])
    theta=0.6
    rot=np.array([
        [np.cos(theta),-np.sin(theta),0],
        [np.sin(theta),np.cos(theta),0],
        [0,0,1],
    ])
    c=np.array([0.05,-0.08,0.16])
    s,m=box_coulomb_moments(x,[0,0,0],np.eye(3),half)
    moved_s,moved_m=box_coulomb_moments(x@rot.T+c,c,rot,half)
    np.testing.assert_allclose(s,moved_s,rtol=3e-13,atol=1e-15)
    np.testing.assert_allclose(m@rot.T,moved_m,rtol=3e-13,atol=1e-15)
