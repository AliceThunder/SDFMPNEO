"""P1-C: analytic Jacobian of real BFZI non-guided curl trial currents."""
import numpy as np
import pytest

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.sea_basis import build_sea_basis, SeaCurrentBasis
from sdfmpneo_bfzi.sea_basis_differential import evaluate_nonguided_curl_jacobian
from sdfmpneo_bfzi.green_coulomb import assemble_basis_coulomb_taylor
# The main P1 driver imports this module only after other long experiments.
# Importing it during pytest catches a syntax regression before any run.
from examples.bfzi_p1_full_sea_taylor import trial as full_sea_trial


def test_bfzi_actual_box_curl_jacobian_matches_physical_perturbations():
    scene=build_demo_scene()
    original=build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    basis=SeaCurrentBasis(
        scene,tuple(original.modes[i] for i in (0,1,12,13)),
        original.wall_scale,original.source_lines,
        original.outer_envelope,original.outer_taper_width,
    )
    points=np.array([
        [0.015,0.018,-0.105],
        [0.04,-0.02,0.105],
        [0.15,0.04,0.03],
    ])
    assert scene.in_seawater(points).all()
    analytical=evaluate_nonguided_curl_jacobian(basis,points)
    numerical=np.empty_like(analytical)
    eps=1e-6
    for k in range(3):
        shift=np.eye(3)[k]*eps
        plus=basis.evaluate(points+shift)
        minus=basis.evaluate(points-shift)
        numerical[:,:,:,k]=(plus-minus)/(2*eps)
    # Comparing actual mode field derivatives (not a hand-coded
    # alternate Hessian) checks the full outer taper, rotating boxes,
    # and Gaussian current profile product rule in one physical scene.
    assert np.linalg.norm(analytical-numerical)/max(
        np.linalg.norm(numerical),1e-30
    ) < 2e-4


def test_full_sea_quadrature_rejects_identical_x_y_axes():
    # The first P1-C experiment must never evaluate the Taylor residual
    # at an identical pair of x/y quadrature nodes.
    with pytest.raises(ValueError, match="disjoint x/y"):
        full_sea_trial(None,(4,6,10),(4,6,10))



def test_actual_box_symmetric_difference_coulomb_small_space_runs():
    from sdfmpneo_bfzi.green_coulomb import assemble_basis_coulomb_difference

    scene = build_demo_scene()
    original = build_sea_basis(scene, modes_per_face=2, face_refinement=1)
    basis = SeaCurrentBasis(
        scene, tuple(original.modes[i] for i in (0, 12)),
        original.wall_scale, original.source_lines,
        original.outer_envelope, original.outer_taper_width,
    )
    x, wx = scene.quadrature(2, 4, 6, wall_layers=0)
    y, wy = scene.quadrature(3, 5, 7, wall_layers=0)
    L, ratio = assemble_basis_coulomb_difference(
        basis, np.eye(basis.size), x, wx, y, wy,
        return_diagnostics=True,
    )
    assert L.shape == (2, 2) and np.isfinite(L).all()
    assert np.isfinite(ratio)
    np.testing.assert_allclose(L, L.T, atol=1e-20, rtol=1e-12)




def test_coulomb_azimuth_panels_follow_both_real_box_poses():
    """Angular sectors use projected OBB vertices and preserve full 2pi."""
    from dataclasses import replace
    from sdfmpneo_bfzi.geometry import RigidPose
    from sdfmpneo_bfzi.green_coulomb import (
        _bfzi_projected_box_azimuth_rule,
    )

    scene=build_demo_scene()
    angle=0.29
    rot=np.array([
        [np.cos(angle),0,np.sin(angle)],
        [0,1,0],
        [-np.sin(angle),0,np.cos(angle)],
    ])
    turned=replace(
        scene,
        transmitter=replace(
            scene.transmitter,
            pose=RigidPose(rot,scene.transmitter.pose.center),
        ),
    )
    p=np.array([0.15,0.14,0.01])
    assert scene.in_seawater(p) and turned.in_seawater(p)
    e1,e2=np.eye(3)[:2]
    unrotated,wu=_bfzi_projected_box_azimuth_rule(scene,p,e1,e2,8)
    rotated,wr=_bfzi_projected_box_azimuth_rule(turned,p,e1,e2,8)
    for nodes,weights in ((unrotated,wu),(rotated,wr)):
        assert np.isfinite(nodes).all() and np.all(weights>0)
        assert np.all((nodes>0)&(nodes<2*np.pi))
        np.testing.assert_allclose(sum(weights),2*np.pi,rtol=1e-14)
    assert not np.array_equal(unrotated,rotated)



def test_actual_box_direct_radial_original_coulomb_small_space_runs():
    """Exact box-ray clipping, no x/y diagonal and no large subtraction."""
    from sdfmpneo_bfzi.green_coulomb import assemble_basis_coulomb_radial

    scene=build_demo_scene()
    original=build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    basis=SeaCurrentBasis(
        scene,tuple(original.modes[i] for i in (0,12)),
        original.wall_scale,original.source_lines,
        original.outer_envelope,original.outer_taper_width,
    )
    x,wx=scene.quadrature(2,3,4,wall_layers=0)
    L,skew=assemble_basis_coulomb_radial(
        basis,np.eye(basis.size),x,wx,
        radial_order=2,polar_order=2,azimuthal_order=6,
        return_diagnostics=True,
    )
    assert L.shape==(2,2) and np.isfinite(L).all()
    assert np.isfinite(skew)
    np.testing.assert_allclose(L,L.T,rtol=0,atol=0)



def test_actual_box_original_coulomb_taylor_small_space_runs():
    scene=build_demo_scene()
    original=build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    basis=SeaCurrentBasis(
        scene,tuple(original.modes[i] for i in (0,12)),
        original.wall_scale,original.source_lines,
        original.outer_envelope,original.outer_taper_width,
    )
    x,wx=scene.quadrature(2,4,6,wall_layers=0)
    y,wy=scene.quadrature(3,5,7,wall_layers=0)
    assert len(x)>0 and len(y)>0
    got,skew=assemble_basis_coulomb_taylor(
        basis,np.eye(basis.size),x,wx,y,wy,
        return_diagnostics=True,
    )
    assert got.shape==(2,2) and np.isfinite(got).all()
    assert np.isfinite(skew)
    np.testing.assert_allclose(got,got.T,rtol=0,atol=0)


def test_real_two_box_physical_outer_rule_is_positive_and_wet():
    """The production P1-C physical x rule keeps exact seawater geometry.

    This guards the actual box/radial/angular/tangent call used for the
    new L candidate; it does not claim electromagnetic convergence.
    """
    from examples.bfzi_p1_full_sea_taylor import _quadrature
    scene=build_demo_scene()
    original=build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    selected=(0,1,12,13,24)
    basis=SeaCurrentBasis(
        scene,tuple(original.modes[i] for i in selected),
        original.wall_scale,original.source_lines,
        original.outer_envelope,original.outer_taper_width,
    )
    x,w=_quadrature(
        scene,basis,("shell_phys",3,4,8,3),{},
    )
    assert len(x)>0
    assert np.isfinite(x).all() and np.isfinite(w).all()
    assert np.all(w>0) and np.all(scene.in_seawater(x))
    exact_volume=4*np.pi*scene.outer_radius**3/3
    for assembly in scene.assemblies:
        exact_volume-=8*np.prod(assembly.enclosure.half_extents)
    # Loose geometry smoke only. High-precision volume and all 5 Gram
    # modes are measured in the P1-C local log, not certified by pytest.
    assert abs(np.sum(w)-exact_volume)/exact_volume<0.03
