"""M1 physical two-port, spatial Joule and phase-combination smoke."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.direct3d_structured_bfzi import axis_aligned_bfzi_tetra_mesh
from sdfmpneo_bfzi.m1_direct import solve_m1


def test_m1_real_dual_umfc_fields_and_same_heat():
    scene = build_demo_scene()
    mesh = axis_aligned_bfzi_tetra_mesh(
        scene, far_step_m=.50, box_layers=1)
    assert len(mesh.tetrahedra) > 1000
    assert np.any(mesh.sigma == 0) and np.any(mesh.sigma > 0)

    ref = solve_m1(scene, mesh, 200000, panels_per_corner=3)
    i = np.array([1., .4+.3j])
    a = np.array([[0., 0., 0.], [.12, 0., 0.]])
    result = ref.fields(a, i)

    assert ref.z_sea.shape == (2, 2)
    assert ref.z_em.shape == (2, 2)
    assert np.isfinite(ref.z_em).all()
    assert ref.physical.reaction.linear_relative_residual < 1e-8
    assert_allclose(ref.z_loss, ref.z_loss.conj().T, rtol=1e-11, atol=1e-14)
    assert np.linalg.eigvalsh(ref.z_loss).min() >= -1e-12

    e = result["unit_E_V_m"]
    b = result["unit_B_T"]
    assert e.shape == (2, len(a), 3)
    assert b.shape == (2, len(a), 3)
    assert_allclose(result["E_V_m"], np.einsum("p,pnc->nc", i, e))
    assert_allclose(result["B_T"], np.einsum("p,pnc->nc", i, b))
    assert_allclose(result["J_A_m2"],
                    result["E_V_m"]*scene.seawater.conductivity)

    xyz, w, q = ref.heat_nodes(i)
    assert len(xyz) == len(w) == len(q)
    p = ref.volume_power(i)
    assert p > 0
    assert_allclose(w @ q, p, rtol=1e-10)
    assert_allclose(.5*np.real(np.vdot(i, ref.z_loss@i)), p, rtol=1e-10)
    assert_allclose(ref.port_sea_power(i), p, rtol=1e-10)
    assert_allclose(ref.z_sea, ref.z_sea.T, rtol=1e-11, atol=1e-13)
    assert np.linalg.eigvalsh(.5*(ref.z_sea+ref.z_sea.conj().T)).min() >= -1e-12
    assert_allclose(.5*(ref.z_sea+ref.z_sea.conj().T), ref.z_loss,
                    rtol=1e-10, atol=1e-12)
    for phase in ((1, 0), (0, 1), (1, 1), (1, -1), (1, 1j)):
        assert_allclose(ref.port_sea_power(phase),
                        ref.volume_power(phase), rtol=1e-10)
    assert np.isfinite(ref.z_closed).all()
    assert ref.closed_port_sea_power(i) != ref.port_sea_power(i)

    # The complex two-port phase interference must enter q before squaring.
    e1 = ref.fields(a, (1, 0))["E_V_m"]
    e2 = ref.fields(a, (0, 1))["E_V_m"]
    assert_allclose(result["q_W_m3"],
                    .5*scene.seawater.conductivity*
                    np.sum(abs(e1+(.4+.3j)*e2)**2, axis=1))
    with pytest.raises(ValueError, match="conducting seawater"):
        ref.fields(np.array([[0., 0., -.13]]), i)


def test_m1_circular_tx_and_variable_seawater():
    from dataclasses import replace
    from sdfmpneo_bfzi.scene import Seawater

    src = build_demo_scene()
    tx = src.transmitter
    circle = replace(tx.coil, corner_radius=tx.coil.outer_half_size,
                     alpha=1.0)
    tx = replace(tx, coil=circle)
    scene = replace(src, transmitter=tx, seawater=Seawater(3.0))
    mesh = axis_aligned_bfzi_tetra_mesh(
        scene, far_step_m=.50, box_layers=1)
    result = solve_m1(scene, mesh, 200000, panels_per_corner=3)
    current = np.array([1., .4+.3j])
    probe = result.fields(np.array([[0., 0., 0.]]), current)
    assert np.isfinite(probe["B_T"]).all()
    assert np.isfinite(probe["J_A_m2"]).all()
    assert np.all(probe["q_W_m3"] >= 0)
    assert_allclose(result.port_sea_power(current),
                    result.volume_power(current), rtol=1e-10)
    assert_allclose(result.z_sea, result.z_sea.T, rtol=1e-10, atol=1e-12)


def test_unified_bfzi_coil_cli_parameter_domain():
    from examples.bfzi_m1_direct import coil_from

    base = build_demo_scene().transmitter.coil
    circle = coil_from([2, .075, .075, .005, 1., 1.], base)
    rounded = coil_from([2, .075, .045, .005, .5, -1.], base)
    assert circle.outer_half_size == circle.corner_radius
    assert circle.alpha == 1.
    assert rounded.winding == -1
    assert rounded.alpha == .5
    assert rounded.corner_radius < rounded.outer_half_size
    assert coil_from(None, base) is base
    with pytest.raises(ValueError):
        coil_from([2.1, .075, .075, .005, 1., 1.], base)


def test_fixed_core_grid_coordinates_across_R():
    from dataclasses import replace

    a = build_demo_scene()
    b = replace(a, outer_radius=.75)
    # Exact core coordinates are independent of exterior R.
    one = axis_aligned_bfzi_tetra_mesh(
        a, far_step_m=.30, box_layers=1, inner_core_radius_m=.30)
    two = axis_aligned_bfzi_tetra_mesh(
        b, far_step_m=.30, box_layers=1, inner_core_radius_m=.30)
    x = lambda m: np.unique(np.round(m.nodes[:,0], 12))
    inner = lambda z: z[(z>=-.30) & (z<=.30)]
    assert_allclose(inner(x(one)), inner(x(two)), atol=1e-12)
