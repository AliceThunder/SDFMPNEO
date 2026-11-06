"""Independent analytic low-induction checks of the REAL 3D A-phi PDE.

No published-number fitting, no fake power PASS from dual source loads.
"""
import numpy as np
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_elliptic_limit import (
    axial_uniform_potential, elliptic_low_induction_phi,
    elliptic_low_induction_e, elliptic_3d_benchmark,
)


def test_nontrivial_phi_gives_analytic_elliptic_electric_field():
    a, b, frequency, B0 = .12, .07, 1000., .001
    points = np.array([[.03,.012,.017],[-.04,.021,-.03]])
    dx = 1e-6
    gradient = np.zeros((len(points),3),complex)
    for k in range(3):
        step = np.zeros(3)
        step[k] = dx
        gradient[:,k] = (
            elliptic_low_induction_phi(points+step,frequency,B0,a,b)
            - elliptic_low_induction_phi(points-step,frequency,B0,a,b)
        )/(2*dx)
    expected = elliptic_low_induction_e(
        points,frequency,B0,a,b)
    reconstructed = (
        -1j*2*np.pi*frequency*axial_uniform_potential(points,B0)
        - gradient)
    assert np.linalg.norm(gradient)>0
    assert_allclose(reconstructed, expected, rtol=0, atol=1e-14)


def test_elliptic_3d_mesh_field_and_joule_converge_to_analytic_limit():
    coarse = elliptic_3d_benchmark(sectors=16,radial=2,axial=2)
    finer = elliptic_3d_benchmark(sectors=32,radial=4,axial=4)
    assert coarse["magnetic_induction_parameter"]<1e-3
    assert finer["relative_e_l2_error"]<coarse["relative_e_l2_error"]
    assert abs(finer["relative_joule_error"])<abs(coarse["relative_joule_error"])
    assert finer["fem_joule_W"]>0
    assert finer["n_tetrahedra"]>coarse["n_tetrahedra"]
    assert_allclose(finer["analytic_ellipse_joule_W"],
                    coarse["analytic_ellipse_joule_W"],rtol=0,atol=0)


def test_swap_ellipse_axes_keeps_physics_not_exact_mesh_algebra():
    first = elliptic_3d_benchmark(
        a=.12,b=.07,sectors=16,radial=2,axial=2)
    rotated = elliptic_3d_benchmark(
        a=.07,b=.12,sectors=16,radial=2,axial=2)
    assert_allclose(first["analytic_ellipse_joule_W"],
                    rotated["analytic_ellipse_joule_W"],rtol=0,atol=1e-16)
    # Two different tetra orientations do not give identical discrete error.
    assert_allclose(first["fem_joule_W"],rotated["fem_joule_W"],
                    rtol=1e-5)


def test_circular_limit_needs_no_scalar_charge_correction():
    reference = elliptic_3d_benchmark(
        a=.1,b=.1,sectors=24,radial=3,axial=3)
    assert reference["relative_e_l2_error"]<1e-3
