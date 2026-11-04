"""P1: original Coulomb kernel against an exactly solvable solenoidal mode."""
import numpy as np

from examples.bfzi_p1_coulomb import (
    ball_rule, vortex_modes, vortex_green_potential,
    exact_vortex_green, trial, diagnose_original_split,
    taylor_subtraction_diagnostics, cubic_vortex_modes,
)
from sdfmpneo_bfzi.source import MU0


def test_p1_ball_vortex_exact_analytic_potential_and_energy():
    # W_i=e_i cross x, div W_i=0 and x dot W_i=0 on the sphere.
    # ΔU_i=-4πW_i with exterior dipole matching.
    radius=0.7
    x,w=ball_rule(radius,5,6,8)
    modes=vortex_modes(x)
    np.testing.assert_allclose(
        np.einsum("nmc,nc->nm",modes,x),
        0.0,rtol=0,atol=2e-16,
    )
    potential=vortex_green_potential(x,radius)
    energy=MU0/(4*np.pi)*np.einsum(
        "n,nmc,nkc->mk",w,modes,potential,optimize=True,
    )
    np.testing.assert_allclose(
        energy,exact_vortex_green(radius),
        rtol=2e-13,atol=1e-22,
    )


def test_p1_original_coulomb_split_is_finite_on_exact_vortex_mode():
    # This is only a coarse smoke test: the independent P1 log must
    # compare real x/y refinement and physical cutoff dependence.
    result=trial(
        1.0,(3,5,8),(4,6,10),
        (6,8,12),0.4,
    )
    assert np.isfinite(result["norm_error"])
    # No arbitrary physical-accuracy threshold here: the first P1
    # experiment must measure the real error without tuning an assertion.
    assert np.isfinite(result["diag_error"])
    assert np.isfinite(result["offdiag_error"])
    assert np.isfinite(result["L11"])


def test_p1_exact_local_ball_moment_and_global_subtraction_operator():
    # For the existing C2 cutoff: int_0^1 s*chi(s) ds == 2/7.
    # Full local sphere contribution for the linear vortex is thus
    # (8*pi*a*a/7) * W(x), independently of local angular nodes.
    from sdfmpneo_bfzi.green_coulomb import _near_window
    g,w=np.polynomial.legendre.leggauss(8)
    integral=0.0
    for start,stop in ((0.0,0.5),(0.5,1.0)):
        radius=0.5*(stop-start)*g+0.5*(start+stop)
        weights=0.5*(stop-start)*w
        integral+=float(np.dot(weights,radius*_near_window(radius)))
    np.testing.assert_allclose(integral,2/7,rtol=1e-13,atol=1e-15)

    # Outer integration of the ANALYTIC ball Newton potential must be
    # exact with this polynomial-positive sphere rule. The remaining
    # inner 1/r quadrature has an independently measurable discrepancy.
    diagnostic=diagnose_original_split(
        1.0,(4,6,8),(4,6,10),0.4,
    )
    assert diagnostic["x_exact_rel"]<1e-12
    assert diagnostic["far_y_rel"]>1e-3
    assert (
        diagnostic["global_constant_subtraction_rel"]
        < diagnostic["far_y_rel"]
    )


def test_nonaffine_solenoidal_cubic_vortex_uses_real_residual_kernel():
    # W=r²(e_i cross x) is nonaffine, div-free, zero sphere-normal.
    # Unlike affine W, its Taylor-subtracted residual is NOT zero.
    xyz=np.array([[0.25,-0.1,0.3],[0.4,0.12,-0.21]])
    modes=cubic_vortex_modes(xyz)
    np.testing.assert_allclose(
        np.einsum("nmc,nc->nm",modes,xyz),
        0.0,rtol=0,atol=1e-15,
    )
    metrics=taylor_subtraction_diagnostics(
        1.0,(5,8,12),(6,9,14),
    )
    assert metrics["linear_affine_rel"]<1e-12
    assert metrics["cubic_outer_exact_rel"]<1e-3
    assert metrics["cubic_affine_rel"]<1e-3
    assert metrics["cubic_const_rel"]>metrics["cubic_affine_rel"]
