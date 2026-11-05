"""P1: original Coulomb kernel against an exactly solvable solenoidal mode."""
import numpy as np

from examples.bfzi_p1_coulomb import (
    ball_rule, vortex_modes, vortex_green_potential,
    exact_vortex_green, trial, diagnose_original_split,
    taylor_subtraction_diagnostics, cubic_vortex_modes,
)
from sdfmpneo_bfzi.source import MU0



def test_symmetric_difference_unsoftened_1_over_r_against_independent_ball_truth():
    """Both affine and NON-affine exact solenoidal balls at fixed rules."""
    from sdfmpneo_bfzi.green_coulomb import assemble_coulomb_difference
    from examples.bfzi_p1_coulomb import exact_cubic_vortex_green

    x, wx = ball_rule(1.0, 5, 7, 12)
    y, wy = ball_rule(1.0, 6, 8, 14)
    scalar = lambda p: 2 * np.pi * (
        1 - np.einsum("nc,nc->n", p, p) / 3
    )
    for evaluate, exact in (
        (vortex_modes, exact_vortex_green(1.0)),
        (cubic_vortex_modes, exact_cubic_vortex_green(1.0)),
    ):
        got, ratio = assemble_coulomb_difference(
            x, wx, y, wy, evaluate, scalar, return_diagnostics=True,
        )
        assert np.isfinite(got).all() and np.isfinite(ratio)
        # Predetermined P1 sphere benchmark, not a two-box G1 claim.
        assert np.linalg.norm(got-exact)/np.linalg.norm(exact) < 0.001
        np.testing.assert_allclose(got, got.T, atol=1e-21, rtol=1e-13)

    # The difference integrand vanishes on the diagonal: identical
    # physical node sets do not require a fake softened self kernel.
    xx, ww = ball_rule(1.0, 3, 5, 8)
    same = assemble_coulomb_difference(xx, ww, xx, ww, vortex_modes, scalar)
    assert np.isfinite(same).all()



def test_unsoftened_x_centered_radial_green_matches_independent_analytic_ball():
    """Exact kernel; inner Jacobian removes 1/r without subtraction."""
    from sdfmpneo_bfzi.green_coulomb import assemble_coulomb_radial
    from examples.bfzi_p1_coulomb import exact_cubic_vortex_green

    x, wx = ball_rule(1.0, 5, 6, 10)

    def ball_rays(point, dirs):
        dot = dirs @ point
        reach = -dot + np.sqrt(dot**2 + 1.0 - np.dot(point,point))
        return np.zeros((len(dirs),1)), reach[:,None]

    for modes, exact in (
        (vortex_modes, exact_vortex_green(1.0)),
        (cubic_vortex_modes, exact_cubic_vortex_green(1.0)),
    ):
        L, skew = assemble_coulomb_radial(
            x,wx,modes,ball_rays,
            radial_order=4,polar_order=4,azimuthal_order=10,
            return_diagnostics=True,
        )
        assert np.isfinite(skew)
        assert np.linalg.norm(L-exact)/np.linalg.norm(exact) < 0.001
        np.testing.assert_allclose(L,L.T,rtol=0,atol=0)



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
