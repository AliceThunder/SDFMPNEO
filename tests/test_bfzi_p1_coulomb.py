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




def test_x_centered_green_rays_resolve_exact_physical_wall_slope():
    """Independent closed-form 1D wall envelope integral through real ray code.

    A shell with a radial density proportional to d/(d+s)' tests the
    exact wall-normal derivative s/(d+s)^2. This is a controlled kernel
    quadrature test, NOT a physical BFZI curl mode or a G1 certificate.
    """
    from sdfmpneo_bfzi.green_coulomb import assemble_coulomb_radial

    a,b,s=0.20,0.52,0.001125
    def current(points):
        radius=np.linalg.norm(points,axis=1)
        amp=np.where(radius<a,1.0,s/(radius-a+s)**2)
        return amp[:,None,None]*np.array([1.0,0.0,0.0])[None,None,:]

    def radial_shell(point,directions):
        return (
            np.full((len(directions),1),a),
            np.full((len(directions),1),b),
        )

    # Physical transformed integrand is exactly r*s/(r-a+s)^2.
    def antiderivative(r):
        z=r-a+s
        return s*np.log(z)-s*(a-s)/z

    truth=antiderivative(b)-antiderivative(a)
    x=np.zeros((1,3))
    wx=np.ones(1)
    raw=assemble_coulomb_radial(
        x,wx,current,radial_shell,
        radial_order=3,polar_order=2,azimuthal_order=4,
    )[0,0]/MU0
    aligned=assemble_coulomb_radial(
        x,wx,current,radial_shell,
        radial_order=3,polar_order=2,azimuthal_order=4,
        radial_wall_scale=s,
    )[0,0]/MU0
    # With identical order, only physical ray endpoint panels change.
    assert abs(raw-truth)/truth > 0.5
    assert abs(aligned-truth)/truth < 0.001



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


def test_true_gaussian_mode_peak_gets_ray_centered_radial_panels():
    """Independent analytic radial integral for a narrow off-center mode.

    A positive Gaussian source has a sharp peak in the MIDDLE of a
    physical wet ray, not at a box wall or near the observer. With the
    same coarse Gauss-3 rule, the original panels nearly miss this
    source; actual projected Gaussian-window knots must resolve it.
    """
    import math
    from sdfmpneo_bfzi.green_coulomb import assemble_coulomb_radial

    nmu,nphi=2,4
    angular,_=np.polynomial.legendre.leggauss(nmu)
    mu=(angular[-1]+1)/2
    phi=2*np.pi*(nphi-0.5)/nphi
    direction=np.array([
        np.sqrt(1-mu**2)*np.sin(phi),
        -np.sqrt(1-mu**2)*np.cos(phi),
        mu,
    ])
    width=0.02
    center=0.57*direction
    boost=1000.0

    def field(y):
        q=np.sum((y-center)**2,axis=1)/width**2
        return (1+boost*np.exp(-q))[:,None,None]*np.array([
            1.0,0.0,0.0,
        ])[None,None,:]

    def full_ball(point,directions):
        return (np.zeros((len(directions),1)),
                np.ones((len(directions),1)))

    # Compute the exact radial contribution for EVERY discrete ray,
    # keeping the same angular rule. Only radial quadrature differs.
    nodes,weights=np.polynomial.legendre.leggauss(nmu)
    mus=np.concatenate(((nodes-1)/2,(nodes+1)/2))
    wmu=np.tile(weights/2,2)
    value=2*np.pi # Gauss-3 integrates the constant kernel r exactly.
    for polar,wpolar in zip(mus,wmu):
        for az in range(nphi):
            a=2*np.pi*(az+0.5)/nphi
            u=np.array([
                np.sqrt(1-polar**2)*np.sin(a),
                -np.sqrt(1-polar**2)*np.cos(a),
                polar,
            ])
            projection=float(u@center)
            transverse=max(0.,float(center@center-projection**2))
            scalar=math.exp(-transverse/width**2)
            integral=(
                -width**2/2*(
                    math.exp(-((1-projection)/width)**2)
                    -math.exp(-(projection/width)**2)
                )
                +projection*width*math.sqrt(math.pi)/2*(
                    math.erf((1-projection)/width)
                    +math.erf(projection/width)
                )
            )
            value+=boost*(2*np.pi/nphi)*wpolar*scalar*integral

    args=dict(
        x=np.zeros((1,3)),wx=np.ones(1),evaluate=field,
        wet_radial_segments=full_ball,radial_order=3,
        polar_order=nmu,azimuthal_order=nphi,
    )
    raw=assemble_coulomb_radial(**args)[0,0]
    adjusted=assemble_coulomb_radial(
        **args,radial_mode_windows=(
            center[None,:],(np.eye(3)/width**2)[None,:,:],
        ),
    )[0,0]
    from sdfmpneo_bfzi.source import MU0
    exact=MU0/(4*np.pi)*value
    assert abs(raw-exact)/exact>0.3
    assert abs(adjusted-exact)/exact<1e-4


def test_source_centered_caps_resolve_narrow_angular_gaussian_against_exact_ball():
    """An actual remote Gaussian is almost invisible to coarse sphere angles.

    Keep the radial Gaussian-window panels IDENTICAL in both trials.
    The only change is source-centred, positive partition-of-unity
    spherical cap charts; compare with a 3D analytic Newton potential
    of an isotropic Gaussian, truncated at radius 1 (>18 widths away).
    """
    import math
    from sdfmpneo_bfzi.green_coulomb import assemble_coulomb_radial
    from sdfmpneo_bfzi.source import MU0

    center=np.array([0.33,0.31,0.33])
    width=0.025
    metric=(np.eye(3)/width**2)[None,:,:]
    windows=(center[None,:],metric)

    def density(y):
        return np.exp(
            -np.sum((y-center)**2,axis=1)/width**2
        )[:,None,None]*np.array([[[1.,0.,0.]]])

    def sphere(point,directions):
        return np.zeros((len(directions),1)),np.ones((len(directions),1))

    # Test the point-centered INNER Green potential U(x) directly.
    # The old failed test mistakenly compared a vanishing self-bilinear
    # L11=mu0/(4*pi)*W(x)*U(x) with a finite analytic U(x).
    common=dict(
        x=np.zeros((1,3)),wx=np.ones(1),evaluate=density,
        wet_radial_segments=sphere,
        radial_order=3,polar_order=4,azimuthal_order=10,
        radial_mode_windows=windows,
        return_point_potentials=True,
    )
    _,plain_u=assemble_coulomb_radial(**common)
    stats={}
    _,corrected_u=assemble_coulomb_radial(
        **common,source_angular_caps=windows,source_cap_stats=stats,
    )
    plain=MU0/(4*math.pi)*plain_u[0,0,0]
    corrected=MU0/(4*math.pi)*corrected_u[0,0,0]
    # Newton potential of a whole-space Gaussian at the observer:
    # pi^(3/2) w^3 erf(|c|/w)/|c|. The omitted sphere exterior is
    # exponentially tiny because c lies >18 Gaussian widths within it.
    exact=(
        MU0/(4*math.pi)*math.pi**1.5*width**3
        *math.erf(np.linalg.norm(center)/width)
        /np.linalg.norm(center)
    )
    assert abs(plain-exact)/exact>0.8
    assert abs(corrected-exact)/exact<0.09
    assert stats["points"]==1
    assert stats["points_with_caps"]==1
    assert stats["cap_charts"]==1
    assert stats["cap_nodes"]>0
    assert stats["max_solid_angle_rel"]<0.01


def test_source_cap_positive_moment_fit_conserves_exact_ball_area_s_and_m():
    """Same narrow source caps, independently exact ball S/M and 4*pi.

    For x=0 in a unit ball, S=2*pi and M=0 analytically.
    Unlike the source-only Gaussian test, this checks the REAL finite
    angular weights after partition-of-unity and positive moment fitting.
    """
    import math
    from sdfmpneo_bfzi.green_coulomb import assemble_coulomb_radial

    center=np.array([0.33,0.31,0.33])
    width=0.025
    windows=(center[None,:],(np.eye(3)/width**2)[None,:,:])

    def source(points):
        return np.broadcast_to(
            np.array([[[1.,0.,0.]]]),(len(points),1,3)
        )

    def unit_ball(point,directions):
        return (
            np.zeros((len(directions),1)),
            np.ones((len(directions),1)),
        )

    stats={}
    _,u=assemble_coulomb_radial(
        np.zeros((1,3)),np.ones(1),source,unit_ball,
        radial_order=3,polar_order=4,azimuthal_order=10,
        source_angular_caps=windows,
        exact_scalar_moments=np.array([2*math.pi]),
        exact_first_moments=np.zeros((1,3)),
        source_cap_stats=stats,
        return_point_potentials=True,
    )
    assert stats["points"]==1
    assert stats["accepted"]==1
    assert stats["fall_back"]==0
    assert stats["max_solid_angle_rel"]<2e-8
    assert stats["max_scalar_moment_rel"]<2e-8
    assert stats["max_first_moment_scaled"]<2e-7
    np.testing.assert_allclose(u[0,0],[2*math.pi,0,0],rtol=1e-9,atol=1e-12)
