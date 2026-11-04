"""G1 reconstruction experiments: real Coulomb, nested curl and exact exterior.

Use: python run.py --mode bfzi-g1-foundations --with-tests
No 229-mode reference is certified by this affordable small-space pilot.
"""
from __future__ import annotations

import math
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi import (
    SeaCurrentBasis, build_sea_basis, build_bfzi_source,
)
from sdfmpneo_bfzi.source import MU0
from sdfmpneo_bfzi.green_coulomb import assemble_basis_coulomb_split
from scipy.special import spherical_kn
from sdfmpneo_bfzi.exterior_dipole import ExteriorDipole, ExteriorSphericalCurrent
from sdfmpneo_bfzi.sea_reference import _assemble_ball_green_inductance


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def _coupling(source, modes, weights, points):
    # C_port is A_port/mu0. Both L methods use precisely this fixed
    # two-port excitation and physical finite domain.
    return MU0*np.einsum(
        "n,nmc,ncp->mp",
        weights,modes,source.potential_kernels(points),optimize=True,
    )


def main():
    scene = build_demo_scene()
    source = build_bfzi_source(scene,200_000.,order=4)
    base = build_sea_basis(
        scene,modes_per_face=2,face_refinement=1,
    )
    systematic1 = build_sea_basis(
        scene,modes_per_face=2,face_refinement=1,structured_level=1,
    )
    systematic2 = build_sea_basis(
        scene,modes_per_face=2,face_refinement=1,structured_level=2,
    )
    if (base.size != 27 or systematic1.size != base.size+44
            or systematic2.size != systematic1.size+140):
        raise RuntimeError("structured current-space hierarchy is not nested")
    # The historical modes are kept byte-for-byte; new charts use an
    # independent smooth C2 window with finite spatial support.
    for current in (systematic1,systematic2):
        for original,mode in zip(base.modes,current.modes):
            if (not np.array_equal(original.center,mode.center)
                or not np.array_equal(original.direction,mode.direction)
                or original.width != mode.width
                or original.profile != mode.profile):
                raise RuntimeError("structured hierarchy changed a baseline mode")
    print(
        f"g1_structured_basis: base={base.size}, "
        f"level1={systematic1.size}, level2={systematic2.size}, "
        f"compact_level1={systematic1.size-base.size}",
        flush=True,
    )

    # Deliberately evaluate a small *physical* mode subspace first:
    # the independent full Coulomb double integral is substantially
    # costlier than the positive cloud approximation.
    selected = tuple(base.modes[k] for k in (0,1,2,-2,-1))
    trial = SeaCurrentBasis(
        scene,selected,base.wall_scale,base.source_lines,
        base.outer_envelope,base.outer_taper_width,
    )
    transform = np.eye(trial.size)
    def quadrature(radial, polar, azimuth):
        return scene.box_shell_quadrature(
            radial,polar,azimuth,
            wall_scale=trial.wall_scale,outer_taper_width=trial.outer_taper_width,
            shell_normal_order=2,shell_tangential_order=2,
        )
    x,wx = quadrature(3,5,8)
    y,wy = quadrature(4,6,10)
    modes=trial.evaluate(x)
    r = np.einsum(
        "n,nmc,nkc->mk",wx/scene.seawater.conductivity,modes,modes,
        optimize=True,
    )
    k=_coupling(source,modes,wx,x)
    lball,_=_assemble_ball_green_inductance(x,wx,modes,128)
    kwargs=dict(
        local_radius=0.055,local_fraction=0.6,
        local_radial_order=4,
    )
    la=assemble_basis_coulomb_split(
        trial,transform,x,wx,y,wy,
        local_polar_order=4,local_azimuthal_order=8,**kwargs,
    )
    lb,skew_base=assemble_basis_coulomb_split(
        trial,transform,x,wx,y,wy,
        local_polar_order=6,local_azimuthal_order=12,
        return_diagnostics=True,**kwargs,
    )
    # The old cutoff shrank to the distance from the boxes, leaving
    # near-wall near-singular work to sparse global y cubature. The new
    # clipped-ray integration uses the same physical radius even across
    # an insulating-box silhouette. Compare both on IDENTICAL x/y nodes,
    # and vary the partition radius without changing the continuous L.
    l_clearance,skew_clearance=assemble_basis_coulomb_split(
        trial,transform,x,wx,y,wy,
        clip_local_to_domain=False,
        local_polar_order=6,local_azimuthal_order=12,
        return_diagnostics=True,**kwargs,
    )
    l_patch,skew_patch=assemble_basis_coulomb_split(
        trial,transform,x,wx,y,wy,
        local_radius=0.08,local_fraction=0.6,local_radial_order=4,
        local_polar_order=6,local_azimuthal_order=12,
        return_diagnostics=True,
    )
    # More important than another local-angle tweak: independently
    # refine BOTH physical x and y volume cubatures. Keep R/K,
    # physical modes and the port forcing from the first rule.
    x2,wx2 = quadrature(4,6,10)
    y2,wy2 = quadrature(5,8,14)
    # Factorial 2x2 x/y volume cubature: isolate each integration axis.
    # This answers whether the 2.98% L change in the last log came
    # primarily from x, from y or from their coupled quadrature aliasing.
    lx,skew_x=assemble_basis_coulomb_split(
        trial,transform,x2,wx2,y,wy,
        local_polar_order=6,local_azimuthal_order=12,
        return_diagnostics=True,**kwargs,
    )
    ly,skew_y=assemble_basis_coulomb_split(
        trial,transform,x,wx,y2,wy2,
        local_polar_order=6,local_azimuthal_order=12,
        return_diagnostics=True,**kwargs,
    )
    lc,skew_refined=assemble_basis_coulomb_split(
        trial,transform,x2,wx2,y2,wy2,
        local_polar_order=6,local_azimuthal_order=12,
        return_diagnostics=True,**kwargs,
    )
    w=source.omega
    def solve(l):
        u=np.linalg.solve(r+1j*w*l,-1j*w*k)
        return 1j*w*k.T @ u,u
    za,ua=solve(la)
    zb,ub=solve(lb)
    zc,uc=solve(lc)
    zx,ux=solve(lx)
    zy,uy=solve(ly)
    zball,uball=solve(lball)
    # The common R-energy metric is meaningful only for this fixed
    # pilot space, not a bound on continuum current error.
    delta_u = ub-ua
    energy_delta = float(np.real(np.vdot(delta_u,r@delta_u)))
    energy_base = float(np.real(np.vdot(ua,r@ua)))
    dj_r = math.sqrt(max(energy_delta,0.0)/max(energy_base,1e-60))
    eigenvalues,eigenvectors = np.linalg.eigh(r)
    if np.min(eigenvalues) <= 0.0:
        raise RuntimeError("pilot R Gram is unresolved; refine volume rule")
    invsqrt = (eigenvectors/np.sqrt(eigenvalues))@eigenvectors.T
    weighted_delta = w*invsqrt@(lc-lb)@invsqrt
    dynamic_dL = float(np.linalg.norm(weighted_delta,ord=2))
    dynamic_x = float(np.linalg.norm(
        w*invsqrt@(lx-lb)@invsqrt,ord=2,
    ))
    dynamic_y = float(np.linalg.norm(
        w*invsqrt@(ly-lb)@invsqrt,ord=2,
    ))
    nonadditive = float(
        np.linalg.norm(lc-lx-ly+lb)
        /max(np.linalg.norm(lb),1e-30)
    )
    refined_residual = (
        (r+1j*w*lc)@ub+1j*w*k
    )
    forcing = 1j*w*k
    driven_residual = (
        np.linalg.norm(invsqrt@refined_residual)
        /max(np.linalg.norm(invsqrt@forcing),1e-30)
    )
    test_currents = np.array([1.0+0.2j,-0.4+0.5j])
    probe=np.array([[0.,0.,0.],[0.15,0.,0.],[0.,0.12,0.],[0.,0.,0.22]])
    probe_modes=trial.evaluate(probe)
    def probe_heat(u):
        current=np.einsum(
            "nmc,m->nc",probe_modes,u@test_currents,optimize=True,
        )
        return np.sum(np.abs(current)**2,axis=1)/(
            2*scene.seawater.conductivity
        )
    print(
        f"g1_coulomb_patch: modes={trial.size}, "
        f"clip_vs_clearance_dL={relative(lb,l_clearance):.6g}, "
        f"radius_055_to_080_dL={relative(l_patch,lb):.6g}, "
        f"raw_skew_clearance={skew_clearance:.6g}, "
        f"raw_skew_clip_055={skew_base:.6g}, "
        f"raw_skew_clip_080={skew_patch:.6g}",
        flush=True,
    )
    print(
        f"g1_coulomb_axes: modes={trial.size}, "
        f"dL_x_only={relative(lx,lb):.6g}, "
        f"dL_y_only={relative(ly,lb):.6g}, "
        f"dL_both={relative(lc,lb):.6g}, "
        f"nonadditive_dL={nonadditive:.6g}, "
        f"raw_skew_base={skew_base:.6g}, "
        f"raw_skew_x_only={skew_x:.6g}, "
        f"raw_skew_y_only={skew_y:.6g}, "
        f"raw_skew_both={skew_refined:.6g}, "
        f"dimensionless_dL_x={dynamic_x:.6g}, "
        f"dimensionless_dL_y={dynamic_y:.6g}, "
        f"dZ_x_only={relative(zx,zb):.6g}, "
        f"dZ_y_only={relative(zy,zb):.6g}, "
        f"dq_x_only={relative(probe_heat(ux),probe_heat(ub)):.6g}, "
        f"dq_y_only={relative(probe_heat(uy),probe_heat(ub)):.6g}",
        flush=True,
    )
    print(
        f"g1_coulomb_split: modes={trial.size}, x_nodes={len(x)}, "
        f"y_nodes={len(y)}, "
        f"local_angular_dL={relative(lb,la):.6g}, "
        f"ball_vs_split_dL={relative(lball,lb):.6g}, "
        f"local_angular_dZ={relative(zb,za):.6g}, "
        f"ball_vs_split_dZ={relative(zball,zb):.6g}, "
        f"local_angular_dJ_R={dj_r:.6g}, "
        f"volume_x_nodes={len(x2)}, volume_y_nodes={len(y2)}, "
        f"volume_dL={relative(lc,lb):.6g}, "
        f"dimensionless_dL={dynamic_dL:.6g}, "
        f"refined_on_coarse_residual={driven_residual:.6g}, "
        f"volume_dZ={relative(zc,zb):.6g}, "
        f"volume_probe_dq={relative(probe_heat(uc),probe_heat(ub)):.6g}, "
        f"min_L_split_eig={np.linalg.eigvalsh(lc)[0]:.6g}",
        flush=True,
    )
    print(
        "g1_coulomb_note: this is a 5-mode fixed-domain, unsoftened "
        "singularity-split pilot; x/y cubatures, near-ray clipping and "
        "patch radius are compared, not yet converged. No 229-mode certificate.",
        flush=True,
    )

    dipole=ExteriorDipole(0.16,200_000.,scene.seawater.conductivity)
    true_j=dipole.current_phi(0.25,math.pi/2)
    for radius in (0.52,0.85,1.3):
        truncated=dipole.current_phi(
            0.25,math.pi/2,outer_radius=radius,
        )
        print(
            f"g1_exact_exterior: outer_radius={radius:.3g}, "
            f"skin_depth={dipole.skin_depth:.6g}, "
            f"dirichlet_rel_dJ={abs(truncated-true_j)/max(abs(true_j),1e-30):.6g}, "
            f"transparent_log_derivative={dipole.transparent_log_derivative(radius):.6g}",
            flush=True,
        )
    # Actual homogeneous seawater outside an arbitrary enclosing sphere
    # has an infinite diagonal vector *spherical-harmonic* DtN (after
    # angular projection), not a single Robin scalar. Check two distinct
    # divergence-free toroidal degrees and the independent tail loss.
    exterior = ExteriorSphericalCurrent(
        0.52, 200_000., scene.seawater.conductivity, max_degree=3,
    )
    mu,phi,_ = exterior.angular_nodes()
    transverse = np.sqrt(1-mu**2)
    tangent = np.column_stack((
        -transverse*np.sin(phi), transverse*np.cos(phi), np.zeros(len(mu)),
    ))
    boundary_current = (1+0.4*mu)[:,None]*tangent
    coefficients,projection_error = exterior.project(boundary_current)
    radius=0.85
    gamma=exterior.gamma
    exact_tail = (
        spherical_kn(1,gamma*radius)/spherical_kn(1,gamma*exterior.radius)
        +0.4*mu*spherical_kn(2,gamma*radius)
        /spherical_kn(2,gamma*exterior.radius)
    )[:,None]*tangent
    print(
        f"g1_exterior_multipole: lmax={exterior.max_degree}, "
        f"boundary_projection_error={projection_error:.6g}, "
        f"l1_l2_tail_error={relative(exterior.evaluate(coefficients,radius,mu,phi),exact_tail):.6g}, "
        f"tail_ohmic_power_for_unit_trace={exterior.tail_ohmic_power(coefficients):.6g}",
        flush=True,
    )
    print(
        "g1_exterior_note: multiple exact toroidal exterior harmonics and "
        "vector spherical-harmonic continuation now implemented; their "
        "DtN must still be coupled to interior BFZI J at a matching sphere "
        "with divergence compatibility. G1 infinite-domain FAIL remains.",
        flush=True,
    )


if __name__=="__main__":
    main()
