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
from sdfmpneo_bfzi.exterior_dipole import ExteriorDipole
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
    lb=assemble_basis_coulomb_split(
        trial,transform,x,wx,y,wy,
        local_polar_order=6,local_azimuthal_order=12,**kwargs,
    )
    w=source.omega
    def solve(l):
        u=np.linalg.solve(r+1j*w*l,-1j*w*k)
        return 1j*w*k.T @ u,u
    za,ua=solve(la)
    zb,ub=solve(lb)
    zball,uball=solve(lball)
    # The common R-energy metric is meaningful only for this fixed
    # pilot space, not a bound on continuum current error.
    chol_r = np.linalg.cholesky(r).T
    dj_r = relative(chol_r@ub,chol_r@ua)
    print(
        f"g1_coulomb_split: modes={trial.size}, x_nodes={len(x)}, "
        f"y_nodes={len(y)}, "
        f"local_angular_dL={relative(lb,la):.6g}, "
        f"ball_vs_split_dL={relative(lball,lb):.6g}, "
        f"local_angular_dZ={relative(zb,za):.6g}, "
        f"ball_vs_split_dZ={relative(zball,zb):.6g}, "
        f"local_angular_dJ_R={dj_r:.6g}, "
        f"min_L_split_eig={np.linalg.eigvalsh(lb)[0]:.6g}",
        flush=True,
    )
    print(
        "g1_coulomb_note: this is a 5-mode fixed-domain, unsoftened "
        "singularity-split pilot; radial, outer cubature and full-space "
        "convergence remain untested. No 229-mode certificate.",
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
    print(
        "g1_exterior_note: exact l=1 toroidal spherical-cavity MQS "
        "transparent boundary is not a general BFZI two-box DtN map.",
        flush=True,
    )


if __name__=="__main__":
    main()
