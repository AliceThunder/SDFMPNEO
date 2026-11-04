"""P1: physically exact two-box geometry Coulomb scalar and first moments.

Compares analytic oriented rectangle-face primitives against an INDEPENDENT
surface Gauss rule near actual seawater-facing face/edge/corner points.
This does not assemble the current-dependent Coulomb L matrix.
"""
from __future__ import annotations

import math
import numpy as np
from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.coulomb_geometry_moments import (
    box_coulomb_moments,sea_coulomb_moments,
)


def _graded_face_rule(lo,hi,order,scale):
    """Independent Gauss face measure with graded panels around query ray."""
    q,w=np.polynomial.legendre.leggauss(order)
    bounds=[float(lo),float(hi)]
    for direction in (-1.0,1.0):
        for mul in (0,1,2,4,8,16,32):
            t=direction*mul*scale
            if lo<t<hi:
                bounds.append(t)
    bounds=sorted(set(bounds))
    nodes,weights=[],[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        nodes.append(0.5*(b-a)*q+0.5*(a+b))
        weights.append(0.5*(b-a)*w)
    return np.concatenate(nodes),np.concatenate(weights)


def _independent_box_surface_moments(point,assembly,order):
    """Direct physical face Gauss: no analytic rectangle corner primitive."""
    pose=assembly.pose
    h=np.asarray(assembly.enclosure.half_extents)
    p=(np.asarray(point)-pose.center)@pose.rotation
    S=0.0
    M=np.zeros(3)
    for axis in range(3):
        j,k=[d for d in range(3) if d!=axis]
        for side in (-1.0,1.0):
            w=side*h[axis]-p[axis]
            scale=max(abs(w),2e-4)
            u,wu=_graded_face_rule(-h[j]-p[j],h[j]-p[j],order,scale)
            v,wv=_graded_face_rule(-h[k]-p[k],h[k]-p[k],order,scale)
            distance=np.sqrt(
                u[:,None]**2+v[None,:]**2+w*w
            )
            weights=wu[:,None]*wv[None,:]
            I0=float(np.sum(weights/distance))
            I1=float(np.sum(weights*distance))
            S+=0.5*side*w*I0
            M[axis]+=side*I1
    return S,M@pose.rotation.T


def _relative(a,b):
    return float(
        np.linalg.norm(np.asarray(a)-np.asarray(b))
        /max(np.linalg.norm(np.asarray(b)),1e-30)
    )


def main():
    scene=build_demo_scene()
    print(
        "p1_geom_exact: domain=ball_minus_two_rotated_insulating_boxes,"
        "S=integral_wet(1/r),"
        "M=integral_wet((y-x)/r),"
        "box_face_integrals=analytic_rectangle_primitives,"
        "ball_moments=analytic,no_softening=True",
        flush=True,
    )
    for i,assembly in enumerate(scene.assemblies):
        half=np.asarray(assembly.enclosure.half_extents)
        specs=(
            ("face",np.array([0.0,0.0,half[2]+0.0025])),
            ("edge",np.array([half[0]+0.0025,0.0,half[2]+0.0025])),
            ("corner",half+0.0025),
        )
        for label,local in specs:
            point=assembly.pose.position(local)
            if not bool(scene.in_seawater(point)):
                raise ValueError("P1 geometry query must be wet")
            s_exact,m_exact=box_coulomb_moments(
                point[None,:],assembly.pose.center,assembly.pose.rotation,
                half,
            )
            s_exact,m_exact=float(s_exact[0]),m_exact[0]
            s12,m12=_independent_box_surface_moments(point,assembly,12)
            s18,m18=_independent_box_surface_moments(point,assembly,18)
            print(
                f"p1_geom_box: port={i},probe={label},"
                f"S_box={s_exact:.11g},M_box_norm={np.linalg.norm(m_exact):.11g},"
                f"S_gauss12_rel={_relative(s12,s_exact):.7g},"
                f"S_gauss18_rel={_relative(s18,s_exact):.7g},"
                f"M_gauss12_rel={_relative(m12,m_exact):.7g},"
                f"M_gauss18_rel={_relative(m18,m_exact):.7g}",
                flush=True,
            )
            s_sea,m_sea=sea_coulomb_moments(
                scene,point[None,:],
            )
            print(
                f"p1_geom_wet: port={i},probe={label},"
                f"S_wet={s_sea[0]:.11g},"
                f"M_wet_norm={np.linalg.norm(m_sea[0]):.11g},"
                f"positive_S={bool(s_sea[0]>0)}",
                flush=True,
            )
    print(
        "p1_geom_note: analytic box face integrals are directly checked "
        "against independent graded surface quadrature near physical "
        "face/edge/corner. This validates geometry moments only; "
        "the induced-current Taylor residual and entire BFZI L "
        "must be separately tested on the SAME wet domain.",
        flush=True,
    )


if __name__=="__main__":
    main()
