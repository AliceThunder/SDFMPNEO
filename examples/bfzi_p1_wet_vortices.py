"""P1-B: two genuine seawater vortex currents with exact SELF and MUTUAL L.

These currents are smooth compact-supported swirls in the REAL wet domain
outside two sealed BFZI boxes. They are divergence free, zero normal on
both enclosures and the outer sphere, and support radii / poses follow
the two physical boxes (no superellipse, no artificial magnetic cloud).

The compact spherical *current-support* is an exactly chosen physical
test CURRENT, NOT a replacement Coulomb kernel or a production basis.
Every 1/|x-y| in this test is the original unsoftened Newton kernel.

We can obtain self L from the exact Newton potential of a polynomial
toroidal vortex and mutual L from the EXACT exterior dipole potential.
This supplies a genuinely two-current off-diagonal analytic target,
independent of the Coulomb quadrature algorithm being tested.

A support-only quadrature is mathematically sufficient because J=0
elsewhere in the actual sea. Thus the subtraction moments are those
of the union of the two support balls, NOT the earlier full-sea S/M.
It does NOT certify wall-hugging/global BFZI W dictionaries or any
finite conductor/internal current distribution.
"""
from __future__ import annotations

import math
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from examples.bfzi_p1_coulomb import ball_rule
from sdfmpneo_bfzi.source import MU0
from sdfmpneo_bfzi.green_coulomb import assemble_coulomb_subtracted


def support_vortices(scene, radius=0.045, wall_gap=0.0025):
    """One smooth toroidal support beside each actual box's facing face."""
    data=[]
    for i, assembly in enumerate(scene.assemblies):
        half=np.asarray(assembly.enclosure.half_extents,float)
        sign=1.0 if i==0 else -1.0
        center=assembly.pose.position(
            [0.0,0.0,sign*(half[2]+radius+wall_gap)]
        )
        axis=assembly.pose.vector([0.0,0.0,1.0])
        axis=axis/np.linalg.norm(axis)
        if not bool(scene.in_seawater(center)):
            raise ValueError("P1 vortex center must be wet")
        clearance=[
            scene.outer_radius-np.linalg.norm(
                center-np.asarray(scene.outer_center)
            )
        ]
        for other in scene.assemblies:
            d2,_=other.enclosure.squared_exterior_distance(
                center[None,:],other.pose,
            )
            clearance.append(float(np.sqrt(d2[0])))
        if min(clearance) < radius+wall_gap-1e-11:
            raise ValueError("P1 vortex support intersects a physical boundary")
        data.append((center,axis,float(radius)))
    sep=np.linalg.norm(data[0][0]-data[1][0])
    if sep <= data[0][2]+data[1][2]+1e-12:
        raise ValueError("P1 vortex supports overlap")
    return tuple(data)


def vortex_fields(supports, points, *, gradient=False):
    """W_j(x)=(1-|x-cj|²/a²)^2 (ej cross (x-cj)) inside its ball."""
    x=np.asarray(points,float).reshape(-1,3)
    out=np.zeros((len(x),len(supports),3))
    dw=np.zeros((len(x),len(supports),3,3)) if gradient else None
    eye=np.eye(3)
    for j,(center,axis,a) in enumerate(supports):
        d=x-center
        q=np.einsum("ni,ni->n",d,d)/(a*a)
        t=np.maximum(1.0-q,0.0)
        f=t*t
        rot=np.cross(axis[None,:],d)
        out[:,j,:]=f[:,None]*rot
        if gradient:
            first=-4.0/(a*a)*t[:,None,None]*rot[:,:,None]*d[:,None,:]
            derivative=np.cross(axis[None,:],eye).T
            dw[:,j,:,:]=first+f[:,None,None]*derivative[None,:,:]
    return dw if gradient else out


def support_rule(supports, radial, polar, azimuth):
    """Independent positive 3D volume rules on physical current supports."""
    points,weights=[],[]
    for center,axis,a in supports:
        x,w=ball_rule(a,radial,polar,azimuth)
        points.append(x+center)
        weights.append(w)
    return np.concatenate(points),np.concatenate(weights)


def support_geometry_moments(supports, points):
    """Analytic unsoftened scalar and FIRST geometry moments of support union.

    Same integral identities as for real sea S/M. Interior polynomial
    ball formula is supplemented by exact exterior ball formulas, since
    a point in one vortex is outside the other current-support ball.
    """
    x=np.asarray(points,float).reshape(-1,3)
    S=np.zeros(len(x))
    M=np.zeros((len(x),3))
    for c,axis,a in supports:
        d=x-c
        r=np.linalg.norm(d,axis=1)
        inside=r<a
        scalar=np.empty(len(x))
        moment=np.empty_like(d)
        rr=np.maximum(r,1e-30)
        scalar[inside]=2*math.pi*(
            a*a-r[inside]**2/3
        )
        moment[inside]=(4*math.pi/15)*(
            r[inside]**2-5*a*a
        )[:,None]*d[inside]
        scalar[~inside]=4*math.pi*a**3/(3*rr[~inside])
        moment[~inside]=(
            -4*math.pi*a**3/(3*rr[~inside])
            +4*math.pi*a**5/(15*rr[~inside]**3)
        )[:,None]*d[~inside]
        S+=scalar
        M+=moment
    return S,M


def exact_vortex_matrix(supports):
    """Analytic physical Coulomb self and mutual L for two support currents.

    A self-current W=f(r) e×r has exact potential h(r)e×r:
      h(r)=4*pi*a²*(1/18-r²/(10a²)+r⁴/(14a⁴)-r⁶/(54a⁶)).
    The exact self energy is 2048*pi*mu0*a⁷/1216215.
    Outside each ball, U(x)=D e×(x-c)/|x-c|³,
      D=32*pi*a⁵/945.
    Hence mutual L is the exact dipole-dipole current energy.
    """
    n=len(supports)
    result=np.zeros((n,n))
    for i,(c,e,a) in enumerate(supports):
        result[i,i]=2048*math.pi*MU0*a**7/1216215
        for j in range(i):
            cj,ej,aj=supports[j]
            separation=c-cj
            distance=np.linalg.norm(separation)
            direction=separation/distance
            di=32*math.pi*a**5/945
            dj=32*math.pi*aj**5/945
            result[i,j]=MU0/(4*math.pi)*di*dj/(distance**3)*(
                3*np.dot(e,direction)*np.dot(ej,direction)-np.dot(e,ej)
            )
            result[j,i]=result[i,j]
    return result


def wet_vortex_trial(supports, x_rule, y_rule):
    x,wx=support_rule(supports,*x_rule)
    y,wy=support_rule(supports,*y_rule)
    moments=support_geometry_moments(supports,x)
    exact=exact_vortex_matrix(supports)
    got,skew=assemble_coulomb_subtracted(
        x,wx,y,wy,
        lambda p:vortex_fields(supports,p),
        lambda p:support_geometry_moments(supports,p)[0],
        gradient=lambda p:vortex_fields(supports,p,gradient=True),
        first_moment=lambda p:support_geometry_moments(supports,p)[1],
        return_diagnostics=True,
    )
    return dict(
        x_nodes=len(x),y_nodes=len(y),
        frob_rel=float(np.linalg.norm(got-exact)/np.linalg.norm(exact)),
        diag_rel=float(np.max(np.abs(
            np.diag(got-exact)/np.diag(exact)
        ))),
        mutual_rel=float(abs(got[0,1]-exact[0,1])/
                         max(abs(exact[0,1]),1e-40)),
        raw_skew=skew,L11=got[0,0],L12=got[0,1],
        true_L11=exact[0,0],true_L12=exact[0,1],
        support_min_S=float(np.min(moments[0])),
    )


def main():
    scene=build_demo_scene()
    supports=support_vortices(scene)
    print(
        "p1_wet_exact: current_support=two_smooth_compact_vortices,"
        "physical_domain=outer_ball_minus_two_actual_boxes,"
        "wall_normal_J=0,divJ=0,"
        "self_and_mutual_L=analytical_original_1/r,"
        "support_only_y_integration=exact_because_J_zero_elsewhere,"
        "not_full_sea_basis=True",
        flush=True,
    )
    for i,(c,e,a) in enumerate(supports):
        print(
            f"p1_wet_source: mode={i},"
            f"center={','.join(f'{t:.9g}' for t in c)},"
            f"radius_m={a:.9g},"
            f"axis={','.join(f'{t:.9g}' for t in e)}",
            flush=True,
        )
    outcomes={}
    # On the exact self potential, the x-radial energy integrand is a
    # degree-14 polynomial: Gauss 8 is sufficient for that outer moment.
    # Keep the y-rule independent; the remaining inner residual is NOT a
    # polynomial and is evaluated against an exact self/mutual matrix.
    for name,x_rule,y_rule in (
        ("base",(3,5,8),(4,6,10)),
        ("both",(8,10,16),(8,11,18)),
    ):
        z=wet_vortex_trial(supports,x_rule,y_rule)
        outcomes[name]=z
        print(
            f"p1_wet_L: variant={name},x_nodes={z['x_nodes']},"
            f"y_nodes={z['y_nodes']},"
            f"frob_rel={z['frob_rel']:.7g},"
            f"diag_rel={z['diag_rel']:.7g},"
            f"mutual_rel={z['mutual_rel']:.7g},"
            f"raw_skew={z['raw_skew']:.7g},"
            f"L11={z['L11']:.12g},true_L11={z['true_L11']:.12g},"
            f"L12={z['L12']:.12g},true_L12={z['true_L12']:.12g}",
            flush=True,
        )
    fine=outcomes["both"]
    print(
        f"p1_wet_gate: target_rel=0.001,"
        f"self_pass={fine['diag_rel']<0.001},"
        f"mutual_pass={fine['mutual_rel']<0.001},"
        f"matrix_pass={fine['frob_rel']<0.001},"
        "scope=two_compact_physical_wet_currents_only,"
        "full_sea_volume_and_wall_hugging_modes_not_certified",
        flush=True,
    )


if __name__=="__main__":
    main()
