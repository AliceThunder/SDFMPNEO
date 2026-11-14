"""Conservative finite-depth sea skin integrated with original BFZI wet FV."""
from types import SimpleNamespace as N

import numpy as np

from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D


def demo():
    # Complete cube FV records. The tiny sealed packages are represented
    # only by positive package-to-water geometry in this unit test.
    edges=np.linspace(-.45,.45,6)
    r=[];pos=[];vol=[]
    for i in range(5):
        for j in range(5):
            for k in range(5):
                lo=edges[[i,j,k]] if False else np.array(
                    [edges[i],edges[j],edges[k]])
                hi=np.array([edges[i+1],edges[j+1],edges[k+1]])
                r.append(dict(cell_min=lo,cell_max=hi))
                pos.append((lo+hi)*.5)
                vol.append(np.prod(hi-lo))
    def coil(y):
        return N(center=np.array([0.,y,0.]),roll_deg=90.,
                 yaw_deg=0.,pitch_deg=0.,plane_angle_deg=90.)
    coils=[coil(-.18),coil(.18)]
    fam=[]
    for c in coils:
        pos0=np.array([[0,c.center[1],-.05],[.03,c.center[1],.03]])
        dl=np.array([[.01,0,0],[0,0,.01]])
        fam.append(N(coil=c,active=N(pos=pos0,dl=dl)))
    model=N(vol=np.array(vol),points=np.array(pos),
            proj=N(edges=[],boundary=[]),
            base=dict(records=r,sample=dict(
                coil1_package_x_half_mm=102.,
                coil1_package_z_pos_half_mm=102.,
                coil1_package_z_neg_half_mm=102.,
                coil2_package_x_half_mm=102.,
                coil2_package_z_pos_half_mm=102.,
                coil2_package_z_neg_half_mm=102.),
                cfg=dict(BSR_PACKAGE_NORMAL_HALF_M=.0055),
                families=fam),
            copper=N(lines=[N(line_elements=f.active.dl) for f in fam],
                     scene=N(assemblies=[N(wire_radius=.001)]*2)))
    return model


def test_skin_positive_C_G_and_exact_shared_energy_capacity():
    m=demo()
    et=BFZIV148Skin3D(m,skin_depth=.002,
                      skin_layers=6,skin_quad=1)
    assert et.n>et.cu.stop and et.skin.stop==et.n
    assert np.min(et.cap)>0
    assert np.all(et.bulk_vol>0)
    assert np.isclose(et.bulk_vol.sum()+et.skin_vol.sum(),
                      m.vol.sum(),atol=1e-12)
    assert np.isclose(et.skin_info["sea_cap_delta_J_K"],0,atol=1e-7)
    assert (et.H-et.H.T).nnz==0
    assert np.allclose(et.H@np.ones(et.n),0,atol=1e-8)
    assert all(len(links)>0 for links in et.pkg_links)
    assert et.skin_info["faces"]>0


def test_skin_em_temperature_and_heat_injection_conserve_each_original_FV():
    m=demo()
    et=BFZIV148Skin3D(m,skin_depth=.002,skin_layers=6,
                      skin_quad=1)
    rng=np.random.default_rng(5)
    state=rng.normal(size=et.n)
    wet=et.sea_temp(state)
    assert np.isclose(wet@m.vol,
                      state[:et.count]@et.bulk_vol+
                      state[et.skin]@et.skin_vol)
    q=np.linspace(.5,2.,et.count)
    cu=np.array([5.,8.])
    power=et.heat_vec(q,cu)
    assert np.isclose(power.sum(),q@m.vol+cu.sum())
    assert np.allclose(power[et.skin],et.skin_mix.T@q)
    assert np.allclose(
        np.asarray(et.skin_mix.sum(axis=0)).ravel(),et.skin_vol)
    assert np.allclose(
        et.bulk_vol+np.asarray(et.skin_mix.sum(axis=1)).ravel(),
        m.vol)
    assert et.skin_info["map_order"]==2


def test_finite_skin_shorttime_fractional_diffusion():
    # Independent unit-area surface heat flux into the physical 8-layer
    # water RC chain, Dirichlet water at 2 mm. For t << L^2/alpha
    # the surface temperature follows the 1/sqrt(t) thermal impedance.
    depth=.002
    ratio=1.4**np.arange(8)
    dx=depth*ratio/ratio.sum()
    k=.6
    rhoc=4e6
    C=rhoc*dx
    H=np.zeros((8,8))
    for j in range(7):
        g=k/(.5*(dx[j]+dx[j+1]))
        H[j,j]+=g;H[j+1,j+1]+=g
        H[j,j+1]-=g;H[j+1,j]-=g
    H[-1,-1]+=k/(.5*dx[-1])
    dt=.02
    A=np.diag(C/dt)+H
    x=np.zeros(8)
    for j in range(1,101):
        force=np.zeros(8)
        force[0]=100.
        x=np.linalg.solve(A,C*x/dt+force)
    surface=x[0]+100.*dx[0]/(2*k)
    exact=2*100.*np.sqrt(2./(np.pi*k*rhoc))
    assert abs(surface/exact-1)<.025
