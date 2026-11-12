"""Local checks of the original-BFZI-form conservative spatial surrogate."""
import numpy as np
import torch
from scipy.sparse import diags

from sdfmpneo_bfzi.geom_memory import GeomFrac,geom_port
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack


def case():
    g=np.array([.09,.09,0.,0.,1.,1.,10.,10.,.00079,.00079,.04,
                16e-6,5e-6,16e-6,35.,35.])
    port=geom_port(g)
    sea=np.array([[.12+.002j,.08+.001j],
                  [.08+.001j,.12+.002j]])
    z=port.total(sea,np.full(2,293.15))
    vol=np.full(4,.03)
    H=diags(np.full(10,.15)).tocsr()
    cap=np.full(10,25.)
    unit=np.ones((4,2,3),complex)*(.1+.01j)
    weak=np.broadcast_to(np.eye(2)[None],(4,2,2))*3.
    xyz=np.array([[-.1,0.,0.],[.1,0.,0.],[0.,0.,-.1],[0.,0.,.1]])
    pkg=np.array([
        [[0.,-.04,0.],[.02,-.04,0.]],
        [[0., .04,0.],[.02, .04,0.]]])
    return GeomHeat(g,pack(z),np.diag([.5,.5]),
                    vol,xyz,pkg,cap,H,unit,weak)


def test_conservative_heat_and_cold_anchor(tmp_path):
    a=case()
    a.save(tmp_path/"anchor.npz")
    b=GeomHeat.load(tmp_path/"anchor.npz")
    assert np.allclose(a.z0,b.z0)
    assert np.allclose(a.cap,b.cap)
    assert np.allclose(a.weak,b.weak)
    old=np.zeros(len(b.cap))
    state,z,I,q,J,ps,pc,bal=b.advance(
        old,30.,60.,np.zeros((2,2),complex))
    assert J.shape==(4,3) and q.shape==(4,)
    assert np.min(q)>=0
    assert np.isclose(q@b.vol,ps,atol=1e-10)
    assert np.all(state>=0)
    assert np.max(abs(bal))<1e-9


def test_neural_field_rollout_is_exactly_power_conservative():
    b=case()
    model=GeomFrac()
    for p in model.net.parameters():
        torch.nn.init.zeros_(p)
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(14),
            port_sea=True)
    v=np.array([50.,50.,0.,50.])
    t=np.array([0.,30.,60.,90.])
    pred=b.rollout(model,ck,v,t)
    assert pred["temp_K"].shape==(4,10)
    assert pred["qsea_W_m3"].shape==(4,4)
    assert pred["Jsea_A_m2"].shape==(4,4,3)
    assert np.all(pred["temp_K"]>=293.15-1e-8)
    assert np.max(pred["fv_node_residual_W"])<1e-8
    assert np.allclose(pred["qsea_W_m3"]@b.vol,
                       pred["Psea_W"],atol=1e-10)
    assert np.all(pred["qsea_W_m3"][2]==0)


def test_continuous_xyz_temperature_and_complex_J_queries():
    h=case()
    state=np.full(len(h.cap),293.15)
    state[h.ns:h.ns+4]=[300.,301.,302.,303.]
    p=np.array([[0.,-.04,0.],[.02,.04,0.]])
    ans=h.query_temp(p,state)
    assert np.allclose(ans,[300.,303.],atol=1e-6)
    sea=h.query_temp(np.array([[0.,0.,0.]]),state)
    assert np.allclose(sea,[293.15])
    j=np.arange(h.ns*3).reshape(h.ns,3)*(1+.1j)
    assert np.allclose(h.query_sea(h.xyz[:1],j),j[:1])
