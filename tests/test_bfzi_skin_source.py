"""Temperature-dependent BFZI quadrature and conservative skin Joule heat."""
import numpy as np
from sdfmpneo_bfzi.near_source import local_j_var,split_heat


def test_var_sigma_native_mean():
    cell=np.array([0,0,1,1])
    w=np.array([.5,.5,1.,1.])
    src=np.array([[[1.,0,0],[0,2,0]],[[3.,0,0],[0,4,0]],
                  [[0,1,0],[3,0,0]],[[1,1,0],[1,0,0]]],float)
    sig=np.array([2.,4.,6.,5.])
    ref=np.array([np.einsum("q,qpc->pc",w[cell==i]*sig[cell==i],
                           src[cell==i])/sum(w[cell==i]) for i in (0,1)])
    avg=np.array([[1.+1j,2.,3.],[2.,1j,0.]])
    cur=np.array([2j,1.+.5j])
    out=local_j_var(avg,sig,cur,src,ref,cell,.25)
    for i in (0,1):
        use=cell==i
        assert np.allclose(np.average(out[use],axis=0,weights=w[use]),avg[i])


def test_resolved_skin_heat_conserves_each_wet_fv():
    q=np.array([2.,8.])
    vol=np.array([4.,3.])
    bulk_vol=np.array([3.,1.])
    cell=np.array([0,0,1,1])
    node=np.array([0,1,1,2])
    wt=np.array([.5,.5,1.,1.])
    raw_bulk=np.array([2.,1.])
    raw_skin=np.array([3.,7.,2.,8.])
    bulk,skin=split_heat(q,vol,bulk_vol,raw_bulk,cell,node,wt,raw_skin,3)
    assert np.all(bulk>=0) and np.all(skin>=0)
    s=wt*raw_skin
    den=bulk_vol*raw_bulk+np.bincount(cell,weights=s,minlength=2)
    bycell=bulk+np.bincount(cell,weights=s*(q*vol/den)[cell],minlength=2)
    assert np.allclose(bycell,q*vol)
    assert np.isclose(bulk.sum()+skin.sum(),q@vol)


def test_uniform_local_joule_reverts_to_volume_heat():
    q=np.array([2.,3.])
    vol=np.array([2.,3.])
    bulk_vol=np.array([1.5,2.])
    bulk,skin=split_heat(q,vol,bulk_vol,np.ones(2),
        np.array([0,1]),np.array([0,1]),
        np.array([.5,1.]),np.ones(2),2)
    assert np.allclose(bulk,q*bulk_vol)
    assert np.allclose(skin,q*np.array([.5,1.]))
