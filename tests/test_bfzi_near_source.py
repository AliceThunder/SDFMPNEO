"""Near-source BFZI quadrature conservation without external BFZI."""
import numpy as np
from sdfmpneo_bfzi.near_source import local_j,local_q


def test_source_mean_and_weak_heat_power():
    w=np.array([.5,.5,1.,1.])
    cell=np.array([0,0,1,1])
    src=np.array([[[1.,0,0],[0,0,0]],[[3.,0,0],[0,0,0]],
                  [[1.,2,0],[0,1,0]],[[3.,0,0],[0,1,0]]])
    avg=np.array([src[:2].mean(axis=0),src[2:].mean(axis=0)])
    jbar=np.array([[1.+2j,1,0],[2.+.1j,3,1.]])
    sig=np.array([4.,3.])
    cur=np.array([2.+1j,1j])
    j=local_j(jbar,sig,cur,src,avg,cell,2.)
    assert np.allclose((w[:2,None]*j[:2]).sum(axis=0)/w[:2].sum(),jbar[0])
    assert np.allclose((w[2:,None]*j[2:]).sum(axis=0)/w[2:].sum(),jbar[1])
    qcell=np.array([2.,5.])
    vol=np.array([1.,2.])
    q,scale=local_q(j,sig,qcell,vol,cell,w)
    assert np.all(q>=0)
    assert np.allclose(np.bincount(cell,weights=w*q,minlength=2),qcell*vol)
    q2,_=local_q(j,sig,qcell,vol,cell,scale=scale)
    assert np.allclose(q,q2)


def test_uniform_source_returns_original_j():
    cell=np.array([0,0])
    avg=np.zeros((1,2,3))
    src=np.zeros((2,2,3))
    bar=np.array([[1j,2j,3j]])
    out=local_j(bar,np.array([5.]),np.array([1.+0j,0j]),src,avg,cell,.3)
    assert np.allclose(out,bar[0])
