"""BFZI sea thermal tangent recovered from archived physical 3-D histories."""
from types import SimpleNamespace
import numpy as np
import pytest

from sdfmpneo_bfzi.geom_tangent import basis
from sdfmpneo_bfzi.geom_tangent_data import data,fit


class Port:
    def copper(self,t):
        d=np.asarray(t)-293.15
        return np.array([.2+.01j,.3+.015j]) + (.002+.001j)*d


def make(tmp_path):
    rng=np.random.default_rng(19)
    n=13
    vol=np.ones(n-2)
    phi=np.eye(n)[:,:8]
    modes=basis(vol,phi[:len(vol)],8)
    jac=rng.normal(size=(8,6))*2e-3
    cold=np.array([1,.1,.04,.02,1.2,.11])
    head=SimpleNamespace(
        ns=n-2,vol=vol,cap=np.ones(n),cu=slice(n-2,n),
        amb=293.15,geom=np.arange(16),z0=cold,
        sea0=np.eye(2),port=Port())
    theta=np.zeros((5,26,n))
    z=np.broadcast_to(cold,(5,26,6)).copy()
    for i in range(5):
        for k in range(1,26):
            beta=rng.normal(size=8)*(.2*k/25)
            theta[i,k,:head.ns]=modes@beta
            theta[i,k,head.cu]=[.7*k/25,.4*k/25]
            dz=beta@jac
            dc=head.port.copper(293.15+theta[i,k,head.cu])-head.port.copper(
                np.full(2,293.15))
            dz[0]+=dc[0].real;dz[1]+=dc[0].imag
            dz[4]+=dc[1].real;dz[5]+=dc[1].imag
            z[i,k]+=dz
    p=tmp_path/"train.npz"
    np.savez_compressed(p,temp_K=293.15+theta,z6=z,
                        geometry=head.geom)
    return head,phi,p


def test_recovers_original_seawater_jacobian_without_em_solves(tmp_path):
    head,phi,path=make(tmp_path)
    tan,stats=fit(head,phi,path,rank=8)
    x,y=data(head,tan.modes,path)
    pred=x@np.column_stack([
        tan.jac[:,0,0].real,tan.jac[:,0,0].imag,
        tan.jac[:,0,1].real,tan.jac[:,0,1].imag,
        tan.jac[:,1,1].real,tan.jac[:,1,1].imag])
    assert tan.nsolve==0
    assert stats["identified_modes"]==8
    assert np.linalg.norm(pred-y)/np.linalg.norm(y)<1e-6


def test_rejects_different_geometry_history(tmp_path):
    head,phi,path=make(tmp_path)
    head.geom=head.geom+1
    with pytest.raises(ValueError,match="geometry"):
        fit(head,phi,path)
