"""BFZI one-sided thermal impedance tangent: cost, accuracy and persistence."""
from types import SimpleNamespace
import numpy as np
import pytest

from sdfmpneo_bfzi.geom_tangent import SeaTangent


def fixture():
    rng=np.random.default_rng(7)
    n=30
    vol=np.full(n,.02)
    phi=rng.normal(size=(n,8))
    cold=np.array([[.31+.28j,.12+.13j],[.12+.13j,.17+.35j]])
    deriv=np.array([[.004+.002j,.001+.0005j],
                    [.001+.0005j,.002+.001j]])
    head=SimpleNamespace(vol=vol,ns=n,amb=293.15,sea0=cold,
                         geom=np.arange(16,dtype=float))
    return head,phi,cold,deriv


class EM:
    def __init__(self,cold,deriv,vol,power=0.):
        self.cold,self.deriv,self.vol,self.power=cold,deriv,vol,power
        self.calls=0

    def solve(self,sea,cu):
        self.calls+=1
        temp=np.asarray(sea)-293.15
        v=self.vol@temp
        return SimpleNamespace(sea_z=self.cold+
                               self.deriv*(v+self.power*v*v))


def test_one_sided_uses_half_original_bfzi_solves(tmp_path):
    head,phi,cold,deriv=fixture()
    em=EM(cold,deriv,head.vol)
    exact=EM(cold,deriv,head.vol)
    one=SeaTangent.build(em,head,phi,rank=8,amplitude=.5,
                         method="forward")
    two=SeaTangent.build(exact,head,phi,rank=8,amplitude=.5,
                         method="central")
    assert em.calls==one.nsolve==8
    assert exact.calls==two.nsolve==16
    assert np.allclose(one.modes,two.modes)
    assert np.allclose(one.jac,two.jac,atol=1e-10)
    for temp in (np.ones(30)*2,np.random.default_rng(4).normal(size=30)):
        assert np.allclose(one.delta(temp),two.delta(temp),atol=1e-10)
    path=tmp_path/"tangent.npz"
    one.save(path)
    read=SeaTangent.load(path,head)
    assert read.method=="forward" and read.nsolve==8
    assert read.amplitude==.5
    assert np.allclose(read.delta(np.ones(30)),one.delta(np.ones(30)))


def test_nonlinear_forward_error_drops_with_smaller_step():
    head,phi,cold,deriv=fixture()
    coarse=SeaTangent.build(EM(cold,deriv,head.vol,power=.01),
                            head,phi,amplitude=.5,method="forward")
    fine=SeaTangent.build(EM(cold,deriv,head.vol,power=.01),
                          head,phi,amplitude=.05,method="forward")
    central=SeaTangent.build(EM(cold,deriv,head.vol,power=.01),
                             head,phi,amplitude=.5)
    jac=np.einsum("nj,n,ab->jab",coarse.modes,head.vol,deriv)
    assert np.linalg.norm(fine.jac-jac)<.2*np.linalg.norm(coarse.jac-jac)
    assert np.linalg.norm(central.jac-jac)<1e-9


def test_invalid_method_and_legacy_tangent(tmp_path):
    head,phi,cold,deriv=fixture()
    em=EM(cold,deriv,head.vol)
    with pytest.raises(ValueError):
        SeaTangent.build(em,head,phi,method="unknown")
    with pytest.raises(ValueError):
        SeaTangent.build(em,head,phi,amplitude=-1)
    old=SeaTangent.build(em,head,phi,rank=4)
    path=tmp_path/"legacy.npz"
    np.savez_compressed(path,modes=old.modes,jac=old.jac,
                        vol=old.vol,cold=old.cold,geometry=old.geom)
    read=SeaTangent.load(path,head)
    assert read.method=="central" and read.nsolve==8
