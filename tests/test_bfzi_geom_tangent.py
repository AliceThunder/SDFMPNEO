"""Original BFZI electromagnetic thermal tangent in physical 3-D sea coordinates."""
import numpy as np
from types import SimpleNamespace
from sdfmpneo_bfzi.geom_tangent import SeaTangent,basis


class ExactEM:
    def __init__(self,z,d,vec):
        self.z=z
        self.d=d
        self.vec=vec

    def solve(self,sea,cu):
        return SimpleNamespace(sea_z=self.z+
                self.d*(self.vec@(sea-293.15)))


def test_cold_bfzi_tangent_exact_linear_geometry(tmp_path):
    vol=np.full(20,.01)
    v=np.linspace(-1.,1.,len(vol))
    v/=np.sqrt((v*v*vol).sum())
    p=np.column_stack((v,np.ones_like(v)))
    z=np.array([[.2+.3j,.05+.2j],[.05+.2j,.1+.3j]])
    dz=np.array([[.002+.001j,.001+0j],[.001+0j,.003+.0005j]])
    em=ExactEM(z,dz,v*vol)
    h=SimpleNamespace(vol=vol,ns=len(vol),amb=293.15,
                      sea0=z,geom=np.arange(16,dtype=float))
    tan=SeaTangent.build(em,h,p,rank=2)
    temp=2*v+1
    assert np.allclose(tan.delta(temp),dz*(v@(vol*temp)))
    assert tan.capture(temp)<1e-8
    tan.save(tmp_path/"t.npz")
    loaded=SeaTangent.load(tmp_path/"t.npz",h)
    assert np.allclose(loaded.delta(temp),tan.delta(temp))


def test_volume_orthonormal_tangent_modes():
    x=basis(np.ones(6),np.eye(6),rank=3)
    assert x.shape==(6,3)
    assert np.allclose(x.T@x,np.eye(3))
