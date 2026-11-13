"""Optional original BFZI FV J correction must preserve cold J and q power."""
import numpy as np
from scipy.sparse import csr_matrix

from test_bfzi_geom_heat import case
from sdfmpneo_bfzi.geom_heat import GeomHeat


def test_fv_J_cold_identity_and_archive(tmp_path):
    head=case()
    n=head.ns
    pairs=[(0,1,0,1.,1.,1.),(1,2,1,1.,1.,1.),
           (2,3,2,1.,1.,1.)]
    B=csr_matrix(([-1.,1.]*3,
                  ([0,0,1,1,2,2],[0,1,1,2,2,3])),
                 shape=(3,n))
    head.sea_proj=dict(
        edges=np.asarray(pairs,float),weights=np.ones(3),
        bdata=B.data.copy(),bind=B.indices.copy(),bptr=B.indptr.copy(),
        bshape=np.asarray(B.shape),sigma0=np.asarray(4.),
        omega_mu=np.asarray(2.))
    path=tmp_path/"anchor.npz"
    head.save(path)
    loaded=GeomHeat.load(path)
    assert loaded.sea_proj is not None
    I=np.array([2+1j,-.3j])
    theta=np.zeros(len(loaded.cap))
    ref=np.einsum("npc,p->nc",loaded.unit,I)
    assert np.allclose(loaded.corrected_J(theta,I),ref,atol=1e-12)
    assert np.all(loaded.corrected_J(theta,np.zeros(2))==0)
    theta[:n]=[2.,3.,5.,7.]
    new=loaded.corrected_J(theta,I)
    assert new.shape==(n,3)
    assert np.all(np.isfinite(new))
    assert np.linalg.norm(new-ref)>0
    _,q,_,psea,_=loaded.source(theta,I,loaded.sea0)
    assert np.isclose(q@loaded.vol,psea,atol=1e-11)
