"""Synthetic physical heat-mode alignment and zero-EM tangent-transfer smoke."""
from types import SimpleNamespace
import numpy as np
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.geom_tangent_transfer import rotate,label,TangentTransfer


def fake(j,mode=None):
    n=10
    vol=np.ones(n)
    xyz=np.column_stack((np.arange(n)*.02,np.zeros(n),np.zeros(n)))
    modes=np.eye(n,8) if mode is None else mode
    g=np.ones(16)
    z0=np.array([1,0,.1,0,1.1,0.],float)
    pose=np.r_[j*.001,np.zeros(17)]
    head=SimpleNamespace(xyz=xyz,vol=vol,ns=n,geom=g,z0=z0,
                         pose=pose,sea0=np.eye(2,dtype=complex),cap=np.ones(n))
    jac=np.stack([np.array([[.01+.001*j+.002j*1j,.001+.001j],
                             [.001+.001j,.02+.003*j+.001j*1j]])
                  for _ in range(8)])
    tan=SeaTangent(modes,jac,vol,head.sea0,g)
    rom=SimpleNamespace(phi=np.eye(n))
    return head,tan,rom


def test_heat_mode_sign_and_permutation_alignment():
    head,tan,_=fake(0)
    order=[5,2,6,4,1,3,7,0]
    sign=np.array([-1,1,-1,1,1,-1,1,-1])
    turned=tan.modes[:,order]*sign
    rotated=rotate(head.xyz,head.vol,tan.modes,head.xyz,turned)
    assert np.allclose(turned@rotated,tan.modes,atol=1e-12)
    moved=SeaTangent(turned,tan.jac[order]*sign[:,None,None],
                     tan.vol,tan.cold,tan.geom)
    got,_=label(head,tan,head,moved)
    expected,_=label(head,tan,head,tan)
    assert np.allclose(got,expected,atol=1e-12)


def test_train_and_predict_aligned_original_bfzi_tangent(tmp_path):
    entries=[]
    for j,name in enumerate(("P01","P02","P04","P05","P06")):
        head,tan,_=fake(j)
        entries.append((name,head,tan))
    net=TangentTransfer.train(entries,epochs=25)
    out=tmp_path/"tangent.npz"
    net.save(out)
    loaded=TangentTransfer.load(out)
    head,_,rom=fake(3)
    pred=loaded.predict(head,rom)
    assert pred.method=="neural-residual" and pred.nsolve==0
    assert pred.jac.shape==(8,2,2)
    assert np.isfinite(pred.jac).all()
    assert np.allclose(pred.delta(np.ones(head.ns)),
                       pred.delta(np.ones(head.ns)).T)
    assert np.allclose(pred.jac,net.predict(head,rom).jac)
