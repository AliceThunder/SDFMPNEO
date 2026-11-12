"""Fast causal-memory checks, independent of the private BFZI checkout."""
import numpy as np
import torch

from sdfmpneo_bfzi.frac_memory import FracNet,build_sources,predict,zmat


def test_fractional_memory_distinguishes_histories():
    torch.set_num_threads(2)
    net=FracNet(10,"frac")
    t=np.arange(0,601,60,dtype=float)
    fresh=np.zeros((2,len(t),3),float)
    fresh[0,1:5,0]=1.
    fresh[1,1:3,0]=1.
    a=net(torch.tensor(fresh,dtype=torch.float32),t)
    assert np.array_equal(a[:,0].detach().numpy(),np.zeros((2,10)))
    assert torch.linalg.vector_norm(a[0,7]-a[1,7])>1e-7
    a.sum().backward()
    assert torch.isfinite(net.raw.grad).all()
    assert torch.linalg.vector_norm(net.raw.grad)>0
    alpha=net.orders().detach().numpy()
    assert np.all(alpha>0) and np.all(alpha<1)


def test_single_exponential_memory_and_circuit():
    net=FracNet(7,"exp")
    u=torch.ones((1,3,3),dtype=torch.float32)
    out=net(u,np.array([0.,5.,20.]))
    assert torch.isfinite(out).all()
    assert out.shape==(1,3,7)
    assert torch.all(out[:,0]==0)
    zr=np.array([1.,2.,.2,.1,2.,3.])
    m=zmat(zr)
    assert m.shape==(2,2) and m[0,1]==m[1,0]
    ex=np.eye(2,dtype=complex)*.5
    v=np.array([[10.,0.,10.]],float)
    zz=np.broadcast_to(zr,(1,3,6))
    src=build_sources(v,zz,ex)
    assert np.all(src>=0)
    assert np.all(src[:,1]==0)
    assert np.all(src[:,2]>0)


def test_fractional_predict_is_cold_exact():
    net=FracNet(9,"frac")
    basis=torch.zeros((6,3))
    basis[0,0]=1
    basis[4,1]=1
    basis[5,2]=1
    ck=dict(
        external_z=torch.eye(2,dtype=torch.complex64)*.5,
        z0=torch.tensor([.4,2.,.1,.2,.4,2.]),
        t_scale=torch.ones(6),
        basis=basis,
        y_scale=torch.ones(9),
        src_scale=torch.ones(3),
        ambient=293.15)
    v=np.array([10.,10.,0.,10.])
    res=predict(net,ck,v,np.array([0.,5.,10.,20.]))
    assert np.array_equal(res["z6"][0],ck["z0"].numpy())
    assert np.allclose(res["temp_K"][0],293.15)
    assert np.array_equal(res["current_A"][2],np.zeros(2))
    assert res["eta"][2]==0.


def test_hybrid_fractional_and_exponential_memory():
    net=FracNet(8,"hybrid")
    t=np.array([0.,30.,60.,120.])
    src=torch.zeros((2,len(t),3),dtype=torch.float32)
    src[0,1,2]=4.
    src[1,1:3,2]=2.
    y=net(src,t)
    assert y.shape==(2,len(t),8)
    assert torch.all(y[:,0]==0)
    y.sum().backward()
    assert torch.linalg.vector_norm(net.raw.grad)>0
    assert torch.linalg.vector_norm(net.raw_tau.grad)>0
    assert torch.isfinite(y).all()
