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



def test_multigrid_teacher_split_and_backprop(tmp_path):
    import json
    from examples.bfzi_frac import dataset
    paths=[]
    ncase=6
    train=np.array([True,True,True,True,False,False])
    cases=[dict(id=i,voltage_v=float(i+5),
                windows_s=[[0,30],[45,60]],
                split="train" if train[i] else "test")
           for i in range(ncase)]
    for step in (30,15):
        t=np.arange(0,60+step,step,dtype=float)
        volt=np.tile(np.array([5.]*len(t)),(ncase,1))
        volt[:,(t>30)&(t<=45)]=0
        temp=np.full((ncase,len(t),10),293.15)
        z=np.broadcast_to(
            np.array([.25,20.,.1,6.,.25,20.]),
            (ncase,len(t),6)).copy()
        for i in range(ncase):
            temp[i]+=((i+1)*t[:,None]/60)*np.linspace(.02,2.,10)
            z[i,:,0]+=(i+1)*t*1e-4
        path=tmp_path/f"teacher_{step}.npz"
        np.savez_compressed(
            path,time_s=t,voltage_v=volt,
            temp_K=temp,z6=z,
            currents_A=np.zeros((ncase,len(t),2),complex),
            eta=np.zeros((ncase,len(t))),
            train=train,nsea=4,npkg=2,
            ambient_K=293.15,external_z=np.eye(2)*.5,
            cases_json=json.dumps(cases))
        paths.append(path)
    d=dataset(paths,rank=5)
    assert len(d["grids"])==2
    assert list(d["train"])==[0,1,2]
    assert list(d["val"])==[3]
    assert list(d["test"])==[4,5]
    assert d["basis"].shape==(10,5)
    assert d["grids"][0]["y"].shape==(6,3,11)
    assert d["grids"][1]["y"].shape==(6,5,11)
    net=FracNet(11,"hybrid")
    preds=[net(torch.tensor(g["src"][:3],dtype=torch.float32),
               g["times"]) for g in d["grids"]]
    loss=sum(torch.mean(p**2) for p in preds)
    loss.backward()
    assert torch.isfinite(net.raw.grad).all()
    assert torch.isfinite(net.raw_tau.grad).all()


def test_sea_field_head_power_and_saved_inference(tmp_path):
    from sdfmpneo_bfzi.frac_field import SeaHead
    vol=np.array([.1,.2,.1,.2])
    basis=np.array([[1.],[0.],[0.],[0.]])
    j0=np.ones((4,2,3),complex)*(.3+.1j)
    q0=np.ones(4)/vol.sum()
    z0=np.array([.2,-.01,.1,-.002,.2,-.01])
    head=SeaHead(vol,basis,np.array([1.]),j0,q0,z0,
                 np.zeros((1,24),complex),
                 np.zeros((1,4)),np.zeros((1,6)))
    a=np.array([3+.2j,.5-.3j])
    j,q,z=head.at(np.full(4,293.15),a)
    assert j.shape==(4,3)
    assert q.shape==(4,)
    assert np.all(q>=0)
    assert np.isclose(q@vol,.5*np.vdot(a,z@a).real,rtol=1e-12)
    j0,q0,z0=head.at(np.full(4,293.15),np.zeros(2,complex))
    assert np.all(j0==0) and np.all(q0==0)
    file=tmp_path/"field.npz"
    head.save(file)
    j1,q1,z1=SeaHead.load(file).at(np.full(4,293.15),a)
    assert np.allclose(j1,j)
    assert np.allclose(q1,q)
    assert np.allclose(z1,z)
