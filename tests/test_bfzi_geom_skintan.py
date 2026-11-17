import numpy as np
import torch

from sdfmpneo_bfzi.geom_skintan import SkinTan
from sdfmpneo_bfzi.geom_free import circuit


def test_skin_tangent_is_cold_zero_and_port_factorized():
    torch.manual_seed(3)
    net=SkinTan(ng=45,sa=np.ones(32),ss=np.ones(16))
    g=torch.randn(4,45)
    a=torch.zeros(4,32)
    s=torch.zeros(4,16)
    assert torch.count_nonzero(net(g,a,s))==0
    with torch.no_grad():
        for head in (net.tx,net.mx,net.rx):
            head[-1].weight.fill_(.01)
    a=torch.rand(4,32)
    s=torch.rand(4,16)
    y=net(g,a,s)
    s2=s.clone()
    s2[:,8:]+=1.
    y2=net(g,a,s2)
    assert y.shape==(4,6)
    assert torch.isfinite(y).all()
    assert torch.allclose(y[:,:2],y2[:,:2],atol=1e-7,rtol=0)
    assert not torch.allclose(y[:,4:],y2[:,4:])


def test_time_conditioned_tangent_receives_interval_voltage():
    class Port:
        omega=2*np.pi*200000
        air=np.zeros((2,2))
        def copper(self,temp):
            return np.zeros(2,dtype=complex)

    seen={}
    def tan(g,a,tm,voltage):
        seen["time"]=tm
        seen["voltage"]=voltage
        return torch.zeros((len(a),6),dtype=a.dtype)
    tan.time_conditioned=True
    init=dict(z0=np.array([1.,0.,0.,0.,1.,0.]),
              air=np.zeros(3),cap=np.ones(5),
              ext=np.zeros((2,2),complex))
    circuit(np.zeros(32),init,Port(),np.zeros(32),tan,
            torch.zeros((1,45)),np.ones(32),12.,293.15,
            tm=30.,heat_voltage=8.)
    assert seen=={"time":30.,"voltage":8.}


def test_v2_periodic_rotation_encoding():
    from sdfmpneo_bfzi.geom_skintan import encode_geom
    from sdfmpneo_bfzi.pose import _rot
    angles=[(23.,-11.,17.),(383.,349.,377.)]
    g=torch.zeros((2,45))
    for i,(yaw,pitch,tilt) in enumerate(angles):
        r=_rot('z',yaw)@_rot('y',pitch)@_rot('x',tilt)
        g[i,25:34]=torch.as_tensor(r.reshape(-1),dtype=torch.float32)
    x=encode_geom(g)
    assert x.shape==(2,42)
    assert torch.allclose(x[0,25:31],x[1,25:31],atol=2e-6,rtol=0)
    assert torch.allclose(x[:,:25],g[:,:25])
    assert torch.allclose(x[:,31:],g[:,34:])


def test_v2_tangent_is_cold_zero_and_finite():
    from sdfmpneo_bfzi.geom_skintan import SkinTanV2
    from sdfmpneo_bfzi.pose import _rot
    net=SkinTanV2(ng=45,sa=np.ones(32),ss=np.ones(16))
    g=torch.zeros((3,45))
    for i,(yaw,pitch,tilt) in enumerate(((0.,0.,0.),(15.,-8.,12.),(-20.,10.,-5.))):
        r=_rot('z',yaw)@_rot('y',pitch)@_rot('x',tilt)
        g[i,25:34]=torch.as_tensor(r.reshape(-1),dtype=torch.float32)
    a=torch.zeros((3,32)); s=torch.zeros((3,16))
    assert torch.count_nonzero(net(g,a,s))==0
    y=net(g,torch.rand_like(a),torch.rand_like(s))
    assert y.shape==(3,6) and torch.isfinite(y).all()


def test_v3_physical_encoder_omits_pose_and_cold_z():
    from sdfmpneo_bfzi.geom_skintan import encode_geom_v3,SkinTanV3
    from sdfmpneo_bfzi.pose import _rot
    g=torch.zeros((2,45))
    g[0,25:34]=torch.as_tensor(np.eye(3).reshape(-1),dtype=torch.float32)
    r=_rot('z',25.)@_rot('y',-12.)@_rot('x',8.)
    g[1,25:34]=torch.as_tensor(r.reshape(-1),dtype=torch.float32)
    g[1,16:22]=2.
    x=encode_geom_v3(g)
    assert x.shape==(2,30)
    assert torch.allclose(x[0],x[1])
    net=SkinTanV3(ng=45,sa=np.ones(32),ss=np.ones(16))
    a=torch.zeros((2,32)); skin=torch.zeros((2,16))
    assert torch.count_nonzero(net(g,a,skin))==0
    y=net(g,torch.rand_like(a),torch.rand_like(skin))
    assert y.shape==(2,6) and torch.isfinite(y).all()


def test_v4_tangent_ignores_geometry_and_keeps_port_heads():
    from sdfmpneo_bfzi.geom_skintan import SkinTanV4
    net=SkinTanV4(ng=45,sa=np.ones(32),ss=np.ones(16))
    g=torch.randn(3,45); g2=torch.randn(3,45)
    a=torch.rand(3,32); skin=torch.rand(3,16)
    with torch.no_grad():
        for head in (net.tx,net.mx,net.rx):
            head[-1].weight.fill_(.01)
    y=net(g,a,skin); y2=net(g2,a,skin)
    assert y.shape==(3,6)
    assert torch.allclose(y,y2,atol=1e-7,rtol=0)
    assert torch.isfinite(y).all()
