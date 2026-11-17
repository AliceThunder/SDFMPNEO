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
