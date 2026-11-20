"""Cold-port copper split keeps winding and external lead losses separate."""
import numpy as np
import torch

from sdfmpneo_bfzi.frac_port import PortModel
from sdfmpneo_bfzi.geom_coldnet import ColdNet,copper_target,target
from sdfmpneo_bfzi.geom_free import circuit
from sdfmpneo_bfzi.geom_heat import pack
from sdfmpneo_bfzi.geom_memory import heat_drive


def test_coldnet_optional_copper_output():
    g=dict(alpha1=.85,alpha2=.93,core1_mm=10,core2_mm=15,
           shift_mm=[2,90,-4],tilt_deg=5,yaw_deg=2,pitch_deg=3,
           box1_mm=220,box2_mm=230)
    row=dict(geometry=g,z0=[.25,18.,.03,1.2,.27,19.],
             air=[14e-6,1e-6,15e-6],
             copper=[[.20,.01],[.21,.012]])
    base=target(row)
    y=np.r_[base,copper_target(row)]
    for vals in (base,y):
        net=ColdNet(np.zeros(16),np.ones(16),vals,np.ones(len(vals)),hidden=8)
        for p in net.parameters():
            torch.nn.init.zeros_(p)
        z,air,cu=net.predict_parts(g)
        np.testing.assert_allclose(z,row["z0"],rtol=1e-6,atol=1e-6)
        assert (cu is None)==(len(vals)==9)
        if cu is not None:
            np.testing.assert_allclose(cu,[.20+.01j,.21+.012j],rtol=1e-6)


def test_terminal_copper_split_preserves_port_and_sea_loss():
    amb=293.15
    omega=2*np.pi*200000
    air=np.array([[15e-6,1e-6],[1e-6,16e-6]])
    port=PortModel(omega,air,[4.,4.],[.00079,.00079],
                   [5.8e7]*2,[.00393]*2,[4e-7*np.pi]*2,[amb]*2)
    wind=port.copper(np.array([amb,amb]))
    lead=np.array([.012+.002j,.016+.003j])
    sea=np.array([[.02+.003j,.004+.001j],[.004+.001j,.025+.002j]])
    total=sea+1j*omega*air+np.diag(wind+lead)
    init=dict(z0=pack(total),air=np.array([air[0,0],air[0,1],air[1,1]]),
              cap=np.ones(5),ext=np.eye(2,dtype=complex)*.5,
              copper0=wind+lead)
    state=np.zeros(5)
    x=(None,init,port,state,None,torch.zeros((1,45)),np.ones(5),12.,amb)
    z,cur,ps,pc,_=circuit(*x)
    np.testing.assert_allclose(z,total,rtol=0,atol=1e-12)
    np.testing.assert_allclose(ps,.5*np.vdot(cur,sea.real@cur).real,
                               rtol=1e-12)
    q=heat_drive(12.,z,init["ext"],port,np.array([amb,amb]),
                 np.ones(2),lead=lead)
    np.testing.assert_allclose(q[0],ps,rtol=1e-12)
    legacy=dict(init);legacy.pop("copper0")
    zold,iold,psold,pcold,_=circuit(None,legacy,port,state,None,
                                    torch.zeros((1,45)),np.ones(5),12.,amb)
    np.testing.assert_allclose(zold,z,rtol=0,atol=1e-12)
    np.testing.assert_allclose(iold,cur,rtol=0,atol=1e-12)
    np.testing.assert_allclose(pcold,pc,rtol=0,atol=1e-12)
    assert psold>ps
    hot=state.copy();hot[3:5]=[20.,15.]
    zh,_,_,_,_=circuit(None,init,port,hot,None,
                        torch.zeros((1,45)),np.ones(5),12.,amb)
    zhold,_,_,_,_=circuit(None,legacy,port,hot,None,
                          torch.zeros((1,45)),np.ones(5),12.,amb)
    np.testing.assert_allclose(zh,zhold,rtol=0,atol=1e-12)
