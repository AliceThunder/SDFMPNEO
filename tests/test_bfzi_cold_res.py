"""Cold residual port keeps frozen self and mutual physics at zero update."""
import numpy as np
import torch

from sdfmpneo_bfzi.geom_coldres import ColdRes
from sdfmpneo_bfzi.geom_portself import PortSelf


def _case(a,core):
    return dict(geometry=dict(alpha1=a,alpha2=a+.01,core1_mm=core,
                              core2_mm=core+1,box1_mm=220,box2_mm=222,
                              shift_mm=[0,90,0],tilt_deg=5,
                              yaw_deg=2,pitch_deg=-1))


def _model(path):
    rows=[]
    for i in range(7):
        case=_case(.72+.03*i,10+i)
        y=np.array([.2+i*.002,16+i*.2,.01,.9,
                    .21+i*.003,16.8+i*.2,
                    14e-6,1e-6,14.5e-6])
        rows.append(dict(case=case,y=y))
    model=PortSelf.fit(rows,dims=2,ell=2.5,lam=.01)
    model.save(path)
    return model


def test_zero_residual_preserves_original_port(tmp_path):
    path=tmp_path/'port.npz'
    base=_model(path)
    net=ColdRes(str(path),np.zeros((2,5)),
                np.ones((2,5)),np.ones((2,2)))
    case=_case(.82,15)
    init=dict(z0=np.zeros(6),air=np.zeros(3))
    a=base.apply(case,init)
    b=net.apply(case,init)
    np.testing.assert_allclose(a['z0'],b['z0'],atol=1e-8)
    np.testing.assert_allclose(a['air'],b['air'],atol=1e-12)


def test_tx_residual_changes_only_tx_and_air(tmp_path):
    path=tmp_path/'port.npz'
    base=_model(path)
    net=ColdRes(str(path),np.zeros((2,5)),
                np.ones((2,5)),np.ones((2,2)))
    with torch.no_grad():
        net.head[0][-1].bias[:] = torch.tensor([.08,.5])
    case=_case(.82,15)
    init=dict(z0=np.zeros(6),air=np.zeros(3))
    a=base.apply(case,init)
    b=net.apply(case,init)
    np.testing.assert_allclose(b['z0'][[0,1]]-a['z0'][[0,1]],
                               [.08,.5],rtol=0,atol=1e-7)
    np.testing.assert_allclose(b['z0'][2:],a['z0'][2:],atol=1e-8)
    np.testing.assert_allclose(b['air'][0]-a['air'][0],
                               .5/(2*np.pi*200000),atol=1e-12)
    np.testing.assert_allclose(b['air'][1:],a['air'][1:],atol=1e-12)
