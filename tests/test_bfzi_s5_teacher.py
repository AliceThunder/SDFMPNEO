"""S5 teacher full thermal fields, material inputs and restored skin nodes."""
import numpy as np
from types import SimpleNamespace as NS

from sdfmpneo_bfzi.s5_bridge import state_order, feature
from examples.bfzi_geom_neural import read


def test_s5_teacher_reorders_without_losing_skin(tmp_path):
    n = 13
    port = NS(copper=lambda T: np.zeros(2,complex), omega=1.,
              air=np.zeros((2,2)))
    z0 = np.array([.13,.02,.04,.005,.15,.04])
    order = state_order(4,2,3)
    rom = NS(amb=293.15,cap=np.ones(n),phi=np.eye(n),ns=7,npkg=2,
             cu=slice(11,13),vol=np.ones(7),port=port,z0=z0,
             ext=np.eye(2),state_order=order,geom=np.ones(16),
             pose=np.ones(18),thermal=np.array([.02,.25,.001,.25,20.]),
             source=lambda *args: (_ for _ in ()).throw(
                 AssertionError("read must not compute teacher source")))
    t=np.array([0.,10.,20.])
    v=np.array([0.,45.,0.])
    T=np.ones((3,n))*rom.amb
    T[1]=rom.amb+np.arange(1,n+1)
    T[2]=rom.amb+np.arange(1,n+1)*.5
    z=np.tile(z0,(3,1))
    z[1:,0]+=.001
    file=tmp_path/'s5.npz'
    np.savez(file,time_s=t,voltage_v=v,temp_K=T,z6=z,
             Tpkg_K=T[:,4:8],Tcu_K=T[:,8:10])
    row=read(file,rom)
    assert row['a'].shape==(3,n)
    assert np.allclose(row['temp_K'],T[:,order])
    assert np.allclose(row['a'],row['temp_K']-rom.amb)
    assert np.isclose(row['y'][1,7],
                      max(row['temp_K'][1,:rom.ns])-rom.amb)
    assert 'q' not in row
    assert feature(rom,45).shape==(45,)


def test_40d_geomfrac_weights_expand_to_45d():
    import torch
    from sdfmpneo_bfzi.geom_memory import GeomFrac
    from sdfmpneo_bfzi.geom_neural import GeomFracET
    from examples.bfzi_geom_neural import load_init
    torch.manual_seed(5)
    src=GeomFrac(ng=40)
    dst=GeomFracET(ng=45,rank=5)
    load_init(dst,{'ng':40,'port_sea':True,'state':src.state_dict()})
    h=src.init_state(1)
    u=torch.tensor([[.8,.1,.2]])
    g=torch.rand((1,40))
    extra=torch.rand((1,5))
    _, orig=src.step(h,u,15.,g)
    _, new=dst.step(h,u,15.,torch.cat((g,extra),dim=1))
    assert torch.allclose(orig,new,atol=1e-7)
    assert torch.allclose(dst.net[0].weight[:,-5:],
                          torch.zeros_like(dst.net[0].weight[:,-5:]))


def test_s5_teacher_detects_bad_package_reorder(tmp_path):
    from sdfmpneo_bfzi.s5_bridge import state_order
    from types import SimpleNamespace as NS
    n=13
    z0=np.array([.13,.02,.04,.005,.15,.04])
    port=NS(copper=lambda T: np.zeros(2,complex), omega=1.,
            air=np.zeros((2,2)))
    rom=NS(amb=293.15,cap=np.ones(n),phi=np.eye(n),ns=7,npkg=2,
           cu=slice(11,13),vol=np.ones(7),port=port,z0=z0,ext=np.eye(2),
           state_order=state_order(4,2,3),
           source=lambda *args:(np.ones(n),None,None,None,None))
    T=np.full((2,n),293.15)
    T[1,4:8]=[300,301,302,303]
    z=np.tile(z0,(2,1))
    path=tmp_path/'wrong.npz'
    np.savez(path,time_s=np.array([0.,15.]),voltage_v=np.array([0.,40.]),
             temp_K=T,z6=z,Tpkg_K=T[:,4:8]+.1,Tcu_K=T[:,8:10])
    import pytest
    with pytest.raises(ValueError,match="package cells"):
        read(path,rom)
