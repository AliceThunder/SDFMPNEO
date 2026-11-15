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
             source=lambda theta,I,sea:
                 (np.ones(n),None,None,None,None))
    t=np.array([0.,10.,20.])
    v=np.array([0.,45.,0.])
    T=np.ones((3,n))*rom.amb
    T[1]=rom.amb+np.arange(1,n+1)
    T[2]=rom.amb+np.arange(1,n+1)*.5
    z=np.tile(z0,(3,1))
    z[1:,0]+=.001
    file=tmp_path/'s5.npz'
    np.savez(file,time_s=t,voltage_v=v,temp_K=T,z6=z)
    row=read(file,rom)
    assert row['a'].shape==(3,n)
    assert np.allclose(row['temp_K'],T[:,order])
    assert np.allclose(row['a'],row['temp_K']-rom.amb)
    assert np.isclose(row['y'][1,7],
                      max(row['temp_K'][1,:rom.ns])-rom.amb)
    assert row['q'].shape==(3,n)
    assert feature(rom,45).shape==(45,)
