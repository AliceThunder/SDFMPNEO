"""BFZI seawater skin SOE must reproduce passive halfspace diffusion."""
import numpy as np
import pytest

from sdfmpneo_bfzi.sea_skin import SeaSkin,exact_step


def test_half_order_step_matches_seawater_analytic_solution():
    s=SeaSkin(dt_min=.5,t_max=3600,modes=32)
    for i in range(1200):
        rise=s.step(100.,.5)
        if i in (0,1,9,39,1199):
            ref=exact_step(100.,(i+1)*.5)
            assert abs(rise/ref-1)<.0005
    assert np.isclose(s.energy,60_000.)


def test_memory_with_pulse_and_reverse_flux():
    s=SeaSkin(dt_min=.5,t_max=3600,modes=32)
    q=np.array([80.,80.,-15.,-15.,100.,0.,30.,30.])
    dt=1.
    for n,v in enumerate(q,1):
        rise=s.step(v,dt)
        offsets=np.sqrt(n-np.arange(n))-np.sqrt(n-1-np.arange(n))
        ref=2/np.sqrt(np.pi*s.k*s.rhoc)*np.dot(q[:n],offsets)
        assert abs(rise-ref)<max(.00025,abs(ref)*.002)
    assert np.isclose(s.energy,np.sum(q))


def test_passive_contact_and_separate_package_faces():
    s=SeaSkin(shape=(3,))
    for _ in range(40):
        q=s.contact(np.array([1.,2.,-1.]),15.,.04)
        assert np.allclose(s.temp+.04*q,[1.,2.,-1.],atol=1e-12)
    assert np.all(s.energy[:2]>0) and s.energy[2]<0
    for w in np.geomspace(1e-4,5.,12):
        assert s.impedance(1j*w).real>0
    with pytest.raises(ValueError):
        SeaSkin(k=0)
    with pytest.raises(ValueError):
        s.step(np.array([1.,2.]),1.)
