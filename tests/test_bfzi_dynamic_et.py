"""Minimal deterministic V2.1 thermal-to-complex-Z feedback tests."""
from types import SimpleNamespace
import numpy as np
import pytest

from sdfmpneo_bfzi.dynamic_et import BFZIDynamicET


class Box:
    half_extents=(.1,.1,.02)

    def contains(self,x,pose):
        return np.all(abs(np.asarray(x)-pose.center)<=self.half_extents,axis=-1)

    def squared_exterior_distance(self,x,pose):
        z=np.maximum(abs(np.asarray(x)-pose.center)-self.half_extents,0)
        return (z*z).sum(axis=-1),z


class Sources:
    def __init__(self):
        self.omega=1.256637e6
        self.inductance_air=np.array([[3e-6,1e-6],[1e-6,3e-6]])
        assys=[
            SimpleNamespace(enclosure=Box(),
                pose=SimpleNamespace(center=np.array([-.25,0,0])),
                wire_radius=.001),
            SimpleNamespace(enclosure=Box(),
                pose=SimpleNamespace(center=np.array([.25,0,0])),
                wire_radius=.001)]
        self.scene=SimpleNamespace(
            seawater=SimpleNamespace(conductivity=5.),
            outer_radius=.5,outer_center=(0.,0.,0.),assemblies=assys)
        self.lines=[SimpleNamespace(
            line_elements=np.array([[.4,0,0],[.4,0,0]]))]*2

    def conductor_internal_impedance(self,temp):
        return (.12+.0005*(np.asarray(temp)-293.15)+1j*.01)*np.ones(2)


def make_et():
    x=np.array([[-.40,0,0],[-.23,.12,0],[0,0,.2],
                [.23,.12,0],[.40,0,0],[0,0,-.2]])
    w=np.ones(len(x))*.01
    modes=np.zeros((len(x),2,3))
    modes[:,0,0]=[1,1,0,.1,.1,.5]
    modes[:,1,1]=[.1,.1,0,1,1,.5]
    modes[:,1,0]=[.2,.2,0,.2,.2,.2]
    ref=SimpleNamespace(
        sources=Sources(),integration_points=x,
        integration_weights=w,integration_modes=modes,
        inductance=np.diag([2e-8,2e-8]),
        coupling=np.eye(2)*5e-8)
    return BFZIDynamicET(ref,sea_step=.12,sea_alpha=.05,
                          sea_rhoc=1,pkg_rhoc=1,cu_rhoc=1,
                          cu_pkg_g=.01,pkg_sea_g=.01)


def test_warm_seawater_changes_real_imag_and_mutual_z():
    et=make_et()
    cold=et.zeros()
    warm=cold.copy()
    warm[:et.count]+=15
    z0,s0,i0,heat,psea,pcu=et.em(cold,[.3,0])
    z1,s1,_,_,_,_=et.em(warm,[.3,0])
    assert abs((s1-s0)[0,0].real)>1e-6
    assert abs((s1-s0)[0,0].imag)>1e-6
    assert abs((s1-s0)[0,1].real)>1e-8
    assert abs((s1-s0)[0,1].imag)>1e-8
    assert np.allclose(s0,s0.T,atol=1e-12)
    assert np.isclose(psea,.5*np.real(np.vdot(i0,s0@i0)),rtol=1e-11)
    assert np.isclose(heat.sum(),psea+pcu.sum(),rtol=1e-11)
    assert np.allclose(z1-z0,s1-s0,atol=1e-12)


def test_dynamic_on_off_cooling_and_history():
    et=make_et()
    t=np.array([0,2,4,8,12],float)
    trace=et.trajectory(t,lambda now: np.array(
        [.3 if now<=4 else 0.,0.]))
    assert trace[2].temp.max()>trace[0].temp.max()
    assert trace[-1].temp.max()<trace[2].temp.max()
    assert np.allclose(trace[-1].currents,0)
    assert np.allclose(trace[-1].copper_power,0)
    assert trace[2].energy>0
    assert trace[2].sea_q.shape==(et.count,)
    assert np.all(trace[2].sea_q >= 0)
    assert np.isfinite(trace[2].impedance).all()
    pts=np.array([[-.4,0,0],[.25,0,0]])
    assert et.temperature_at(pts,trace[2].temp).shape==(2,)
    with pytest.raises(ValueError):
        et.em(et.zeros(),[1,2,3])
