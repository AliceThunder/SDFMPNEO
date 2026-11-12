"""Small direct tests for BFZI v148 spatial-temperature integration.

The original BFZI repository has a separate dependency tree. These tests
exercise the adapter's weighted FV projector and the time integrator without
requiring heavy full v148 Green matrix construction.
"""
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp

from sdfmpneo_bfzi.bfzi_v148_adapter import WeightedFV
from sdfmpneo_bfzi.v148_et import BFZIV148ET


class Grid:
    def __init__(self):
        # Four sea FV volumes, with a physical closed face loop.
        self.n = 4
        self.meta = {}
        self.edges = [
            (0,1,0,1.,2.,1.),
            (2,3,0,1.,2.,1.),
            (0,2,1,1.,1.,1.),
            (1,3,1,1.,1.,1.),
        ]
        self.weights = np.array([a*d for i,j,k,d,a,s in self.edges])
        rows,cols,vals=[],[],[]
        for e,(i,j,*_) in enumerate(self.edges):
            rows.extend([e,e])
            cols.extend([i,j])
            vals.extend([-1.,1.])
        self.B=sp.csr_matrix(
            (vals,(rows,cols)),shape=(len(self.edges),self.n))
        self.boundary=[(i,0,1,1.,True) for i in range(4)]


def test_temperature_weighted_fv_projection():
    base=Grid()
    cold=WeightedFV(base,np.ones(4))
    scaled=WeightedFV(base,np.ones(4)*5.)
    warm=WeightedFV(base,np.array([1.,2.,4.,8.]))
    field=np.array([
        [1+1j,.2,0],[2-1j,.5+1j,0],
        [-1+2j,1.1-2j,0],[.5+.5j,-.3,0]
    ])
    a=cold.project_one(field)
    b=scaled.project_one(field)
    c=warm.project_one(field)
    assert np.allclose(a,b,atol=1e-12)
    assert np.linalg.norm(a-c)/np.linalg.norm(a) > .01


class Model:
    def __init__(self):
        self.proj=Grid()
        self.points=np.array([
            [-.28,0,0],[-.18,0,0],[.18,0,0],[.28,0,0]])
        self.vol=np.ones(4)*.01
        self.omega=1.256637e6
        fam=[
            SimpleNamespace(coil=SimpleNamespace(center=(-.2,0,0))),
            SimpleNamespace(coil=SimpleNamespace(center=(.2,0,0))),
        ]
        self.base={
            "families":fam,"sample":{},
            "cfg":{"BSR_PACKAGE_NORMAL_HALF_M":.0055}
        }
        lines=[
            SimpleNamespace(line_elements=np.array([[.1,0,0]])),
            SimpleNamespace(line_elements=np.array([[.1,0,0]])),
        ]
        assys=[
            SimpleNamespace(wire_radius=.001),
            SimpleNamespace(wire_radius=.001),
        ]
        self.copper=SimpleNamespace(lines=lines,
            scene=SimpleNamespace(assemblies=assys))

    def solve(self,sea_t,cu_t):
        rise=np.mean(sea_t)-293.15
        sea_z=np.array([
            [.06+.002*rise+1j*.01,.01+.001j*rise],
            [.01+.001j*rise,.06+.002*rise+1j*.01]
        ])
        cu_z=np.array([.1,.1])+.001*(cu_t-293.15)
        return SimpleNamespace(
            sea_z=sea_z,copper_z=cu_z,
            total_z=sea_z+np.diag(cu_z))

    def heat(self,state,currents):
        q=np.ones(4)*.3*abs(currents[0])**2
        cu=.5*state.copper_z.real*abs(currents)**2
        return q,cu


def test_v148_fv_heat_and_off_cooling():
    m=Model()
    et=BFZIV148ET(m,sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
                  sea_k=.05,cu_pkg_g=.01,pkg_sea_g=.01,outer_h=.1,
                  external=np.eye(2)*.5)
    t=np.array([0.,1.,2.,5.,8.])
    steps=et.trajectory(t,lambda tm: [1 if tm<=2 else 0,0])
    assert steps[2].temp.max()>0
    assert steps[-1].temp.max()<steps[2].temp.max()
    assert np.allclose(steps[-1].currents,0)
    assert np.allclose(steps[-1].sea_q,0)
    assert steps[2].energy>0
    assert abs(steps[2].impedance[0,0]-steps[0].impedance[0,0])>1e-9
    # Identical drive at cold start and after a different thermal history
    # must give a distinct Z and current on restart.
    restart=et.trajectory([0.,1.,2.,3.,4.,6.,7.],
        lambda tm: [1 if tm<=2 or tm>=7 else 0,0])
    assert abs(restart[-1].currents[0])>0
    assert restart[-1].temp.max()>0
    assert abs(restart[-1].impedance[0,0]-restart[0].impedance[0,0])>1e-6
    assert restart[-1].sea_q.shape==(et.count,)


def test_coupled_heat_balance():
    et=BFZIV148ET(Model(),sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
                  sea_k=.05,cu_pkg_g=.01,pkg_sea_g=.01,outer_h=.1,
                  external=np.eye(2)*.5)
    dt=1.
    new=et.advance(et.zeros(),[1,0],dt)
    _,_,q,_,_=et.em(new,[1,0])
    deposited=dt*q.sum()
    stored=et.cap@new
    cooling=dt*np.sum(et.H@new)
    assert abs(deposited-stored-cooling) < 1e-5*deposited
