"""Small direct tests for BFZI v148 spatial-temperature integration.

The original BFZI repository has a separate dependency tree. These tests
exercise the adapter's weighted FV projector and the time integrator without
requiring heavy full v148 Green matrix construction.
"""
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp

from sdfmpneo_bfzi.bfzi_v148_adapter import WeightedFV, BFZIV148Adapter, V148State
from sdfmpneo_bfzi.v148_et import BFZIV148ET, circuit_power


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
    # Independently reconstruct original BFZI fv_internal_only projector.
    A=(base.B.T@sp.diags(base.weights)@base.B).tolil()
    A[0,:]=0.; A[:,0]=0.; A[0,0]=1.
    from scipy.sparse.linalg import factorized
    solve=factorized(A.tocsc())
    edge=np.array([.5*(field[i,k]+field[j,k])*direction
                   for i,j,k,d,area,direction in base.edges])
    rhs=np.asarray(base.B.T@(base.weights*edge))
    rhs-=rhs.sum()/base.n
    rhs[0]=0
    phi=solve(rhs.real)+1j*solve(rhs.imag)
    grad=np.zeros_like(field)
    denom=np.zeros((base.n,3))
    for i,j,k,d,area,direction in base.edges:
        ge=(phi[j]-phi[i])*direction/d
        grad[i,k]+=area*ge; grad[j,k]+=area*ge
        denom[i,k]+=area; denom[j,k]+=area
    good=denom>0
    grad[good]/=denom[good]
    assert np.allclose(a,field-grad,atol=1e-12)
    assert np.linalg.norm(a-c)/np.linalg.norm(a) > .01


class Model:
    def __init__(self):
        self.proj=Grid()
        self.points=np.array([
            [-.28,0,0],[-.18,0,0],[.18,0,0],[.28,0,0]])
        self.vol=np.ones(4)*.01
        self.omega=1.256637e6
        fam=[
            SimpleNamespace(
                coil=SimpleNamespace(center=(-.2,0,0)),
                active=SimpleNamespace(pos=np.array([[-.23,0,-.02],[-.17,0,.02]]),
                                       dl=np.array([[.03,0,0],[.03,0,0]]))),
            SimpleNamespace(
                coil=SimpleNamespace(center=(.2,0,0)),
                active=SimpleNamespace(pos=np.array([[.17,0,-.02],[.23,0,.02]]),
                                       dl=np.array([[.03,0,0],[.03,0,0]]))),
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
            unit_j=np.zeros((4,2,3),complex),
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
    assert restart[-1].sea_j.shape==(et.count,3)
    assert restart[-1].unit_j.shape==(et.count,2,3)
    assert np.allclose(
        restart[-1].sea_j,
        np.einsum("npc,p->nc",restart[-1].unit_j,
                  restart[-1].currents))
    assert np.allclose(steps[-1].sea_j,0)


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



def test_package_flux_and_material_energy_budget():
    # The two enclosure nodes store heat independently and route copper
    # dissipation through package and seawater FV cells.
    et=BFZIV148ET(Model(),sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
                  sea_k=.05,cu_pkg_g=.08,pkg_sea_g=.04,outer_h=.1,
                  external=np.diag([.5,.5]))
    dt=.1
    old=et.zeros()
    new=et.advance(old,[8.,0.],dt)
    _,_,heat,_,_=et.em(new,[8.,0.])
    cp,ps,amb=et.heat_flux(new)
    cond=et.H@new
    assert cp.shape==ps.shape==(2,)
    assert np.allclose(cond[et.cu],cp,atol=1e-12)
    assert np.allclose(cond[et.pkg],ps-cp,atol=1e-12)
    assert np.isclose(cond[:et.count].sum(),amb-ps.sum(),atol=1e-12)
    for ids,gain in (
        (slice(0,et.count),heat[:et.count].sum()+ps.sum()-amb),
        (et.pkg,cp.sum()-ps.sum()),
        (et.cu,heat[et.cu].sum()-cp.sum())
    ):
        de=np.sum(et.cap[ids]*(new[ids]-old[ids]))
        assert np.isclose(de,dt*gain,rtol=1e-4,atol=1e-6)
    assert et.cap[et.pkg].shape==(2,)
    assert np.all(et.cap[et.pkg]>0)



def test_heat_rom_conserves_each_package(tmp_path):
    from sdfmpneo_bfzi.v148_heat_rom import BFZIV148HeatROM
    # Synthetic temperatures span sea gradients and both independent boxes.
    p=tmp_path/"thermal.npz"
    sea=293.15+np.array([[0,.1,.2,.4],[.2,.3,.4,.5],[.4,.5,.6,.8]])
    pkg=293.15+np.array([[0,.3],[.05,.4],[.15,.5]])
    cu=293.15+np.array([[0,1],[.25,1.2],[.3,1.4]])
    np.savez(p,Tsea_K=sea,Tpkg_K=pkg,Tcu_K=cu)
    et=BFZIV148HeatROM(
        Model(),[p],rank=6,sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
        sea_k=.05,cu_pkg_g=.08,pkg_sea_g=.04,outer_h=.1,
        external=np.diag([.5,.5]))
    assert et.hgroups==5
    old=et.zeros()
    dt=.1
    new=et.advance(old,[8.,0.],dt)
    _,_,heat,_,_=et.em(new,[8.,0.])
    residual=et.cap*(new-old)/dt+et.H@new-heat
    for idx in [np.arange(et.count),*[np.array([j]) for j in range(et.count,et.n)]]:
        assert abs(np.sum(residual[idx]))<1e-5
    assert np.isfinite(et.hretained)



def test_spatial_package_3d_flux_and_heat_rom(tmp_path):
    from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
    from sdfmpneo_bfzi.v148_heat_rom import BFZIV148PkgHeatROM
    model=Model()
    args=dict(pkg_shape=(2,2,2),sea_rhoc=1.,pkg_rhoc=1.,
              cu_rhoc=1.,sea_k=.05,pkg_k=.25,cu_pkg_g=.08,
              pkg_sea_g=.04,outer_h=.1,external=np.diag([.5,.5]))
    et=BFZIV148Pkg3D(model,**args)
    assert et.pkg_n==8 and et.n==model.vol.size+18
    temp=et.advance(et.zeros(),[8,0],.1)
    _,_,heat,_,_=et.em(temp,[8,0])
    cp,ps,amb=et.heat_flux(temp)
    res=et.cap*temp/.1+et.H@temp-heat
    assert np.max(abs(res))<1e-5
    assert np.isclose((et.H@temp).sum(),amb,atol=1e-12)
    assert cp.shape==ps.shape==(2,)
    assert np.isclose(et.cap[et.pkg].sum(),
                      np.sum([8*np.prod(half) for _,half in et.boxes]),rtol=1e-12)
    xyz=et.pkg_xyz[0]
    vals=et.temperature_at(xyz,temp)
    assert np.allclose(vals,et.amb+temp[et.count:et.count+et.pkg_n])
    for p in range(2):
        idx=slice(et.pkg.start+p*et.pkg_n,et.pkg.start+(p+1)*et.pkg_n)
        assert abs(np.sum(res[idx]))<1e-5
    # Train the material-preserving POD on the same spatial shape.
    path=tmp_path/"pkg3d.npz"
    snap=np.stack([et.zeros(),temp,temp*1.5])
    np.savez(path,Tsea_K=et.amb+snap[:,:et.count],
             Tpkg_cells_K=et.amb+snap[:,et.pkg].reshape(3,2,et.pkg_n),
             Tcu_K=et.amb+snap[:,et.cu])
    rom=BFZIV148PkgHeatROM(model,[path],rank=7,**args)
    reduced=rom.advance(rom.zeros(),[8,0],.1)
    _,_,heat_r,_,_=rom.em(reduced,[8,0])
    res_r=rom.cap*reduced/.1+rom.H@reduced-heat_r
    groups=[np.arange(rom.count)]
    groups.extend(np.arange(rom.pkg.start+p*rom.pkg_n,
                            rom.pkg.start+(p+1)*rom.pkg_n) for p in range(2))
    groups.extend([np.array([rom.cu.start+p]) for p in range(2)])
    assert max(abs(res_r[g].sum()) for g in groups)<1e-5
    assert np.allclose(rom.cap,et.cap)



def test_v148_circuit_efficiency():
    et=BFZIV148ET(Model(),sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
                  sea_k=.05,cu_pkg_g=.01,pkg_sea_g=.01,outer_h=.1,
                  external=np.diag([.5,.5]))
    voltage=np.array([1.,0.],complex)
    m,I,_,_,cu=et.em(et.zeros(),voltage)
    pin,pload,psrc,eta=circuit_power(voltage,I,.5,.5)
    psea=.5*np.real(np.vdot(I,m.sea_z@I))
    assert 0 < eta < 1
    assert np.isclose(pin,pload+psrc+psea+cu.sum(),atol=1e-12)
    assert circuit_power([0,0],[0,0],.5,.5)[-1] == 0.


def test_v148_local_mass_heat_matches_port():
    # Original v148 local source variance adds a real source-mass term to Z.
    # Its physically located Joule power must be present in the heat source.
    adapter=BFZIV148Adapter.__new__(BFZIV148Adapter)
    adapter.omega=1.25e6
    adapter.mu=4e-7*np.pi
    adapter.vol=np.array([.006,.009])
    adapter.mass=np.array([
        [[.0006,.0001],[.0001,.0004]],
        [[.0005,-.0002],[-.0002,.0006]],
    ])
    sigma=np.array([5.,4.])
    state=V148State(
        np.zeros((2,2),complex),np.zeros((2,2),complex),
        np.array([.08+.001j,.12+.002j]),
        np.zeros((2,2,3),complex),sigma,None)
    I=np.array([1.,.4+.3j])
    q,pcu=adapter.heat(state,I)
    dz=(adapter.omega*adapter.mu)**2*np.einsum(
        "p,pij->ij",sigma,adapter.mass)
    p_mass=float(.5*np.real(np.vdot(I,dz@I)))
    assert p_mass>0
    assert np.isclose(np.dot(q,adapter.vol),p_mass,rtol=2e-14)
    assert np.all(q>=0)
    assert np.isclose(pcu[0],.5*.08*abs(I[0])**2)



def test_bfzi_port_weak_heat_closes_two_port_power():
    adapter=BFZIV148Adapter.__new__(BFZIV148Adapter)
    adapter.heat_mode="port-weak"
    adapter.vol=np.array([.02,.04])
    adapter.omega=2*np.pi*200000
    adapter.mu=4e-7*np.pi
    adapter.mass=np.zeros((2,2,2))
    weak=np.array([
        [[2.,.2],[.2,1.]],
        [[1.,-.1],[-.1,.5]],
    ])
    state=V148State(
        np.zeros((2,2),complex),np.zeros((2,2),complex),
        np.array([.08,.09],complex),
        np.zeros((2,2,3),complex),np.ones(2)*5.,None,weak)
    i=np.array([1+.2j,.5-.4j])
    q,cu=adapter.heat(state,i)
    R=np.einsum("p,pij->ij",adapter.vol,weak)
    assert np.all(q>0)
    assert np.isclose(q@adapter.vol,.5*np.vdot(i,R@i).real,atol=1e-14)
    assert np.isclose(cu.sum(),.5*np.sum(state.copper_z.real*abs(i)**2))

def test_v148_rejects_failed_original_resolvent():
    adapter=BFZIV148Adapter.__new__(BFZIV148Adapter)
    adapter.omega=2*np.pi*200000.
    adapter.mu=4e-7*np.pi
    adapter.vol=np.ones(4)*.01
    adapter.ambient=293.15
    adapter.alpha=.02
    adapter.sigma0=5.
    adapter.proj=Grid()
    adapter.base={"C":[np.zeros((4,3)),np.zeros((4,3))]}
    core=SimpleNamespace(solve_operator=lambda *args,**kwargs: (
        [], {"gmres_infos":"[1, 0]"}))
    adapter.calc=SimpleNamespace(core=core)
    with pytest.raises(RuntimeError,match="GMRES"):
        adapter.solve(np.full(4,293.15),np.full(2,293.15))


def test_m4_archive_circuit_power_reference():
    """Original BFZI M4 archive, 200 kHz: sea+air only, no copper.

    Independent of the expensive original Green solve; checks the complex
    mutual coupling, compensation convention, and four-way power split.
    """
    w=2*np.pi*200000.
    sea_r=np.array([[.12781381449951726,.10805885891894453],
                    [.10805885891894453,.12781381449951723]])
    sea_l=np.array([[-.0030673350819896755,-.002969536493368606],
                    [-.002969536493368606,-.003067335081989676]])*1e-6
    air_l=np.array([[15.827666708487106,5.077003979629416],
                    [5.077003979629416,15.827666708487106]])*1e-6
    zsea=sea_r+1j*w*sea_l
    zem=zsea+1j*w*air_l
    ext=np.diag([.5-1j*w*air_l[0,0],
                 .5-1j*w*air_l[1,1]])
    v=np.array([3.,0.],complex)
    i=np.linalg.solve(zem+ext,v)
    pin,pload,psrc,eta=circuit_power(v,i,.5,.5)
    psea=float(.5*np.real(np.vdot(i,zsea@i)))
    assert np.isclose(pin,pload+psrc+psea,atol=1e-12)
    assert np.isclose(eta,.789055655984629,rtol=1e-12)
    assert np.isclose(psea,.013982265015249606,rtol=1e-12)


def test_thermal_fixed_point_must_converge():
    et=BFZIV148ET(Model(),sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
                  sea_k=.05,cu_pkg_g=.01,pkg_sea_g=.01,outer_h=.1,
                  external=np.diag([.5,.5]))
    with pytest.raises(RuntimeError,match="did not converge"):
        et.advance(et.zeros(),[8.,0.],.1,iterations=1,tol=1e-14)


def test_thermal_budget_controls_large_temperature_tolerance():
    et=BFZIV148ET(Model(),sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
                  sea_k=.05,cu_pkg_g=.01,pkg_sea_g=.01,outer_h=.1,
                  external=np.diag([.5,.5]))
    with pytest.raises(RuntimeError,match="heat residual"):
        et.advance(et.zeros(),[8.,0.],.1,iterations=1,
                   tol=1e6,budget_tol=1e-12)


def test_external_terminal_lead_loss_is_not_winding_heat():
    adapter=BFZIV148Adapter.__new__(BFZIV148Adapter)
    adapter.heat_mode="port-weak"
    adapter.vol=np.array([.1])
    state=V148State(
        np.zeros((2,2),complex),np.zeros((2,2),complex),
        np.array([.3+.01j,.5+.02j]),
        np.zeros((1,2,3),complex),np.ones(1),None,
        np.zeros((1,2,2)),
        winding_z=np.array([.2+.01j,.35+.02j]),
        lead_z=np.array([.1,.15]))
    current=np.array([3.+2j,2.-1j])
    q,cu=adapter.heat(state,current)
    pl=.5*state.lead_z.real*abs(current)**2
    assert np.allclose(q,0.)
    assert np.allclose(cu,.5*state.winding_z.real*abs(current)**2)
    assert np.allclose(cu+pl,.5*state.copper_z.real*abs(current)**2)


def test_switch_off_applies_after_previous_interval():
    model=Model()
    et=BFZIV148ET(model,sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
                  sea_k=.05,cu_pkg_g=.01,pkg_sea_g=.01,outer_h=.1,
                  external=np.eye(2)*.5)
    t=np.array([0.,1.,2.,3.])
    steps=et.trajectory(t,lambda now: [1. if now<2. else 0.,0.])
    # 1..2 s must still receive the ON drive. Off current begins at 2 s.
    assert steps[2].temp[et.cu][0]>steps[1].temp[et.cu][0]
    assert np.allclose(steps[2].currents,0.)
    assert steps[3].temp[et.cu][0]<steps[2].temp[et.cu][0]


def test_cn_thermal_step_midpoint_energy_budget():
    et=BFZIV148ET(Model(),sea_rhoc=1.,pkg_rhoc=1.,cu_rhoc=1.,
                  sea_k=.05,cu_pkg_g=.01,pkg_sea_g=.01,outer_h=.1,
                  external=np.eye(2)*.5)
    old=et.zeros()
    dt=.1
    v=np.array([2.,0.])
    new=et.advance(old,v,dt,method="cn")
    q0=et.em(old,v)[2]
    q1=et.em(new,v)[2]
    residual=et.cap*(new-old)/dt+.5*(et.H@old+et.H@new)-.5*(q0+q1)
    assert np.max(abs(residual))<1e-4
    assert abs(np.sum(residual))<1e-4
    assert np.max(new)>0.
