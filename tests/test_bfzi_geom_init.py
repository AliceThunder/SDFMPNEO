"""Geometry-only cold/S5 shared operator, independent of BFZI runtime."""
import numpy as np
import pytest
from sdfmpneo_bfzi.geom_init import GeomInit,encode


def _case(a=0.8):
    return dict(donor='g1',alpha_per_K=.020,
        geometry=dict(alpha1=a,alpha2=.95,core1_mm=19,core2_mm=15,
            shift_mm=[2,94,0],tilt_deg=10,yaw_deg=2,pitch_deg=3,
            box1_mm=221,box2_mm=230),
        thermal=dict(ins_k=.2,ins_t=.0004,pkg_k=.25,pkg_sea_h=20.))


def test_encode_uses_only_geometric_and_material_inputs():
    assert encode(_case()).shape==(18,)
    assert not np.allclose(encode(_case(.8)),encode(_case(.9)))


def test_shared_operator_stays_symmetric_and_dissipative(tmp_path):
    cases=[_case(.8),_case(.9),_case(.95)]
    x=np.stack([encode(c) for c in cases])
    y=np.zeros((3,14))
    y[:,:6]=[1,10,.1,.5,1,10]
    y[:,6:9]=[1e-5,1e-6,1e-5]
    y[:,9:14]=np.log([1e5,500,500,20,20])
    k=np.stack([np.eye(32)*v for v in (.01,.02,.03)])
    f=np.ones((3,32,3))
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    model=GeomInit.fit(x,y,k,f,ext)
    got=model.predict(_case(.86))
    assert got['k'].shape==(32,32)
    assert np.linalg.eigvalsh(got['k'])[0]>0
    assert np.allclose(got['k'],got['k'].T)
    assert np.all(got['cap']>0)
    assert got['src'].shape==(32,3)
    assert got['z0'].shape==(6,)
    model.save(tmp_path/'atlas.npz')
    restored=GeomInit.load(tmp_path/'atlas.npz')
    assert np.allclose(restored.predict(_case(.86))['k'],got['k'])
    with pytest.raises(ValueError):
        model.predict(_case(),external=np.ones((3,3)))

def test_shared_modal_order_accepts_64_coordinates():
    cases=[_case(.8),_case(.9),_case(.95)]
    x=np.stack([encode(c) for c in cases])
    y=np.zeros((3,14));y[:,9:]=np.log([1e5,400,400,20,20])
    k=np.stack([np.eye(64)*v for v in (.01,.02,.03)])
    f=np.ones((3,64,3))
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    model=GeomInit.fit(x,y,k,f,ext)
    pred=model.predict(_case(.86))
    assert model.rank==64
    assert pred['k'].shape==(64,64)
    assert pred['src'].shape==(64,3)
    assert np.linalg.eigvalsh(pred['k'])[0]>0

def test_physical_thermal_scaling_preserves_dissipation():
    cases=[_case(.8),_case(.9),_case(.95)]
    x=np.stack([encode(c) for c in cases])
    y=np.zeros((3,14));y[:,9:]=np.log([1e5,400,400,20,20])
    k=np.stack([np.eye(32)*v for v in (.01,.02,.03)])
    src=np.ones((3,32,3))
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    atlas=GeomInit.fit(x,y,k,src,ext).set_kscale((.5,.25,1.))
    for c in cases:
        op=atlas.predict(c)['k']
        assert np.linalg.eigvalsh(op)[0]>0
        assert np.allclose(op,op.T)


def test_two_input_coil_predictor_preserves_cold_and_load(tmp_path):
    cases=[_case(.8),_case(.9),_case(.95)]
    x=np.stack([encode(c) for c in cases])
    y=np.zeros((3,14))
    y[:,0]=[.15,.17,.18]
    y[:,1]=[19,18,17]
    y[:,4]=[.14,.16,.17]
    y[:,5]=[18,17,16]
    y[:,9:]=np.log([1e5,400,400,20,20])
    k=np.stack([np.eye(32)*v for v in (.01,.02,.03)])
    src=np.ones((3,32,3))
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    rows=[dict(case=c,y=t) for c,t in zip(cases,y)]
    m=GeomInit.fit(x,y,k,src,ext).fit_self(rows,1.5,.1,dims=2)
    z=m.predict(_case(.86))['z0']
    assert z.shape==(6,) and np.isfinite(z).all()
    assert m.sx.shape==(6,2)
    m.save(tmp_path/'model.npz')
    assert np.allclose(GeomInit.load(tmp_path/'model.npz').predict(_case(.86))['z0'],z)


def test_cold_mix_keeps_material_conservation_operator():
    from sdfmpneo_bfzi.geom_init import mix_init
    cases=[_case(.8),_case(.9)]
    x=np.stack([encode(c) for c in cases])
    y=np.zeros((2,14));y[:,1]=[15,18]
    y[:,9:]=np.log([1e5,400,400,20,20])
    k=np.stack([np.eye(32)*v for v in (.01,.02)])
    src=np.ones((2,32,3))
    ext=np.stack([np.eye(2,dtype=complex)]*2)
    a=GeomInit.fit(x,y,k,src,ext)
    b=GeomInit.fit(x,y+0.01,k,src,ext)
    base=a.predict(cases[0])
    val=mix_init(cases[0],a,b,.2)
    assert np.allclose(val['z0'],.8*base['z0']+.2*b.predict(cases[0])['z0'])
    assert np.allclose(val['k'],base['k'])
    assert np.allclose(val['src'],base['src'])
    with pytest.raises(ValueError):
        mix_init(cases[0],a,b,1.2)

def test_thermal_kernel_bandwidth_is_independent_and_persistent(tmp_path):
    cases=[_case(.8),_case(.9),_case(.95)]
    x=np.stack([encode(c) for c in cases])
    y=np.zeros((3,14))
    y[:,9:]=np.log([1e5,500,500,20,20])
    k=np.stack([np.eye(32)*v for v in (.01,.02,.04)])
    src=np.stack([np.full((32,3),v) for v in (.2,.4,.8)])
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    base=GeomInit.fit(x,y,k,src,ext,ell=6.)
    local=GeomInit.fit(x,y,k,src,ext,ell=6.,kell=.6)
    q=_case(.86)
    a=base.predict(q)
    b=local.predict(q)
    assert not np.allclose(a['k'],b['k'])
    assert np.allclose(a['src'],b['src'])
    assert np.linalg.eigvalsh(b['k'])[0]>0
    assert np.allclose(b['k'],b['k'].T)
    local.save(tmp_path/'local.npz')
    restored=GeomInit.load(tmp_path/'local.npz')
    assert restored.kell==.6
    assert np.allclose(restored.predict(q)['k'],b['k'])


def test_copper_balance_accepts_rank48_operator():
    from sdfmpneo_bfzi.geom_init import balance_cu
    init=dict(cap=np.array([1e5,500.,500.,20.,20.]),
              k=np.eye(48)*.01)
    got=balance_cu(init)
    assert got['k'].shape==(48,48)
    assert np.allclose(got['k'],got['k'].T)
    assert np.linalg.eigvalsh(got['k'])[0]>0


def test_rank32_source_correction_preserves_rank48_atlas_modes():
    from sdfmpneo_bfzi.geom_srcfix import SrcFix

    model=SrcFix(dict(idx=np.array([0]),x=np.zeros((1,1)),
        mu=np.zeros(1),std=np.ones(1),ym=np.zeros(96),
        beta=np.zeros((1,96)),ell=1.,lam=.01))
    model.predict=lambda case: np.full((32,3),2.)
    case=_case()
    src=np.arange(48*3,dtype=float).reshape(48,3)
    got=model.apply(case,dict(src=src))['src']
    assert got.shape==(48,3)
    assert np.all(got[:32]==2.)
    assert np.array_equal(got[32:],src[32:])


def test_material_mean_operator_preserves_physical_energy_graph():
    from sdfmpneo_bfzi.geom_init import conductance_mix, thermal_mix

    cases=[_case(.8),_case(.9),_case(.95)]
    x=np.stack([encode(c) for c in cases])
    cap=np.array([[1e5,500.,510.,20.,21.],
                  [1.1e5,520.,530.,22.,23.],
                  [1.2e5,540.,550.,24.,25.]])
    edges=np.array([[12.8,2.0,2.2,13.0,12.0],
                    [12.8,2.5,2.3,14.0,13.0],
                    [12.8,3.0,2.4,15.0,14.0]])
    mats=[]
    for c,e in zip(cap,edges):
        h,a,b,d,f=e
        g=np.zeros((5,5))
        g[0,0]=h+a+b
        g[1,1]=a+d
        g[2,2]=b+f
        g[3,3]=d
        g[4,4]=f
        g[0,1]=g[1,0]=-a
        g[0,2]=g[2,0]=-b
        g[1,3]=g[3,1]=-d
        g[2,4]=g[4,2]=-f
        k=np.eye(32)*.02
        mass=np.sqrt(c)
        k[:5,:5]=g/(mass[:,None]*mass[None,:])
        mats.append(k)
    y=np.zeros((3,14))
    y[:,9:14]=np.log(cap)
    src=np.ones((3,32,3))
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    atlas=GeomInit.fit(x,y,np.stack(mats),src,ext,ell=6.,kell=.6)
    assert atlas.kmm_edges is not None

    def check(k,c):
        mass=np.sqrt(c)
        g=mass[:,None]*k[:5,:5]*mass[None,:]
        assert np.allclose(g.sum(axis=1),[12.8,0.,0.,0.,0.],atol=1e-9)
        assert np.allclose(g[[0,0,1,2],[3,4,2,3]],0.,atol=1e-12)
        assert np.all(g[[0,0,1,2],[1,2,3,4]]<0.)

    case=_case(.86)
    init=atlas.predict(case)
    check(init['k'],init['cap'])
    phys=conductance_mix(atlas,init)
    check(phys['k'],init['cap'])
    mixed=thermal_mix(atlas,init,.25)
    check(mixed['k'],init['cap'])
