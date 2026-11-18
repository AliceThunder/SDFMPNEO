"""Copper/package conductance conservation in shared thermal ROM."""
import numpy as np

from sdfmpneo_bfzi.geom_init import balance_cu


def test_copper_exchange_and_no_mutation():
    cap=np.array([100.,20.,25.,4.,5.])
    k=np.eye(32)
    k[3,3]=3.
    k[4,4]=4.
    for p,c,g in ((1,3,6.),(2,4,11.)):
        k[p,c]=k[c,p]=-g/np.sqrt(cap[p]*cap[c])
    original=k.copy()
    result=balance_cu(dict(k=k,cap=cap))
    out=result["k"]
    assert np.array_equal(k,original)
    assert np.allclose(out,out.T,rtol=0,atol=1e-14)
    for p,c in ((1,3),(2,4)):
        old=-original[p,c]*np.sqrt(cap[p]*cap[c])
        goal=original[c,c]*cap[c]
        assert np.isclose(-out[p,c]*np.sqrt(cap[p]*cap[c]),goal)
        old_b=original[p,c]
        new_b=out[p,c]
        assert np.isclose(out[p,p],original[p,p]+
                          (new_b**2-old_b**2)/original[c,c])
        assert np.isclose(out[c,c],original[c,c])
    assert np.linalg.eigvalsh(original)[0]>0
    assert np.linalg.eigvalsh(out)[0]>0


def test_consistent_operator_remains_unchanged():
    cap=np.array([100.,20.,25.,4.,5.])
    k=np.eye(32)
    for p,c in ((1,3),(2,4)):
        k[p,c]=k[c,p]=-k[c,c]*np.sqrt(cap[c]/cap[p])
    out=balance_cu(dict(k=k,cap=cap))["k"]
    assert np.allclose(out,k,rtol=0,atol=1e-14)


def test_conductance_mixing_is_passive_and_conservative():
    from types import SimpleNamespace
    from sdfmpneo_bfzi.geom_canon import labels
    from sdfmpneo_bfzi.geom_init import conductance_mix
    lab=labels(32)
    cap=np.array([100.,20.,25.,4.,5.])
    m=np.sqrt(cap[lab])
    bank=[]
    for g in (4.,9.):
        h=np.eye(32)*0.05
        h[3,3]=h[4,4]=0.
        for p,c in ((1,3),(2,4)):
            v=np.zeros(32);v[p]=-1.;v[c]=1.
            h+=g*np.outer(v,v)
        bank.append(h/(m[:,None]*m[None,:]))
    atlas=SimpleNamespace(rank=32,k=np.stack(bank),
        y=np.tile(np.r_[np.zeros(9),np.log(cap)],(2,1)))
    raw=np.eye(32)
    raw[3,3]=3.;raw[4,4]=3.
    init=dict(cap=cap,k=raw,norm_weights=np.array([.4,.6]))
    out=conductance_mix(atlas,init)["k"]
    assert np.allclose(out,out.T,atol=1e-12)
    assert np.linalg.eigvalsh(out)[0]>0
    for p,c in ((1,3),(2,4)):
        gd=out[c,c]*cap[c]
        gx=-out[p,c]*np.sqrt(cap[p]*cap[c])
        assert np.isclose(gd,gx,rtol=1e-12)
        assert np.isclose(gd,raw[c,c]*cap[c],rtol=1e-12)
