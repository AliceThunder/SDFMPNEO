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
