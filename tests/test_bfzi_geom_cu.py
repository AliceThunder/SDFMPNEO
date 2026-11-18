"""Analytic continuous-winding copper conduction and capacity."""
import numpy as np
from sdfmpneo_bfzi.geom_cu import properties,apply


def case():
    return dict(geometry=dict(alpha1=.75,alpha2=.94,
        core1_mm=19.,core2_mm=11.),
        thermal=dict(ins_k=.27,ins_t=.00041))


def test_copper_continuous_winding():
    length,cap,cond=properties(case())
    assert length.shape==(2,)
    assert np.all((length>3.5)&(length<4.5))
    assert np.all((cap>20)&(cap<35))
    assert np.all((cond>10)&(cond<25))


def test_heat_capacity_only_preserves_physical_conductance():
    c=case()
    k=np.eye(32)
    cold=np.array([100.,20.,25.,27.,26.])
    init=dict(k=k,cap=cold,extra=7)
    out=apply(c,init,exchange=False)
    assert out["extra"]==7
    assert np.array_equal(init["k"],k)
    assert np.array_equal(init["cap"],cold)
    assert np.linalg.eigvalsh(out["k"])[0]>0
    assert np.allclose(out["k"][3,3]*out["cap"][3],27.)
    assert np.allclose(out["k"][4,4]*out["cap"][4],26.)


def test_analytic_conductance_exchange_is_passive():
    from sdfmpneo_bfzi.geom_canon import labels
    c=case()
    k=np.eye(32)
    cap=np.array([100.,20.,25.,27.,26.])
    lab=labels(32)
    for p,cu in ((1,3),(2,4)):
        k[p,cu]=k[cu,p]=-.65*np.sqrt(cap[cu]/cap[p])
        k[p,p]=4.
    out=apply(c,dict(k=k,cap=cap),exchange=True)
    z=out["k"]
    _,cp,g=properties(c)
    assert np.allclose(out["cap"][3:5],cp)
    assert np.linalg.eigvalsh(z)[0]>0
    for j,(p,cu) in enumerate(((1,3),(2,4))):
        assert np.isclose(z[cu,cu]*cp[j],g[j])
        assert np.isclose(-z[p,cu]*np.sqrt(out["cap"][p]*cp[j]),g[j])
