"""S5 state layout, weak-source transfer and geometry conditioning."""
import numpy as np
from scipy.sparse import csr_matrix
import pytest

from sdfmpneo_bfzi.s5_bridge import state_order, skin_field, teacher_temp


def test_s5_layout_preserves_all_nodes():
    order = state_order(4, 2, 3)
    assert order.tolist() == [0, 1, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8, 9]
    assert sorted(order.tolist()) == list(range(13))
    class Head:
        state_order = order
    x = np.arange(39).reshape(3, 13)
    assert np.array_equal(teacher_temp(x, Head()), x[:, order])
    with pytest.raises(ValueError, match="thermal node count"):
        teacher_temp(np.zeros((3, 14)), Head())


def test_skin_weak_quadrature_is_volume_conservative():
    mix = csr_matrix([[.1, .0, .2], [.0, .15, .05], [.2, .0, .1], [.1, .1, .0]])
    vol = np.asarray(mix.sum(axis=0)).ravel()
    weak = np.arange(4*2*2).reshape(4,2,2).astype(float)/20
    for array in [weak, weak[..., None]*np.array([1+2j,2-1j,3+4j])]:
        mapped = skin_field(mix, array, vol)
        expect = np.tensordot(np.asarray(mix.sum(axis=1)).ravel(), array, axes=([0],[0]))
        got = np.tensordot(vol, mapped, axes=([0],[0]))
        assert np.allclose(got, expect)


def test_bad_s5_layout_rejected():
    for size in [(0, 2, 4), (4, 0, 4), (4, 2, 0)]:
        with pytest.raises(ValueError, match="nonempty"):
            state_order(*size)


def test_from_skin_anchor_retains_exact_heat_operator(monkeypatch):
    from types import SimpleNamespace as NS
    from scipy.sparse import diags
    import sdfmpneo_bfzi.s5_bridge as mod
    g = np.arange(13) + 1.
    cap = np.array([3., 4., 2., 5., 6., 7., 8., 1., 1., 1., 1., 1., 1.])
    mix = csr_matrix([[.3, 0., .2], [0., .2, .3], [.2, .1, 0.], [0., .2, .2]])
    shell = np.asarray(mix.sum(axis=0)).ravel()
    et = NS(count=4, pkg_n=2, n=13, skin=slice(10, 13), amb=293.15,
            cap=cap, H=diags(g).tocsr(), bulk_vol=np.array([.1,.2,.3,.4]),
            skin_vol=shell, skin_mix=mix, skin_xyz=np.array([[0.,0.,.01],[0.,0.,.02],[0.,0.,.03]]),
            pkg_xyz=np.zeros((2,2,3)), cu=slice(8,10),
            pkg_axes=np.tile(np.eye(3),(2,1,1)),
            boxes=[(np.zeros(3),np.ones(3)),(np.ones(3),np.ones(3))])
    coil = lambda cen: NS(outer_radius_m=.09,rounded_square_core_half_m=.0,
                            rounded_square_alpha=1.,turns=10,wire_radius_m=.00079,
                            center=cen)
    cold=NS(total_z=np.eye(2,dtype=complex),
            unit_j=np.ones((4,2,3),complex),weak_q=np.tile(np.eye(2),(4,1,1)))
    em=NS(base={'families':[NS(coil=coil((0,0,0))),NS(coil=coil((0,.04,0)))]},
          points=np.ones((4,3)),air=np.eye(2),alpha=.02,solve=lambda *args:cold)
    class Head:
        def __init__(self,*args,**kwargs):
            self.args=args
            self.kwargs=kwargs
    monkeypatch.setattr(mod,'GeomHeat',Head)
    out=mod.from_skin(em,et,np.eye(2),thermal=[.02,.25,.0006,.25,20])
    order=state_order(4,2,3)
    assert np.array_equal(out.state_order,order)
    assert np.allclose(out.args[6], cap[order])
    assert np.allclose(out.args[7].toarray(),et.H.toarray()[order,:][:,order])
    assert out.args[8].shape==(7,2,3)
    assert out.args[9].shape==(7,2,2)
    assert out.args[0].shape == (16,)
