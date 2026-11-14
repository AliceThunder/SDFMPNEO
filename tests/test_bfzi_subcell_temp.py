"""BFZI subcell temperature quadrature and near-source covariance."""
import numpy as np
import pytest

from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter


def adapter():
    # Build one physical quadrature cell without initializing original BFZI.
    em=object.__new__(BFZIV148Adapter)
    em.vol=np.array([2.])
    em.sigma0=4.
    em.alpha=.02
    em.ambient=293.15
    pts=np.array([[-1.,0,0],[1.,0,0]])
    w=np.ones(2)
    # The two winding projections vary differently within the wet FV.
    moments=np.array([[[1.,0.],[0.,2.]],
                      [[4.,1.],[1.,3.]]])
    em.subcells=[(pts,w,moments)]
    return em,moments


def test_uniform_subcell_temperature_has_zero_extra_mass():
    em,mom=adapter()
    t,cov=em.subcell_temp(
        lambda q:np.full(len(q),em.ambient+5))
    assert np.allclose(t,[em.ambient+5])
    assert np.allclose(cov,0,atol=1e-13)


def test_linear_subcell_temperature_covariance_is_source_weighted():
    em,mom=adapter()
    def field(q):
        return em.ambient+5+2*q[:,0]
    t,cov=em.subcell_temp(field)
    sig=em.sigma0*em.alpha*2*np.array([-1,1])
    expected=np.einsum("q,qij->ij",sig,mom)
    assert np.isclose(t[0],em.ambient+5)
    assert np.allclose(cov[0],expected,atol=1e-12)
    assert np.allclose(cov[0],cov[0].T)
    # Port loss density integrates exactly to the same near-source Z mass.
    w=2.3
    loss=w*w*cov[0]/em.vol[0]
    assert np.allclose(loss*em.vol[0],w*w*cov.sum(axis=0))


def test_subcell_source_requires_valid_temperature_and_geometry():
    em,_=adapter()
    with pytest.raises(ValueError,match="temperature"):
        em.subcell_temp(lambda q:np.zeros((len(q),3)))
    with pytest.raises(ValueError,match="temperature"):
        em.subcell_temp(lambda q:np.full(len(q),np.nan))
    em.subcells=None
    with pytest.raises(ValueError,match="not built"):
        em.subcell_temp(lambda q:np.zeros(len(q)))


def test_subcell_conservative_target_mean_keeps_local_covariance():
    em,_=adapter()
    field=lambda q: em.ambient+5+2*q[:,0]
    avg,old=em.subcell_temp(field)
    target=avg+7.
    mean,local=em.subcell_temp(field,mean=target)
    assert np.array_equal(mean,target)
    assert np.allclose(old,local,rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match="mean"):
        em.subcell_temp(field,mean=np.zeros(2))


def test_cached_native_source_reconstruction_recovers_original_mean():
    from sdfmpneo_bfzi.near_source import local_j,local_q
    em,_=adapter()
    xyz,w,_=em.subcells[0]
    src=np.array([[[1.,0,0],[0,2,0]],[[3.,0,0],[0,4,0]]])
    em.subcells=[(xyz,w,np.einsum("qic,qjc->qij",src,src),src)]
    em.source_mean=np.einsum("q,qpc->pc",w,src)[None]/w.sum()
    em.omega=2.
    em.mu=.25
    coords,weights,owner,fields=em.source_native()
    sig=np.array([3.])
    bar=np.array([[1.+2j,2,0]])
    cur=np.array([2j,1.+0j])
    j=em.sample_j(bar,sig,cur,coords,owner,source=fields)
    assert np.allclose(np.sum(weights[:,None]*j,axis=0)/weights.sum(),bar[0])
    heat,scale=local_q(j,sig,np.array([2.]),em.vol,owner,weights)
    assert np.isclose(weights@heat,2.*em.vol[0])
