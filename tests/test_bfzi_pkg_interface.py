"""Oriented package-face FV transfer, thermal resistance and conservation."""
import numpy as np
import pytest

from sdfmpneo_bfzi.pkg_interface import face_links


def sample():
    rng=np.random.default_rng(43)
    dims=(2,2,2)
    half=np.array([.10,.04,.12])
    pts=np.array(np.meshgrid(*[np.linspace(-.35,.35,9)]*3)).reshape(3,-1).T
    pts+=rng.normal(size=pts.shape)*.003
    pts=pts[np.any(abs(pts)>np.array([.16,.10,.17]),axis=1)]
    size=2*half/np.array(dims)
    cells=np.array([-half+(np.array(t)+.5)*size
                    for t in np.ndindex(*dims)])
    return pts,cells,half,dims


def test_face_flux_conserves_and_responds_to_physical_resistance():
    wet,cells,half,dims=sample()
    I=np.eye(3)
    low=face_links(wet,cells,half,I,dims,100,.25,.6,5.)
    high=face_links(wet,cells,half,I,dims,100,.25,.6,30.)
    assert len(low)==len(high)==6*4*4*4
    assert all(a>=0 and 100<=b<108 and g>0
               for a,b,g in high)
    assert sum(x[2] for x in high)>sum(x[2] for x in low)
    assert sum(x[2] for x in high)<.5
    ts=np.linspace(.1,.5,len(wet))
    tp=np.arange(8)*.3
    sea=np.zeros(len(wet))
    pkg=np.zeros(8)
    for a,b,g in high:
        flow=g*(tp[b-100]-ts[a])
        sea[a]+=flow
        pkg[b-100]-=flow
    assert abs(sea.sum()+pkg.sum())<1e-13


def test_rotation_equivariance_and_cold_geometry_validation():
    wet,cells,half,dims=sample()
    angle=np.deg2rad(27)
    R=np.array([[np.cos(angle),-np.sin(angle),0],
                [np.sin(angle),np.cos(angle),0],[0,0,1]])
    a=face_links(wet,cells,half,np.eye(3),dims,10,.25,.6,20.)
    b=face_links(wet@R.T,cells@R.T,half,R,dims,10,.25,.6,20.)
    def sums(links):
        d={}
        for i,j,v in links:
            d[(i,j)]=d.get((i,j),0.)+v
        return d
    x,y=sums(a),sums(b)
    assert x.keys()==y.keys()
    assert all(np.isclose(x[k],y[k],rtol=1e-10,atol=1e-12)
               for k in x)
    with pytest.raises(ValueError,match="positive"):
        face_links(wet,cells,half,np.eye(3),dims,10,.25,.6,0.)
