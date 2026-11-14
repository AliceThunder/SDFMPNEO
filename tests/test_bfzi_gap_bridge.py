"""Physically adjacent tilted BFZI package skin gap coupling."""
import numpy as np

from sdfmpneo_bfzi.gap_bridge import bridge


def faces(gap=.006,shift=0.,angle=0.):
    a=np.deg2rad(angle)
    n=np.array([np.sin(a),0.,-np.cos(a)])
    return [
        dict(id=0,pkg=0,center=np.array([0.,0.,0.]),
             normal=np.array([0.,0.,1.]),
             v1=np.array([.1,0.,0.]),v2=np.array([0.,.1,0.]),
             area=.01,last=5),
        dict(id=1,pkg=1,center=np.array([shift,0.,gap]),
             normal=n,
             v1=np.array([.1*np.cos(a),0.,.1*np.sin(a)]),
             v2=np.array([0.,.1,0.]),area=.01,last=13)]


def test_true_gap_bridge_passive_faces_and_nonduplication():
    f=faces()
    edges,used,meta=bridge(
        f,depth=.002,last=.0005,gap=.012,quad=4)
    assert len(edges)==16
    assert all(i==5 and j==13 and g>0 for i,j,g in edges)
    assert np.allclose(used,[.01,.01])
    assert np.isclose(meta["area_m2"],.01)
    assert np.isclose(meta["min_gap_m"],.006)
    assert np.isclose(sum(e[2] for e in edges),.6*.01/.0025)
    flux=sum(g*(1.-0.) for _,_,g in edges)
    net=np.zeros(14)
    for i,j,g in edges:
        net[i]-=g
        net[j]+=g
    assert abs(net.sum())<1e-14
    assert np.isclose(net[5],-flux)
    assert np.isclose(net[13],flux)


def test_partial_overlap_and_no_bridge_at_wide_gap():
    face=faces(shift=.05)
    edges,used,meta=bridge(
        face,depth=.002,last=.0005,gap=.012,quad=4)
    assert len(edges)==8
    assert np.isclose(meta["area_m2"],.005)
    assert np.allclose(used,[.005,.005])
    edge,used,meta=bridge(
        faces(gap=.04),depth=.002,last=.0005,gap=.012,quad=4)
    assert len(edge)==0
    assert not np.any(used)
    assert meta["bridges"]==0


def test_tilted_bridge_remains_positive_and_face_limited():
    face=faces(gap=.006,angle=25.)
    edges,used,meta=bridge(
        face,depth=.002,last=.0005,gap=.012,quad=8)
    assert len(edges)>0
    assert np.all(used>0)
    assert np.all(used<=.01+1e-12)
    assert meta["G_W_K"]>0
    assert all(g>0 for _,_,g in edges)
