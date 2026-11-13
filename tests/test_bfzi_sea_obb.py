"""Original BFZI FV volume must exclude the true rotated package solids."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from sdfmpneo_bfzi.pose import axes,isinside
from sdfmpneo_bfzi.sea_obb import build


class FS:
    @staticmethod
    def _quadrature_for_subboxes(boxes,order):
        q=[];w=[]
        u,v=np.polynomial.legendre.leggauss(order)
        for mn,mx in boxes:
            for i,j,k in product(range(order),repeat=3):
                q.append(.5*(mn+mx)+.5*(mx-mn)*u[[i,j,k]])
                w.append(float(np.prod(mx-mn)*v[i]*v[j]*v[k]/8.))
        return np.asarray(q),np.asarray(w)


def data(tilt=27.,distance=.14):
    coils=[
        SimpleNamespace(center=(0.,-.06,0.),roll_deg=90.,
                        yaw_deg=0.,pitch_deg=0.,plane_angle_deg=90.),
        SimpleNamespace(center=(0.,-.06+distance,0.),
                        roll_deg=90.+tilt,yaw_deg=0.,pitch_deg=0.,
                        plane_angle_deg=90.)]
    e=np.linspace(-.35,.35,8)
    records=[]
    for i,j,k in product(range(7),repeat=3):
        mn=np.array([e[i],e[j],e[k]])
        mx=np.array([e[i+1],e[j+1],e[k+1]])
        records.append(dict(idx=(i,j,k),cell_min=mn,cell_max=mx))
    meta=dict(box_min_m=[-.35]*3,box_max_m=[.35]*3)
    sample={}
    for p in (1,2):
        sample[f"coil{p}_package_x_half_mm"]=102.
        sample[f"coil{p}_package_z_pos_half_mm"]=102.
        sample[f"coil{p}_package_z_neg_half_mm"]=102.
    cfg=dict(BSR_PACKAGE_NORMAL_HALF_M=.0055,
             BSR_PACKAGE_TOP_RISER_ENABLE=False)
    return coils,records,meta,sample,cfg


def test_original_grid_exactly_excludes_rotated_packages():
    coils,records,meta,sample,cfg=data()
    pts,vol,m,out=build(FS,records,meta,coils,sample,cfg)
    expect=.7**3-2*8*.102*.0055*.102
    assert abs(vol.sum()-expect)<1e-11
    assert abs(m["relative_volume_error"])<1e-11
    assert m["clipped_cell_count"]>0
    assert len(pts)==len(out)
    assert all(np.all(r["quadrature_weights"]>0) for r in out)
    for coil in coils:
        R=axes(coil)
        assert not isinside(pts,np.asarray(coil.center),np.array([.102,.0055,.102]),R).any()
        for r in out:
            assert not isinside(r["quadrature_points"],np.asarray(coil.center),
                                np.array([.102,.0055,.102]),R).any()


def test_oriented_water_grid_rejects_collision_and_node_budget():
    coils,records,meta,sample,cfg=data(distance=.01)
    with pytest.raises(ValueError,match="overlap"):
        build(FS,records,meta,coils,sample,cfg)
    coils,records,meta,sample,cfg=data(distance=.14)
    with pytest.raises(ValueError,match="budget"):
        build(FS,records,meta,coils,sample,cfg,max_nodes=10)
