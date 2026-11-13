"""Conservative true wet-FV package-face adjacency on original BFZI grid."""
from itertools import product

import numpy as np
import pytest

from sdfmpneo_bfzi.pkg_interface import adjacent_links


def grid():
    pts=[]; rec=[]
    ends=(-.3,-.1,.1,.3)
    for i,j,k in product(range(3),repeat=3):
        if (i,j,k)==(1,1,1):
            continue
        lo=np.array([ends[i],ends[j],ends[k]])
        hi=np.array([ends[i+1],ends[j+1],ends[k+1]])
        pts.append(.5*(lo+hi))
        rec.append(dict(cell_min=lo,cell_max=hi))
    return np.array(pts),rec


def test_adjacent_cut_face_conservation_and_positivity():
    wet,rec=grid()
    half=np.array([.1,.1,.1])
    dims=(2,2,2)
    cells=np.array([-half+(np.asarray(t)+.5)*half
                    for t in np.ndindex(*dims)])
    links=adjacent_links(wet,rec,cells,half,np.eye(3),
                         dims,100,.25,.6,20.,quad=2)
    assert len(links)==96
    assert len(set(k for k,_,_ in links))==6
    assert all(0<=i<len(wet) and 100<=j<108 and g>0
               for i,j,g in links)
    T=np.linspace(1.,4.,len(wet))
    P=np.linspace(5.,7.,8)
    net=np.zeros(len(wet)+8)
    for i,j,g in links:
        flux=g*(P[j-100]-T[i])
        net[i]+=flux
        net[len(wet)+j-100]-=flux
    assert abs(net.sum())<1e-12


def test_invalid_face_mapping_is_exposed():
    wet,rec=grid()
    half=np.array([.1,.1,.1])
    cells=np.zeros((1,3))
    with pytest.raises(ValueError,match="mismatch"):
        adjacent_links(wet,rec,cells,half,np.eye(3),(2,2,2),
                       10,.25,.6,20.)
    with pytest.raises(ValueError,match="no adjacent wet"):
        adjacent_links(wet[:1],rec[:1],np.zeros((8,3)),
                       half,np.eye(3),(2,2,2),
                       10,.25,.6,20.)
