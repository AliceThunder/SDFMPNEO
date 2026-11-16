"""Optional sea-skin 48-mode temperature reconstruction from a 32-mode state."""
from __future__ import annotations

import numpy as np

from .geom_canon import basis, mode_scale


IDX=[0,*range(5,18)]


def fit(x,y,reg=1e-5):
    """Ridge fit using autonomous neural heat states as inputs."""
    x=np.asarray(x,float)
    y=np.asarray(y,float)
    g=x.T@x
    lam=reg*np.trace(g)/len(g)
    return np.linalg.solve(g+lam*np.eye(len(g)),x.T@y)


def reconstruct(head,temp,state,w):
    """Return enhanced 3-D temperature; unchanged physical-loop fields."""
    t=np.asarray(temp,float)
    a=np.asarray(state,float)
    w=np.asarray(w,float)
    if a.ndim!=2 or a.shape[1]!=32 or w.shape!=(14,16):
        raise ValueError("expected 32-state neural heat and 14x16 lifting map")
    if t.shape!=(len(a),len(head.cap)):
        raise ValueError("temperature and S5 head dimensions do not match")
    phi=basis(head,48)[:,32:]
    sc=mode_scale(head,48)[32:]
    return t+((a[:,IDX]@w)*sc)@phi.T
