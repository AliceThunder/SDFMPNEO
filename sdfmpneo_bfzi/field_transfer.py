"""Compare native BFZI FV fields on shared physical wet-water quadrature."""
from __future__ import annotations

import numpy as np


def quadrature(rec):
    """Symmetric wet-volume sample cloud from the original BFZI records."""
    q = []
    w = []
    for cell in rec:
        p = np.asarray(cell["quadrature_points"], float)
        wt = np.asarray(cell["quadrature_weights"], float)
        if p.ndim != 2 or p.shape[1] != 3 or wt.shape != (len(p),):
            raise ValueError("invalid original BFZI wet FV quadrature")
        if np.any(wt <= 0):
            raise ValueError("positive BFZI wet FV weights required")
        q.append(p)
        w.append(wt)
    return np.concatenate(q), np.concatenate(w)


def owner(rec, pts, *, batch=256, allow_outside=False, allow_face=False):
    """Locate exact containing native wet FV box; no neighbor smoothing."""
    lo = np.array([r["cell_min"] for r in rec], float)
    hi = np.array([r["cell_max"] for r in rec], float)
    xyz = np.asarray(pts, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("common physical points must be (n,3)")
    ids = np.empty(len(xyz), int)
    for i in range(0, len(xyz), batch):
        x = xyz[i:i+batch]
        mask = np.all((x[:,None,:] >= lo[None,:,:]-1e-12) &
                      (x[:,None,:] < hi[None,:,:]+1e-12), axis=2)
        count = mask.sum(axis=1)
        if (not allow_face and np.any(count>1)) or (not allow_outside and np.any(count==0)):
            raise ValueError("shared quadrature is outside or ambiguous in BFZI wet cells")
        ids[i:i+batch] = np.where(count>0,mask.argmax(axis=1),-1)
    return ids


def shared(rec_a, rec_b):
    """Union of both native wet quadratures, each with half-volume weight."""
    a,wa = quadrature(rec_a)
    b,wb = quadrature(rec_b)
    xyz = np.vstack((a,b))
    w = .5*np.r_[wa,wb]
    # Exact face ties can arise at wet OBB quadrature nodes. Their
    # continuous-volume measure is zero; take first containing FV.
    ia=owner(rec_a,xyz,allow_outside=True,allow_face=True)
    ib=owner(rec_b,xyz,allow_outside=True,allow_face=True)
    good=(ia>=0)&(ib>=0)
    if not np.any(good):
        raise ValueError("BFZI grids have no common physical wet volume")
    return xyz[good],w[good],ia[good],ib[good]


def norm(x,w):
    """Weighted spatial L2 norm for scalar or vector fields."""
    vals = np.asarray(x)
    if vals.shape[0] != len(w):
        raise ValueError("spatial field and physical volume weights mismatch")
    return float(np.sqrt(np.dot(w, np.sum(
        abs(vals.reshape(len(w),-1))**2, axis=1))))


def rel(a,b,w):
    return norm(a-b,w)/max(norm(b,w),1e-30)


def shape_j(j,w,ref=None):
    """Remove drive magnitude and global phase, preserving field pattern."""
    size = norm(j,w)
    if size < 1e-20:
        return None
    out = j/size
    if ref is not None:
        refmat = np.asarray(ref).reshape(len(w),-1)
        mat = out.reshape(len(w),-1)
        z = np.sum(w[:,None]*np.conj(refmat)*mat)
        if abs(z)>1e-15:
            out = out * np.conj(z)/abs(z)
    return out


def shape_q(q,w):
    """Joule power density normalized by total seawater power."""
    total=float(w@np.asarray(q,float))
    if total <= 1e-20:
        return None
    return q/total


def compare(a,b,w):
    """Time-series comparison of raw fields, shapes, and thermal perturbation."""
    out = {"raw":[], "shape":[], "thermal":[],
           "delta_over_cold":[], "delta_ref_over_cold":[]}
    if a.shape != b.shape:
        raise ValueError("shared fields must have identical shape")
    baseline_a = baseline_b = None
    for k,(x,y) in enumerate(zip(a,b)):
        out["raw"].append(rel(x,y,w) if norm(y,w)>1e-20 else None)
        if x.ndim==2:
            sx = shape_j(x,w,baseline_a)
            sy = shape_j(y,w,baseline_b)
        else:
            sx = shape_q(x,w)
            sy = shape_q(y,w)
        if k==0:
            baseline_a,baseline_b=sx,sy
        out["shape"].append(rel(sx,sy,w) if sx is not None and sy is not None else None)
        if k and sx is not None and sy is not None and baseline_a is not None and baseline_b is not None:
            da = sx-baseline_a
            db = sy-baseline_b
            ref=norm(db,w)
            out["thermal"].append(rel(da,db,w) if ref>1e-12 else None)
            out["delta_over_cold"].append(norm(da-db,w)/max(norm(baseline_b,w),1e-30))
            out["delta_ref_over_cold"].append(ref/max(norm(baseline_b,w),1e-30))
        else:
            out["thermal"].append(None)
            out["delta_over_cold"].append(None)
            out["delta_ref_over_cold"].append(None)
    return out
