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


def project(w, ia, ib, n_a, n_b, field):
    """Conservative shared-volume projection from fine to coarse wet FV.

    Returns coarse-cell reconstructed averages and their sampled wet
    volumes, using the same symmetric physical quadrature for both.
    """
    from scipy.sparse import coo_matrix
    mat=coo_matrix((w,(ia,ib)),shape=(n_a,n_b)).tocsr()
    vol=np.asarray(mat.sum(axis=1)).ravel()
    if np.any(vol<=0):
        raise ValueError("native coarse wet FV has no shared physical volume")
    src=np.asarray(field)
    if src.shape[0]!=n_b:
        raise ValueError("source wet FV field count mismatch")
    dst=(mat@src.reshape(n_b,-1))/vol[:,None]
    return dst.reshape((n_a,)+src.shape[1:]),vol


def overlap(rec_a, rec_b, meta):
    """Exact overlap volumes of two native OBB-cut BFZI wet-FV grids."""
    from scipy.sparse import coo_matrix
    from .sea_obb import _wet
    mid=np.asarray(meta["physical_package_center_m"],float).reshape(-1,3)
    half=np.asarray(meta["physical_package_half_m"],float).reshape(-1,3)
    rot=np.asarray(meta["physical_package_rot"],float).reshape(-1,3,3)
    boxes=list(zip(mid,half,rot))
    limits=[(m-np.abs(r)@h,m+np.abs(r)@h)
            for m,h,r in boxes]
    lo_b=np.asarray([r["cell_min"] for r in rec_b],float)
    hi_b=np.asarray([r["cell_max"] for r in rec_b],float)
    ii=[];jj=[];vv=[]
    for i,a in enumerate(rec_a):
        lo_a=np.asarray(a["cell_min"],float)
        hi_a=np.asarray(a["cell_max"],float)
        js=np.flatnonzero(np.all((lo_b<hi_a)&(hi_b>lo_a),axis=1))
        for j in js:
            lo=np.maximum(lo_a,lo_b[j]);hi=np.minimum(hi_a,hi_b[j])
            if not np.all(hi>lo):
                continue
            active=[box for box,(l,h) in zip(boxes,limits)
                    if np.all(hi>l) and np.all(lo<h)]
            if active:
                _,w=_wet(lo,hi,active)
                volume=float(w.sum())
            else:
                volume=float(np.prod(hi-lo))
            if volume>1e-15:
                ii.append(i);jj.append(j);vv.append(volume)
    mat=coo_matrix((vv,(ii,jj)),shape=(len(rec_a),len(rec_b))).tocsr()
    va=np.asarray([r["volume"] for r in rec_a],float)
    vb=np.asarray([r["volume"] for r in rec_b],float)
    row=np.asarray(mat.sum(axis=1)).ravel()
    col=np.asarray(mat.sum(axis=0)).ravel()
    erow=float(np.sum(abs(row-va))/va.sum())
    ecol=float(np.sum(abs(col-vb))/vb.sum())
    if max(erow,ecol)>1e-6:
        raise ValueError(f"wet overlap volume inconsistency: {erow:.3g}, {ecol:.3g}")
    return mat,erow,ecol


def project_mat(mat, field):
    """Exact passive volume average of fine field onto native coarse FV."""
    n_a,n_b=mat.shape
    val=np.asarray(field)
    if val.shape[0]!=n_b:
        raise ValueError("source wet FV field count mismatch")
    vol=np.asarray(mat.sum(axis=1)).ravel()
    if np.any(vol<=0):
        raise ValueError("native coarse FV lacks positive wet overlap")
    out=(mat@val.reshape(n_b,-1))/vol[:,None]
    return out.reshape((n_a,)+val.shape[1:]),vol


def pair_l2(a,b,mat):
    """Exact integral of squared difference of piecewise constant fields."""
    n_a,n_b=mat.shape
    x=np.asarray(a).reshape(n_a,-1)
    y=np.asarray(b).reshape(n_b,-1)
    if x.shape[1]!=y.shape[1]:
        raise ValueError("BFZI spatial field vector dimensions mismatch")
    va=np.asarray(mat.sum(axis=1)).ravel()
    vb=np.asarray(mat.sum(axis=0)).ravel()
    power_a=float(va@np.sum(abs(x)**2,axis=1))
    power_b=float(vb@np.sum(abs(y)**2,axis=1))
    cross=np.sum(np.conj(x)*(mat@y)).real
    err=np.sqrt(max(0.,power_a+power_b-2*cross))
    return float(err),float(np.sqrt(power_b))


def compare_exact(a,b,mat):
    """Shared-volume raw, drive-normalized and thermal-shape errors."""
    va=np.asarray(mat.sum(axis=1)).ravel()
    vb=np.asarray(mat.sum(axis=0)).ravel()
    out={k:[] for k in ("raw","shape","thermal",
                         "delta_over_cold","delta_ref_over_cold")}
    cold_a=cold_b=None
    for k,(x,y) in enumerate(zip(a,b)):
        err,den=pair_l2(x,y,mat)
        out["raw"].append(err/den if den>1e-20 else None)
        if x.ndim==2:
            sx=shape_j(x,va,cold_a);sy=shape_j(y,vb,cold_b)
        else:
            sx=shape_q(x,va);sy=shape_q(y,vb)
        if k==0:
            cold_a,cold_b=sx,sy
        if sx is None or sy is None:
            out["shape"].append(None)
            out["thermal"].append(None)
            out["delta_over_cold"].append(None)
            out["delta_ref_over_cold"].append(None)
            continue
        err,den=pair_l2(sx,sy,mat)
        out["shape"].append(err/max(den,1e-30))
        if k and cold_a is not None and cold_b is not None:
            err,den=pair_l2(sx-cold_a,sy-cold_b,mat)
            out["thermal"].append(err/den if den>1e-12 else None)
            out["delta_over_cold"].append(err/max(norm(cold_b,vb),1e-30))
            out["delta_ref_over_cold"].append(den/max(norm(cold_b,vb),1e-30))
        else:
            out["thermal"].append(None)
            out["delta_over_cold"].append(None)
            out["delta_ref_over_cold"].append(None)
    return out


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
