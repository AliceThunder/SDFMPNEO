"""Geometry-shared thermal coordinates with five material means and 27 shapes.

Fixed spatial functions replace geometry-dependent POD eigenvector ordering.
The same 32 indices describe the same modes for every BFZI geometry.
"""
from __future__ import annotations

import numpy as np

from .geom_rom import groups


def _shapes(head):
    """Temperature shapes in sea (13) and package A/B (7+7)."""
    sea = np.asarray(head.xyz,float)
    pkg = np.asarray(head.pkg_xyz,float)
    mid = pkg.mean(axis=1)
    origin = mid.mean(axis=0)
    span = np.maximum(np.ptp(sea,axis=0),.03)
    x,y,z = ((sea-origin)/span).T
    dist = np.linalg.norm(sea[:,None,:]-mid[None,:,:],axis=-1)
    extra=[np.sin(np.pi*x),np.sin(np.pi*y)]
    if hasattr(head,"nskin"):
        nwet,nskin=int(head.nwet),int(head.nskin)
        if nwet+nskin!=len(sea) or nskin%2:
            raise ValueError("invalid S5 sea-skin partition")
        # S5 traverses package surfaces in TX/RX order. Older anchors
        # retain this ordering even without explicit skin_pkg labels.
        pkg_id=(np.asarray(head.skin_pkg,int) if hasattr(head,"skin_pkg")
                else np.repeat(np.arange(2),nskin//2))
        if pkg_id.shape!=(nskin,) or np.any((pkg_id<0)|(pkg_id>1)):
            raise ValueError("invalid BFZI sea-skin package labels")
        extra=[]
        for j in range(2):
            mask=np.zeros(len(sea))
            mask[nwet:]=(pkg_id==j).astype(float)
            extra.append(mask)
    vals=[x,y,z,x*x,y*y,z*z,x*y,x*z,y*z,
          *extra,
          np.exp(-.5*(dist[:,0]/.09)**2),
          np.exp(-.5*(dist[:,1]/.09)**2)]
    features = [np.stack(vals,axis=1)]
    for j in range(2):
        v=(pkg[j]-mid[j])@np.asarray(head._pkg_axes[j],float)
        v/=np.maximum(2*np.asarray(head._pkg_half[j],float),1e-4)
        a,b,c=v.T
        # The original S5 package uses 4x2x4 cells. b**2 is constant
        # with two symmetric thickness layers, so use cross terms instead.
        r2=a*a+c*c
        features.append(np.stack([a,b,c,r2,r2*r2,b*r2,a*c],axis=1))
    return features


def labels(rank=32):
    if rank not in (5,32):
        raise ValueError("canonical thermal basis uses 5 or 32 modes")
    return np.array([0,1,2,3,4]+([0]*13+[1]*7+[2]*7 if rank==32 else []),int)


def basis(head,rank=32):
    """Capacity-orthonormal geometric basis; first five are material means."""
    lab=labels(rank)
    cap=np.asarray(head.cap,float)
    sl=groups(head)
    if np.min(cap)<=0:
        raise ValueError("thermal capacities must be positive")
    phi=np.zeros((len(cap),rank))
    for j,span in enumerate(sl):
        phi[span,j]=1/np.sqrt(np.sum(cap[span]))
    if rank==5:
        return phi
    for mat,raw in enumerate(_shapes(head)):
        span=sl[mat]
        weight=cap[span]
        ids=np.flatnonzero(lab==mat)
        n_extra=len(ids)-1
        if raw.shape!=(len(weight),n_extra):
            raise ValueError("canonical geometry features mismatch the thermal mesh")
        local=phi[span,:]
        for col,idx in enumerate(ids[1:]):
            vec=raw[:,col].copy()
            for prev in ids[:col+1]:
                vec-=(weight*vec)@local[:,prev]*local[:,prev]
            norm=np.sqrt((weight*vec)@vec)
            if norm<1e-10*np.sqrt(weight.sum()):
                raise ValueError(f"canonical material {mat} mode {col} is degenerate")
            phi[span,idx]=vec/norm
    return phi


def mode_scale(head,rank=32):
    """Convert energy-weighted physical coefficients to shared kelvin modes."""
    lab=labels(rank)
    cap=np.asarray(head.cap,float)
    energy=np.array([cap[sl].sum() for sl in groups(head)])
    return np.sqrt(energy[lab])


def project(head,temp,phi):
    """Project a temperature-rise field onto shared kelvin coefficients."""
    t=np.asarray(temp,float)
    if t.shape[-1]!=len(head.cap):
        raise ValueError("temperature field and canonical FV do not match")
    a=(t*np.asarray(head.cap))@phi
    return a/mode_scale(head,phi.shape[1])


def make_rom(head,rank=32):
    """Build a small Galerkin operator without per-geometry thermal snapshots."""
    from .geom_rom import GeomHeatROM
    return GeomHeatROM(head,basis(head,rank))
