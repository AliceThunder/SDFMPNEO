"""BFZI-free analytic spatial temperature field from common 32-mode states.

Fixed canonical raw functions and geometry-interpolated orthogonalization
coefficients replace the per-geometry FV mesh and basis at query time.
"""
from __future__ import annotations

import numpy as np

from .geom_init import encode, _w


def raw_field(points,region,origin,span,mid,axes,half):
    p=np.asarray(points,float).reshape(-1,3)
    reg=np.asarray(region,int).reshape(-1)
    if len(p)!=len(reg):
        raise ValueError("point/region size mismatch")
    out=np.zeros((len(p),32),float)
    out[:,0]=1.
    sea=(reg==0)|(reg==5)|(reg==6)
    if np.any(sea):
        pts=p[sea]
        xyz=(pts-origin)/span
        x,y,z=xyz.T
        dist=np.linalg.norm(pts[:,None,:]-mid[None,:,:],axis=-1)
        rr=reg[sea]
        raw=np.stack([x,y,z,x*x,y*y,z*z,x*y,x*z,y*z,
                      (rr==5).astype(float),(rr==6).astype(float),
                      np.exp(-.5*(dist[:,0]/.09)**2),
                      np.exp(-.5*(dist[:,1]/.09)**2)],axis=1)
        out[sea,1:14]=raw
    for j in (1,2):
        mask=reg==j
        if not np.any(mask):
            continue
        local=(p[mask]-mid[j-1])@axes[j-1]
        local/=np.maximum(2*half[j-1],1e-4)
        x,y,z=local.T
        r2=x*x+z*z
        out[mask,1:8]=np.stack([x,y,z,r2,r2*r2,y*r2,x*z],axis=1)
    return out


def raw_skin(points,region,mid,axes,half):
    """V7 16 seawater-skin canonical raw modes on user-supplied world points."""
    p=np.asarray(points,float).reshape(-1,3)
    r=np.asarray(region,int).reshape(-1)
    out=np.zeros((len(p),16),float)
    for j in range(2):
        if not np.any(r==j+5):
            continue
        local=(p-mid[j])@axes[j]
        local/=np.maximum(2*half[j],1e-4)
        a,b,c=local.T
        f=np.stack((a,b,c,a*a,c*c,a*c,a*b,b*c),axis=1)
        out[:,j*8:(j+1)*8]=f*(r==j+5)[:,None]
    return out


class FieldAtlas:
    """Continuous per-material temperature-shape interpolation."""

    def __init__(self,raw):
        self.x=np.asarray(raw['x'],float)
        self.mu=np.asarray(raw['mu'],float)
        self.std=np.asarray(raw['std'],float)
        self.ell=float(raw['ell'])
        self.qsea=np.asarray(raw['qsea'],float)
        self.qpkg=np.asarray(raw['qpkg'],float)
        self.qskin=np.asarray(raw.get('qskin',[]),float)
        self.origin=np.asarray(raw['origin'],float)
        self.span=np.asarray(raw['span'],float)
        self.mid=np.asarray(raw['mid'],float)
        self.axes=np.asarray(raw['axes'],float)
        self.half=np.asarray(raw['half'],float)
        if (self.qsea.shape!=(len(self.x),14,13) or
                self.qpkg.shape!=(len(self.x),2,8,7) or
                (self.qskin.size and
                 self.qskin.shape!=(len(self.x),30,16))):
            raise ValueError("invalid spatial temperature shape atlas")

    def weights(self,case):
        feat=(encode(case)-self.mu)/self.std
        return _w(feat[None],self.x,self.ell)[0]

    def predict(self,case,points,region,weights=None):
        """Return a [num_points,32] temperature shape basis in kelvin modes."""
        p=np.asarray(points,float).reshape(-1,3)
        r=np.asarray(region,int).reshape(-1)
        if len(p)!=len(r) or np.any(~np.isin(r,[0,1,2,3,4,5,6])):
            raise ValueError("regions: 0 bulk sea, 5/6 sea skin, 1/2 pkg, 3/4 copper")
        w=self.weights(case)
        origin=np.einsum('n,ni->i',w,self.origin)
        span=np.einsum('n,ni->i',w,self.span)
        mid=np.einsum('n,nij->ij',w,self.mid)
        axes=np.einsum('n,nijk->ijk',w,self.axes)
        for j in range(2):
            u,_,v=np.linalg.svd(axes[j])
            axes[j]=u@v
        half=np.einsum('n,nij->ij',w,self.half)
        raw=raw_field(p,r,origin,span,mid,axes,half)
        mat=np.where(r>=5,0,r)
        shapes=np.zeros((len(p),32),float)
        for m in range(5):
            idx=np.flatnonzero(mat==m)
            if not len(idx):
                continue
            shapes[idx,m]=1.
            if m==0:
                Q=np.einsum('n,nij->ij',w,self.qsea)
                shapes[np.ix_(idx,np.arange(5,18))]=raw[idx,:14]@Q
            elif m in (1,2):
                Q=np.einsum('n,nij->ij',w,self.qpkg[:,m-1])
                start=18+(m-1)*7
                shapes[np.ix_(idx,np.arange(start,start+7))]=raw[idx,:8]@Q
        if weights is not None:
            cap=np.asarray(weights,float).reshape(-1)
            if cap.shape!=(len(p),) or np.any(cap<=0) or not np.isfinite(cap).all():
                raise ValueError("positive point capacities are required")
            for m,lo,hi in ((0,5,18),(1,18,25),(2,25,32)):
                idx=np.flatnonzero(mat==m)
                if len(idx):
                    mean=np.average(shapes[idx,lo:hi],axis=0,weights=cap[idx])
                    shapes[idx,lo:hi]-=mean
        return shapes

    def predict_skin(self,case,points,region,weights=None):
        if not self.qskin.size:
            raise ValueError("16-mode seawater field atlas is not trained")
        p=np.asarray(points,float).reshape(-1,3)
        r=np.asarray(region,int).reshape(-1)
        w=self.weights(case)
        origin=w@self.origin
        span=w@self.span
        mid=np.einsum('n,nij->ij',w,self.mid)
        axes=np.einsum('n,nijk->ijk',w,self.axes)
        for j in range(2):
            u,_,v=np.linalg.svd(axes[j])
            axes[j]=u@v
        half=np.einsum('n,nij->ij',w,self.half)
        old=raw_field(p,r,origin,span,mid,axes,half)[:,:14]
        skin=raw_skin(p,r,mid,axes,half)
        raw=np.column_stack((old,skin))
        Q=np.einsum('n,nij->ij',w,self.qskin)
        out=raw@Q
        sea=(r==0)|(r==5)|(r==6)
        out[~sea]=0.
        if weights is not None and np.any(sea):
            cap=np.asarray(weights,float).reshape(-1)
            if cap.shape!=(len(p),) or np.any(cap<=0):
                raise ValueError("positive point capacities are required")
            out[sea]-=np.average(out[sea],axis=0,weights=cap[sea])
        return out

    def query(self,case,points,region,state,ambient=293.15,weights=None,
              skin=None):
        st=np.asarray(state,float)
        if st.ndim==1:
            st=st[None]
        if st.ndim!=2 or st.shape[1]!=32:
            raise ValueError("shared heat state must have 32 kelvin coordinates")
        b=self.predict(case,points,region,weights)
        out=ambient+st@b.T
        if skin is not None:
            extra=np.asarray(skin,float)
            if extra.shape!=(len(st),16):
                raise ValueError("sea-skin temporal modes must match [time,16]")
            out+=extra@self.predict_skin(case,points,region,weights).T
        return out

    def save(self,path):
        np.savez_compressed(path,x=self.x,mu=self.mu,std=self.std,
            ell=self.ell,qsea=self.qsea,qpkg=self.qpkg,qskin=self.qskin,
            origin=self.origin,span=self.span,mid=self.mid,
            axes=self.axes,half=self.half)

    @classmethod
    def load(cls,path):
        with np.load(path) as f:
            return cls({k:f[k] for k in f.files})


def learn(rows,init,skin=False):
    """Fit the exact canonical spatial modes on the original training anchors."""
    from .geom_canon import basis,mode_scale
    from .geom_heat import GeomHeat
    from .geom_rom import groups
    qs=[];qp=[];qsk=[];orig=[];span=[];mid=[];axes=[];half=[]
    for row in rows:
        head=GeomHeat.load(row['anchor'])
        phi=basis(head,32)
        scale=mode_scale(head,32)
        center=head.pkg_xyz.mean(axis=1)
        org=center.mean(axis=0)
        sp=np.maximum(np.ptp(head.xyz,axis=0),.03)
        points=np.concatenate((head.xyz,*head.pkg_xyz),axis=0)
        reg=np.r_[np.zeros(head.nwet,int),
                  np.repeat([5,6],head.nskin//2),
                  np.repeat(1,head.npkg),np.repeat(2,head.npkg),
                  3,4]
        # Copper has no spatial shape functions; its coordinates are
        # placeholders needed only for aligning the material vector.
        if len(points)!=len(reg)-2:
            raise ValueError("unexpected thermal field layout")
        reg=reg[:-2]
        f=raw_field(points,reg,org,sp,center,
                    head._pkg_axes,head._pkg_half)
        sl=groups(head)
        q0=np.linalg.lstsq(f[sl[0],:14],
                           phi[sl[0],5:18]*scale[5:18],rcond=1e-10)[0]
        q12=[]
        for j in (1,2):
            a=18+(j-1)*7
            q12.append(np.linalg.lstsq(
                f[sl[j],:8],phi[sl[j],a:a+7]*scale[a:a+7],rcond=1e-10)[0])
        if skin:
            phi48=basis(head,48)
            scale48=mode_scale(head,48)
            rskin=raw_skin(points,reg,center,
                           head._pkg_axes,head._pkg_half)
            R=np.column_stack((f[sl[0],:14],rskin[sl[0]]))
            qsk.append(np.linalg.lstsq(
                R,phi48[sl[0],32:48]*scale48[32:48],rcond=1e-10)[0])
        qs.append(q0);qp.append(q12);orig.append(org);span.append(sp)
        mid.append(center);axes.append(head._pkg_axes);half.append(head._pkg_half)
    return FieldAtlas(dict(x=init.x,mu=init.mu,std=init.std,ell=init.ell,
        qsea=np.asarray(qs),qpkg=np.asarray(qp),
        qskin=np.asarray(qsk),
        origin=np.asarray(orig),span=np.asarray(span),
        mid=np.asarray(mid),axes=np.asarray(axes),half=np.asarray(half)))
