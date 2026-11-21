"""Geometry-only cold port and shared 32/48/64-mode thermal operator emulator.

No BFZI solve, geometry-specific FV grid or POD basis is needed at query time.
The frozen training atlas contains geometry-disjoint cold anchors.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def encode(case):
    """The user-controlled M4 geometry, pose, package and thermal parameters."""
    g=case['geometry']
    t=case['thermal']
    vals=[g['alpha1'],g['alpha2'],
          g['core1_mm'],g['core2_mm'],*g['shift_mm'],
          g['tilt_deg'],g['yaw_deg'],g['pitch_deg'],
          g['box1_mm'],g['box2_mm'],
          case['alpha_per_K'],t['ins_k'],t.get('ins_t',.0004),
          t['pkg_k'],t['pkg_sea_h'],
          1. if case['donor']=='g2' else 0.]
    out=np.asarray(vals,float)
    if out.shape!=(18,) or not np.isfinite(out).all():
        raise ValueError('expected 18 finite geometry/material input scalars')
    return out


def _rbf(a,b,ell):
    delta=a[:,None,:]-b[None,:,:]
    d=np.sqrt(np.mean(delta*delta,axis=-1)+1e-20)
    r=np.sqrt(3.)*d/ell
    return (1+r)*np.exp(-r)


def _w(a,b,ell):
    s=_rbf(a,b,ell)
    return s/np.maximum(s.sum(1,keepdims=True),1e-20)


class GeomInit:
    """Cold/source interpolants with an independently weighted SPD thermal atlas."""

    def __init__(self,raw):
        self.x=np.asarray(raw['x'],float)
        self.mu=np.asarray(raw['mu'],float)
        self.std=np.asarray(raw['std'],float)
        self.y=np.asarray(raw['y'],float)
        self.ym=np.asarray(raw['ym'],float)
        self.ys=np.asarray(raw['ys'],float)
        self.beta=np.asarray(raw['beta'],float)
        self.k=np.asarray(raw['k'],float)
        self.src=np.asarray(raw['src'],float)
        self.ext=np.asarray(raw['ext'],complex)
        self.ell=float(raw['ell'])
        self.kell=float(raw.get('kell',self.ell))
        self.lam=float(raw['lam'])
        self.sx=np.asarray(raw.get('sx',[]),float)
        self.smu=np.asarray(raw.get('smu',np.zeros(3)),float)
        self.sstd=np.asarray(raw.get('sstd',np.ones(3)),float)
        self.sy=np.asarray(raw.get('sy',np.zeros(2)),float)
        self.sbeta=np.asarray(raw.get('sbeta',np.zeros((0,2))),float)
        self.sell=float(raw.get('sell',1.))
        self.slam=float(raw.get('slam',0.))
        self.guard=np.asarray(raw.get('guard',[]),float)
        self.gmat=np.repeat(np.arange(5,self.k.shape[1]),2)
        self.rank=int(self.k.shape[1])
        self.kexp=np.asarray(raw.get('kexp',np.zeros(3)),float)
        self.knorm=None
        if (self.x.shape[1]!=18 or self.rank<5 or
                self.k.shape[1:]!=(self.rank,self.rank) or
                self.src.shape[1:]!=(self.rank,3)):
            raise ValueError('invalid cold/thermal atlas dimensions')
        self.set_kscale(self.kexp)
        self.kmm_edges=self._build_kmm_edges()


    def _kscale(self,x):
        from .geom_canon import labels
        a=np.asarray(x,float)
        pkg=(a[...,15]/.25)**self.kexp[0]*(a[...,16]/20.)**self.kexp[1]
        cu=(a[...,13]/a[...,14]/625.)**self.kexp[2]
        m=np.ones(a.shape[:-1]+(5,))
        m[...,1:3]=pkg[...,None]
        m[...,3:]=cu[...,None]
        return np.sqrt(m[...,labels(self.rank)])

    def set_kscale(self,exp):
        exp=np.asarray(exp,float)
        if exp.shape!=(3,):
            raise ValueError("thermal physical parameter exponents must have length three")
        self.kexp=exp
        if self.k.size:
            raw=self.x*self.std+self.mu
            scale=self._kscale(raw)
            self.knorm=self.k/(scale[:,:,None]*scale[:,None,:])
        return self

    @staticmethod
    def fit(x,y,k,src,ext,ell=1.5,lam=.03,kell=None):
        x=np.asarray(x,float)
        y=np.asarray(y,float)
        mu=x.mean(0)
        std=np.maximum(x.std(0),np.array([.02]*2+[1.]*2+[1.]*3+
                              [1.]*3+[2.]*2+[.001,.01,.00001,.01,1.,.15]))
        x=(x-mu)/std
        ym=y.mean(0)
        ys=np.maximum(y.std(0),1e-10)
        z=(y-ym)/ys
        h=_rbf(x,x,ell)
        beta=np.linalg.solve(h+lam*np.eye(len(x)),z)
        return GeomInit(dict(x=x,mu=mu,std=std,y=y,ym=ym,ys=ys,
                             beta=beta,k=k,src=src,ext=ext,
                             ell=ell,kell=ell if kell is None else kell,lam=lam))

    def set_kell(self,ell):
        if not np.isfinite(ell) or ell<=0:
            raise ValueError("thermal kernel length must be positive")
        self.kell=float(ell)
        return self

    def set_guard(self,guard):
        guard=np.asarray(guard,float)
        if guard.shape!=(len(self.x),2*(self.rank-5),self.rank-5):
            raise ValueError("inconsistent canonical geometry positivity samples")
        self.guard=guard
        return self

    def fit_self(self,rows,ell,lam,dims=2):
        if dims not in (2,3):
            raise ValueError('coil self predictor uses two or three inputs')
        x=[];y=[]
        for row in rows:
            g=row['case']['geometry']
            for j in (1,2):
                v=[g[f'alpha{j}'],g[f'core{j}_mm'],g[f'box{j}_mm']]
                x.append(v[:dims])
                idx=0 if j==1 else 4
                y.append(row['y'][idx:idx+2])
        x=np.asarray(x,float)
        y=np.asarray(y,float)
        self.smu=x.mean(0)
        self.sstd=np.maximum(x.std(0),[.01,1.,2.][:dims])
        self.sx=(x-self.smu)/self.sstd
        self.sy=y.mean(0)
        scale=np.maximum(y.std(0),1e-8)
        self.sbeta=np.linalg.solve(_rbf(self.sx,self.sx,ell)+
                   lam*np.eye(len(x)),(y-self.sy)/scale)*scale
        self.sell=float(ell)
        self.slam=float(lam)
        return self

    def _build_kmm_edges(self):
        """Extract the conservative five-material conductance graph from anchors.

        The physical topology is sea<->TX/RX package<->TX/RX copper.
        Only seawater exchanges heat with ambient. Interpolating these five
        positive conductances preserves every internal material energy sum.
        """
        cap=np.exp(self.y[:,9:14])
        mass=np.sqrt(cap)
        g=mass[:,:,None]*self.k[:,:5,:5]*mass[:,None,:]
        row=g.sum(axis=2)
        forbidden=((0,3),(0,4),(1,2),(1,4),(2,3),(3,4))
        allowed=((0,1),(0,2),(1,3),(2,4))
        scale=max(float(np.max(abs(g))),1.)
        tol=1e-8*scale
        if np.max(abs(g-np.swapaxes(g,1,2)))>tol:
            return None
        if max(float(np.max(abs(g[:,i,j]))) for i,j in forbidden)>tol:
            return None
        if np.max(abs(row[:,1:]))>tol or np.min(row[:,0])<=tol:
            return None
        edges=np.column_stack((row[:,0],
            *[-g[:,i,j] for i,j in allowed]))
        if np.min(edges)<=tol or not np.isfinite(edges).all():
            return None
        return edges

    def _predict_kmm(self,w,cap):
        """Build Kmm from positive ambient/interface conductances."""
        h,g1,g2,g3,g4=np.asarray(w,float)@self.kmm_edges
        g=np.zeros((5,5),float)
        g[0,0]=h+g1+g2
        g[1,1]=g1+g3
        g[2,2]=g2+g4
        g[3,3]=g3
        g[4,4]=g4
        g[0,1]=g[1,0]=-g1
        g[0,2]=g[2,0]=-g2
        g[1,3]=g[3,1]=-g3
        g[2,4]=g[4,2]=-g4
        mass=np.sqrt(np.asarray(cap,float))
        return g/(mass[:,None]*mass[None,:])

    def predict(self,case,external=None):
        raw=encode(case)
        a=((raw-self.mu)/self.std)[None]
        h=_rbf(a,self.x,self.ell)
        y=self.ym+(h@self.beta)[0]*self.ys
        w=_w(a,self.x,self.ell)[0]
        wk=w if self.kell==self.ell else _w(a,self.x,self.kell)[0]
        # A separate thermal bandwidth leaves cold, capacity and source fits intact.
        scale=self._kscale(raw)
        k=np.einsum('n,nij->ij',wk,self.knorm,optimize=True)
        k*=scale[:,None]*scale[None,:]
        k=.5*(k+k.T)
        src=np.einsum('n,nij->ij',w,self.src,optimize=True)
        # The external matching network belongs to a known circuit family.
        # Do not interpolate the calibrated external reactance across families.
        donor=(self.x[:,-1]*self.std[-1]+self.mu[-1]>.5)
        family=case['donor']=='g2'
        inds=np.flatnonzero(donor==family)
        ext=self.ext[inds[0]] if len(inds) else self.ext[np.argmax(w)]
        z=y[:6].copy()
        if self.sx.size:
            g=case['geometry']
            q=np.array([[g[f'alpha{i}'],g[f'core{i}_mm'],g[f'box{i}_mm']]
                        for i in (1,2)])[:,:self.sx.shape[1]]
            t=_rbf((q-self.smu)/self.sstd,self.sx,self.sell)
            values=self.sy+t@self.sbeta
            z[[0,1]]=values[0]
            z[[4,5]]=values[1]
        air=y[6:9]
        cap=np.exp(np.clip(y[9:14],-30,30))
        if self.kmm_edges is not None:
            k[:5,:5]=self._predict_kmm(wk,cap)
        if external is not None:
            ext=np.asarray(external,complex)
            if ext.shape!=(2,2):
                raise ValueError("external circuit must be 2x2 complex matrix")
        chosen=np.argsort(-wk)[:min(3,len(wk))]
        guard=self.guard[chosen].reshape(-1,self.rank-5) if self.guard.size else None
        gmat=np.tile(self.gmat,len(chosen)) if guard is not None else None
        from .geom_canon import labels
        gmat=labels(self.rank)[gmat] if guard is not None else None
        return dict(z0=z,air=air,cap=cap,k=k,src=src,ext=ext,
                    guard=guard,gmat=gmat,norm_weights=w,
                    thermal_norm_weights=wk)

    def save(self,path):
        np.savez_compressed(path,x=self.x,mu=self.mu,std=self.std,
              y=self.y,ym=self.ym,ys=self.ys,beta=self.beta,
              k=self.k,src=self.src,ext=self.ext,
              ell=self.ell,kell=self.kell,lam=self.lam,sx=self.sx,smu=self.smu,
              sstd=self.sstd,sy=self.sy,sbeta=self.sbeta,
              sell=self.sell,slam=self.slam,guard=self.guard,
              kexp=self.kexp)

    @classmethod
    def load(cls,path):
        with np.load(Path(path)) as f:
            raw={k:f[k] for k in f.files}
        return cls(raw)


def balance_cu(init):
    """Reconcile Cu/package conductance while retaining a positive heat ROM.

    The copper self block D specifies heat exchange G=Kcc*Ccu.
    Replace the interpolated Cu-to-package mean coupling B by the
    physically required -G/sqrt(Cpkg*Ccu), preserving the original
    non-copper Schur complement A-B*inv(D)*B.T. This maintains the
    dissipative shared-rank operator instead of altering an isolated 5x5 block.
    """
    out=dict(init)
    cap=np.asarray(init["cap"],float)
    k=np.asarray(init["k"],float).copy()
    if (cap.shape!=(5,) or k.ndim!=2 or k.shape[0]!=k.shape[1]
            or k.shape[0] not in (32,48,64)):
        raise ValueError("expected five material capacities and a 32/48/64-mode K")
    if np.min(cap)<=0 or not np.isfinite(k).all():
        raise ValueError("invalid capacity or thermal operator")
    cu=np.array([3,4])
    rest=np.array([i for i in range(len(k)) if i not in cu])
    dc=k[np.ix_(cu,cu)]
    old=k[np.ix_(rest,cu)]
    new=old.copy()
    for j,p in enumerate((1,2)):
        new[int(np.flatnonzero(rest==p)[0]),j]=(
            -dc[j,j]*np.sqrt(cap[cu[j]]/cap[p]))
    delta=new@np.linalg.solve(dc,new.T)-old@np.linalg.solve(dc,old.T)
    k[np.ix_(rest,rest)]+=delta
    k[np.ix_(rest,cu)]=new
    k[np.ix_(cu,rest)]=new.T
    out["k"]=.5*(k+k.T)
    return out


def conductance_mix(atlas,init):
    """Cross-geometry heat ROM in physical conductance coordinates.

    Interpolate H=M K M rather than K, with M=sqrt(material capacity).
    The Cu-package exchange identity is linear in H and survives mixing.
    Scale each conservative copper branch to the independently predicted
    copper self-conductance, retaining the non-copper Schur complement.
    """
    from .geom_canon import labels
    lab=labels(atlas.rank)
    m=np.sqrt(np.exp(atlas.y[:,9:14])[:,lab])
    bank=atlas.k*m[:,:,None]*m[:,None,:]
    # The thermal atlas has its own selected bandwidth (kell). Keep the
    # conductance-coordinate branch on the same thermal weights as predict().
    w=np.asarray(init.get("thermal_norm_weights",init["norm_weights"]),float)
    h=np.einsum("n,nij->ij",w,bank,optimize=True)
    cu=np.array([3,4])
    rest=np.array([i for i in range(atlas.rank) if i not in cu])
    base=h[np.ix_(cu,cu)]
    b=h[np.ix_(rest,cu)]
    target=np.array([init["k"][i,i]*init["cap"][i] for i in cu])
    if np.min(target)<=0:
        raise ValueError("invalid copper self-conductance")
    factors=target/np.diag(base)
    bnew=b*factors[None,:]
    dnew=np.diag(target)
    anew=(h[np.ix_(rest,rest)]-
          b@np.linalg.solve(base,b.T)+
          bnew@np.linalg.solve(dnew,bnew.T))
    h[np.ix_(rest,rest)]=anew
    h[np.ix_(rest,cu)]=bnew
    h[np.ix_(cu,rest)]=bnew.T
    h[np.ix_(cu,cu)]=dnew
    mass=np.sqrt(np.asarray(init["cap"],float)[lab])
    k=h/(mass[:,None]*mass[None,:])
    if atlas.kmm_edges is not None:
        w=np.asarray(init.get("thermal_norm_weights",init["norm_weights"]),float)
        k[:5,:5]=atlas._predict_kmm(w,init["cap"])
    out=dict(init)
    out["k"]=.5*(k+k.T)
    return out



def thermal_mix(atlas,init,weight):
    """Blend two SPD thermal atlas representations in shared ROM coordinates."""
    w=float(weight)
    if not 0.<=w<=1.:
        raise ValueError("thermal blend must be in [0,1]")
    if w==0.:
        return init
    phys=conductance_mix(atlas,init)
    out=dict(init)
    out["k"]=(1.-w)*np.asarray(init["k"])+w*phys["k"]
    return out


def mix_init(case,base,extra=None,weight=0.):
    """Cold electrical blending; preserve the base dissipative thermal operator."""
    w=float(weight)
    if not 0<=w<=1:
        raise ValueError('cold interpolation weight must be in [0,1]')
    if w and extra is None:
        raise ValueError('cold blending requires an additional atlas')
    init=base.predict(case)
    if w:
        rhs=extra.predict(case)
        for key in ('z0','air'):
            init[key]=(1-w)*init[key]+w*rhs[key]
    return init
