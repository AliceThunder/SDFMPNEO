"""Finite-depth passive seawater thermal skin on original BFZI wet FVs.

Each physical package face has a positive-C / positive-G thermal ladder.
The water heat capacity of every shell layer is removed from its owning
original BFZI wet FV. Original weak Joule heating is redistributed by the
same water volumes, while the EM solve sees the volume-averaged water T.
There is no new BFZI electromagnetic cell or direction solve.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags,block_diag

from .v148_pkg3d import BFZIV148Pkg3D


def _faces(et, quad, near):
    """Coil-attached package face patches and weighted original wet cells."""
    wet=et.model.points
    tree=cKDTree(wet)
    k=min(len(wet),max(16,near*4))
    dim=np.asarray(et.pkg_shape)
    for p,(center,half) in enumerate(et.boxes):
        rot=et.pkg_axes[p]
        size=2*half/dim
        cells=et.pkg_xyz[p]
        start=et.count+p*et.pkg_n
        for t in np.ndindex(*dim):
            a=np.ravel_multi_index(t,dim)
            for ax in range(3):
                cross=[j for j in range(3) if j!=ax]
                for end,sgn in ((0,-1),(dim[ax]-1,1)):
                    if t[ax]!=end:
                        continue
                    n=sgn*rot[:,ax]
                    loc=cells[a]+sgn*.5*size[ax]*rot[:,ax]
                    area=float(np.prod(size[cross]))
                    dst={}
                    dist=0.
                    for i in range(quad):
                        for j in range(quad):
                            pt=(loc+
                                ((i+.5)/quad-.5)*size[cross[0]]*rot[:,cross[0]]+
                                ((j+.5)/quad-.5)*size[cross[1]]*rot[:,cross[1]])
                            d,ids=tree.query(pt,k=k)
                            d=np.atleast_1d(d)
                            ids=np.atleast_1d(ids)
                            valid=((wet[ids]-pt)@n)>1e-12
                            if not np.any(valid):
                                raise ValueError("no outward BFZI wet FV for skin")
                            d=d[valid][:near]
                            ids=ids[valid][:near]
                            w=1/(d*d+(.1*min(size))**2)
                            w/=w.sum()
                            dist+=float(w@d)/(quad*quad)
                            for idx,wt in zip(ids,w):
                                dst[int(idx)]=dst.get(int(idx),0.)+float(wt)/(quad*quad)
                    inds=np.array(list(dst),int)
                    weights=np.array(list(dst.values()),float)
                    weights/=weights.sum()
                    yield int(start+a),loc,n,area,float(size[ax]),inds,weights,dist


def _owner(pt, lo, hi):
    ids=np.flatnonzero(np.all(
        (lo<=pt+1e-12)&(pt<hi+1e-12),axis=1))
    if len(ids)!=1:
        raise ValueError(f"water skin layer must have one BFZI wet cell: {len(ids)}")
    return int(ids[0])


class BFZIV148Skin3D(BFZIV148Pkg3D):
    """Conservative skin RC ladder replacing direct package-sea heat links.

    A single ladder per package surface FV face retains the original
    weak-source power, seawater total heat capacity, and two-port dynamics.
    """
    def __init__(self, model, *, skin_depth=.002, skin_layers=8,
                 skin_ratio=1.4, skin_quad=2, skin_near=4,
                 sea_rhoc=4e6, sea_k=.6, pkg_k=.25,
                 pkg_sea_h=20., **kwargs):
        if (skin_depth<=0 or skin_layers<2 or skin_ratio<1 or
                skin_quad<1 or skin_near<1 or
                min(sea_k,sea_rhoc,pkg_k,pkg_sea_h)<=0):
            raise ValueError("positive seawater skin physical dimensions required")
        super().__init__(
            model,sea_rhoc=sea_rhoc,sea_k=sea_k,pkg_k=pkg_k,
            pkg_sea_model="face-resistance",pkg_sea_h=pkg_sea_h,
            pkg_sea_quad=skin_quad,pkg_sea_near=skin_near,**kwargs)
        self.skin_depth=float(skin_depth)
        self.skin_layers=int(skin_layers)
        self.skin_ratio=float(skin_ratio)
        frac=skin_ratio**np.arange(skin_layers)
        dx=skin_depth*frac/frac.sum()
        self.skin_dx=dx
        self.skin=slice(self.n,self.n)
        orig_n=self.n
        records=model.base["records"]
        lo=np.array([v["cell_min"] for v in records])
        hi=np.array([v["cell_max"] for v in records])
        cap=self.cap.copy()
        H=self.H.tolil()
        # Replace every old package-water edge; preserve all other heat
        # conductances, copper-package links and ambient leakage.
        for links in self.pkg_links:
            for i,j,g in links:
                H[i,i]-=g;H[j,j]-=g
                H[i,j]+=g;H[j,i]+=g
        edges=[]
        owner=[]
        shell_vol=[]
        self.pkg_links=[[],[]]
        def link(i,j,g):
            if g<=0 or not np.isfinite(g):
                raise ValueError("passive skin thermal link requires positive G")
            edges.extend(((i,j,g),))
        for pk,face,n,area,pitch,inds,weights,dist in _faces(
                self,skin_quad,skin_near):
            center=face.copy()
            start=orig_n+len(owner)
            x=0.
            for l,width in enumerate(dx):
                x+=.5*width
                pt=center+x*n
                cell=_owner(pt,lo,hi)
                owner.append(cell)
                shell_vol.append(area*width)
                x+=.5*width
            first=start
            pkg_r=1/pkg_sea_h+.5*pitch/pkg_k+.5*dx[0]/sea_k
            g=area/pkg_r
            link(pk,first,g)
            # Use this as the real package-to-skin flux interface.
            self.pkg_links[0 if pk<self.count+self.pkg_n else 1].append(
                (first,pk,g))
            for l in range(skin_layers-1):
                link(start+l,start+l+1,
                     sea_k*area/(.5*(dx[l]+dx[l+1])))
            # At steady state, the total water conduction distance from
            # physical face to wet FV center remains the original dist.
            tail=max(dist-skin_depth+.5*dx[-1],.5*dx[-1])
            g=sea_k*area/tail
            for j,w in zip(inds,weights):
                link(start+skin_layers-1,int(j),g*float(w))
        owner=np.asarray(owner,int)
        shell_vol=np.asarray(shell_vol,float)
        take=np.bincount(owner,weights=shell_vol,minlength=self.count)
        bulk=model.vol-take
        if np.any(bulk<=0):
            ids=np.flatnonzero(bulk<=0)
            raise ValueError(
                f"skin water capacity exceeds original BFZI wet FV: {ids[:8]}")
        self.bulk_vol=bulk
        self.skin_owner=owner
        self.skin_vol=shell_vol
        self.skin=slice(orig_n,orig_n+len(owner))
        self.n=self.skin.stop
        cap[:self.count]=sea_rhoc*bulk
        self.cap=np.r_[cap,sea_rhoc*shell_vol]
        self.amb_g=np.pad(self.amb_g,(0,len(owner)))
        adds_i=[];adds_j=[];adds_v=[]
        for i,j,g in edges:
            adds_i.extend((i,j,i,j))
            adds_j.extend((i,j,j,i))
            adds_v.extend((g,g,-g,-g))
        self.H=(block_diag(
            (H.tocsr(),coo_matrix((len(owner),len(owner))).tocsr()),
            format="csr")+
            coo_matrix((adds_v,(adds_i,adds_j)),
                       shape=(self.n,self.n)).tocsr())
        self._factor={}
        self.skin_info=dict(
            layers=skin_layers,depth_m=skin_depth,ratio=skin_ratio,
            faces=len(owner)//skin_layers,
            redistributed_water_m3=float(shell_vol.sum()),
            max_wet_fraction=float(np.max(take/model.vol)),
            sea_cap_delta_J_K=float(np.sum(self.cap[:self.count])+
                np.sum(self.cap[self.skin])-
                sea_rhoc*np.sum(model.vol)))

    def sea_temp(self,theta):
        """Original BFZI sea-cell conductivity sees exact volume-mean T."""
        t=np.asarray(theta,float)
        n=self.count
        acc=(self.bulk_vol*t[:n]+np.bincount(
            self.skin_owner,weights=self.skin_vol*t[self.skin],
            minlength=n))
        return acc/self.model.vol

    def heat_vec(self,q,cu):
        """Original spatial Joule heat, redistributed without power change."""
        q=np.asarray(q,float)
        heat=np.zeros(self.n)
        heat[:self.count]=q*self.bulk_vol
        heat[self.skin]=q[self.skin_owner]*self.skin_vol
        heat[self.cu]=cu
        return heat

    def em(self,state,drive):
        theta=np.asarray(state,float)
        v=np.asarray(drive,complex)
        if theta.shape!=(self.n,) or v.shape!=(2,):
            raise ValueError("BFZI layered thermal state or voltage mismatch")
        warm=self.model.solve(self.amb+self.sea_temp(theta),
                              self.amb+theta[self.cu])
        I=np.linalg.solve(warm.total_z+self.external,v)
        q,cu=self.model.heat(warm,I)
        return warm,I,self.heat_vec(q,cu),q,cu
