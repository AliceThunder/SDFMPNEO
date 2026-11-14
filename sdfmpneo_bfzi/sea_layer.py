"""Finite-depth passive seawater thermal skin on original BFZI wet FVs.

Each physical package face has a positive-C / positive-G thermal ladder.
The water heat capacity of every shell layer is removed from its owning
original BFZI wet FV. Original weak Joule heating is redistributed by the
same water volumes, while the EM solve sees the volume-averaged water T.
There is no new BFZI electromagnetic cell or direction solve.
"""
from __future__ import annotations

import numpy as np
from itertools import product
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags,block_diag,csr_matrix

from .v148_pkg3d import BFZIV148Pkg3D
from .gap_bridge import bridge
from .near_source import local_j_var,split_heat


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
                    yield (int(start+a),loc,n,area,float(size[ax]),
                           inds,weights,dist,
                           rot[:,cross[0]]*size[cross[0]],
                           rot[:,cross[1]]*size[cross[1]])


def _owner(pt, lo, hi):
    ids=np.flatnonzero(np.all(
        (lo<=pt+1e-12)&(pt<hi+1e-12),axis=1))
    if len(ids)!=1:
        raise ValueError(f"water skin layer must have one BFZI wet cell: {len(ids)}")
    return int(ids[0])


def _qmap(subcells, faces, dx):
    """Map native wet-FV EM quadrature points to physical 3-D skin nodes.

    Off-skin points retain their original wet FV bulk temperature. Each
    layer prism is defined by its actual rotated package-face patch.
    """
    if subcells is None:
        return {}
    sizes=[len(item[0]) for item in subcells]
    cuts=np.r_[0,np.cumsum(sizes)]
    xyz=np.concatenate([item[0] for item in subcells])
    ids=np.repeat(np.arange(len(subcells)),sizes)
    seen=np.zeros(len(ids),bool)
    depths=np.cumsum(dx)
    for face in faces:
        v1,v2=face["v1"],face["v2"]
        d=xyz-face["center"]
        depth=d@face["normal"]
        u=(d@v1)/(v1@v1)
        v=(d@v2)/(v2@v2)
        hit=((depth>=-1e-12)&(depth<depths[-1]-1e-12)&
             (abs(u)<=.5+1e-12)&(abs(v)<=.5+1e-12))
        if np.any(hit&seen):
            raise ValueError("overlapping physical seawater skin prisms")
        layer=np.searchsorted(depths,depth[hit],side="right")
        ids[hit]=face["last"]-len(dx)+1+layer
        seen|=hit
    return {id(item[0]):ids[cuts[i]:cuts[i+1]].copy()
            for i,item in enumerate(subcells)}


class BFZIV148Skin3D(BFZIV148Pkg3D):
    """Conservative skin RC ladder replacing direct package-sea heat links.

    A single ladder per package surface FV face retains the original
    weak-source power, seawater total heat capacity, and two-port dynamics.
    """
    def __init__(self, model, *, skin_depth=.002, skin_layers=8,
                 skin_ratio=1.4, skin_quad=2, skin_near=4, skin_map=2,
                 gap_bridge=False, gap_max=.012, gap_quad=4, gap_align=.5,
                 sea_rhoc=4e6, sea_k=.6, pkg_k=.25,
                 pkg_sea_h=20., skin_heat="volume", **kwargs):
        if skin_heat not in ("volume","source"):
            raise ValueError("skin heat must be volume or source")
        if skin_heat=="source" and getattr(model,"subcells",None) is None:
            raise ValueError("source-resolved skin heating needs BFZI subcell source")
        self.skin_heat=skin_heat
        if (skin_depth<=0 or skin_layers<2 or skin_ratio<1 or
                skin_quad<1 or skin_near<1 or skin_map not in (1,2) or
                min(sea_k,sea_rhoc,pkg_k,pkg_sea_h)<=0 or
                gap_max<=0 or gap_quad<1 or not 0<=gap_align<1):
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
        share_i=[];share_j=[];share_v=[];share_pt=[]
        # Tensor Gauss sampling conservatively distributes every shell
        # prism to its original wet FV, avoiding centroid ownership jumps.
        frac_pts=([.5] if skin_map==1 else
                  [.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
        vwt=1/len(frac_pts)**3
        self.pkg_links=[[],[]]
        faces=[]
        tails=[]
        def link(i,j,g):
            if g<=0 or not np.isfinite(g):
                raise ValueError("passive skin thermal link requires positive G")
            edges.extend(((i,j,g),))
        for pk,face,n,area,pitch,inds,weights,dist,v1,v2 in _faces(
                self,skin_quad,skin_near):
            center=face.copy()
            start=orig_n+len(owner)
            x=0.
            for l,width in enumerate(dx):
                mid=center+(x+.5*width)*n
                cell=_owner(mid,lo,hi)
                owner.append(cell)
                shell_vol.append(area*width)
                j=start+l-orig_n
                for u,v,t in product(frac_pts,repeat=3):
                    pt=(center+(u-.5)*v1+(v-.5)*v2+
                        (x+t*width)*n)
                    cell=_owner(pt,lo,hi)
                    share_i.append(cell)
                    share_j.append(j)
                    share_v.append(area*width*vwt)
                    if skin_heat=="source":
                        share_pt.append(pt)
                x+=width
            faces.append(dict(
                id=len(faces),pkg=0 if pk<self.count+self.pkg_n else 1,
                center=center.copy(),normal=np.asarray(n,float).copy(),
                v1=np.asarray(v1,float).copy(),v2=np.asarray(v2,float).copy(),
                area=float(area),last=int(start+skin_layers-1)))
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
            tails.append((start+skin_layers-1,inds.copy(),
                          weights.copy(),g))
        gap_edges,covered,gap_info=(bridge(
            faces,depth=skin_depth,last=dx[-1],k=sea_k,
            gap=gap_max,quad=gap_quad,align=gap_align)
            if gap_bridge else
            ([],np.zeros(len(faces)),dict(
                bridges=0,area_m2=0.,G_W_K=0.,
                min_gap_m=None,max_gap_m=None)))
        for i,j,g in gap_edges:
            link(i,j,g)
        for f,(end,inds,weights,g) in enumerate(tails):
            exposed=max(0.,1-covered[f]/faces[f]["area"])
            for j,w in zip(inds,weights):
                if exposed*g*float(w)>0:
                    link(end,int(j),exposed*g*float(w))
        owner=np.asarray(owner,int)
        shell_vol=np.asarray(shell_vol,float)
        mix=coo_matrix(
            (share_v,(share_i,share_j)),
            shape=(self.count,len(owner))).tocsr()
        take=np.asarray(mix.sum(axis=1)).ravel()
        bulk=model.vol-take
        if np.any(bulk<=0):
            ids=np.flatnonzero(bulk<=0)
            raise ValueError(
                f"skin water capacity exceeds original BFZI wet FV: {ids[:8]}")
        self.bulk_vol=bulk
        self.skin_owner=owner
        self.skin_vol=shell_vol
        self.skin_mix=mix
        self._qmap=_qmap(getattr(model,"subcells",None),faces,dx)
        if skin_heat=="source":
            (self._src_pts,self._src_w,self._src_cell,
             self._src_C)=model.source_native()
            self._src_node=np.concatenate([
                self._qmap[id(item[0])] for item in model.subcells])
            self._shell_cell=np.asarray(share_i,int)
            self._shell_node=orig_n+np.asarray(share_j,int)
            self._shell_w=np.asarray(share_v,float)
            self._shell_C=model.source_at(np.asarray(share_pt,float))
            self._src_vol=np.bincount(
                self._src_cell,weights=self._src_w,minlength=self.count)
            if not np.allclose(self._src_vol,model.vol,rtol=1e-8,atol=1e-12):
                raise ValueError("native BFZI source quadrature volume mismatch")
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
            map_order=skin_map,heat_mode=skin_heat,
            faces=len(owner)//skin_layers,
            gap_bridge=bool(gap_bridge),gap=gap_info,
            gap_bulk_area_covered_m2=float(sum(covered)),
            redistributed_water_m3=float(shell_vol.sum()),
            max_wet_fraction=float(np.max(take/model.vol)),
            sea_cap_delta_J_K=float(np.sum(self.cap[:self.count])+
                np.sum(self.cap[self.skin])-
                sea_rhoc*np.sum(model.vol)))

    def sea_temp(self,theta):
        """Original BFZI sea-cell conductivity sees exact volume-mean T."""
        t=np.asarray(theta,float)
        n=self.count
        acc=self.bulk_vol*t[:n]+self.skin_mix@t[self.skin]
        return acc/self.model.vol

    def heat_vec(self,q,cu):
        """Original spatial Joule heat, redistributed without power change."""
        q=np.asarray(q,float)
        heat=np.zeros(self.n)
        heat[:self.count]=q*self.bulk_vol
        heat[self.skin]=self.skin_mix.T@q
        heat[self.cu]=cu
        return heat


    def source_heat(self,theta,unit_j,currents,q,cu):
        """Conserve wet-FV weak power while resolving 8-layer local sigma/J."""
        if self.skin_heat=="volume":
            return self.heat_vec(q,cu)
        theta=np.asarray(theta,float)
        q=np.asarray(q,float)
        I=np.asarray(currents,complex)
        unit_j=np.asarray(unit_j,complex)
        sea=self.amb+self.sea_temp(theta)
        raw=self.amb+theta[self._src_node]
        native_avg=np.bincount(
            self._src_cell,weights=self._src_w*raw,
            minlength=self.count)/self._src_vol
        offset=sea-native_avg
        temp=raw+offset[self._src_cell]
        sig=self.model.sigma0*(1+self.model.alpha*
                               (temp-self.model.ambient))
        skin_temp=self.amb+theta[self._shell_node]+offset[self._shell_cell]
        sig_skin=self.model.sigma0*(1+self.model.alpha*
                                    (skin_temp-self.model.ambient))
        if np.any(sig<=0) or np.any(sig_skin<=0):
            raise ValueError("nonpositive local BFZI seawater conductivity")
        ref=np.empty((self.count,2,3))
        for p in range(2):
            for ax in range(3):
                ref[:,p,ax]=np.bincount(
                    self._src_cell,weights=self._src_w*sig*
                    self._src_C[:,p,ax],minlength=self.count)/self._src_vol
        jbar=np.einsum("npc,p->nc",unit_j,I,optimize=True)
        omega_mu=self.model.omega*self.model.mu
        j=local_j_var(jbar,sig,I,self._src_C,ref,
                      self._src_cell,omega_mu)
        j_skin=local_j_var(jbar,sig_skin,I,self._shell_C,ref,
                           self._shell_cell,omega_mu)
        raw_q=.5*np.sum(abs(j)**2,axis=1)/sig
        raw_skin=.5*np.sum(abs(j_skin)**2,axis=1)/sig_skin
        base=np.bincount(
            self._src_cell,weights=self._src_w*raw_q,
            minlength=self.count)/self._src_vol
        bulk,skin=split_heat(
            q,self.model.vol,self.bulk_vol,base,
            self._shell_cell,self._shell_node-self.skin.start,
            self._shell_w,raw_skin,self.skin.stop-self.skin.start)
        heat=np.zeros(self.n)
        heat[:self.count]=bulk
        heat[self.skin]=skin
        heat[self.cu]=cu
        return heat

    def em(self,state,drive):
        theta=np.asarray(state,float)
        v=np.asarray(drive,complex)
        if theta.shape!=(self.n,) or v.shape!=(2,):
            raise ValueError("BFZI layered thermal state or voltage mismatch")
        sea=self.amb+self.sea_temp(theta)
        copper=self.amb+theta[self.cu]
        if self._qmap:
            # Same thermal state: exact conserved FV mean for Green,
            # physical eight-layer values at native BFZI source quadrature.
            def at_q(q):
                return self.amb+theta[self._qmap[id(q)]]
            warm=self.model.solve(sea,copper,sea_temp=at_q,
                                  sea_mean=True)
        else:
            warm=self.model.solve(sea,copper)
        I=np.linalg.solve(warm.total_z+self.external,v)
        q,cu=self.model.heat(warm,I)
        return warm,I,self.source_heat(theta,warm.unit_j,I,q,cu),q,cu
