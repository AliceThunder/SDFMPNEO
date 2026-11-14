"""Three-dimensional finite-volume thermal field inside both BFZI packages.

The original BFZI electromagnetic geometry and 3-D seawater FV stay unchanged.
Each package box is a Cartesian conduction grid coupled to the original sea
cells across its outer surfaces and to its own copper winding.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.spatial import cKDTree

from .v148_et import BFZIV148ET
from .pose import axes,isinside
from .pkg_interface import face_links,adjacent_links


class BFZIV148Pkg3D(BFZIV148ET):
    """3-D conduction in two independently sized coil-attached package boxes."""

    def __init__(self,model,*,pkg_shape=(4,2,4),pkg_k=.25,
                 sea_rhoc=4e6,sea_k=.6,pkg_rhoc=1.5e6,
                 pkg_sea_g=.6,cu_pkg_g=None,outer_h=10.,
                  cu_ins_k=.25,cu_ins_t=None,
                 pkg_sea_model="nearest",pkg_sea_h=20.,
                 pkg_sea_quad=2,pkg_sea_near=4,**kwargs):
        if (not np.isfinite(cu_ins_k) or cu_ins_k<=0 or
                (cu_ins_t is not None and
                 (not np.isfinite(cu_ins_t) or cu_ins_t<=0)) or
                (cu_pkg_g is not None and
                 (not np.isfinite(cu_pkg_g) or cu_pkg_g<=0))):
            raise ValueError("positive finite winding insulation parameters required")
        super().__init__(
            model,sea_rhoc=sea_rhoc,sea_k=sea_k,pkg_rhoc=pkg_rhoc,
            pkg_sea_g=pkg_sea_g,
            cu_pkg_g=.8 if cu_pkg_g is None else cu_pkg_g,
            outer_h=outer_h,**kwargs)
        dims=tuple(int(x) for x in pkg_shape)
        if len(dims)!=3 or min(dims)<1 or pkg_k<=0:
            raise ValueError("positive 3-D package grid and conductivity required")
        if pkg_sea_model not in ("nearest","face-resistance","face-adjacent","face-normal"):
            raise ValueError("BFZI package-sea interface mode invalid")
        self.pkg_sea_model=pkg_sea_model
        self.pkg_sea_h=float(pkg_sea_h)
        self.pkg_shape=dims
        self.pkg_n=int(np.prod(dims))
        cap_cu=self.cap[self.cu].copy()
        self.pkg=slice(self.count,self.count+2*self.pkg_n)
        self.cu=slice(self.pkg.stop,self.pkg.stop+2)
        self.n=self.cu.stop
        self.cap=np.zeros(self.n)
        self.cap[:self.count]=sea_rhoc*model.vol
        self.cap[self.cu]=cap_cu
        self.pkg_xyz=[]
        self.pkg_vol=[]
        self.pkg_axes=[axes(fam.coil) for fam in model.base["families"]]
        self.pkg_links=[]
        self.cu_links=[]
        self.cu_pkg_total=[]
        self.cu_ins_layers=[]
        rows,cols,vals=[],[],[]
        def link(i,j,g):
            if g<=0:
                return
            rows.extend((i,j,i,j))
            cols.extend((i,j,j,i))
            vals.extend((g,g,-g,-g))
        for i,j,ax,dist,area,direction in model.proj.edges:
            link(i,j,sea_k*area/dist)
        for p,(center,half) in enumerate(self.boxes):
            center=np.asarray(center,float)
            half=np.asarray(half,float)
            rot=self.pkg_axes[p]
            size=2*half/np.asarray(dims)
            vol=float(np.prod(size))
            cells=np.array([center+rot@(-half+(np.asarray(t)+.5)*size)
                            for t in np.ndindex(*dims)])
            self.pkg_xyz.append(cells)
            self.pkg_vol.append(np.full(self.pkg_n,vol))
            start=self.count+p*self.pkg_n
            self.cap[start:start+self.pkg_n]=pkg_rhoc*vol
            tree=cKDTree(cells)
            # Conservative in-package face conductances.
            for t in np.ndindex(*dims):
                a=np.ravel_multi_index(t,dims)
                for ax in range(3):
                    if t[ax]+1>=dims[ax]:
                        continue
                    nb=list(t)
                    nb[ax]+=1
                    b=np.ravel_multi_index(tuple(nb),dims)
                    area=vol/size[ax]
                    link(start+a,start+b,pkg_k*area/size[ax])
            if pkg_sea_model=="face-adjacent":
                links=adjacent_links(
                    model.points,model.base["records"],cells,half,rot,
                    dims,start,pkg_k,sea_k,pkg_sea_h,quad=pkg_sea_quad)
            elif pkg_sea_model in ("face-resistance","face-normal"):
                links=face_links(
                    model.points,cells,half,rot,dims,start,
                    pkg_k,sea_k,pkg_sea_h,
                    quad=pkg_sea_quad,near=pkg_sea_near,
                    distance=("normal" if pkg_sea_model=="face-normal"
                              else "euclidean"))
            else:
                # Original BFZI V150 thermal baseline, unchanged.
                faces=[]
                for t in np.ndindex(*dims):
                    a=np.ravel_multi_index(t,dims)
                    for ax in range(3):
                        area=vol/size[ax]
                        for end,sign in ((0,-1),(dims[ax]-1,1)):
                            if t[ax]!=end:
                                continue
                            loc=cells[a]+sign*.5*size[ax]*rot[:,ax]
                            sea=int(self.tree.query(loc)[1])
                            faces.append((sea,start+a,area))
                area_all=sum(item[2] for item in faces)
                links=[(sea,pk,pkg_sea_g*area/area_all)
                       for sea,pk,area in faces]
            for sea,pk,cond in links:
                link(sea,pk,cond)
            self.pkg_links.append(links)
            # Put copper heat into boxes near the true BFZI winding,
            # weighted by each physical winding element's length.
            fam=model.base["families"][p]
            pos=np.asarray(fam.active.pos,float)
            dl=np.asarray(fam.active.dl,float)
            if len(pos)==0:
                raise ValueError("BFZI winding source must contain line elements")
            near=min(4,self.pkg_n)
            dist,ind=tree.query(pos,k=near)
            dist=np.asarray(dist).reshape(len(pos),near)
            ind=np.asarray(ind).reshape(len(pos),near)
            weight=1/np.maximum(dist,min(size)/2)
            weight/=np.sum(weight,axis=1)[:,None]
            length=np.linalg.norm(dl,axis=1)
            mass=np.bincount(ind.ravel(),
                             weights=(weight*length[:,None]).ravel(),
                             minlength=self.pkg_n)
            if np.sum(mass)<=0:
                raise ValueError("BFZI winding segment lengths are zero")
            # Radial conduction across the real wire insulation/potting:
            # R = ln((r+t)/r)/(2*pi*k*L), independent of FV grid size.
            # An explicit cu_pkg_g retains historical lumped comparisons.
            radius=float(model.copper.scene.assemblies[p].wire_radius)
            if cu_pkg_g is not None:
                # Historical lumped-thermal tests have no wire envelope.
                layer=0.
            elif cu_ins_t is None:
                # Original BFZI V150 describes both bare/insulated radii.
                cfg=model.base["cfg"]
                sample=model.base["sample"]
                ins_r=cfg.get("BSR_INSULATED_WIRE_RADIUS_M")
                if ins_r is None and "insulated_wire_d_mm" in sample:
                    ins_r=.0005*float(sample["insulated_wire_d_mm"])
                if ins_r is None:
                    raise ValueError("BFZI insulated wire radius is missing")
                layer=float(ins_r)-radius
            else:
                layer=float(cu_ins_t)
            if cu_pkg_g is None and (not np.isfinite(layer) or layer<=0):
                raise ValueError("BFZI wire insulation must exceed copper radius")
            self.cu_ins_layers.append(layer)
            if cu_pkg_g is None:
                # Only enclosed winding length conducts into this package.
                length=float(np.sum(np.linalg.norm(dl,axis=1)))
                if radius<=0 or length<=0:
                    raise ValueError("invalid physical copper wire length/radius")
                total_g=2*np.pi*cu_ins_k*length/np.log1p(layer/radius)
            else:
                total_g=float(cu_pkg_g)
            self.cu_pkg_total.append(total_g)
            g=total_g*mass/np.sum(mass)
            couplings=[]
            for a,cond in enumerate(g):
                if cond>0:
                    link(start+a,self.cu.start+p,cond)
                    couplings.append((start+a,float(cond)))
            self.cu_links.append(couplings)
        amb_g=np.zeros(self.n)
        for i,ax,side,area,is_outer in model.proj.boundary:
            if is_outer:
                amb_g[i]+=outer_h*area
        self.amb_g=amb_g
        self.H=coo_matrix((vals,(rows,cols)),
                          shape=(self.n,self.n)).tocsr()+diags(amb_g)
        self._factor={}

    def temperature_at(self,points,state):
        pts=np.asarray(points,float).reshape(-1,3)
        temp=np.asarray(state,float)
        if temp.shape!=(self.n,):
            raise ValueError("invalid thermal state")
        out=self.amb+temp[self.tree.query(pts)[1]]
        for p,(center,half) in enumerate(self.boxes):
            inside=isinside(pts,np.asarray(center,float),
                            np.asarray(half,float),self.pkg_axes[p])
            if np.any(inside):
                sub=cKDTree(self.pkg_xyz[p]).query(pts[inside])[1]
                out[inside]=self.amb+temp[self.count+p*self.pkg_n+sub]
        return out

    def heat_flux(self,state):
        temp=np.asarray(state,float)
        if temp.shape!=(self.n,):
            raise ValueError("invalid thermal state")
        cp=np.array([
            sum(g*(temp[self.cu.start+p]-temp[k])
                for k,g in self.cu_links[p])
            for p in range(2)])
        ps=np.array([
            sum(g*(temp[pk]-temp[sea])
                for sea,pk,g in self.pkg_links[p])
            for p in range(2)])
        return cp,ps,float(self.amb_g@temp)

    def package_data(self,state):
        t=np.asarray(state,float)[self.pkg].reshape(2,self.pkg_n)
        c=self.cap[self.pkg].reshape(2,self.pkg_n)
        return (
            self.amb+np.sum(c*t,axis=1)/np.sum(c,axis=1),
            np.sum(c*t,axis=1),
            self.amb+t
        )
