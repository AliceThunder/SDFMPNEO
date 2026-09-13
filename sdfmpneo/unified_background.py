"""Fixed multiscale Cartesian Maxwell/thermal background.

Geometry changes material occupancy and impressed coil currents, never the
background topology. Seawater remains a full three-dimensional conductive
medium in both Maxwell and thermal operators.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
import scipy.sparse as sp
from .unified_geometry import UnifiedUWPTGeometry

MU0=4e-7*np.pi
EPS0=8.8541878128e-12


def stretched_axis(bounds,core_half,fine_step,growth,max_step,center=0.0):
    lo,hi=map(float,bounds); c=float(center); h=float(core_half); fine=float(fine_step)
    if not lo<c<hi or h<=0 or fine<=0 or growth<1 or max_step<fine:
        raise ValueError("invalid background axis settings")
    a=max(lo,c-h); b=min(hi,c+h); n=max(1,int(np.ceil((b-a)/fine))); core=np.linspace(a,b,n+1)
    left=[]; x=a; step=fine
    while x>lo:
        step=min(max_step,step*growth); x2=max(lo,x-step); left.append(x2); x=x2
    right=[]; x=b; step=fine
    while x<hi:
        step=min(max_step,step*growth); x2=min(hi,x+step); right.append(x2); x=x2
    return np.asarray(list(reversed(left))+core.tolist()+right,float)


@dataclass
class BackgroundContext:
    geometry: UnifiedUWPTGeometry
    fractions: dict
    source_shape: np.ndarray
    line_heat_weights: tuple
    thermal_mass_full: sp.csr_matrix
    thermal_stiffness_full: sp.csr_matrix
    thermal_mass_reduced: np.ndarray
    thermal_stiffness_reduced: np.ndarray


class FixedMultiscaleBackground:
    def __init__(self,x,y,z,*,frequency_hz,materials,coil_materials,package_materials,seawater_material,thermal_rank,ambient_temperature=293.15):
        self.x=np.asarray(x,float); self.y=np.asarray(y,float); self.z=np.asarray(z,float)
        for q in (self.x,self.y,self.z):
            if q.ndim!=1 or len(q)<3 or np.any(np.diff(q)<=0):
                raise ValueError("background axes must be strictly increasing")
        self.dx=np.diff(self.x); self.dy=np.diff(self.y); self.dz=np.diff(self.z)
        self.nx,self.ny,self.nz=len(self.dx),len(self.dy),len(self.dz)
        self.frequency_hz=float(frequency_hz); self.omega=2*np.pi*self.frequency_hz
        self.ambient_temperature=float(ambient_temperature)
        self.materials={str(k):dict(v) for k,v in materials.items()}
        self.coil_materials=tuple(map(str,coil_materials)); self.package_materials=tuple(map(str,package_materials)); self.seawater_material=str(seawater_material)
        if len(self.coil_materials)!=len(self.package_materials): raise ValueError("coil/package material lists must match")
        required=set(self.coil_materials+self.package_materials+(self.seawater_material,))
        if required-set(self.materials): raise ValueError("background region materials are incomplete")
        self._build_cells(); self._build_edges(); self._build_curl(); self._build_reconstruction(); self.thermal_basis=self._thermal_basis(int(thermal_rank))

    @classmethod
    def from_config(cls,cfg,*,frequency_hz,materials,coil_materials,package_materials,seawater_material,thermal_rank,ambient_temperature):
        bounds=np.asarray(cfg["bounds"],float); core=np.asarray(cfg["core_half_extent"],float); center=np.asarray(cfg.get("core_center",[0,0,0]),float)
        if bounds.shape!=(3,2) or core.shape!=(3,) or center.shape!=(3,): raise ValueError("BACKGROUND bounds/core dimensions are invalid")
        axes=[stretched_axis(bounds[k],core[k],cfg["fine_step"],cfg.get("growth",1.4),cfg.get("max_step",4*cfg["fine_step"]),center[k]) for k in range(3)]
        return cls(*axes,frequency_hz=frequency_hz,materials=materials,coil_materials=coil_materials,package_materials=package_materials,seawater_material=seawater_material,thermal_rank=thermal_rank,ambient_temperature=ambient_temperature)

    @property
    def n_cells(self): return self.nx*self.ny*self.nz
    @property
    def n_edges(self): return len(self.edge_lengths)
    def _cell_id(self,i,j,k): return (i*self.ny+j)*self.nz+k

    def _build_cells(self):
        xc=(self.x[:-1]+self.x[1:])/2; yc=(self.y[:-1]+self.y[1:])/2; zc=(self.z[:-1]+self.z[1:])/2
        X,Y,Z=np.meshgrid(xc,yc,zc,indexing="ij"); self.cell_centers=np.column_stack([X.ravel(),Y.ravel(),Z.ravel()])
        DX,DY,DZ=np.meshgrid(self.dx,self.dy,self.dz,indexing="ij"); self.cell_widths=np.column_stack([DX.ravel(),DY.ravel(),DZ.ravel()]); self.cell_volumes=np.prod(self.cell_widths,axis=1)

    def _build_edges(self):
        full=[]; lengths=[]; maps=[]
        for axis in range(3):
            amap={}
            ranges=(range(self.nx),range(self.ny+1),range(self.nz+1)) if axis==0 else ((range(self.nx+1),range(self.ny),range(self.nz+1)) if axis==1 else (range(self.nx+1),range(self.ny+1),range(self.nz)))
            for i in ranges[0]:
                for j in ranges[1]:
                    for k in ranges[2]:
                        boundary=(axis==0 and (j in (0,self.ny) or k in (0,self.nz))) or (axis==1 and (i in (0,self.nx) or k in (0,self.nz))) or (axis==2 and (i in (0,self.nx) or j in (0,self.ny)))
                        if boundary: continue
                        amap[(i,j,k)]=len(full); full.append((axis,i,j,k)); lengths.append((self.dx[i],self.dy[j],self.dz[k])[axis])
            maps.append(amap)
        self.edge_tuples=tuple(full); self.edge_lengths=np.asarray(lengths); self.edge_maps=maps
        rows=[]; cols=[]; data=[]
        for e,(axis,i,j,k) in enumerate(self.edge_tuples):
            cells=[]
            if axis==0:
                for jj in (j-1,j):
                    for kk in (k-1,k):
                        if 0<=jj<self.ny and 0<=kk<self.nz: cells.append(self._cell_id(i,jj,kk))
            elif axis==1:
                for ii in (i-1,i):
                    for kk in (k-1,k):
                        if 0<=ii<self.nx and 0<=kk<self.nz: cells.append(self._cell_id(ii,j,kk))
            else:
                for ii in (i-1,i):
                    for jj in (j-1,j):
                        if 0<=ii<self.nx and 0<=jj<self.ny: cells.append(self._cell_id(ii,jj,k))
            for c in cells:
                rows.append(e); cols.append(c); data.append(self.cell_volumes[c]/(4.0*self.edge_lengths[e]**2))
        self.edge_cell_hodge=sp.csr_matrix((data,(rows,cols)),shape=(self.n_edges,self.n_cells))

    def _build_curl(self):
        rows=[]; cols=[]; vals=[]; hrows=[]; hcols=[]; hdata=[]; face=0
        def add(edge_map,key,sign):
            idx=edge_map.get(key)
            if idx is not None: rows.append(face); cols.append(idx); vals.append(sign)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz+1):
                    add(self.edge_maps[0],(i,j,k),1); add(self.edge_maps[1],(i+1,j,k),1); add(self.edge_maps[0],(i,j+1,k),-1); add(self.edge_maps[1],(i,j,k),-1); area=self.dx[i]*self.dy[j]
                    if k>0: hrows.append(face); hcols.append(self._cell_id(i,j,k-1)); hdata.append(.5*self.dz[k-1]/area)
                    if k<self.nz: hrows.append(face); hcols.append(self._cell_id(i,j,k)); hdata.append(.5*self.dz[k]/area)
                    face+=1
        for i in range(self.nx):
            for j in range(self.ny+1):
                for k in range(self.nz):
                    add(self.edge_maps[0],(i,j,k),1); add(self.edge_maps[2],(i+1,j,k),1); add(self.edge_maps[0],(i,j,k+1),-1); add(self.edge_maps[2],(i,j,k),-1); area=self.dx[i]*self.dz[k]
                    if j>0: hrows.append(face); hcols.append(self._cell_id(i,j-1,k)); hdata.append(.5*self.dy[j-1]/area)
                    if j<self.ny: hrows.append(face); hcols.append(self._cell_id(i,j,k)); hdata.append(.5*self.dy[j]/area)
                    face+=1
        for i in range(self.nx+1):
            for j in range(self.ny):
                for k in range(self.nz):
                    add(self.edge_maps[1],(i,j,k),1); add(self.edge_maps[2],(i,j+1,k),1); add(self.edge_maps[1],(i,j,k+1),-1); add(self.edge_maps[2],(i,j,k),-1); area=self.dy[j]*self.dz[k]
                    if i>0: hrows.append(face); hcols.append(self._cell_id(i-1,j,k)); hdata.append(.5*self.dx[i-1]/area)
                    if i<self.nx: hrows.append(face); hcols.append(self._cell_id(i,j,k)); hdata.append(.5*self.dx[i]/area)
                    face+=1
        self.curl=sp.csr_matrix((vals,(rows,cols)),shape=(face,self.n_edges)); self.face_cell_hodge=sp.csr_matrix((hdata,(hrows,hcols)),shape=(face,self.n_cells))

    def _build_reconstruction(self):
        mats=[]
        for axis in range(3):
            rows=[]; cols=[]; data=[]; emap=self.edge_maps[axis]
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        c=self._cell_id(i,j,k)
                        keys=((i,j,k),(i,j+1,k),(i,j,k+1),(i,j+1,k+1)) if axis==0 else (((i,j,k),(i+1,j,k),(i,j,k+1),(i+1,j,k+1)) if axis==1 else ((i,j,k),(i+1,j,k),(i,j+1,k),(i+1,j+1,k)))
                        found=[emap[q] for q in keys if q in emap]
                        if found:
                            length=(self.dx[i],self.dy[j],self.dz[k])[axis]
                            for e in found: rows.append(c); cols.append(e); data.append(1.0/(len(found)*length))
            mats.append(sp.csr_matrix((data,(rows,cols)),shape=(self.n_cells,self.n_edges)))
        self.reconstruct=tuple(mats)

    def _thermal_basis(self,rank):
        if rank<1 or rank>self.n_cells: raise ValueError("invalid thermal rank")
        lo=np.array([self.x[0],self.y[0],self.z[0]]); hi=np.array([self.x[-1],self.y[-1],self.z[-1]]); xi=(self.cell_centers-lo)/(hi-lo); candidates=[]; m=1
        while len(candidates)<rank:
            candidates=[(a*a+b*b+c*c,a,b,c) for a in range(1,m+1) for b in range(1,m+1) for c in range(1,m+1)]; candidates.sort(); m+=1
        modes=[q[1:] for q in candidates[:rank]]; Phi=np.column_stack([np.sin(a*np.pi*xi[:,0])*np.sin(b*np.pi*xi[:,1])*np.sin(c*np.pi*xi[:,2]) for a,b,c in modes]); gram=Phi.T@(self.cell_volumes[:,None]*Phi); L=np.linalg.cholesky(gram); return Phi@np.linalg.inv(L.T)

    def _require_inside(self,points,label):
        p=np.asarray(points,float); lo=np.array([self.x[0],self.y[0],self.z[0]]); hi=np.array([self.x[-1],self.y[-1],self.z[-1]])
        if p.ndim!=2 or p.shape[1]!=3 or np.any(~np.isfinite(p)) or np.any(p<=lo) or np.any(p>=hi):
            raise ValueError(f"{label} lies outside the fixed physical background; enlarge BACKGROUND['bounds']")

    def _package_fraction(self,package):
        signs=np.asarray(list(itertools.product((-1.0,1.0),repeat=3))); fraction=np.zeros(self.n_cells)
        for sign in signs:
            points=self.cell_centers+0.25*self.cell_widths*sign
            fraction+=package.contains(points)
        return fraction/len(signs)

    def _deposit_line(self,points):
        self._require_inside(points,"coil centerline"); source=np.zeros(self.n_edges,float); heat=np.zeros(self.n_cells,float)
        for p0,p1 in zip(points[:-1],points[1:]):
            d=p1-p0; length=float(np.linalg.norm(d))
            if length==0: continue
            mid=.5*(p0+p1); i=int(np.searchsorted(self.x,mid[0])-1); j=int(np.searchsorted(self.y,mid[1])-1); k=int(np.searchsorted(self.z,mid[2])-1); heat[self._cell_id(i,j,k)]+=length
            for axis,component in enumerate(d):
                if component==0: continue
                candidates=[(i,j,k),(i,j+1,k),(i,j,k+1),(i,j+1,k+1)] if axis==0 else ([(i,j,k),(i+1,j,k),(i,j,k+1),(i+1,j,k+1)] if axis==1 else [(i,j,k),(i+1,j,k),(i,j+1,k),(i+1,j+1,k)])
                avail=[self.edge_maps[axis][q] for q in candidates if q in self.edge_maps[axis]]
                for e in avail: source[e]+=component/(self.edge_lengths[e]*len(avail))
        if heat.sum()<=0 or np.linalg.norm(source)==0: raise ValueError("coil deposition produced a zero physical source")
        heat/=heat.sum(); return source,heat

    def geometry_context(self,geometry):
        g=geometry if isinstance(geometry,UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
        if g.n_ports!=len(self.coil_materials): raise ValueError("geometry port count differs from configured materials")
        spacing=.45*min(np.min(self.dx),np.min(self.dy),np.min(self.dz)); fractions={name:np.zeros(self.n_cells) for name in set(self.coil_materials+self.package_materials+(self.seawater_material,))}; sources=[]; heat_weights=[]
        for coil,mat in zip(g.coils,self.coil_materials):
            points=coil.centerline(spacing); source,heat=self._deposit_line(points); sources.append(source); heat_weights.append(heat); area=coil.conductor_width*coil.conductor_thickness; length=np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1)); fractions[mat]+=heat*(area*length)/self.cell_volumes
        total=np.zeros(self.n_cells)
        for mat in self.coil_materials: total+=fractions[mat]
        scale=np.ones(self.n_cells); mask=total>1.0; scale[mask]=1.0/total[mask]
        for mat in self.coil_materials: fractions[mat]*=scale
        occupied=np.minimum(total,1.0)
        for index,(package,mat) in enumerate(zip(g.packages,self.package_materials)):
            corners=package.pose.apply(np.asarray(list(itertools.product((-1.0,1.0),repeat=3)))*package.half_extent); self._require_inside(corners,f"package {index}"); raw=self._package_fraction(package); add=raw*np.clip(1.0-occupied,0.0,1.0); fractions[mat]+=add; occupied+=add
        fractions[self.seawater_material]=np.clip(1.0-occupied,0.0,1.0)
        total_fraction=sum(fractions.values())
        if np.max(np.abs(total_fraction-1.0))>1e-10: raise FloatingPointError("material fractions do not close to unity")
        M,K=self.thermal_operator_full(fractions); Phi=self.thermal_basis; Mr=Phi.T@(M@Phi); Kr=Phi.T@(K@Phi)
        return BackgroundContext(g,fractions,np.column_stack(sources),tuple(heat_weights),M,K,Mr,Kr)

    def _temperature_material(self,material,temperature):
        m=self.materials[material]; sigma=float(m.get("electrical_conductivity",0)); alpha=float(m.get("resistivity_temperature_coefficient",0)); tref=float(m.get("reference_temperature",self.ambient_temperature)); denominator=1+alpha*(np.asarray(temperature,float)-tref)
        if np.any(denominator<=0): raise ValueError("temperature-dependent conductivity left its physical constitutive range")
        return sigma/denominator

    def cell_properties(self,context,state=None,*,em=False):
        T=np.full(self.n_cells,self.ambient_temperature) if state is None else self.ambient_temperature+self.thermal_basis@np.asarray(state,float); sigma=np.zeros(self.n_cells); eps=np.zeros(self.n_cells); mu_inv=np.zeros(self.n_cells); k=np.zeros(self.n_cells); cap=np.zeros(self.n_cells)
        for name,f in context.fractions.items():
            m=self.materials[name]; local_sigma=self._temperature_material(name,T)
            if em and name in self.coil_materials: local_sigma=0.0
            sigma+=f*local_sigma; eps+=f*EPS0*float(m.get("relative_permittivity",1.0)); mu_inv+=f/(MU0*float(m.get("relative_permeability",1.0))); k+=f*float(m.get("thermal_conductivity",0)); cap+=f*float(m.get("volumetric_heat_capacity",0))
        return sigma,eps,mu_inv,k,cap,T

    def em_operator(self,context,state):
        sigma,eps,mu_inv,_,_,_=self.cell_properties(context,state,em=True); h2=np.asarray(self.face_cell_hodge@mu_inv).ravel(); hs=np.asarray(self.edge_cell_hodge@sigma).ravel(); he=np.asarray(self.edge_cell_hodge@eps).ravel(); return (self.curl.T@sp.diags(h2)@self.curl-self.omega**2*sp.diags(he)+1j*self.omega*sp.diags(hs)).tocsr()
    def rhs_matrix(self,context): return (-1j*self.omega)*np.asarray(context.source_shape,complex)

    def thermal_operator_full(self,fractions):
        k=np.zeros(self.n_cells); cap=np.zeros(self.n_cells)
        for name,f in fractions.items():
            m=self.materials[name]; k+=f*float(m.get("thermal_conductivity",0)); cap+=f*float(m.get("volumetric_heat_capacity",0))
        if np.any(k<=0) or np.any(cap<=0): raise ValueError("thermal material properties must be positive")
        M=sp.diags(cap*self.cell_volumes,format="csr"); rows=[]; cols=[]; data=[]; diag=np.zeros(self.n_cells)
        def pair(c1,c2,area,distance):
            kf=2*k[c1]*k[c2]/(k[c1]+k[c2]); conductance=kf*area/distance; diag[c1]+=conductance; diag[c2]+=conductance; rows.extend([c1,c2]); cols.extend([c2,c1]); data.extend([-conductance,-conductance])
        for i in range(self.nx):
            for j in range(self.ny):
                for kk in range(self.nz):
                    c=self._cell_id(i,j,kk)
                    if i+1<self.nx: pair(c,self._cell_id(i+1,j,kk),self.dy[j]*self.dz[kk],.5*(self.dx[i]+self.dx[i+1]))
                    if j+1<self.ny: pair(c,self._cell_id(i,j+1,kk),self.dx[i]*self.dz[kk],.5*(self.dy[j]+self.dy[j+1]))
                    if kk+1<self.nz: pair(c,self._cell_id(i,j,kk+1),self.dx[i]*self.dy[j],.5*(self.dz[kk]+self.dz[kk+1]))
                    if i in (0,self.nx-1): diag[c]+=k[c]*self.dy[j]*self.dz[kk]/(.5*self.dx[i])
                    if j in (0,self.ny-1): diag[c]+=k[c]*self.dx[i]*self.dz[kk]/(.5*self.dy[j])
                    if kk in (0,self.nz-1): diag[c]+=k[c]*self.dx[i]*self.dy[j]/(.5*self.dz[kk])
        return M,sp.csr_matrix((data,(rows,cols)),shape=(self.n_cells,self.n_cells))+sp.diags(diag)

    def field_components(self,X): return tuple(R@X for R in self.reconstruct)
    def material_joule_cells(self,context,state,X):
        sigma,_,_,_,_,_=self.cell_properties(context,state,em=True); ex,ey,ez=self.field_components(X); return ex,ey,ez,.5*sigma*self.cell_volumes
    def wire_resistances(self,context,state):
        _,_,_,_,_,T=self.cell_properties(context,state); out=[]; spacing=.45*min(np.min(self.dx),np.min(self.dy),np.min(self.dz))
        for coil,mat,w in zip(context.geometry.coils,self.coil_materials,context.line_heat_weights):
            temp=float(np.dot(w,T)); sigma=float(self._temperature_material(mat,np.array(temp))); length=coil.length(spacing); area=coil.conductor_width*coil.conductor_thickness; mu=MU0*float(self.materials[mat].get("relative_permeability",1)); delta=np.sqrt(2/(self.omega*mu*max(sigma,np.finfo(float).tiny))); effective=min(area,2*(coil.conductor_width+coil.conductor_thickness)*delta); out.append(length/(max(sigma,np.finfo(float).tiny)*max(effective,np.finfo(float).tiny)))
        return np.asarray(out)

    def save_arrays(self): return {"background_x":self.x,"background_y":self.y,"background_z":self.z,"thermal_basis":self.thermal_basis}

__all__=["BackgroundContext","FixedMultiscaleBackground","stretched_axis"]
