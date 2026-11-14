"""Transfer a BFZI thermal/EM surrogate to a new native geometry.

The target needs one original-BFZI *cold* solve. Its thermal operator,
ROM and temperature-directional EM probes are shared from a donor. No target
dynamic history or target perturbed-EM call enters the prediction.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .frac_rom import FracROM
from .heat_rom import HeatROM

class SharedROM:
    def __init__(self, rom, fr, cold_corr, sea_xyz):
        self.rom=rom
        self.fr=fr
        self.corr=np.asarray(cold_corr,float)
        self.sea_xyz=np.asarray(sea_xyz,float)
        n=len(rom.cap)
        self.ns=n-64-2-1024
        if (self.corr.shape!=(6,) or self.sea_xyz.shape!=(self.ns,3)
                or len(fr.wsea)!=n or fr.cu.tolist()!=[self.ns+64,self.ns+65]):
            raise ValueError("incompatible donor BFZI thermal topology")

    @classmethod
    def load(cls,runs,donor):
        base=Path(runs)/f"bfzi_d_{donor}"
        with np.load(base.with_name(base.name+"_cold.npz")) as f:
            corr=f["cold_correction"].copy()
            xyz=f["sea_xyz"].copy()
        return cls(HeatROM.load(base.with_name(base.name+"_rom.npz")),
                   FracROM.load(base.with_name(base.name+"_tangent.npz")),
                   corr,xyz)

    def target(self,cold_z,*,alpha_scale=1.):
        """Cold 6-port Z and material-dependent sea conductivity slope."""
        model=copy.copy(self.fr)
        a=np.asarray(cold_z,float)
        if a.shape!=(6,) or not np.isfinite(alpha_scale) or alpha_scale<=0:
            raise ValueError("BFZI cold Z or seawater material scaling invalid")
        model.z0=a+self.corr
        if alpha_scale!=1.:
            # Separate the original temperature copper derivative from the
            # seawater derivative. Only sea sigma'(T) changes with alpha.
            from .geom_heat import pack
            B=self.rom.basis[self.fr.cu]
            jcu=np.stack([pack(np.diag(self.fr.dcu*B[:,j]))
                           for j in range(self.rom.rank)],axis=1)
            model.jac=jcu+alpha_scale*(self.fr.jac-jcu)
        return model

    def project(self,temp,xyz):
        """Map shared donor sea onto target native wet FV coordinates.

        Body-attached dual package / copper / skin state indices are shared.
        The donor ROM and its conservation law remain the online integrator.
        """
        x=np.asarray(xyz,float)
        val=np.asarray(temp,float)
        if x.ndim!=2 or x.shape[1]!=3 or val.ndim!=2 or val.shape[1]!=len(self.rom.cap):
            raise ValueError("BFZI target sea coordinates or donor thermal shape invalid")
        dist,idx=cKDTree(self.sea_xyz).query(x,k=min(8,self.ns))
        dist=np.asarray(dist).reshape(len(x),-1)
        idx=np.asarray(idx).reshape(len(x),-1)
        weights=1./np.maximum(dist,1e-9)**2
        weights/=weights.sum(axis=1,keepdims=True)
        sea=np.einsum("ij,tij->ti",weights,val[:,idx])
        exact=dist[:,0]<1e-10
        sea[:,exact]=val[:,idx[exact,0]]
        return np.column_stack((sea,val[:,self.ns:]))

    def rollout(self,cold_z,times,voltage,*,net=None,ck=None,sea_xyz=None,
                alpha_scale=1.):
        pred=self.target(cold_z,alpha_scale=alpha_scale).rollout(
            self.rom,times,voltage,net,ck)
        if sea_xyz is not None:
            pred["temp_K"]=self.project(pred["temp_K"],sea_xyz)
        return pred
