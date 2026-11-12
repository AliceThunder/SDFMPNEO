"""Energy-conserving BFZI sea/package/copper transient heat POD.

The original 3-D FV conduction matrix and Joule source are retained. A
capacity-weighted material indicator modes preserve each component's heat budget.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .v148_et import BFZIV148ET
from .v148_pkg3d import BFZIV148Pkg3D


class BFZIV148HeatROM(BFZIV148ET):
    def __init__(self, model, paths, *, rank=8, **kwargs):
        super().__init__(model, **kwargs)
        if rank < 2:
            raise ValueError("heat ROM rank must be at least 2")
        frames = []
        for path in paths:
            with np.load(Path(path)) as d:
                sea = np.asarray(d["Tsea_K"], float)
                key="Tpkg_cells_K" if self.pkg.stop-self.pkg.start>2 else "Tpkg_K"
                pkg=np.asarray(d[key],float).reshape(len(sea),-1)
                cu=np.asarray(d["Tcu_K"],float)
            if (sea.ndim!=2 or sea.shape[1]!=self.count or
                    pkg.shape!=(len(sea),self.pkg.stop-self.pkg.start) or
                    cu.shape!=(len(sea),2)):
                raise ValueError("heat ROM snapshots do not match sea/package/copper cells")
            frames.extend(np.hstack((sea,pkg,cu))-self.amb)
        if not frames:
            raise ValueError("missing thermal snapshots")
        # Five capacity-weighted indicator modes make the projected
        # equations satisfy individual material power budgets: sea,
        # package 1/2, copper 1/2. Orthogonal POD modes describe the
        # remaining spatial gradient within seawater.
        self.hrank = min(int(rank),self.n)
        if self.hrank < 5:
            raise ValueError("heat ROM needs at least five material modes")
        w = np.sqrt(self.cap)
        mark = np.zeros((self.n,5))
        mark[:self.count,0] = w[:self.count]
        pn=(self.pkg.stop-self.pkg.start)//2
        for p in range(2):
            a=self.pkg.start+p*pn
            mark[a:a+pn,p+1]=w[a:a+pn]
        for p in range(2):
            a=self.cu.start+p
            mark[a,3+p]=w[a]
        mark /= np.linalg.norm(mark,axis=0)[None,:]
        A = w[:,None]*np.column_stack(frames)
        Ares = A-mark@(mark.T@A)
        u,_,_ = np.linalg.svd(Ares,full_matrices=False)
        self.phi = np.column_stack(
            (mark,u[:,:self.hrank-5]))/w[:,None]
        U = self.phi*w[:,None]
        defect=A-U@(U.T@A)
        total=float(np.sum(A*A))
        self.hretained=1-float(np.sum(defect*defect))/max(total,1e-30)
        self.hk = self.phi.T@(self.H@self.phi)
        self._hfac = {}
        self.hsensors = len(frames)
        self.hgroups = 5

    def advance(self,state,drive,dt,*,iterations=8,tol=1e-5):
        if not np.isfinite(dt) or dt<=0:
            raise ValueError("positive thermal time step required")
        old=np.asarray(state,float)
        if old.shape!=(self.n,):
            raise ValueError("invalid thermal state")
        if dt not in self._hfac:
            self._hfac[dt]=cho_factor(
                np.eye(self.hrank)+dt*self.hk,check_finite=False)
        fact=self._hfac[dt]
        theta=old.copy()
        for _ in range(iterations):
            _,_,heat,_,_=self.em(theta,drive)
            b=self.phi.T@(self.cap*old+dt*heat)
            nxt=self.phi@cho_solve(fact,b,check_finite=False)
            if np.max(abs(nxt-theta))<tol:
                break
            theta=nxt
        return nxt



class BFZIV148PkgHeatROM(BFZIV148HeatROM, BFZIV148Pkg3D):
    """Material-conservative POD with 3-D box FV conduction."""
