"""Energy-conserving BFZI sea/package/copper transient heat POD.

The original 3-D FV conduction matrix and Joule source are retained. A
capacity-weighted constant test/trial mode preserves global heat balance.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .v148_et import BFZIV148ET


class BFZIV148HeatROM(BFZIV148ET):
    def __init__(self, model, paths, *, rank=8, **kwargs):
        super().__init__(model, **kwargs)
        if rank < 2:
            raise ValueError("heat ROM rank must be at least 2")
        frames = []
        for path in paths:
            with np.load(Path(path)) as d:
                sea = np.asarray(d["Tsea_K"], float)
                pkg = np.asarray(d["Tpkg_K"], float)
                cu = np.asarray(d["Tcu_K"], float)
            if sea.ndim != 2 or sea.shape[1] != self.count or (pkg.shape,cu.shape) != ((len(sea),2),(len(sea),2)):
                raise ValueError("heat ROM snapshots do not match sea/package/copper cells")
            frames.extend(np.hstack((sea,pkg,cu))-self.amb)
        if not frames:
            raise ValueError("missing thermal snapshots")
        w = np.sqrt(self.cap)
        u0 = w/np.linalg.norm(w)
        A = w[:,None]*np.column_stack(frames)
        A -= u0[:,None]*(u0@A)[None,:]
        u,s,_ = np.linalg.svd(A, full_matrices=False)
        self.hrank = min(int(rank),self.n)
        # Include the spatially constant temperature mode for exact total
        # stored-heat versus Joule/ambient-flux balance in the reduced space.
        U = np.column_stack((u0,u[:,:self.hrank-1]))
        self.phi = U/w[:,None]
        self.hk = self.phi.T@(self.H@self.phi)
        self._hfac = {}
        self.hretained = float(np.sum(s[:self.hrank-1]**2)/max(np.sum(s**2),1e-30))
        self.hsensors = len(frames)

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
