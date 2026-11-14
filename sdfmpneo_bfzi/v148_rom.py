"""Projected original BFZI V148 sea resolvent ROM from true thermal snapshots.

The operator, cellwise conductivity, FV projection, local source mass, air
inductance and copper impedance are reused from the full BFZI adapter.
Only the expensive 3N Green/resolvent solve is projected to a snapshot space.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .bfzi_v148_adapter import V148State, WeightedFV, _weak_q


class BFZIV148ROM:
    """Fixed-geometry, temperature-updated Petrov/Galerkin resolvent ROM."""

    def __init__(self, model, paths, *, rank=8, stride=1):
        if rank < 2 or stride < 1:
            raise ValueError("rank >= 2 and stride >= 1 required")
        self.model = model
        n = len(model.vol)
        cols = []
        C = model.base["C"]
        G0 = model.base["G0"]
        for path in paths:
            with np.load(Path(path)) as d:
                t = np.asarray(d["Tsea_K"], float)[::stride]
                j = np.asarray(d["Jsea_unit_A_m2_per_A"], complex)[::stride]
            if t.ndim != 2 or t.shape[1] != n or j.shape != (len(t), n, 2, 3):
                raise ValueError("training snapshots must match BFZI seawater cells")
            for temp, unit_j in zip(t, j):
                sig = model.sigma0 * (1 + model.alpha * (temp-model.ambient))
                if np.min(sig) <= 0:
                    raise ValueError("invalid snapshot conductivity")
                gamma = 1j * model.omega * model.mu * sig
                Y = -unit_j / gamma[:, None, None]
                vw = gamma * model.vol
                for p in range(2):
                    # Full solver identity: X + G0 @ diag(vw) P(X) = C.
                    ky = np.column_stack([G0 @ (vw*Y[:, p, k])
                                          for k in range(3)])
                    cols.append((C[p]-ky).reshape(-1))
        if not cols:
            raise ValueError("at least one snapshot is required")
        w = np.repeat(np.sqrt(model.vol), 3)
        mat = w[:, None]*np.column_stack(cols)
        u, s, _ = np.linalg.svd(mat, full_matrices=False)
        self.rank = min(int(rank), len(s))
        self.q = u[:, :self.rank]
        self.v = self.q / w[:, None]
        self.w = w
        self.retained = float(np.sum(s[:self.rank]**2)/np.sum(s**2))
        self.sources = len(cols)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def heat(self, state, currents):
        return self.model.heat(state, currents)

    def solve(self, sea_t, cu_t):
        model = self.model
        temp = np.asarray(sea_t, float)
        cu = np.asarray(cu_t, float)
        if temp.shape != model.vol.shape or cu.shape != (2,):
            raise ValueError("temperature field dimension mismatch")
        sig = model.sigma0*(1+model.alpha*(temp-model.ambient))
        if np.any(sig<=0) or not np.all(np.isfinite(sig)):
            raise ValueError("invalid seawater conductivity")
        gamma = 1j*model.omega*model.mu*sig
        vw = gamma*model.vol
        proj = WeightedFV(model.proj, sig)
        n = len(sig)
        A = np.empty_like(self.v)
        G0 = model.base["G0"]
        for k in range(self.rank):
            x = self.v[:, k].reshape(n, 3)
            px = proj.project_one(x)
            kx = np.column_stack([G0@(vw*px[:, a]) for a in range(3)])
            A[:, k] = (x+kx).reshape(-1)
        mat = self.q.conj().T @ (self.w[:, None]*A)
        C = list(model.base["C"])
        rhs = np.column_stack([x.reshape(-1) for x in C])
        b = self.q.conj().T @ (self.w[:, None]*rhs)
        coef = np.linalg.solve(mat, b)
        X = [(self.v@coef[:, p]).reshape(n, 3) for p in range(2)]
        P = proj.project(X)
        GC = [a-b for a,b in zip(C,proj.project(C))]
        GX = [a-b for a,b in zip(X,P)]
        core = model.calc.core
        z = core.Z_rect(C,X,vw,model.mu,model.omega)
        z -= core.Z_rect(GC,GX,vw,model.mu,model.omega)
        dz = -1j*model.omega*model.mu*np.einsum(
            "p,pij->ij",gamma,model.mass,optimize=True)
        sea_z = core.sym_complex(z+dz)
        zw = model.copper.conductor_internal_impedance(cu)
        zl = model.lead_copper.conductor_internal_impedance(
            np.full(2,model.ambient))
        zcu = zw+zl
        total = sea_z+1j*model.omega*model.air+np.diag(zcu)
        j = np.stack([-gamma[:, None]*px for px in P],axis=1)
        res = {"rom_rank":self.rank,"retained_energy":self.retained}
        weak=(_weak_q(C,X,GC,GX,sig,model.mass,model.vol,
                      model.omega,model.mu)
              if model.heat_mode=="port-weak" else None)
        return V148State(sea_z,total,zcu,j,sig,res,weak,zw,zl)
