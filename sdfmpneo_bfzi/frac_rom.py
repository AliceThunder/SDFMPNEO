"""Causal BFZI impedance tangent, neural SOE residual and conservative heat ROM."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from .geom_heat import pack
from .frac_memory import zmat

class FracROM:
    """Cold BFZI calls during build only. The complete rollout is algebraic."""

    def __init__(self, z0, jac, air, ext, cu0, dcu, wsea, cu, amb,
                 cu_grid=None, cu_table=None):
        self.cu_grid = None if cu_grid is None else np.asarray(cu_grid, float)
        self.cu_table = None if cu_table is None else np.asarray(cu_table, complex)
        self.z0 = np.asarray(z0, float)
        self.jac = np.asarray(jac, float)
        self.air = np.asarray(air, complex)
        self.ext = np.asarray(ext, complex)
        self.cu0 = np.asarray(cu0, complex)
        self.dcu = np.asarray(dcu, complex)
        self.wsea = np.asarray(wsea, float)
        self.cu = np.asarray(cu, int)
        self.amb = float(amb)
        if (self.jac.shape[0] != 6 or self.cu.shape != (2,) or
                self.cu0.shape != (2,) or self.wsea.ndim != 1 or
                self.ext.shape != (2, 2)):
            raise ValueError("BFZI ROM surrogate dimensions mismatch")

    @classmethod
    def build(cls, et, rom, *, step_K=1.):
        """Original BFZI one-sided temperature probes along 32 ROM modes."""
        if step_K <= 0 or rom.basis.shape != (et.n, rom.rank):
            raise ValueError("thermal ROM/physical geometry mismatch")
        drive = np.array([1., 0.], complex)
        cold, cur, heat, _, _ = et.em(np.zeros(et.n), drive)
        sea0 = .5 * float(np.vdot(cur, cold.sea_z @ cur).real)
        if sea0 <= 0:
            raise ValueError("nonpositive cold seawater power")
        wsea = heat.copy()
        wsea[et.cu] = 0.
        wsea /= sea0
        if not np.isclose(wsea.sum(), 1., atol=1e-8):
            raise ValueError("BFZI source heat and port sea power disagree")
        jac = []
        for j in range(rom.rank):
            phi = rom.basis[:, j]
            amp = step_K / max(float(np.max(abs(phi))), 1e-30)
            warm = et.em(amp*phi, drive)[0]
            jac.append((pack(warm.total_z)-pack(cold.total_z))/amp)
        cu0 = cold.copper_z
        warm_cu = et.model.copper.conductor_internal_impedance(
            np.full(2, et.amb + 1.))
        # Original skin-effect copper law sampled offline; no online BFZI.
        cu_grid = np.linspace(0., 600., 121)
        cu_table = np.array([
            et.model.copper.conductor_internal_impedance(
                np.full(2, et.amb+dt)) for dt in cu_grid])
        return cls(pack(cold.total_z), np.stack(jac, axis=1),
                   1j*et.model.omega*et.model.air, et.external,
                   cu0, warm_cu-cu0, wsea,
                   np.arange(et.cu.start, et.cu.stop), et.amb,
                   cu_grid=cu_grid, cu_table=cu_table)

    def impedance(self, rom, state, residual=None):
        delta = self.jac @ state
        if residual is not None:
            delta = delta + residual
        z = zmat(self.z0 + delta)
        theta = rom.basis[self.cu] @ state
        cu_lin = self.cu0 + self.dcu*theta
        cu = cu_lin
        if self.cu_grid is not None:
            cu = np.array([
                np.interp(theta[i], self.cu_grid, self.cu_table[:, i].real)
                +1j*np.interp(theta[i], self.cu_grid, self.cu_table[:, i].imag)
                for i in range(2)])
            z = z + np.diag(cu-cu_lin)
        sea = z - self.air - np.diag(cu)
        return z, sea, cu

    def power(self, rom, state, voltage, residual=None):
        z, sea, cu = self.impedance(rom, state, residual)
        I = np.linalg.solve(z+self.ext, np.array([voltage, 0.], complex))
        ps = .5 * float(np.vdot(I, sea @ I).real)
        pc = .5 * np.real(cu)*abs(I)**2
        if ps < -1e-10 or np.min(pc) < -1e-10:
            raise ValueError("nonpassive two-port surrogate")
        heat = self.wsea*max(ps, 0.)
        heat[self.cu] = np.maximum(pc, 0.)
        return z, I, ps, pc, heat

    @torch.no_grad()
    def rollout(self, rom, times, voltage, net=None, ck=None, *, niter=8):
        """Independent causal voltage history, exact 32D Galerkin heat solve."""
        t = np.asarray(times, float)
        v = np.asarray(voltage, float)
        if (t.ndim != 1 or t.size < 2 or t[0] != 0 or
                np.any(np.diff(t) <= 0) or v.shape != t.shape):
            raise ValueError("aligned increasing time and voltage required")
        if rom.rank != self.jac.shape[1] or rom.basis.shape[0] != len(self.wsea):
            raise ValueError("loaded BFZI tangent/heat ROM mismatch")
        if net is not None and ck is None:
            raise ValueError("neural model needs trained scaling")
        h = net.init_state(1) if net is not None else None
        vref = float(ck["vref"]) if ck is not None else 1.
        yscale = np.asarray(ck["scale"], float) if ck is not None else None
        state = np.zeros(rom.rank)
        last_v = v[0]
        temp, zhist, ihist, pshist, pchist, budgets = [], [], [], [], [], []
        for k, now in enumerate(t):
            res = None
            if k:
                dt = float(now-t[k-1])
                if net is not None:
                    u = np.array([v[k]/vref, (v[k]/vref)**2,
                                  (v[k]-last_v)/vref], np.float32)
                    h, y = net.step(h, torch.from_numpy(u[None]), dt)
                    res = y[0].cpu().numpy()*yscale
                old = state.copy()
                guess = old.copy()
                fac = cho_factor(rom.m + dt*rom.k, check_finite=False)
                for _ in range(niter):
                    heat = self.power(rom, guess, v[k], res)[-1]
                    nxt = cho_solve(fac, rom.m @ old + dt*(rom.basis.T@heat),
                                    check_finite=False)
                    if np.max(abs(rom.basis@(nxt-guess))) < 1e-7:
                        guess = nxt
                        break
                    guess = nxt
                state = guess
                heat = self.power(rom, state, v[k], res)[-1]
                r = rom.m@(state-old)/dt + rom.k@state - rom.basis.T@heat
                budget = float(abs(r[0])*np.sqrt(np.sum(rom.cap)))
            else:
                budget = 0.
            z, cur, ps, pc, heat = self.power(rom, state, v[k], res)
            temp.append(self.amb+rom.basis@state)
            zhist.append(pack(z))
            ihist.append(cur)
            pshist.append(ps)
            pchist.append(pc)
            budgets.append(budget)
            last_v = v[k]
        return dict(time_s=t, voltage_v=v, temp_K=np.asarray(temp),
                    z6=np.asarray(zhist), current_A=np.asarray(ihist),
                    Psea_W=np.asarray(pshist), Pcu_W=np.asarray(pchist),
                    total_budget_W=np.asarray(budgets))

    def save(self, path):
        extra = ({} if self.cu_grid is None else
                 dict(cu_grid=self.cu_grid, cu_table=self.cu_table))
        np.savez_compressed(Path(path), z0=self.z0, jac=self.jac, air=self.air,
                            ext=self.ext, cu0=self.cu0, dcu=self.dcu,
                            wsea=self.wsea, cu=self.cu, amb=self.amb, **extra)

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            kw = (dict(cu_grid=d["cu_grid"], cu_table=d["cu_table"])
                  if "cu_grid" in d else {})
            return cls(*(d[k] for k in (
                "z0", "jac", "air", "ext", "cu0", "dcu", "wsea", "cu", "amb")), **kw)
