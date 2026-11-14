"""Capacity-weighted conservative Galerkin thermal ROM.

The full BFZI skin/package/sea heat operator is projected without changing
material couplings.  Full-state source vectors remain in physical watts.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class HeatROM:
    def __init__(self, cap, H, snapshots, *, rank=32, ambient=293.15):
        self.cap = np.asarray(cap, float)
        self.amb = float(ambient)
        if np.any(self.cap <= 0):
            raise ValueError("thermal capacities must be positive")
        n = len(self.cap)
        snap = np.asarray(snapshots, float)
        if snap.ndim != 2 or snap.shape[1] != n:
            raise ValueError("snapshot columns must match thermal nodes")
        if not 1 <= rank <= n:
            raise ValueError("invalid ROM rank")
        # Include uniform temperature explicitly to retain total heat.
        w = np.sqrt(self.cap)
        u0 = w / np.linalg.norm(w)
        a = (snap - self.amb).T * w[:, None]
        a -= u0[:, None] * (u0 @ a)[None, :]
        u, s, _ = np.linalg.svd(a, full_matrices=False)
        cols = [u0]
        for j in range(u.shape[1]):
            v = u[:, j].copy()
            for q in cols:
                v -= q * (q @ v)
            nv = np.linalg.norm(v)
            if nv > 1e-10:
                cols.append(v / nv)
            if len(cols) == rank:
                break
        if len(cols) < rank:
            # Enrich short trajectories using the physical thermal operator.
            h = H.tocsr() if hasattr(H, "tocsr") else np.asarray(H)
            seed = a[:, -1].copy()
            for j in range(4 * rank):
                seed = w * (h @ (seed / w)) / self.cap
                for q in cols:
                    seed -= q * (q @ seed)
                norm = np.linalg.norm(seed)
                if norm < 1e-12 or not np.isfinite(norm):
                    seed = np.zeros(n)
                    seed[j % n] = 1.
                    for q in cols:
                        seed -= q * (q @ seed)
                    norm = np.linalg.norm(seed)
                if norm > 1e-12:
                    cols.append(seed / norm)
                    seed = cols[-1].copy()
                if len(cols) == rank:
                    break
        self.basis = np.column_stack(cols) / w[:, None]
        self.m = self.basis.T @ (self.cap[:, None] * self.basis)
        self.k = self.basis.T @ (H @ self.basis)
        self.k = .5 * (self.k + self.k.T)
        self.rank = self.basis.shape[1]
        self.retained = float(np.sum(s[:max(0, self.rank-1)]**2) /
                              max(np.sum(s**2), 1e-30))

    def encode(self, temp):
        d = np.asarray(temp, float) - self.amb
        return self.basis.T @ (self.cap * d)

    def decode(self, state):
        return self.amb + self.basis @ np.asarray(state, float)

    def step(self, state, power, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")
        rhs = self.m @ state + dt * (self.basis.T @ power)
        fac = cho_factor(self.m + dt * self.k, check_finite=False)
        return cho_solve(fac, rhs, check_finite=False)


    def rollout(self, et, times, voltage, *, iterations=8, tol=1e-5):
        """Coupled reduced heat and original BFZI electromagnetic solve.

        This is the physical-EM reference coupling for B; neural replacement
        of the EM solve is handled by the subsequent surrogate stage.
        """
        t = np.asarray(times, float)
        v = np.asarray(voltage, complex)
        if (t.ndim != 1 or len(t) < 2 or t[0] != 0 or
                np.any(np.diff(t) <= 0) or v.shape != (len(t), 2)):
            raise ValueError("invalid thermal ROM time/voltage history")
        if len(et.cap) != len(self.cap) or not np.allclose(et.cap, self.cap):
            raise ValueError("ROM and BFZI thermal operators mismatch")
        state = np.zeros(self.rank)
        temps, z6, powers = [], [], []
        from .geom_heat import pack
        for j, now in enumerate(t):
            if j:
                dt = float(now - t[j-1])
                old = state.copy()
                guess = old.copy()
                for _ in range(iterations):
                    _, _, heat, _, _ = et.em(self.basis @ guess, v[j])
                    nxt = self.step(old, heat, dt)
                    if np.max(np.abs(self.basis @ (nxt - guess))) < tol:
                        guess = nxt
                        break
                    guess = nxt
                state = guess
            theta = self.basis @ state
            em, cur, heat, q, cu = et.em(theta, v[j])
            temps.append(self.amb + theta)
            z6.append(pack(em.total_z))
            powers.append([float(np.dot(q, et.model.vol)), *np.asarray(cu, float)])
        return dict(time_s=t, temp_K=np.asarray(temps),
                    z6=np.asarray(z6), power_W=np.asarray(powers))

    def save(self, path):
        np.savez_compressed(Path(path), cap=self.cap, ambient=self.amb,
                            basis=self.basis, m=self.m, k=self.k,
                            retained=self.retained)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with np.load(path) as d:
            obj.cap = d["cap"]
            obj.amb = float(d["ambient"])
            obj.basis = d["basis"]
            obj.m = d["m"]
            obj.k = d["k"]
            obj.retained = float(d["retained"])
        obj.rank = obj.basis.shape[1]
        return obj
