"""Single GeomFrac with a neural heat step and conservative material budgets.

An experimental student of the existing 32-state thermal ROM.  The thermal
basis is still used for 3-D reconstruction; its implicit dynamics is replaced
by a neural step.  Seawater impedance comes from GeomFrac, with no 8-mode
online tangent call.  Five material-balance equations are enforced in a
small (5x5) correction rather than solving the original heat equation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .frac_memory import zmat
from .geom_heat import pack
from .geom_memory import G_SCALE, GeomFrac, geom_feat, heat_drive
from .geom_rom import GeomHeatROM, groups
from .geom_canon import mode_scale
from .pose import default_desc
from .s5_bridge import feature


class GeomFracET(GeomFrac):
    """One shared SOE/MLP trunk and two outputs: sea Z and thermal ROM state."""

    def __init__(self, ng=40, rank=32, hidden=128, nrate=14):
        super().__init__(ng=ng, hidden=hidden, nrate=nrate)
        if rank < 5:
            raise ValueError("heat rank must include five material modes")
        self.rank = int(rank)
        self.heat_head = nn.Sequential(
            nn.Linear(hidden + rank + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, rank))
        nn.init.zeros_(self.heat_head[-1].weight)
        nn.init.zeros_(self.heat_head[-1].bias)

    def step_et(self, h, u, dt, geom, a, scale):
        """Causal one-step sea impedance and normalized heat-state prediction."""
        if a.shape != (len(u), self.rank):
            raise ValueError("expected one heat state per history")
        if dt <= 0:
            raise ValueError("positive time step required")
        scale = torch.as_tensor(scale, dtype=a.dtype, device=a.device)
        if scale.shape != (self.rank,) or torch.any(scale <= 0):
            raise ValueError("invalid heat normalization")
        h, y = super().step(h, u, dt, geom)
        w = torch.softmax(-self.orders()[:, None] *
                          torch.log(self.rates), dim=-1)
        mf = torch.sum(h[..., :-1] * w[None], dim=-1)
        x = torch.cat((u, mf, h[..., -1], geom), dim=-1)
        latent = self.net[:-1](x)
        clock = a.new_full((len(a), 1), np.log1p(dt)/np.log1p(600.))
        f = torch.cat((latent, a/scale, clock), dim=-1)
        da = self.heat_head(f)
        return h, y, a + (dt/30.) * scale * da


def _balance(rom, old, pred, dt, voltage, dsea, B, D):
    """Correct five material energy budgets without a ROM implicit solve."""
    if dt <= 0:
        raise ValueError("positive time step required")
    a = np.asarray(pred, float).copy()
    jac = B[:, :5]/dt + D[:, :5]
    for _ in range(12):
        theta = rom.phi @ a
        sea, z, I = rom.circuit(theta, dsea, voltage)
        heat, q, J, ps, pc = rom.source(theta, I, sea)
        qg = np.array([heat[sl].sum() for sl in groups(rom)])
        err = B @ ((a-old)/dt) + D @ a - qg
        if np.max(abs(err)) < 1e-8:
            break
        a[:5] += np.linalg.solve(jac, -err)
    theta = rom.phi @ a
    sea, z, I = rom.circuit(theta, dsea, voltage)
    heat, q, J, ps, pc = rom.source(theta, I, sea)
    res = rom.cap * (theta - rom.phi @ old)/dt + rom.H @ theta - heat
    budget = np.array([res[sl].sum() for sl in groups(rom)])
    return a, theta, z, I, q, J, ps, pc, res, budget


@torch.no_grad()
def rollout(rom, net, ck, voltage, times, correct_J=False):
    """Fully autonomous neural impedance + neural thermal evolution.

    Requires a precomputed cold FV anchor and spatial thermal basis, but uses
    neither the BFZI Green solve nor the 8-mode tangent nor an implicit ROM
    solve inside the time loop.  A small correction enforces material budgets.
    """
    if not isinstance(rom, GeomHeatROM) or not isinstance(net, GeomFracET):
        raise TypeError("GeomHeatROM and GeomFracET required")
    if net.rank != rom.rank or not ck.get("port_sea", False):
        raise ValueError("incompatible neural heat rank / BFZI sea checkpoint")
    t, v = np.asarray(times, float), np.asarray(voltage, float)
    if t.ndim != 1 or t.size == 0 or t[0] != 0 or v.shape != t.shape or np.any(np.diff(t) <= 0):
        raise ValueError("invalid BFZI time/voltage history")
    g = feature(rom, net.ng)
    g = torch.tensor(g[None], dtype=net.rates.dtype)
    sc = np.asarray(ck["src_scale"], float)
    ys = np.asarray(ck["y_scale"], float)
    ac = np.asarray(ck["a_scale"], float)
    if sc.shape != (3,) or ys.shape != (14,) or ac.shape != (rom.rank,):
        raise ValueError("invalid GeomFracET normalization")
    cap_phi = rom.cap[:, None] * rom.phi
    h_phi = rom.H @ rom.phi
    B = np.vstack([cap_phi[sl].sum(axis=0) for sl in groups(rom)])
    D = np.vstack([h_phi[sl].sum(axis=0) for sl in groups(rom)])
    h = net.init_state(1)
    a = np.zeros(rom.rank)
    norm = mode_scale(rom, rom.rank) if ck.get('canonical',False) else np.ones(rom.rank)
    theta = np.zeros(len(rom.cap))
    zprev = zmat(rom.z0)
    out = {key: [] for key in (
        "temp_K", "z6", "current_A", "qsea_W_m3", "Jsea_A_m2",
        "Psea_W", "Pcu_W", "fv_node_residual_W", "material_budget_W", "heat_state")}
    for k, tm in enumerate(t):
        if k:
            dt = float(tm - t[k-1])
            u = heat_drive(float(v[k-1]), zprev, rom.ext, rom.port,
                           rom.amb + theta[rom.cu], rom.cap[rom.cu])
            h, y, ahat = net.step_et(
                h, torch.tensor((u/sc)[None], dtype=g.dtype), dt, g,
                torch.tensor(a[None], dtype=g.dtype), ac)
            dsea = zmat(y[0, :6].cpu().numpy() * ys[:6])
            a_phys, theta, z, I, q, J, ps, pc, res, budget = _balance(
                rom, a*norm, ahat[0].cpu().numpy()*norm, dt, float(v[k-1]), dsea, B, D)
            a = a_phys/norm
            # Heat evolves with left-end interval drive; port outputs and
            # Joule density at t[k] are sampled with the present voltage.
            sea, z, I = rom.circuit(theta, dsea, float(v[k]))
            _, q, J, ps, pc = rom.source(theta, I, sea)
        else:
            sea, z, I = rom.circuit(theta, np.zeros((2, 2), complex), float(v[k]))
            _, q, J, ps, pc = rom.source(theta, I, sea)
            res = np.zeros(len(rom.cap))
            budget = np.zeros(5)
        if correct_J and rom.sea_proj is not None:
            J = rom.corrected_J(theta, I)
        zprev = z
        for name, value in (
            ("temp_K", rom.amb+theta), ("z6", pack(z)),
            ("current_A", I), ("qsea_W_m3", q), ("Jsea_A_m2", J),
            ("Psea_W", ps), ("Pcu_W", pc),
            ("fv_node_residual_W", np.max(abs(res))),
            ("material_budget_W", budget), ("heat_state", a)):
            out[name].append(np.asarray(value).copy())
    res = {k: np.asarray(v) for k, v in out.items()}
    res.update(rom_rank=rom.rank, neural_heat=True,
               em_directional_solves_online=0, heat_implicit_solves_online=0)
    return res
