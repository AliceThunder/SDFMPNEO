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

    def __init__(self, ng=40, rank=32, hidden=128, nrate=14,
                 soe_quad="legacy", heat_dt=0., heat_z=False):
        super().__init__(ng=ng, hidden=hidden, nrate=nrate,soe_quad=soe_quad)
        if heat_dt<0:
            raise ValueError("heat internal step must be nonnegative")
        self.heat_dt=float(heat_dt)
        if rank < 5:
            raise ValueError("heat rank must include five material modes")
        self.rank = int(rank)
        self.heat_z = bool(heat_z)
        self.heat_head = nn.Sequential(
            nn.Linear(hidden + rank + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, rank))
        nn.init.zeros_(self.heat_head[-1].weight)
        nn.init.zeros_(self.heat_head[-1].bias)
        if self.heat_z:
            # Thermal correction is exactly zero for ambient heat state.
            # The final layer starts at zero to preserve a V4 warm start.
            self.heat_z_head = nn.Sequential(
                nn.Linear(hidden + rank, hidden // 2), nn.SiLU(),
                nn.Linear(hidden // 2, 6))
            nn.init.zeros_(self.heat_z_head[-1].weight)
            nn.init.zeros_(self.heat_z_head[-1].bias)

    def step_et(self, h, u, dt, geom, a, scale):
        """Causal one-step sea impedance and normalized heat-state prediction."""
        if a.shape != (len(u), self.rank):
            raise ValueError("expected one heat state per history")
        if dt <= 0:
            raise ValueError("positive time step required")
        scale = torch.as_tensor(scale, dtype=a.dtype, device=a.device)
        if scale.shape != (self.rank,) or torch.any(scale <= 0):
            raise ValueError("invalid heat normalization")
        n=max(1,int(np.ceil(dt/self.heat_dt-1e-12))) if self.heat_dt else 1
        hdt=dt/n
        for _ in range(n):
            h,y=super().step(h,u,hdt,geom)
            mf=torch.sum(h[...,:-1]*self.soe_weights()[None],dim=-1)
            x=torch.cat((u,mf,h[...,-1],geom),dim=-1)
            latent=self.net[:-1](x)
            clock=a.new_full((len(a),1),np.log1p(hdt)/np.log1p(600.))
            f=torch.cat((latent,a/scale,clock),dim=-1)
            a=a+(hdt/30.)*scale*self.heat_head(f)
            if self.heat_z:
                # Use the interval-end temperature, consistently in
                # teacher-forced training and autonomous inference.
                thermal=a/scale
                hot=torch.cat((latent,thermal),dim=-1)
                cold=torch.cat((latent,torch.zeros_like(thermal)),dim=-1)
                dz=self.heat_z_head(hot)-self.heat_z_head(cold)
                y=torch.cat((y[:,:6]+dz,y[:,6:]),dim=-1)
        return h,y,a


def _balance_raw(rom, old, pred, dt, voltage, dsea, B, D, phi=None):
    """Correct five material energy budgets without a ROM implicit solve."""
    if dt <= 0:
        raise ValueError("positive time step required")
    a = np.asarray(pred, float).copy()
    phi=rom.phi if phi is None else phi
    jac = B[:, :5]/dt + D[:, :5]
    for _ in range(12):
        theta = phi @ a
        sea, z, I = rom.circuit(theta, dsea(a) if callable(dsea) else dsea, voltage)
        heat, q, J, ps, pc = rom.source(theta, I, sea)
        qg = np.array([heat[sl].sum() for sl in groups(rom)])
        err = B @ ((a-old)/dt) + D @ a - qg
        if np.max(abs(err)) < 1e-8:
            break
        a[:5] += np.linalg.solve(jac, -err)
    theta = phi @ a
    sea, z, I = rom.circuit(theta, dsea(a) if callable(dsea) else dsea, voltage)
    heat, q, J, ps, pc = rom.source(theta, I, sea)
    res = rom.cap * (theta - phi @ old)/dt + rom.H @ theta - heat
    budget = np.array([res[sl].sum() for sl in groups(rom)])
    return a, theta, z, I, q, J, ps, pc, res, budget


def _pos_beta(rom, a, phi=None):
    """Largest admissible scale of zero-mean spatial thermal modes."""
    if rom.rank <= 5:
        return 1.
    phi=rom.phi if phi is None else phi
    base=phi[:,:5]@a[:5]
    fluct=phi[:,5:]@a[5:]
    neg=fluct<0.
    if not np.any(neg):
        return 1.
    return min(1.,float(np.min(np.maximum(base[neg],0.)/(-fluct[neg]))))


def _balance(rom, old, pred, dt, voltage, dsea, B, D, phi=None):
    """Enforce five energy budgets and nonnegative FV temperature rises."""
    out=_balance_raw(rom,old,pred,dt,voltage,dsea,B,D,phi)
    for _ in range(5):
        a=out[0]
        beta=_pos_beta(rom,a,phi)
        if beta>=1.-1e-9:
            break
        a=a.copy()
        a[5:]*=max(beta*.999,0.)
        out=_balance_raw(rom,old,a,dt,voltage,dsea,B,D,phi)
    return out


@torch.no_grad()
def rollout(rom, net, ck, voltage, times, correct_J=False, tangent=None,
            pkg=None, lift=None, skin=None, skin_gain=1.):
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
    phi=rom.phi
    skin_scale=None
    skin_old=None
    if lift is not None:
        from .geom_lift import lift_basis
        phi=lift_basis(rom,lift)
    if skin is not None:
        if rom.rank!=32 or not ck.get('canonical',False):
            raise ValueError("V7 neural sea-skin feedback requires canonical rank 32")
        from .geom_canon import basis
        phi=np.column_stack((phi,basis(rom,48)[:,32:]))
        skin_scale=mode_scale(rom,48)[32:]
        skin_old=np.zeros(16)
    cap_phi = rom.cap[:, None] * phi
    h_phi = rom.H @ phi
    B = np.vstack([cap_phi[sl].sum(axis=0) for sl in groups(rom)])
    D = np.vstack([h_phi[sl].sum(axis=0) for sl in groups(rom)])
    pkg_ref=None
    if pkg is not None or skin is not None:
        # Both absolute spatial corrections are conditioned on the same
        # uncorrected autonomous neural reference trajectory.
        pkg_ref=rollout(rom,net,ck,v,t,correct_J=False,
                        tangent=tangent,pkg=None)["heat_state"]
    h = net.init_state(1)
    a = np.zeros(rom.rank)
    norm = mode_scale(rom, rom.rank) if ck.get('canonical',False) else np.ones(rom.rank)
    theta = np.zeros(len(rom.cap))
    zprev = zmat(rom.z0)
    keys=["temp_K", "z6", "current_A", "qsea_W_m3", "Jsea_A_m2",
          "Psea_W", "Pcu_W", "fv_node_residual_W",
          "material_budget_W", "heat_state"]
    if skin is not None:
        keys.append("skin_state")
    out = {key: [] for key in keys}
    for k, tm in enumerate(t):
        if k:
            dt = float(tm - t[k-1])
            u = heat_drive(float(v[k-1]), zprev, rom.ext, rom.port,
                           rom.amb + theta[rom.cu], rom.cap[rom.cu])
            h, y, ahat = net.step_et(
                h, torch.tensor((u/sc)[None], dtype=g.dtype), dt, g,
                torch.tensor(a[None], dtype=g.dtype), ac)
            if tangent is None:
                dsea = zmat(y[0, :6].cpu().numpy() * ys[:6])
            else:
                def sea_at(phys):
                    state=torch.as_tensor((phys[:rom.rank]/norm)[None],dtype=g.dtype)
                    sea6=tangent(g,state)[0].detach().cpu().numpy()
                    return zmat(sea6)
                dsea=sea_at
            prior=a*norm
            predicted=ahat[0].cpu().numpy()*norm
            if skin is not None:
                tm_in=torch.as_tensor([[tm]],dtype=g.dtype)
                v_in=torch.as_tensor([[v[k-1]]],dtype=g.dtype)
                ref=torch.as_tensor(pkg_ref[k:k+1],dtype=g.dtype)
                extra=skin(g,ref,tm_in,v_in)[0].cpu().numpy()
                if lift is not None:
                    from .geom_lift import IDX
                    extra-=pkg_ref[k,IDX]@np.asarray(lift,float)
                extra*=float(skin_gain)
                prior=np.r_[prior,skin_old*skin_scale]
                predicted=np.r_[predicted,extra*skin_scale]
            a_phys, theta, z, I, q, J, ps, pc, res, budget = _balance(
                rom, prior, predicted, dt, float(v[k-1]), dsea, B, D, phi)
            if pkg is not None:
                if rom.rank!=32:
                    raise ValueError("package correction needs 32 thermal modes")
                ref=torch.as_tensor(pkg_ref[k:k+1],dtype=g.dtype)
                delta=pkg(g,ref)[0].cpu().numpy()
                a_new=a_phys.copy()
                a_new[18:32]=(pkg_ref[k,18:32]+delta)*norm[18:32]
                a_phys, theta, z, I, q, J, ps, pc, res, budget = _balance(
                    rom, prior, a_new, dt, float(v[k-1]), dsea, B, D, phi)
            a = a_phys[:rom.rank]/norm
            if skin is not None:
                skin_old=a_phys[rom.rank:]/skin_scale
            if tangent is not None:
                dsea=sea_at(a_phys)
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
        if skin is not None:
            out["skin_state"].append(skin_old.copy())
    res = {k: np.asarray(v) for k, v in out.items()}
    res.update(rom_rank=rom.rank, neural_heat=True,
               em_directional_solves_online=0, heat_implicit_solves_online=0)
    return res
