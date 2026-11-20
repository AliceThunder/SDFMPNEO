"""Cold-Joule-weighted seawater resistance without a new neural head.

The cold weak sea source projected to rank48 yields the linear thermal
resistance response alpha * integral(q_ij^cold * dT) dV. Mix it with the
existing impedance tangent while keeping the learned reactance unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from .geom_canon import labels


class WeakSea:
    """Geometry-local BFZI-free impedance correction for existing rollout."""

    shared_geom = True

    def __init__(self, tan, par, init, alpha, mix=0.5, gain=None,
                 react_gain=None, react_mix=0.):
        if not 0.0 <= mix <= 1.0:
            raise ValueError("weak sea mixing fraction must be in [0,1]")
        src = np.asarray(init["src"], float)
        cap = np.asarray(init["cap"], float)
        if src.shape != (48, 3) or cap.shape != (5,):
            raise ValueError("weak source needs rank48 and five capacities")
        self.tan = tan
        self.par = par
        self.mix = float(mix)
        self.alpha = float(alpha)
        self.scale = torch.tensor(np.sqrt(cap[labels(48)]), dtype=torch.float32)
        self.src = torch.tensor(src, dtype=torch.float32)
        coef=np.ones((3,3)) if gain is None else np.asarray(gain,float)
        if coef.shape!=(3,3) or not np.isfinite(coef).all():
            raise ValueError("regional seawater gains must have shape [3,3]")
        self.gain=torch.tensor(coef,dtype=torch.float32)
        xcoef=np.zeros((3,3)) if react_gain is None else np.asarray(react_gain,float)
        if xcoef.shape!=(3,3) or not np.isfinite(xcoef).all():
            raise ValueError("regional reactance coefficients must be [3,3]")
        self.react_gain=torch.tensor(xcoef,dtype=torch.float32)
        if not 0. <= react_mix <= 1.:
            raise ValueError("reactance physical mix must be in [0,1]")
        self.react_mix=float(react_mix)

    def __call__(self, g, a, t, v):
        if a.ndim != 2 or a.shape[1] != 48:
            raise ValueError("weak sea requires [batch,48] heat coordinates")
        old = (self.par(g, a, t, v) if self.par is not None else
               a.new_zeros((len(a), 6)))
        if self.mix == 0 and self.react_mix == 0:
            return old
        val = self.tan(g, a)
        x=a*self.scale.to(a)
        src=self.src.to(a)
        pieces=torch.stack((x[:,:1]@src[:1],
                            x[:,1:32]@src[1:32],
                            x[:,32:48]@src[32:48]),dim=1)
        phy=self.alpha*(pieces*self.gain.to(a)[None]).sum(dim=1)
        out = old.clone()
        idx = [0, 2, 4]
        if self.mix:
            out[:, idx] = ((1.0-self.mix)*old[:, idx] +
                           self.mix*(phy-val[:, idx]))
        if self.react_mix:
            xphy=self.alpha*(pieces*self.react_gain.to(a)[None]).sum(dim=1)
            xidx=[1,3,5]
            out[:,xidx]=((1.0-self.react_mix)*old[:,xidx]+
                          self.react_mix*(xphy-val[:,xidx]))
        return out
