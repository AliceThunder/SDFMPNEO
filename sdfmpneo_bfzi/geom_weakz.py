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

    def __init__(self, tan, par, init, alpha, mix=0.5):
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

    def __call__(self, g, a, t, v):
        if a.ndim != 2 or a.shape[1] != 48:
            raise ValueError("weak sea requires [batch,48] heat coordinates")
        old = (self.par(g, a, t, v) if self.par is not None else
               a.new_zeros((len(a), 6)))
        if self.mix == 0:
            return old
        val = self.tan(g, a)
        phy = self.alpha * ((a * self.scale.to(a)) @ self.src.to(a))
        out = old.clone()
        idx = [0, 2, 4]
        out[:, idx] = ((1.0-self.mix)*old[:, idx] +
                       self.mix*(phy-val[:, idx]))
        return out
