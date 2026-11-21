"""Compact geometry/state parallel correction for six dynamic impedance channels."""
from __future__ import annotations

from pathlib import Path
import math
import torch
from torch import nn


class ParZ48(nn.Module):
    """Geometry, full rank48 heat state, time and drive conditioned residual."""

    shared_geom = True

    def __init__(self, gm, gs, am, ass, ys, hidden=32, strength=1.):
        super().__init__()
        self.register_buffer("gm", torch.as_tensor(gm, dtype=torch.float32))
        self.register_buffer("gs", torch.as_tensor(gs, dtype=torch.float32))
        self.register_buffer("am", torch.as_tensor(am, dtype=torch.float32))
        self.register_buffer("ass", torch.as_tensor(ass, dtype=torch.float32))
        self.register_buffer("ys", torch.as_tensor(ys, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(95, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.strength = float(strength)
        self.mask_xself = False

    def _feat(self, g, a, t, v):
        gt = (g - self.gm) / self.gs
        at = (a - self.am) / self.ass
        tt = torch.log1p(torch.clamp(t, min=0.) / 15.) / math.log1p(20.)
        vv = v / 80.
        return torch.cat((gt, at, tt, vv), dim=-1)

    def forward(self, g, a, t, v):
        if (g.ndim != 2 or g.shape[1] != 45 or
                a.ndim != 2 or a.shape[1] != 48 or
                t.shape != (len(g), 1) or v.shape != (len(g), 1) or
                len(g) != len(a)):
            raise ValueError("ParZ48 expects geometry [B,45], state [B,48], time/drive [B,1]")
        x = self._feat(g, a, t, v)
        x0 = self._feat(g, torch.zeros_like(a), t, v)
        out=(self.net(x)-self.net(x0))*self.ys*self.strength
        if self.mask_xself:
            mask=out.new_tensor([1.,0.,1.,1.,1.,0.])
            out=out*mask
        return out

    def save(self, path):
        torch.save(dict(state=self.state_dict(), hidden=self.net[0].out_features,
                        strength=self.strength,mask_xself=self.mask_xself), Path(path))

    @classmethod
    def load(cls, path):
        ck = torch.load(Path(path), map_location="cpu", weights_only=True)
        sd = ck["state"]
        model = cls(sd["gm"], sd["gs"], sd["am"], sd["ass"], sd["ys"],
                    hidden=int(ck["hidden"]), strength=float(ck["strength"]))
        model.load_state_dict(sd)
        model.mask_xself=bool(ck.get("mask_xself",False))
        model.eval()
        return model
