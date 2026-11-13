"""Cold-physics thermal response subspace for fewer original-BFZI EM probes.

Constructs a low-rank space from only original cold heat operators,
weak Joule forcing and copper sources; no held-out dynamic truth is used.
"""
from __future__ import annotations

import time
import numpy as np

from .geom_tangent import SeaTangent


TIMES = (1., 4., 12., 30., 75., 180., 420., 900., 1800., 3600.)


def response(head, rom, modes, times=TIMES):
    """Heat-mode coefficients of cold, unit-step material sources."""
    ns, n = head.ns, len(head.cap)
    phi = np.asarray(rom.phi, float)
    modes = np.asarray(modes, float)
    if phi.shape[0] != n or modes.shape[0] != ns:
        raise ValueError("cold BFZI geometry/heat modes differ")
    k = np.asarray(rom.K, float)
    if k.shape != (phi.shape[1], phi.shape[1]):
        raise ValueError("cold BFZI thermal ROM operator dimensions differ")
    lam, u = np.linalg.eigh(.5*(k+k.T))
    if lam.min() < -1e-8:
        raise ValueError("non-passive BFZI thermal ROM")
    lam = np.maximum(lam, 0.)
    proj = modes.T @ (head.vol[:, None]*phi[:ns])
    cols = []
    for a, b in ((0, 0), (0, 1), (1, 1)):
        f = np.zeros(n)
        f[:ns] = head.weak[:, a, b]*head.vol
        cols.append(f)
    for i in range(2):
        f = np.zeros(n)
        f[head.cu.start+i] = 1.
        cols.append(f)
    out = []
    for f in cols:
        drive = u.T @ (phi.T@f)
        samples = []
        for tm in times:
            if tm <= 0:
                raise ValueError("positive cold BFZI step-response times required")
            g = np.empty_like(lam)
            np.divide(-np.expm1(-tm*lam), lam, out=g, where=lam>1e-12)
            g[lam<=1e-12] = tm
            samples.append(proj @ (u @ (g*drive)))
        x = np.stack(samples, axis=1)
        size = float(np.linalg.norm(x))
        if size > 1e-20:
            out.append(x/size)
    if not out:
        raise ValueError("cold BFZI has no physical sea/copper heat response")
    return np.concatenate(out, axis=1)


def directions(head, rom, modes, rank):
    """Choose physical probe directions; full rank keeps the original span."""
    r = np.asarray(modes).shape[1]
    if not 1<=rank<=r:
        raise ValueError("requested thermal response rank exceeds sea modes")
    B = response(head, rom, modes)
    U, s, _ = np.linalg.svd(B, full_matrices=True)
    var = (s*s)/max(float(np.sum(s*s)), 1e-30)
    return np.asarray(modes)@U[:, :rank], dict(
        explained_fraction=float(np.sum(var[:rank])),
        cold_response_singular_values=s.tolist(), n_probes=rank)


def build(em, head, rom, modes, rank, *, amplitude=.2):
    """Compute only rank BFZI forward EM probes, preserving physical units."""
    if amplitude<=0 or not np.isfinite(amplitude):
        raise ValueError("positive finite BFZI thermal perturbation required")
    probe, info = directions(head, rom, modes, rank)
    cu = np.full(2, head.amb)
    t0 = time.perf_counter()
    jac = []
    for i in range(rank):
        h = amplitude/max(float(np.max(abs(probe[:, i]))), 1e-12)
        plus = em.solve(np.full(head.ns, head.amb)+h*probe[:, i], cu).sea_z
        dz = (plus-head.sea0)/h
        jac.append(.5*(dz+dz.T))
    info["directional_sec"] = time.perf_counter()-t0
    tan = SeaTangent(probe, np.asarray(jac), head.vol, head.sea0, head.geom)
    tan.method = "cold-response-forward"
    tan.amplitude = float(amplitude)
    tan.nsolve = rank
    return tan, info
