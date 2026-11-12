"""Time-dependent BFZI Green-Galerkin and spatial thermal feedback.

Reuse the existing BFZI R/L/K basis, air inductance and copper internal
impedance. A spatially varying sea temperature reweights the SAME Galerkin
resistance operator and re-solves the full complex two-port response.
Sea temperatures use volume-weighted 3-D cells; copper and insulation have
separate thermal nodes. This is a deterministic reduced thermal prototype,
not a continuum-verified three-material heat reference or original v148 port.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized


@dataclass
class ETStep:
    time: float
    temp: np.ndarray  # temperature rise above ambient: sea cells, pkg0/1, cu0/1
    impedance: np.ndarray
    sea_z: np.ndarray
    currents: np.ndarray
    sea_power: float
    copper_power: np.ndarray
    heat_power: float
    energy: float  # stored thermal energy above ambient, J


class BFZIDynamicET:
    """Spatial seawater thermal field and continuous-time coupled impedance.

    The sea current basis, Green inductance L, and source K remain fixed
    for fixed geometry and mu. Every thermal state reweights R via local
    sigma(T), re-solves (R+jwL) X = -jwK, and updates ALL complex Z entries.
    A passive external complex two-port models the chosen circuit.
    """

    def __init__(self, ref, *, ambient=293.15, sea_step=0.16,
                 sea_alpha=0.02, sea_k=0.6, sea_rhoc=4.0e6,
                 pkg_rhoc=1.5e6, cu_rhoc=3.45e6,
                 cu_pkg_g=0.8, pkg_sea_g=0.6, outer_h=10.0,
                 external=None):
        self.ref = ref
        self.amb = float(ambient)
        self.alpha = float(sea_alpha)
        self.omega = ref.sources.omega
        self.sigma0 = float(ref.sources.scene.seawater.conductivity)
        x = np.asarray(ref.integration_points, dtype=float)
        w = np.asarray(ref.integration_weights, dtype=float)
        modes = np.asarray(ref.integration_modes, dtype=float)
        if (sea_step <= 0 or self.sigma0 <= 0 or np.any(w <= 0)
                or x.ndim != 2 or x.shape[1] != 3 or not len(x)
                or modes.shape[0] != len(x) or modes.shape[2] != 3):
            raise ValueError("Invalid spatial seawater cubature or cell size")
        if (not np.isfinite(self.amb) or not np.isfinite(self.alpha)
                or min(sea_k,sea_rhoc,pkg_rhoc,cu_rhoc,cu_pkg_g,
                       pkg_sea_g,outer_h) < 0
                or min(sea_rhoc,pkg_rhoc,cu_rhoc) <= 0):
            raise ValueError("Invalid thermal material coefficients")
        _, group = np.unique(np.floor(x/sea_step).astype(int),
                             axis=0, return_inverse=True)
        self.group = group
        self.count = int(group.max()) + 1
        self.weights = w
        self.modes = modes
        self.vol = np.bincount(group, weights=w, minlength=self.count)
        self.centers = np.column_stack([
            np.bincount(group, weights=w*x[:,k], minlength=self.count)/self.vol
            for k in range(3)
        ])
        self.tree = cKDTree(self.centers)
        nm = modes.shape[1]
        gram = np.zeros((self.count, nm, nm), dtype=float)
        for g in range(self.count):
            idx = np.flatnonzero(group == g)
            gram[g] = np.einsum("n,nic,njc->ij",
                               w[idx]/self.sigma0, modes[idx], modes[idx],
                               optimize=True)
        self.gram = gram
        self.L = np.asarray(ref.inductance, dtype=float)
        self.K = np.asarray(ref.coupling, dtype=float)
        self.air = np.asarray(ref.sources.inductance_air, dtype=float)
        self.external = (np.zeros((2,2), complex) if external is None
                         else np.asarray(external, dtype=complex))
        if self.external.shape != (2,2) or not np.isfinite(self.external).all():
            raise ValueError("external must be a finite 2x2 matrix")
        if np.linalg.eigvalsh((self.external+self.external.conj().T)/2).min() < -1e-12:
            raise ValueError("external circuit must be passive")

        self.n = self.count + 4   # sea cells, pkg0/1, copper0/1
        self.pkg = slice(self.count, self.count+2)
        self.cu = slice(self.count+2, self.count+4)
        cap = list(sea_rhoc*self.vol)
        for assy in ref.sources.scene.assemblies:
            cap.append(pkg_rhoc*8*np.prod(assy.enclosure.half_extents))
        for assy, line in zip(ref.sources.scene.assemblies, ref.sources.lines):
            length = np.linalg.norm(line.line_elements, axis=1).sum()
            cap.append(cu_rhoc*np.pi*assy.wire_radius**2*length)
        self.cap = np.asarray(cap, dtype=float)
        if np.any(self.cap <= 0):
            raise ValueError("thermal heat capacities must be positive")

        rows, cols, vals = [], [], []
        def add(i,j,val):
            if val <= 0:
                return
            rows.extend((i,j,i,j))
            cols.extend((i,j,j,i))
            vals.extend((val,val,-val,-val))
        k_near = min(7, self.count)
        _, near = self.tree.query(self.centers, k=k_near)
        if k_near == 1:
            near = near.reshape(-1,1)
        scene = ref.sources.scene
        for i in range(self.count):
            for j in np.atleast_1d(near[i]):
                j = int(j)
                if j <= i:
                    continue
                mid = (self.centers[i]+self.centers[j])/2
                if any(a.enclosure.contains(mid[None,:],a.pose)[0]
                       for a in scene.assemblies):
                    continue
                dist = np.linalg.norm(self.centers[i]-self.centers[j])
                area = min(self.vol[i],self.vol[j])**(2/3)
                add(i,j,sea_k*area/max(dist,1e-12))
        for p,assy in enumerate(scene.assemblies):
            dist2,_ = assy.enclosure.squared_exterior_distance(
                self.centers, assy.pose)
            pick = np.argsort(dist2)[:min(8,self.count)]
            strength = 1/np.maximum(np.sqrt(dist2[pick]),0.005)
            strength /= strength.sum()
            for cell, coeff in zip(pick,strength):
                add(int(cell),self.count+p,pkg_sea_g*float(coeff))
            add(self.count+p,self.count+2+p,cu_pkg_g)
        center = np.asarray(scene.outer_center)
        radius = float(scene.outer_radius)
        ambient_g = np.zeros(self.n)
        outer = np.linalg.norm(self.centers-center,axis=1) > .75*radius
        ambient_g[:self.count][outer] = outer_h*self.vol[outer]**(2/3)
        self.H = coo_matrix((vals,(rows,cols)),shape=(self.n,self.n)).tocsr()
        self.H += diags(ambient_g)
        self._solve_cache = {}

    def zeros(self):
        return np.zeros(self.n)

    def temperature_at(self, points, state):
        """Query 3-D seawater/package temperature in kelvin."""
        pts = np.asarray(points,float).reshape(-1,3)
        state = np.asarray(state,float)
        if state.shape != (self.n,):
            raise ValueError("expected full thermal state")
        idx = self.tree.query(pts)[1]
        temp = self.amb + state[idx]
        for p,assy in enumerate(self.ref.sources.scene.assemblies):
            inside = assy.enclosure.contains(pts,assy.pose)
            temp[inside] = self.amb + state[self.count+p]
        return temp

    def em(self, state, drive):
        """Rebuild sigma(T), complex Z, circuit I and same-state heat source."""
        state = np.asarray(state,float)
        voltage = np.asarray(drive,complex)
        if state.shape != (self.n,) or voltage.shape != (2,):
            raise ValueError("Expected full thermal state and two-port voltage")
        ratio = 1 + self.alpha*state[:self.count]
        if np.min(ratio) <= 0:
            raise ValueError("seawater conductivity must remain positive")
        R = np.einsum("g,gij->ij",1/ratio,self.gram,optimize=True)
        X = np.linalg.solve(R+1j*self.omega*self.L,
                            -1j*self.omega*self.K)
        sea_z = 1j*self.omega*self.K.T@X
        sea_z = (sea_z+sea_z.T)/2
        cu_temp = self.amb+state[self.cu]
        zcu = self.ref.sources.conductor_internal_impedance(cu_temp)
        Z = 1j*self.omega*self.air+sea_z+np.diag(zcu)
        current = np.linalg.solve(Z+self.external,voltage)
        J = np.einsum("nmc,mp,p->nc",self.modes,X,current,optimize=True)
        sigma = self.sigma0*ratio[self.group]
        q = .5*np.sum(abs(J)**2,axis=1)/sigma
        sea_heat = np.bincount(self.group,weights=self.weights*q,
                               minlength=self.count)
        copper_heat = .5*zcu.real*abs(current)**2
        heat = np.zeros(self.n)
        heat[:self.count] = sea_heat
        heat[self.cu] = copper_heat
        return Z,sea_z,current,heat,float(sea_heat.sum()),copper_heat

    def advance(self, state, drive, dt, *, iterations=3):
        """Implicit heat diffusion with repeated coupled EM/circuit updates."""
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("positive thermal dt required")
        base = np.asarray(state,dtype=float)
        if base.shape != (self.n,):
            raise ValueError("invalid thermal state")
        if dt not in self._solve_cache:
            A = diags(self.cap)+dt*self.H
            self._solve_cache[dt] = factorized(A.tocsc())
        solve = self._solve_cache[dt]
        guess = base.copy()
        for _ in range(iterations):
            *_, heat, _, _ = self.em(guess,drive)
            new = solve(self.cap*base+dt*heat)
            if np.max(abs(new-guess)) < 1e-7:
                break
            guess = new
        return new

    def trajectory(self, times, voltage_at, *, state=None):
        """Integrate history; voltage_at(t) applies on the interval ending at t."""
        t = np.asarray(times,float)
        if t.ndim != 1 or len(t)<2 or t[0] != 0 or np.any(np.diff(t)<=0):
            raise ValueError("times must start at zero and strictly increase")
        state = self.zeros() if state is None else np.asarray(state,float).copy()
        out = []
        for k,now in enumerate(t):
            drive = np.asarray(voltage_at(float(now)),complex)
            if k:
                state = self.advance(state,drive,float(now-t[k-1]))
            Z,sea_z,current,heat,psea,pcu = self.em(state,drive)
            out.append(ETStep(float(now),state.copy(),Z,sea_z,
                              current,psea,pcu,float(heat.sum()),
                              float(self.cap@state)))
        return out
