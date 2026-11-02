"""Compact BFZI dual-port response snapshots, independent of heat cubature.

This is exact-in-parent Galerkin compression at ONE scene/frequency, not
a certified spatial reference or a learned cross-geometry surrogate.
For two phasor ports, at most four REAL response directions span the
real/imaginary parts of both fully solved current snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.linalg import eigh, solve

from .sea_basis import SeaCurrentBasis
from .source import BFZISources


def _scene_signature(sources: BFZISources, basis: SeaCurrentBasis) -> str:
    """Bind saved arrays to actual source paths, conductivity and modes."""
    digest = hashlib.sha256()
    def put(array):
        values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())

    scene = sources.scene
    put([sources.omega, scene.seawater.conductivity,
         scene.outer_radius, basis.wall_scale, basis.outer_taper_width])
    put(scene.outer_center)
    digest.update(basis.outer_envelope.encode("utf-8"))
    for assembly in scene.assemblies:
        put(assembly.pose.center)
        put(assembly.pose.rotation)
        put(assembly.enclosure.half_extents)
    for mode in basis.modes:
        put(mode.center)
        put(mode.direction)
        put([mode.width, mode.source_port])
    for line in sources.lines:
        put(line.points)
        put(line.line_elements)
    put(sources.inductance_air)
    return digest.hexdigest()


@dataclass(frozen=True)
class BFZISnapshotROM:
    """Real, passivity-preserving Galerkin ROM of two complex MQS port states.

    Physical fields are decoded through the original geometry-aware curl
    basis, with only four real reduced states in the full-rank case.
    Input R/L/K MUST come from a common physical Galerkin trial space;
    fine R/K operator-only integrals are supported, avoiding a retained
    hundreds-of-thousands-of-points current/Joule tensor.
    """

    sources: BFZISources
    basis: SeaCurrentBasis
    raw_projection: np.ndarray
    resistance: np.ndarray
    inductance: np.ndarray
    coupling: np.ndarray
    unit_modes: np.ndarray
    impedance_seawater: np.ndarray
    signature: str
    inductance_kernel: str

    @classmethod
    def from_operators(
        cls, sources: BFZISources, basis: SeaCurrentBasis,
        transform, resistance, inductance, coupling, *,
        rank: int = 4, inductance_kernel: str = "uniform_ball",
    ) -> "BFZISnapshotROM":
        if basis.scene is not sources.scene:
            raise ValueError("basis and complete port sources must share a scene")
        if inductance_kernel not in ("uniform_ball", "plummer"):
            raise ValueError("unrecognized Green kernel")
        t = np.asarray(transform, dtype=float)
        r = np.asarray(resistance, dtype=float)
        l = np.asarray(inductance, dtype=float)
        k = np.asarray(coupling, dtype=float)
        n = r.shape[0] if r.ndim == 2 else -1
        if (n < 1 or r.shape != (n,n) or l.shape != (n,n)
                or k.shape != (n,2) or t.shape != (basis.size,n)
                or not all(np.isfinite(z).all() for z in (t,r,l,k))):
            raise ValueError("invalid shared real Galerkin R/L/K dimensions")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= min(4,n):
            raise ValueError("snapshot rank must be between 1 and min(4,modes)")
        if not np.allclose(r,r.T,atol=1e-10,rtol=1e-10):
            raise ValueError("R must be symmetric")
        if not np.allclose(l,l.T,atol=1e-10,rtol=1e-10):
            raise ValueError("L must be symmetric")
        eigen, vectors = eigh(0.5*(r+r.T),check_finite=True)
        if eigen[0] <= 0:
            raise ValueError("snapshot R must be positive definite")
        if np.linalg.eigvalsh(0.5*(l+l.T))[0] < -1e-12:
            raise ValueError("snapshot L must be positive semidefinite")
        root = (vectors*np.sqrt(eigen)[None,:]) @ vectors.T
        omega = sources.omega
        # Solve the two physical unit excitations exactly ONCE.
        full = solve(
            r + 1j*omega*l, -1j*omega*k,
            assume_a="gen",check_finite=True,
        )
        state_snapshots = np.column_stack((full.real,full.imag))
        u, _, _ = np.linalg.svd(root @ state_snapshots,full_matrices=False)
        reduction = solve(root,u[:,:rank],assume_a="pos",check_finite=True)
        # The root is symmetric positive definite. Keep both the reduced
        # operators and the same real linear map for the field decoder.
        rr = reduction.T @ r @ reduction
        ll = reduction.T @ l @ reduction
        kk = reduction.T @ k
        rr = 0.5*(rr+rr.T)
        ll = 0.5*(ll+ll.T)
        unit = solve(
            rr + 1j*omega*ll, -1j*omega*kk,
            assume_a="gen",check_finite=True,
        )
        zsea = 1j*omega*kk.T @ unit
        zsea = 0.5*(zsea+zsea.T)
        projection = t @ reduction
        # Full four-real-vector reduction must exactly reconstruct both
        # port solutions IN THE SELECTED PARENT (not the continuum).
        if rank == min(4,n):
            mismatch = np.linalg.norm(reduction @ unit - full)/max(
                np.linalg.norm(full),1e-30,
            )
            if mismatch > 1e-7:
                raise RuntimeError(
                    "four-real-snapshot compression failed to span parent "
                    f"unit currents (relative error {mismatch:g})"
                )
        return cls(
            sources,basis,projection,rr,ll,kk,unit,zsea,
            _scene_signature(sources,basis),inductance_kernel,
        )

    @property
    def rank(self) -> int:
        return self.resistance.shape[0]

    def current_density(self, xyz, currents=(1.0,0.0)):
        i = np.asarray(currents,dtype=complex)
        if i.shape != (2,) or not np.isfinite(i).all():
            raise ValueError("currents must be a finite complex two-vector")
        pts = np.asarray(xyz,dtype=float)
        if pts.shape[-1:] != (3,):
            raise ValueError("positions must have a final dimension of 3")
        original = pts.shape
        points = pts.reshape(-1,3)
        # Do not allocate a (n_points, n_raw, 3) heat tensor on disk:
        # evaluate only at requested locations.
        currents_raw = self.raw_projection @ (self.unit_modes @ i)
        field = np.einsum(
            "nmc,m->nc",self.basis.evaluate(points),currents_raw,
            optimize=True,
        )
        return field.reshape(original)

    def seawater_loss(self, xyz, currents=(1.0,0.0)):
        j = self.current_density(xyz,currents)
        return np.sum(np.abs(j)**2,axis=-1)/(
            2*self.sources.scene.seawater.conductivity
        )

    def port_seawater_power(self, currents=(1.0,0.0)) -> float:
        i = np.asarray(currents,dtype=complex)
        if i.shape != (2,) or not np.isfinite(i).all():
            raise ValueError("currents must be a finite complex two-vector")
        h = 0.5*(self.impedance_seawater+self.impedance_seawater.conj().T)
        return 0.5*float(np.real(np.vdot(i,h @ i)))

    def modal_seawater_power(self, currents=(1.0,0.0)) -> float:
        i = np.asarray(currents,dtype=complex)
        if i.shape != (2,) or not np.isfinite(i).all():
            raise ValueError("currents must be a finite complex two-vector")
        z = self.unit_modes @ i
        return 0.5*float(np.real(np.vdot(z,self.resistance @ z)))

    def save(self,path) -> None:
        """Portable numeric artifact; no pickled Python objects or volume tensor."""
        np.savez_compressed(
            path,version=np.array(1,dtype=np.int64),
            signature=np.array(self.signature),
            inductance_kernel=np.array(self.inductance_kernel),
            raw_projection=self.raw_projection,resistance=self.resistance,
            inductance=self.inductance,coupling=self.coupling,
            unit_modes=self.unit_modes,
            impedance_seawater=self.impedance_seawater,
        )

    @classmethod
    def load(cls,path,sources: BFZISources,basis: SeaCurrentBasis):
        """Refuse stale artifacts from other geometry/pose/port sources."""
        signature = _scene_signature(sources,basis)
        with np.load(path,allow_pickle=False) as data:
            if int(data["version"]) != 1 or str(data["signature"]) != signature:
                raise ValueError("snapshot artifact geometry/source signature mismatch")
            kernel = str(data["inductance_kernel"])
            values = {
                name:np.array(data[name],copy=True)
                for name in (
                    "raw_projection","resistance","inductance",
                    "coupling","unit_modes","impedance_seawater",
                )
            }
        if kernel not in ("uniform_ball","plummer"):
            raise ValueError("snapshot artifact contains unsupported Green kernel")
        t = values["raw_projection"]
        r = values["resistance"]
        l = values["inductance"]
        k = values["coupling"]
        u = values["unit_modes"]
        z = values["impedance_seawater"]
        m = r.shape[0] if r.ndim == 2 else -1
        if (m < 1 or m > 4 or t.shape != (basis.size,m)
                or l.shape != (m,m) or k.shape != (m,2)
                or u.shape != (m,2) or z.shape != (2,2)
                or not all(np.isfinite(a).all() for a in values.values())
                or not np.allclose(r,r.T,atol=1e-11,rtol=1e-9)
                or not np.allclose(l,l.T,atol=1e-11,rtol=1e-9)
                or np.linalg.eigvalsh(r)[0] <= 0
                or np.linalg.eigvalsh(l)[0] < -1e-12):
            raise ValueError("invalid snapshot artifact operators")
        unit_check = solve(
            r+1j*sources.omega*l,-1j*sources.omega*k,
            assume_a="gen",
        )
        if not np.allclose(unit_check,u,rtol=1e-7,atol=1e-11):
            raise ValueError("snapshot artifact unit-port solution is inconsistent")
        return cls(sources,basis,t,r,l,k,u,z,signature,kernel)
