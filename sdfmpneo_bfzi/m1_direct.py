"""M1 physical two-port BFZI seawater direct solution.

Analytic closed primary sources drive the 3-D A-phi PDE. Queried
E/J/B/q and source-J overlap Z use the same seawater reaction. A finite
outer boundary may be improved with an optional free-space Green trace.
Physical h/R, boundary Green and source-chord convergence remain separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .direct3d_aphi import TetraMesh
from .direct3d_primary_secondary import solve_closed_primary_secondary
from .source_reference import polygonal_port_vertices
from .source import build_bfzi_source


def _currents(i):
    a = np.asarray(i, dtype=complex)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("Expected two finite complex port currents")
    return a


@dataclass
class M1Direct:
    scene: object
    frequency_hz: float
    physical: object
    sources: object
    paths: tuple
    open_steps: int = 0

    @property
    def z_sea(self):
        """Free-space reaction Z of induced seawater J, using source reciprocity.

        Z[p,q] = i*omega*integral A_primary[p] dot J_sea[q] dV.
        The SAME total E/J determines this matrix and spatial Joule heat.
        The induced J itself is approximated on the finite A-phi mesh.
        """
        return self.physical.volume_overlap_z

    @property
    def z_closed(self):
        """Finite-domain FE reaction-A closed-path readout, for comparison."""
        return self.physical.closed_source_reaction_z

    @property
    def z_loss(self):
        """Hermitian sea dissipation channel from the same PDE quadrature."""
        return self.physical.reaction.power_from_fields

    @property
    def z_em(self):
        """Air filament/GMD convention + seawater reaction + wire internal Z."""
        return (1j * 2*np.pi*self.frequency_hz*self.sources.inductance_air
                + self.z_sea
                + np.diag(self.sources.conductor_internal_impedance()))

    def volume_power(self, i=(1, 0)):
        """Integral of actual PDE E-field Joule heating in the seawater."""
        return self.physical.volume_power(_currents(i))

    def port_sea_power(self, i=(1, 0)):
        """Source-overlap reaction power; agrees with same-current Joule heat."""
        i = _currents(i)
        return float(.5*np.real(np.vdot(i, self.z_sea@i)))

    def closed_port_sea_power(self, i=(1, 0)):
        """Finite-boundary reaction-A line readout, independent of loss dual."""
        i = _currents(i)
        return float(.5*np.real(np.vdot(i, self.z_closed@i)))

    def heat_nodes(self, i=(1, 0)):
        """Conductor-only original tetra Gauss nodes, dV weights and q[W/m3]."""
        r = self.physical.reaction
        m = r.mesh
        wet = np.flatnonzero(m.sigma > 0)
        xyz = m.nodes[m.tetrahedra[wet]]
        mat = (xyz[:, 1:] - xyz[:, 0, None, :]).transpose(0, 2, 1)
        from .direct3d_aphi import _BARY4
        points = np.einsum("qi,tij->tqj", _BARY4, xyz)
        vol = np.abs(np.linalg.det(mat))/6
        q = self.physical.joule_density_at_quadrature(_currents(i))[wet]
        return (points.reshape(-1, 3), np.repeat(vol/4, 4), q.reshape(-1))

    def fields(self, xyz, i=(1, 0), *, nearest=96):
        """Physical seawater E/B/H/J/q, with unit-port fields for reuse.

        The electromagnetic E of this A-phi MQS model is queried only in
        conductive seawater; dry regions carry no solved scalar potential.
        """
        i = _currents(i)
        p = np.asarray(xyz, float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("Expected finite query coordinates (n,3)")
        sample = self.physical.sample_fields(p, nearest=nearest)
        if np.any(sample["sigma_S_m"] <= 0):
            raise ValueError("E/J/q queries must lie inside conducting seawater")
        electric = sample["E_total_V_m"]
        magnetic = sample["B_total_T"]
        e = np.einsum("p,pnc->nc", i, electric)
        b = np.einsum("p,pnc->nc", i, magnetic)
        s = sample["sigma_S_m"]
        return {
            "unit_E_V_m": electric,
            "unit_B_T": magnetic,
            "E_V_m": e,
            "B_T": b,
            "H_A_m": b/(4e-7*np.pi),
            "J_A_m2": s[:, None]*e,
            "q_W_m3": .5*s*np.sum(np.abs(e)**2, axis=-1),
            "sigma_S_m": s,
        }


def solve_m1(scene, mesh: TetraMesh, frequency_hz, *,
             panels_per_corner=6, open_steps=0):
    """Compute two independently excited closed UMFC ports on a real 3-D mesh.

    The mesh must conform to both insulating packages and the outer domain.
    Use the existing Gmsh physical-tag importer for rotated SE(3) packages.
    """
    if not isinstance(mesh, TetraMesh):
        raise TypeError("Expected a physical material-conforming tetra mesh")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Positive finite frequency required")
    paths = tuple(((1., polygonal_port_vertices(
        a, panels_per_corner=panels_per_corner)),) for a in scene.assemblies)
    physical = solve_closed_primary_secondary(
        mesh, frequency_hz, paths, gauge="tree", open_steps=open_steps)
    sources = build_bfzi_source(scene, frequency_hz, order=3)
    return M1Direct(scene, float(frequency_hz), physical, sources, paths, open_steps)
