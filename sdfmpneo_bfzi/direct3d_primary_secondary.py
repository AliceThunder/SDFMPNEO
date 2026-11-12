"""PDE-consistent analytic-primary Maxwell-MQS A-phi 3D field solution.

Unlike replacing the vacuum part of an ALREADY-solved coarse total-field
Galerkin solution, the analytic closed-filament A_primary enters the
conductivity forcing inside the FEM weak form BEFORE solving for
A_reaction and phi. This preserves the actual material PDE/charge equation
(up to finite elements, quadrature, boundary truncation and source chords).

Designed for spatial B/E/J/q reference development, not a certified
infinite-seawater BFZI teacher or finite-copper self impedance.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .direct3d_aphi import closed_polyline_potential, solve_aphi
from .direct3d_probe import closed_polyline_magnetic, sample_cell_fields
from .direct3d_port import closed_source_port_impedance_exact


@dataclass
class PrimarySecondary3D:
    reaction: object
    frequency_hz: float
    source_A: tuple
    source_B: tuple
    closed_source_reaction_z: np.ndarray
    volume_overlap_z: np.ndarray

    def volume_power(self, currents):
        """Physical total-field Joule integral of the analytic-primary PDE."""
        return self.reaction.volume_power(np.asarray(currents, complex))

    def joule_density_at_quadrature(self, currents):
        return self.reaction.joule_density_at_quadrature(
            np.asarray(currents, complex))

    def sample_fields(self, points, *, prefer_conductor=False, nearest=96):
        """Return actual-position E/J/q and analytic-primary plus reaction B."""
        return sample_cell_fields(
            self.reaction, points, prefer_conductor=prefer_conductor,
            nearest=nearest, incident_magnetic=self.source_B,
            incident_potentials=self.source_A,
            frequency_hz=self.frequency_hz)


def _weighted_closed_source(strands, field_factory):
    if not strands:
        raise ValueError("A port must have a nonempty closed winding")
    terms = []
    for weight, polygon in strands:
        if not np.isfinite(weight) or not np.isreal(weight):
            raise ValueError("Strand weights must be finite real turns")
        terms.append((float(weight), field_factory(polygon)))

    def field(points):
        x = np.asarray(points, float)
        return sum((w*f(x) for w,f in terms),
                   np.zeros_like(x, dtype=float))
    return field


def _green_boundary(mesh, reaction):
    """Free-space secondary A on the outer surface from FEM seawater J.

    One constant volume-current moment per wet tetra; exterior queries
    stay away from cell centroids. This is a boundary *approximation*;
    the full solution is refined by fixed-point re-solves.
    """
    wet=np.flatnonzero(mesh.sigma>0)
    xyz=mesh.nodes[mesh.tetrahedra[wet]]
    pts=xyz.mean(axis=1)
    v=abs(np.linalg.det(xyz[:,1:,:]-xyz[:,0,None,:]))/6
    jv=reaction.current_at_cells[:,wet,:]*v[None,:,None]
    from .direct3d_aphi import MU0

    def field(x):
        x=np.asarray(x,dtype=float)
        out=np.zeros((len(jv),len(x),3),dtype=complex)
        for start in range(0,len(x),16):
            r=np.linalg.norm(x[start:start+16,None,:]-pts[None,:,:],axis=2)
            if np.any(r<=0):
                raise ValueError("Green boundary point intersects current cell center")
            out[:,start:start+16,:]=(
                MU0/(4*np.pi)*np.einsum(
                    'kn,pnc->pkc',1/r,jv,optimize=True))
        return out
    return field


def solve_closed_primary_secondary(mesh, frequency_hz, source_strands, *,
                                   gauge='tree', open_steps=0):
    """Solve analytic-primary / FEM-material-reaction A-phi directly.

    Uniform mu=mu0, exp(+iwt), insulating enclosed ideal closed strands.
    With open_steps=0, the outer reaction tangential A is zero.
    With open_steps>0, use free-space volume-current Green traces
    (one moment per wet tetra) as prescribed reaction boundary data
    and re-solve the same PDE. This is a finite-step approximate
    open-boundary coupling, requiring independent h/R verification.
    Returned volume_overlap_z is an auxiliary finite-domain energy dual,
    NOT automatically the actual closed-wire port impedance:
    finite-boundary terms and discretization may separate it from
    closed_source_reaction_z. Check both without forcing equality.

    The closed filaments must be outside conducting material. Do not
    interpret a filament as finite-radius copper skin/proximity physics.
    """
    a = tuple(_weighted_closed_source(
        port, closed_polyline_potential) for port in source_strands)
    b = tuple(_weighted_closed_source(
        port, closed_polyline_magnetic) for port in source_strands)
    if not isinstance(open_steps,int) or isinstance(open_steps,bool) or not 0<=open_steps<=10:
        raise ValueError("open_steps must be an integer 0..10")
    response = solve_aphi(
        mesh, frequency_hz, list(a), outer_bc="secondary_zero",
        gauge=gauge)
    for _ in range(open_steps):
        boundary=_green_boundary(mesh,response)
        prescribed=tuple(
            (lambda x,k=k,boundary=boundary: a[k](x)+boundary(x)[k])
            for k in range(len(a))
        )
        response=solve_aphi(
            mesh,frequency_hz,list(a),
            outer_bc="prescribed_total",
            boundary_total_potentials=prescribed,
            gauge=gauge)
    line_z = closed_source_port_impedance_exact(
        response, frequency_hz, source_strands)
    return PrimarySecondary3D(
        reaction=response, frequency_hz=frequency_hz,
        source_A=a, source_B=b,
        closed_source_reaction_z=line_z,
        volume_overlap_z=response.delta_z)
