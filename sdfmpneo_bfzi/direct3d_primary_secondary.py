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


def solve_closed_primary_secondary(mesh, frequency_hz, source_strands):
    """Solve analytic-primary / FEM-material-reaction A-phi directly.

    Uniform mu=mu0, exp(+iwt), insulating enclosed ideal closed strands.
    The outer *reaction* tangential A is zero on the finite domain.
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
    response = solve_aphi(
        mesh, frequency_hz, list(a), outer_bc="secondary_zero")
    line_z = closed_source_port_impedance_exact(
        response, frequency_hz, source_strands)
    return PrimarySecondary3D(
        reaction=response, frequency_hz=frequency_hz,
        source_A=a, source_B=b,
        closed_source_reaction_z=line_z,
        volume_overlap_z=response.delta_z)
