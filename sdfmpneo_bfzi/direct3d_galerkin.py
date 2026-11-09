"""Galerkin-consistent independent 3D MQS A-phi closed-port direct reference.

Solve the TOTAL vector potential with an EXACT oriented Whitney/Nedelec
source functional L. The identical L reads source voltage, ensuring the
discrete field/power identity is not defeated by two unrelated source
representations. Subtract the same-mesh VACUUM solution before reporting
conductor-induced Delta Z. Still requires h, exterior and source geometry
convergence before reference certification.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .direct3d_aphi import AphiResult, TetraMesh, solve_aphi
from .direct3d_edge_source import closed_edge_load, elemental_current_edge_load


@dataclass
class ClosedPortResult:
    medium: AphiResult
    vacuum: AphiResult
    source_edge_load: np.ndarray
    medium_port_z: np.ndarray
    vacuum_port_z: np.ndarray
    material_increment_z: np.ndarray

    def port_power(self, currents):
        """Physical material-induced port power, peak-phasor convention."""
        i = np.asarray(currents, dtype=complex)
        if i.shape != (len(self.source_edge_load),) or not np.isfinite(i).all():
            raise ValueError("Expected finite complex current for each closed port")
        return float((.5*np.vdot(i, self.material_increment_z@i)).real)

    def conductor_power(self, currents):
        """Independent volume Joule power from the solved TOTAL E field."""
        return self.medium.volume_power(currents)

    def joule_density(self, currents):
        return self.medium.joule_density_at_quadrature(currents)

    def analytic_source_corrected_joule(self, frequency_hz, source_potentials):
        """Return independent E-field Joule matrix using analytic vacuum source.

        Replace the coarse mesh representation of the *known vacuum source*
        with its exact closed-filament potential, retaining only the
        finite-element difference A_medium-A_vacuum and induced grad(phi).
        This is a spatial-accuracy diagnostic (and an improved field
        reconstruction candidate), NOT the Galerkin port-energy identity.
        It must converge independently before using corrected J/q as
        high-fidelity thermal labels.

        source_potentials[p](xyz[n,3]) returns analytic unit-port A[n,3].
        Only electrically conducting cells are evaluated: avoids an
        ideal-wire singularity inside insulating source boxes.
        """
        from .direct3d_aphi import _BARY4
        nports = len(self.source_edge_load)
        if (len(source_potentials) != nports or not np.isfinite(frequency_hz)
                or frequency_hz <= 0):
            raise ValueError("Expected one analytic closed-source potential per port")
        mesh = self.medium.mesh
        selected = np.flatnonzero(mesh.sigma > 0)
        matrix = np.zeros((nports,nports),complex)
        if not len(selected):
            return matrix
        corners = mesh.nodes[mesh.tetrahedra[selected]]
        qpoints = np.einsum("qi,tij->tqj", _BARY4, corners)
        total_points = qpoints.reshape(-1,3)
        reaction = (self.medium.electric_at_quadrature[:,selected]
                    - self.vacuum.electric_at_quadrature[:,selected])
        omega = 2*np.pi*frequency_hz
        for port,source in enumerate(source_potentials):
            incident = np.asarray(source(total_points),dtype=complex)
            if incident.shape != total_points.shape or not np.isfinite(incident).all():
                raise ValueError("Analytic source must return finite vector potential")
            reaction[port] -= 1j*omega*incident.reshape(len(selected),4,3)
        volume = np.abs(np.linalg.det(
            (corners[:,1:]-corners[:,0,None,:]).transpose(0,2,1)))/6
        matrix = np.einsum(
            "t,ptqc,rtqc->pr",
            mesh.sigma[selected]*volume/4,
            reaction.conj(),reaction,
            optimize=True)
        return matrix


def solve_closed_ports(mesh: TetraMesh, frequency_hz, source_strands, *,
                       gauge: str = 'tree'):
    """Independent 3D two- or n-port edge-FEM reference, closed filaments.

    Args:
        mesh: material-conforming tetrahedral domain. Non-magnetic mu=mu0.
        frequency_hz: harmonic frequency, e^(+i omega t).
        source_strands: for each port, list of (turns, exactly closed xyz
            polygon). The identical strand is used in forcing and voltage.

    Boundary: n x A_TOTAL = 0 on finite computational boundary.
    Default gauge='tree' removes redundant gradient Lagrange multipliers,
    while preserving physical E/B/Joule and closed-port Z. Set 'mixed'
    to compare against the original saddle-point reference.
    Not an infinite-domain electromagnetic solution.
    """
    L = closed_edge_load(mesh, source_strands)
    return _solve_with_edge_source_load(mesh,frequency_hz,L,gauge=gauge)


def solve_elemental_current_ports(mesh: TetraMesh, frequency_hz,
                                  element_current_density, *,
                                  gauge: str = 'tree'):
    """Solve with the SAME prescribed unit-port elemental J_s as Elmer.

    Only valid for true 1-A, closed, divergence-compatible physical
    coil-current distributions on exactly mapped tetrahedra. The caller
    must independently verify those gates; a 3D array alone is not proof.
    Copper transport/skin and finite-volume source construction are not
    computed here. The returned impedance is the generalized incremental
    response dual to the specified impressed-current functional.
    """
    L=elemental_current_edge_load(mesh,element_current_density)
    return _solve_with_edge_source_load(mesh,frequency_hz,L,gauge=gauge)


def _solve_with_edge_source_load(mesh, frequency_hz, L, *, gauge):
    if L.shape[0] < 1:
        raise ValueError("At least one physical source port is required")

    def zero_source(points):
        return np.zeros_like(points, dtype=complex)

    zeros = [zero_source]*len(L)
    medium = solve_aphi(mesh, frequency_hz, zeros,
                        outer_bc="secondary_zero", edge_source_load=L,
                        gauge=gauge)
    empty = TetraMesh(mesh.nodes, mesh.tetrahedra,
                      np.zeros_like(mesh.sigma))
    vacuum = solve_aphi(empty, frequency_hz, zeros,
                        outer_bc="secondary_zero", edge_source_load=L,
                        gauge=gauge)

    omega = 2*np.pi*frequency_hz
    medium_z = 1j*omega*(L@medium.edge_coefficients.T)
    vacuum_z = 1j*omega*(L@vacuum.edge_coefficients.T)
    return ClosedPortResult(
        medium, vacuum, L, medium_z, vacuum_z, medium_z-vacuum_z)
