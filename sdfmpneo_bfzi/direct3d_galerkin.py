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
from .direct3d_edge_source import closed_edge_load


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


def solve_closed_ports(mesh: TetraMesh, frequency_hz, source_strands):
    """Independent 3D two- or n-port edge-FEM reference, closed filaments.

    Args:
        mesh: material-conforming tetrahedral domain. Non-magnetic mu=mu0.
        frequency_hz: harmonic frequency, e^(+i omega t).
        source_strands: for each port, list of (turns, exactly closed xyz
            polygon). The identical strand is used in forcing and voltage.

    Boundary: n x A_TOTAL = 0 on finite computational boundary.
    Not an infinite-domain electromagnetic solution.
    """
    L = closed_edge_load(mesh, source_strands)
    if L.shape[0] < 1:
        raise ValueError("At least one physical source port is required")

    def zero_source(points):
        return np.zeros_like(points, dtype=complex)

    zeros = [zero_source]*len(L)
    medium = solve_aphi(mesh, frequency_hz, zeros,
                        outer_bc="secondary_zero", edge_source_load=L)
    empty = TetraMesh(mesh.nodes, mesh.tetrahedra,
                      np.zeros_like(mesh.sigma))
    vacuum = solve_aphi(empty, frequency_hz, zeros,
                        outer_bc="secondary_zero", edge_source_load=L)

    omega = 2*np.pi*frequency_hz
    medium_z = 1j*omega*(L@medium.edge_coefficients.T)
    vacuum_z = 1j*omega*(L@vacuum.edge_coefficients.T)
    return ClosedPortResult(
        medium, vacuum, L, medium_z, vacuum_z, medium_z-vacuum_z)
