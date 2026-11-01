"""Structure-preserving BFZI/seawater MQS Galerkin solver.

This is a finite-domain, finite-basis REFERENCE *prototype*, not a certified
infinite-seawater solution. It preserves the useful vNext design: one coupled
unit-port solve, spatial Joule fields, PSD dissipative port channels and
a reusable response object. Accuracy requires basis/cubature/domain refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist

from .source import BFZISources, MU0
from .sea_basis import SeaCurrentBasis, build_sea_basis


@dataclass(frozen=True)
class SeaMQSConfig:
    radial_order: int = 7
    polar_order: int = 10
    azimuthal_order: int = 16
    field_softening: float = 0.018
    resistance_rank_tolerance: float = 1e-9

    def __post_init__(self):
        if min(self.radial_order, self.polar_order, self.azimuthal_order) < 2:
            raise ValueError("quadrature orders must be >= 2")
        if not math.isfinite(self.field_softening) or self.field_softening <= 0:
            raise ValueError("field_softening must be positive")
        if not 0 < self.resistance_rank_tolerance < 1:
            raise ValueError("resistance_rank_tolerance must be in (0,1)")


@dataclass(frozen=True)
class SeawaterResponse:
    sources: BFZISources
    basis: SeaCurrentBasis
    transform: np.ndarray
    resistance: np.ndarray
    inductance: np.ndarray
    coupling: np.ndarray
    unit_current_modes: np.ndarray
    impedance_seawater: np.ndarray
    integration_points: np.ndarray
    integration_weights: np.ndarray
    integration_modes: np.ndarray
    field_softening: float
    raw_basis_count: int

    @property
    def port_impedance(self) -> np.ndarray:
        return 1j * self.sources.omega * self.sources.inductance_air + self.impedance_seawater

    def _currents(self, port_currents) -> np.ndarray:
        i = np.asarray(port_currents, dtype=complex)
        if i.shape != (2,):
            raise ValueError("port_currents must be a complex 2-vector")
        return i

    def current_density(self, xyz, port_currents=(1.0, 0.0)):
        """J_sea(x), valid only for x in the conducting seawater region."""
        x = np.asarray(xyz, dtype=float)
        single = x.ndim == 1
        locations = x.reshape(-1, 3)
        modes = self.basis.evaluate(locations) @ self.transform
        z = self.unit_current_modes @ self._currents(port_currents)
        field = np.einsum("nmc,m->nc", modes, z)
        return field[0] if single else field.reshape(x.shape)

    def electric_field(self, xyz, port_currents=(1.0, 0.0)):
        return self.current_density(xyz, port_currents) / self.sources.scene.seawater.conductivity

    def seawater_loss(self, xyz, port_currents=(1.0, 0.0)):
        j = self.current_density(xyz, port_currents)
        return np.sum(np.abs(j)**2, axis=-1) / (2 * self.sources.scene.seawater.conductivity)

    def seawater_power(self, port_currents=(1.0, 0.0)) -> float:
        z = self.unit_current_modes @ self._currents(port_currents)
        return 0.5 * float(np.real(np.vdot(z, self.resistance @ z)))

    def port_seawater_power(self, port_currents=(1.0, 0.0)) -> float:
        i = self._currents(port_currents)
        h = 0.5 * (self.impedance_seawater + self.impedance_seawater.conj().T)
        return 0.5 * float(np.real(np.vdot(i, h @ i)))

    def quadrature_power(self, port_currents=(1.0, 0.0)) -> float:
        z = self.unit_current_modes @ self._currents(port_currents)
        j = np.einsum("nmc,m->nc", self.integration_modes, z)
        q = np.sum(np.abs(j)**2, axis=-1) / (2 * self.sources.scene.seawater.conductivity)
        return float(self.integration_weights @ q)

    def dissipation_channel(self) -> np.ndarray:
        """Single vNext-compatible Hermitian PSD channel: seawater only."""
        return self.unit_current_modes.conj().T @ self.resistance @ self.unit_current_modes

    def as_structured_prediction(self):
        # Reuses the existing vNext channel/port contract without passing
        # BFZI geometry off as a historical SuperellipseSpiral.
        from sdfmpneo_vnext.prediction import StructuredPortPrediction
        return StructuredPortPrediction(
            self.port_impedance,
            self.dissipation_channel()[None, :, :],
            channel_labels=("seawater",),
        )

    def magnetic_field(self, xyz, port_currents=(1.0, 0.0), chunk_size: int = 96):
        """Biot-Savart source + finite-cubature induced seawater field."""
        x = np.asarray(xyz, dtype=float)
        flat = x.reshape(-1, 3)
        currents = self._currents(port_currents)
        b = np.asarray(self.sources.magnetic_field(flat, currents), dtype=complex)
        z = self.unit_current_modes @ currents
        jsea = np.einsum("nmc,m->nc", self.integration_modes, z)
        weighted = self.integration_weights[:, None] * jsea
        for start in range(0, len(flat), chunk_size):
            offset = flat[start:start + chunk_size, None, :] - self.integration_points[None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", offset, offset) + self.field_softening**2
            b[start:start + chunk_size] += (
                MU0 / (4 * math.pi)
                * np.sum(
                    np.cross(weighted[None, :, :], offset) / dist2[..., None]**1.5,
                    axis=1,
                )
            )
        return b.reshape(x.shape)


def solve_seawater_reference(
    sources: BFZISources,
    *,
    basis: SeaCurrentBasis | None = None,
    config: SeaMQSConfig | None = None,
) -> SeawaterResponse:
    """Assemble physically linked R,L,K; solve both unit-port excitations.

    A shared real basis and identical positive spatial cubature ensure exact
    *discrete* equality between port seawater loss and integrated Joule heat.
    Softened 3D Green kernels approximate self/near-field volume interactions.
    They must be refined rather than mistaken for an exact continuum reference.
    """
    cfg = config or SeaMQSConfig()
    scene = sources.scene
    basis = basis or build_sea_basis(scene)
    points, weights = scene.quadrature(
        cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
    )
    if len(points) == 0:
        raise ValueError("no seawater integration nodes")
    modes = basis.evaluate(points)
    sigma = scene.seawater.conductivity
    wres = weights / sigma
    r_full = np.einsum("n,nic,njc->ij", wres, modes, modes, optimize=True)
    r_full = 0.5 * (r_full + r_full.T)

    # Remove unresolvable modes before normalizing. The chosen basis
    # remains a subspace of divergence-free, zero-normal physical modes.
    eigenvalues, eigenvectors = eigh(r_full, check_finite=True)
    largest = max(float(np.max(eigenvalues)), 0.0)
    keep = eigenvalues > max(largest * cfg.resistance_rank_tolerance, 1e-28)
    if not np.any(keep):
        raise ValueError("seawater modes not resolved by the current cubature")
    transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    w = modes @ transform
    resistance = np.einsum("n,nic,njc->ij", wres, w, w, optimize=True)
    resistance = 0.5 * (resistance + resistance.T)

    # Softened Coulomb kernel 1/sqrt(r^2+a^2) is positive definite. Using
    # a common kernel across all basis modes preserves L >= 0.
    distances = cdist(points, points, metric="sqeuclidean")
    kernel = MU0 / (4 * math.pi) / np.sqrt(distances + cfg.field_softening**2)
    weighted_modes = weights[:, None, None] * w
    n_modes = transform.shape[1]
    inductance = np.zeros((n_modes, n_modes), dtype=float)
    for axis in range(3):
        block = weighted_modes[:, :, axis]
        inductance += block.T @ (kernel @ block)
    inductance = 0.5 * (inductance + inductance.T)

    csource = sources.potential_kernels(points)
    coupling = MU0 * np.einsum(
        "n,nmc,ncp->mp", weights, w, csource, optimize=True,
    )
    matrix = resistance + 1j * sources.omega * inductance
    unit_modes = solve(matrix, -1j * sources.omega * coupling,
                       assume_a="gen", check_finite=True)
    z_sea = 1j * sources.omega * coupling.T @ unit_modes
    z_sea = 0.5 * (z_sea + z_sea.T)
    return SeawaterResponse(
        sources, basis, transform, resistance, inductance, coupling,
        unit_modes, z_sea, points, weights, w, cfg.field_softening, basis.size,
    )
