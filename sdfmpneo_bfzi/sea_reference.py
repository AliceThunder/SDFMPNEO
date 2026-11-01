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
    wall_layers: int = 2
    resistance_rank_tolerance: float = 1e-9
    kernel_chunk_size: int = 128

    def __post_init__(self):
        if min(self.radial_order, self.polar_order, self.azimuthal_order) < 2:
            raise ValueError("quadrature orders must be >= 2")
        if not math.isfinite(self.field_softening) or self.field_softening <= 0:
            raise ValueError("field_softening must be positive")
        if not isinstance(self.wall_layers, int) or not 0 <= self.wall_layers <= 6:
            raise ValueError("wall_layers must be an integer from 0 to 6")
        if not 0 < self.resistance_rank_tolerance < 1:
            raise ValueError("resistance_rank_tolerance must be in (0,1)")
        if not isinstance(self.kernel_chunk_size, int) or self.kernel_chunk_size < 1:
            raise ValueError("kernel_chunk_size must be a positive integer")


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
        return self.port_impedance_at()

    def port_impedance_at(self, temperatures=None) -> np.ndarray:
        """Closed-loop port impedance including finite conductor internal Z."""
        return (
            1j * self.sources.omega * self.sources.inductance_air
            + self.impedance_seawater
            + np.diag(self.sources.conductor_internal_impedance(temperatures))
        )

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
        modes = np.einsum("nmc,mr->nrc", self.basis.evaluate(locations), self.transform)
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

    def seawater_heat_nodes(self, port_currents=(1.0, 0.0)):
        """Positive seawater heat cubature (xyz, dV, q in W/m^3).

        Uses exactly the current-mode samples and positive volume weights
        which define the port dissipative matrix. The integral of this field
        therefore equals the seawater port power up to floating-point error.
        """
        z = self.unit_current_modes @ self._currents(port_currents)
        field = np.einsum("nmc,m->nc", self.integration_modes, z)
        q = np.sum(np.abs(field)**2, axis=1) / (
            2.0 * self.sources.scene.seawater.conductivity
        )
        return self.integration_points.copy(), self.integration_weights.copy(), q

    def thermal_channel_power(self, port_currents=(1.0, 0.0),
                              temperatures=None) -> np.ndarray:
        """Two conductor heat powers followed by total seawater Joule heat."""
        return np.r_[
            self.port_conductor_power(port_currents, temperatures),
            self.port_seawater_power(port_currents),
        ]

    def dissipation_channel(self) -> np.ndarray:
        """Seawater Hermitian PSD channel only, excluding the conductors."""
        return self.unit_current_modes.conj().T @ self.resistance @ self.unit_current_modes

    def port_conductor_power(self, port_currents=(1.0, 0.0),
                             temperatures=None) -> np.ndarray:
        """Per-coil peak-phasor Joule power in each finite circular wire."""
        i = self._currents(port_currents)
        resistances = self.sources.conductor_internal_impedance(temperatures).real
        return 0.5 * resistances * np.abs(i)**2

    def port_total_power(self, port_currents=(1.0, 0.0),
                         temperatures=None) -> float:
        return float(
            self.port_conductor_power(port_currents, temperatures).sum()
            + self.port_seawater_power(port_currents)
        )

    def as_structured_prediction(self, temperatures=None):
        # Keep distinct coil channels so the existing vNext channel-resolved
        # thermal tools can identify the dissipative conductor ports.
        from sdfmpneo_vnext.prediction import StructuredPortPrediction
        conductor = self.sources.conductor_internal_impedance(temperatures).real
        coil_channels = np.zeros((2, 2, 2), dtype=complex)
        coil_channels[0, 0, 0] = conductor[0]
        coil_channels[1, 1, 1] = conductor[1]
        return StructuredPortPrediction(
            self.port_impedance_at(temperatures),
            np.concatenate((coil_channels, self.dissipation_channel()[None]), axis=0),
            channel_labels=("conductor:0", "conductor:1", "seawater"),
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
    if basis.scene is not scene:
        raise ValueError("seawater current basis must belong to the source scene")
    points, weights = scene.quadrature(
        cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
        wall_layers=cfg.wall_layers, wall_scale=basis.wall_scale,
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
    w = np.einsum("nmc,mr->nrc", modes, transform)
    resistance = np.einsum("n,nic,njc->ij", wres, w, w, optimize=True)
    resistance = 0.5 * (resistance + resistance.T)

    # Softened Coulomb kernel 1/sqrt(r^2+a^2) is positive definite. Using
    # a common kernel across all basis modes preserves L >= 0.
    # Matrix-free-in-memory Green contraction: only a small block of the
    # Q-by-Q volume kernel is materialized at once. This borrows the vNext
    # memory-bounded reference pattern without changing the energy kernel.
    weighted_modes = weights[:, None, None] * w
    n_modes = transform.shape[1]
    # Symmetric two-block Green assembly. Compute each pair of cubature
    # blocks once: the transposed off-diagonal contribution is its exact
    # discrete reciprocal counterpart. Peak kernel memory is O(chunk**2)
    # rather than O(chunk * total_nodes); no full QxQ kernel is formed.
    packed = weighted_modes.reshape(len(points), 3 * n_modes)
    inductance = np.zeros((n_modes, n_modes), dtype=float)
    chunk = cfg.kernel_chunk_size
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        left = weighted_modes[start:stop]
        for other in range(start, len(points), chunk):
            end = min(other + chunk, len(points))
            distance2 = cdist(
                points[start:stop], points[other:end], metric="sqeuclidean",
            )
            kernel = MU0 / (4 * math.pi) / np.sqrt(
                distance2 + cfg.field_softening**2
            )
            projected = (kernel @ packed[other:end]).reshape(
                stop - start, n_modes, 3,
            )
            block = np.einsum("pic,pjc->ij", left, projected, optimize=True)
            if other == start:
                inductance += block
            else:
                inductance += block + block.T
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
