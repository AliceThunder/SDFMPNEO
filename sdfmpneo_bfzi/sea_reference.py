"""Structure-preserving BFZI/seawater MQS Galerkin solver.

This is a finite-domain, finite-basis REFERENCE *prototype*, not a certified
infinite-seawater solution. It preserves the useful vNext design: one coupled
unit-port solve, spatial Joule fields, PSD dissipative port channels and
a reusable response object. Accuracy requires basis/cubature/domain refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    quadrature_scheme: str = "box_shell"
    shell_width_factor: float = 32.0
    shell_normal_order: int = 3
    shell_tangential_order: int = 4
    # Stokes/integration-by-parts dual source-coupling form avoids
    # differentiating the sharp box-wall curl modes inside K integrals.
    # Historical current.potential form remains the default.
    coupling_form: str = "current_potential"
    # Independent composite angular rules for the spherical portion.
    # Defaults reproduce the historical global-mu/midpoint-phi rule.
    polar_panel_count: int = 1
    azimuthal_panel_order: int = 1
    # Optional Gauss degree on each geometry-projected polar panel.
    # Panel edges follow actual enclosures and blending-shell silhouettes.
    polar_geometry_order: int | None = None
    # Optional physical-length cap for spherical radial Gauss intervals.
    # Especially useful when increasing the finite outer domain.
    max_radial_panel_width: float | None = None
    # Extra spherical radial panels at exact oriented enclosure and
    # blending-shell min/max radii. In independent-L configurations
    # only primary R/K inherits these geometry-local radial cuts.
    radial_geometry_cuts: bool = False
    # Optional five orders for the Green double integral only:
    # (radial, polar, azimuthal, shell_normal, shell_tangential).
    # R/K/Joule remain on the primary, potentially finer cubature.
    # When specified, these L angles use their own global mu Gauss and
    # midpoint azimuth, never the potentially far costlier R/K composite
    # or geometry-projected angular settings. Green accuracy is separate.
    inductance_quadrature: tuple[int, int, int, int, int] | None = None

    def __post_init__(self):
        if min(self.radial_order, self.polar_order, self.azimuthal_order) < 2:
            raise ValueError("quadrature orders must be >= 2")
        if not math.isfinite(self.field_softening) or self.field_softening <= 0:
            raise ValueError("field_softening must be positive")
        if not isinstance(self.wall_layers, int) or not 0 <= self.wall_layers <= 6:
            raise ValueError("wall_layers must be an integer from 0 to 6")
        if self.quadrature_scheme not in ("ray", "box_shell"):
            raise ValueError("quadrature_scheme must be 'ray' or 'box_shell'")
        if self.coupling_form not in ("current_potential", "curl_dual"):
            raise ValueError(
                "coupling_form must be 'current_potential' or 'curl_dual'"
            )
        if (not math.isfinite(self.shell_width_factor)
                or self.shell_width_factor <= 0):
            raise ValueError("shell_width_factor must be finite and positive")
        if (not isinstance(self.shell_normal_order, int)
                or self.shell_normal_order < 2
                or not isinstance(self.shell_tangential_order, int)
                or self.shell_tangential_order < 2):
            raise ValueError("shell Gauss orders must be integers >= 2")
        if (not isinstance(self.polar_panel_count, int)
                or isinstance(self.polar_panel_count, bool)
                or not 1 <= self.polar_panel_count <= self.polar_order
                or self.polar_order % self.polar_panel_count != 0):
            raise ValueError("polar_panel_count must be in [1, polar_order]")
        if (not isinstance(self.azimuthal_panel_order, int)
                or isinstance(self.azimuthal_panel_order, bool)
                or self.azimuthal_panel_order < 1):
            raise ValueError("azimuthal_panel_order must be a positive integer")
        if self.polar_geometry_order is not None and (
            not isinstance(self.polar_geometry_order, int)
            or isinstance(self.polar_geometry_order, bool)
            or self.polar_geometry_order < 2
            or self.polar_panel_count != 1
        ):
            raise ValueError(
                "polar_geometry_order must be >= 2 and "
                "requires polar_panel_count=1"
            )
        if not isinstance(self.radial_geometry_cuts, bool):
            raise ValueError("radial_geometry_cuts must be a boolean")
        if self.max_radial_panel_width is not None and (
            not math.isfinite(self.max_radial_panel_width)
            or self.max_radial_panel_width <= 0
        ):
            raise ValueError("max_radial_panel_width must be positive and finite")
        if not 0 < self.resistance_rank_tolerance < 1:
            raise ValueError("resistance_rank_tolerance must be in (0,1)")
        if not isinstance(self.kernel_chunk_size, int) or self.kernel_chunk_size < 1:
            raise ValueError("kernel_chunk_size must be a positive integer")
        orders = self.inductance_quadrature
        if orders is not None and (
            not isinstance(orders, tuple) or len(orders) != 5
            or any(not isinstance(value, int) or value < 2 for value in orders)
        ):
            raise ValueError(
                "inductance_quadrature must be five integer orders >= 2 "
                "or None"
            )
        if orders is not None and self.quadrature_scheme != "box_shell":
            raise ValueError(
                "inductance_quadrature currently requires box_shell scheme"
            )



def _assemble_green_inductances(
    points: np.ndarray, weights: np.ndarray, modes: np.ndarray,
    softenings: tuple[float, ...], chunk_size: int,
) -> tuple[np.ndarray, ...]:
    """Shared-distance symmetric Green assembly for one or more softenings.

    Each L uses the same real Galerkin modes and positive volume cubature.
    Distances are computed once per block, including when comparing several
    regularization lengths. The discrete kernel is positive definite for
    every strictly positive softening.
    """
    n_points, n_modes, _ = modes.shape
    weighted = weights[:, None, None] * modes
    packed = weighted.reshape(n_points, 3 * n_modes)
    matrices = [np.zeros((n_modes, n_modes), dtype=float) for _ in softenings]
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        left = weighted[start:stop]
        for other in range(start, n_points, chunk_size):
            end = min(other + chunk_size, n_points)
            distance2 = cdist(
                points[start:stop], points[other:end],
                metric="sqeuclidean",
            )
            for softening, inductance in zip(softenings, matrices):
                kernel = MU0 / (4 * math.pi) / np.sqrt(
                    distance2 + softening**2
                )
                projected = (kernel @ packed[other:end]).reshape(
                    stop - start, n_modes, 3,
                )
                block = np.einsum(
                    "pic,pjc->ij", left, projected, optimize=True,
                )
                if other == start:
                    inductance += block
                else:
                    inductance += block + block.T
    return tuple(0.5 * (matrix + matrix.T) for matrix in matrices)


def _assemble_port_coupling(
    sources: BFZISources, basis: SeaCurrentBasis,
    transform: np.ndarray, points: np.ndarray, weights: np.ndarray,
    *, coupling_form: str, current_modes: np.ndarray | None = None,
) -> np.ndarray:
    """Two equivalent continuous volume expressions for physical K.

    With J_m=curl(F_m), F_m=0 on both insulation and outer boundary:
        K_mp = mu0 * integral J_m.C_port
             = integral F_m.B_port.
    The second weak form removes sharp wall derivatives from the K
    integrand. Both are Galerkin coupling integrals with the SAME modes,
    not a fitted or independently invented source approximation.
    """
    if coupling_form == "current_potential":
        if current_modes is None:
            current_modes = np.einsum(
                "nmc,mr->nrc", basis.evaluate(points), transform,
                optimize=True,
            )
        return MU0 * np.einsum(
            "n,nmc,ncp->mp", weights, current_modes,
            sources.potential_kernels(points), optimize=True,
        )
    if coupling_form == "curl_dual":
        primitive = np.einsum(
            "nmc,mr->nrc", basis.potential_primitive(points), transform,
            optimize=True,
        )
        source_magnetic = np.stack(
            [np.real(line.magnetic_field(points)) for line in sources.lines],
            axis=-1,
        )
        return np.einsum(
            "n,nmc,ncp->mp", weights, primitive,
            source_magnetic, optimize=True,
        )
    raise ValueError(
        "coupling_form must be 'current_potential' or 'curl_dual'"
    )


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

    def reintegrate_source_coupling(self, points, weights, *,
                                     coupling_form="curl_dual") -> np.ndarray:
        """Integrate K independently from R/L using the selected weak form."""
        x = np.asarray(points, dtype=float)
        w = np.asarray(weights, dtype=float)
        if (x.ndim != 2 or x.shape[1] != 3 or w.shape != (len(x),)
                or len(x) == 0 or not np.isfinite(x).all()
                or not np.isfinite(w).all() or np.any(w <= 0)
                or not self.sources.scene.in_seawater(x).all()):
            raise ValueError("coupling cubature needs positive seawater nodes")
        return _assemble_port_coupling(
            self.sources, self.basis, self.transform, x, w,
            coupling_form=coupling_form,
        )

    def reintegrate_local_response(self, points, weights, *,
                                   coupling_form="current_potential") -> SeawaterResponse:
        """Re-solve on independent R/K/Joule nodes while holding Green L fixed.

        This is a controlled single-axis cubature comparison, not a full
        second MQS reference solve. The trial basis, source, outer domain,
        physical-mode transform, omega and nonlocal inductance remain fixed.
        Only the one-volume R and K integrals are changed; the new positive
        heat nodes match the new dissipative operator.
        """
        x = np.asarray(points, dtype=float)
        w = np.asarray(weights, dtype=float)
        if (x.ndim != 2 or x.shape[1] != 3 or w.shape != (len(x),)
                or len(x) == 0 or not np.isfinite(x).all()
                or not np.isfinite(w).all() or np.any(w <= 0)):
            raise ValueError(
                "independent cubature requires nonempty (n,3) points "
                "and strictly positive finite (n,) volume weights"
            )
        if not self.sources.scene.in_seawater(x).all():
            raise ValueError("independent cubature points must be in seawater")
        modes = np.einsum(
            "nmc,mr->nrc", self.basis.evaluate(x), self.transform,
            optimize=True,
        )
        sigma = self.sources.scene.seawater.conductivity
        r = np.einsum(
            "n,nic,njc->ij", w / sigma, modes, modes, optimize=True,
        )
        r = 0.5 * (r + r.T)
        k = _assemble_port_coupling(
            self.sources, self.basis, self.transform, x, w,
            coupling_form=coupling_form, current_modes=modes,
        )
        omega = self.sources.omega
        coefficients = solve(
            r + 1j * omega * self.inductance,
            -1j * omega * k, assume_a="gen", check_finite=True,
        )
        z = 1j * omega * k.T @ coefficients
        z = 0.5 * (z + z.T)
        return SeawaterResponse(
            self.sources, self.basis, self.transform, r,
            self.inductance, k, coefficients, z,
            x, w, modes, self.field_softening, self.raw_basis_count,
        )

    def resolve_with_coupling(self, coupling) -> SeawaterResponse:
        """Re-solve with an independently integrated physical port coupling.

        R, Green L, physical basis and the original R/Joule heat quadrature
        remain untouched. Energy closure stays exact because it is the same
        R matrix that determines both modal Joule heat and port dissipation.
        A new coupling matrix should be integrated on the *same physical*
        two-port sources and retained Galerkin basis, not fitted freely.
        """
        k = np.asarray(coupling, dtype=float)
        if k.shape != self.coupling.shape or not np.isfinite(k).all():
            raise ValueError("refined coupling must be a finite real (m,2) matrix")
        omega = self.sources.omega
        coefficients = solve(
            self.resistance + 1j * omega * self.inductance,
            -1j * omega * k, assume_a="gen", check_finite=True,
        )
        impedance = 1j * omega * k.T @ coefficients
        impedance = 0.5 * (impedance + impedance.T)
        return replace(
            self, coupling=k, unit_current_modes=coefficients,
            impedance_seawater=impedance,
        )

    def reintegrate_green_responses(
        self, points, weights, *, softenings=None, kernel_chunk_size=128,
    ) -> tuple[SeawaterResponse, ...]:
        """Re-solve with new Green L only; R/K/Joule cubature stays fixed.

        The provided nodes are an independent *double* Green cubature. Every
        response retains the original physical basis, R whitening, K, port
        sources, and heat nodes, so port/volume dissipation still closes.
        Distances are reused across regularization lengths to keep a
        softening sweep cheaper than separate full reference solves.
        """
        x = np.asarray(points, dtype=float)
        w = np.asarray(weights, dtype=float)
        if (x.ndim != 2 or x.shape[1] != 3 or w.shape != (len(x),)
                or len(x) == 0 or not np.isfinite(x).all()
                or not np.isfinite(w).all() or np.any(w <= 0)):
            raise ValueError(
                "Green cubature requires nonempty (n,3) points "
                "and strictly positive finite (n,) weights"
            )
        if not self.sources.scene.in_seawater(x).all():
            raise ValueError("Green cubature points must be in seawater")
        if (not isinstance(kernel_chunk_size, int)
                or kernel_chunk_size < 1):
            raise ValueError("kernel_chunk_size must be a positive integer")
        if softenings is None:
            softenings = (self.field_softening,)
        softenings = tuple(softenings)
        if not softenings or any(
            not math.isfinite(a) or a <= 0 for a in softenings
        ):
            raise ValueError("softenings must be positive finite lengths")
        modes = np.einsum(
            "nmc,mr->nrc", self.basis.evaluate(x), self.transform,
            optimize=True,
        )
        inductances = _assemble_green_inductances(
            x, w, modes, softenings, kernel_chunk_size,
        )
        omega = self.sources.omega
        responses = []
        for softening, inductance in zip(softenings, inductances):
            coefficients = solve(
                self.resistance + 1j * omega * inductance,
                -1j * omega * self.coupling,
                assume_a="gen", check_finite=True,
            )
            impedance = 1j * omega * self.coupling.T @ coefficients
            impedance = 0.5 * (impedance + impedance.T)
            responses.append(replace(
                self, inductance=inductance,
                unit_current_modes=coefficients,
                impedance_seawater=impedance,
                field_softening=softening,
            ))
        return tuple(responses)

    def independent_green_metrics(
        self, points, weights, port_currents=(1.0, 0.0), *,
        probes=None, softenings=None, kernel_chunk_size=128,
    ) -> list[dict[str, float | None]]:
        """Independent fixed-R/K Green changes, not a continuum error bound."""
        currents = self._currents(port_currents)
        held = self.reintegrate_green_responses(
            points, weights, softenings=softenings,
            kernel_chunk_size=kernel_chunk_size,
        )
        previous_power = self.port_seawater_power(currents)
        # The exact coincident-node contribution is proportional to
        # w_i**2/a. A large trace share flags point-collocation aliasing;
        # it is NOT the continuum's integrable near-field self energy.
        x = np.asarray(points, dtype=float)
        w = np.asarray(weights, dtype=float)
        modes = np.einsum(
            "nmc,mr->nrc", self.basis.evaluate(x), self.transform,
            optimize=True,
        )
        self_trace_numerator = (
            MU0 / (4 * math.pi)
            * np.einsum("n,nmc,nmc->", w**2, modes, modes,
                        optimize=True)
        )
        old_q = (
            np.asarray(self.seawater_loss(probes, currents))
            if probes is not None else None
        )
        result = []
        for updated in held:
            power = updated.port_seawater_power(currents)
            new_q = (
                np.asarray(updated.seawater_loss(probes, currents))
                if probes is not None else None
            )
            residual = (
                1j * self.sources.omega
                * (updated.inductance - self.inductance)
                @ self.unit_current_modes
            )
            result.append({
                "softening": updated.field_softening,
                "green_diagonal_trace_fraction": float(
                    (self_trace_numerator / updated.field_softening)
                    / max(float(np.trace(updated.inductance)), 1e-30)
                ),
                "relative_inductance_difference": float(
                    np.linalg.norm(updated.inductance - self.inductance)
                    / max(np.linalg.norm(self.inductance), 1e-30)
                ),
                "relative_resolved_impedance_change": float(
                    np.linalg.norm(
                        updated.impedance_seawater - self.impedance_seawater
                    ) / max(np.linalg.norm(self.impedance_seawater), 1e-30)
                ),
                "relative_resolved_probe_change": (
                    float(np.linalg.norm(new_q - old_q)
                          / max(np.linalg.norm(old_q), 1e-30))
                    if old_q is not None else None
                ),
                "relative_resolved_power_change": abs(
                    power - previous_power
                ) / max(abs(previous_power), 1e-30),
                "relative_heldout_residual": float(
                    np.linalg.norm(residual)
                    / max(
                        np.linalg.norm(
                            1j * self.sources.omega * self.coupling
                        ), 1e-30,
                    )
                ),
                "resolved_power": power,
            })
        return result

    def independent_cubature_metrics(
        self, points, weights, port_currents=(1.0, 0.0),
        *, probes=None,
    ) -> dict[str, float | None]:
        """Held-out R/K integration with both frozen-current and re-solved error.

        Report whether the *original* current still integrates accurately
        and how the port response changes when R and K are reassembled,
        without redoing the expensive nonlocal Green L double integral.
        These metrics do not establish convergence of L or basis completeness.
        """
        currents = self._currents(port_currents)
        held = self.reintegrate_local_response(points, weights)
        amplitudes = self.unit_current_modes @ currents
        power = 0.5 * float(np.real(np.vdot(
            amplitudes, held.resistance @ amplitudes,
        )))
        port_power = self.port_seawater_power(currents)
        updated_power = held.port_seawater_power(currents)
        original_system_residual = (
            (held.resistance + 1j * self.sources.omega * self.inductance)
            @ self.unit_current_modes
            + 1j * self.sources.omega * held.coupling
        )
        probe_change = None
        if probes is not None:
            old_q = np.asarray(self.seawater_loss(probes, currents))
            new_q = np.asarray(held.seawater_loss(probes, currents))
            probe_change = float(
                np.linalg.norm(new_q - old_q)
                / max(np.linalg.norm(old_q), 1e-30)
            )
        return {
            "relative_resolved_probe_change": probe_change,
            "heldout_power": power,
            "on_grid_power": port_power,
            "resolved_power": updated_power,
            "relative_power_difference": abs(power - port_power) / max(
                abs(port_power), 1e-30,
            ),
            "relative_resistance_difference": float(
                np.linalg.norm(held.resistance - self.resistance)
                / max(np.linalg.norm(self.resistance), 1e-30)
            ),
            "relative_coupling_difference": float(
                np.linalg.norm(held.coupling - self.coupling)
                / max(np.linalg.norm(self.coupling), 1e-30)
            ),
            "relative_resolved_impedance_change": float(
                np.linalg.norm(held.impedance_seawater - self.impedance_seawater)
                / max(np.linalg.norm(self.impedance_seawater), 1e-30)
            ),
            "relative_resolved_power_change": abs(
                updated_power - port_power
            ) / max(abs(port_power), 1e-30),
            "relative_heldout_residual": float(
                np.linalg.norm(original_system_residual)
                / max(
                    np.linalg.norm(1j * self.sources.omega * held.coupling),
                    1e-30,
                )
            ),
        }

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

    A shared real basis with the same positive primary cubature for R, K
    and spatial Joule heat ensures exact *discrete* port/volume loss closure.
    The optional separate L cubature is only a controllable approximation
    for the expensive nonlocal magnetic term; it does not redefine that
    dissipative identity. Softened 3D Green kernels approximate self/near-field
    volume interactions.
    They must be refined rather than mistaken for an exact continuum reference.
    """
    cfg = config or SeaMQSConfig()
    scene = sources.scene
    basis = basis or build_sea_basis(scene)
    if basis.scene is not scene:
        raise ValueError("seawater current basis must belong to the source scene")
    # Resolve the outer cutoff on a dedicated radial integration interval.
    taper_width = (
        basis.outer_taper_width
        if basis.outer_envelope == "boundary_layer" else None
    )
    if cfg.quadrature_scheme == "box_shell":
        points, weights = scene.box_shell_quadrature(
            cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
            wall_scale=basis.wall_scale,
            shell_width_factor=cfg.shell_width_factor,
            shell_normal_order=cfg.shell_normal_order,
            shell_tangential_order=cfg.shell_tangential_order,
            outer_taper_width=taper_width,
            max_radial_panel_width=cfg.max_radial_panel_width,
            radial_geometry_cuts=cfg.radial_geometry_cuts,
            polar_panel_count=cfg.polar_panel_count,
            azimuthal_panel_order=cfg.azimuthal_panel_order,
            polar_geometry_order=cfg.polar_geometry_order,
        )
    else:
        points, weights = scene.quadrature(
            cfg.radial_order, cfg.polar_order, cfg.azimuthal_order,
            wall_layers=cfg.wall_layers, wall_scale=basis.wall_scale,
            outer_taper_width=taper_width,
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

    # R and K above use the primary accurate quadrature. The nonlocal
    # Green double integral has quadratic node cost; optionally evaluate
    # only L on an independent, usually coarser positive quadrature. The
    # same physical basis and R-derived normalization are kept, so L
    # remains a symmetric PSD Galerkin approximation without perturbing
    # the dissipative port/volume identity.
    if cfg.inductance_quadrature is None:
        magnetic_points, magnetic_weights, magnetic_modes = points, weights, w
    else:
        radial, polar, azimuthal, normal, tangential = cfg.inductance_quadrature
        magnetic_points, magnetic_weights = scene.box_shell_quadrature(
            radial, polar, azimuthal,
            wall_scale=basis.wall_scale,
            shell_width_factor=cfg.shell_width_factor,
            shell_normal_order=normal,
            shell_tangential_order=tangential,
            outer_taper_width=taper_width,
            max_radial_panel_width=cfg.max_radial_panel_width,
            # The independent five-order L rule must NOT implicitly
            # inherit the potentially much finer geometry-angle R/K
            # rule. A 60k-node L rule costs billions of Green pairs.
            # This intentionally uses the default global mu / midpoint
            # azimuth for the separately specified L integration.
        )
        magnetic_modes = np.einsum(
            "nmc,mr->nrc", basis.evaluate(magnetic_points), transform,
            optimize=True,
        )
    # Nonlocal L is assembled on its own positive cubature with the same
    # physical basis and R normalization. Compare its resolution and the
    # softening independently; neither is proven converged by power closure.
    inductance = _assemble_green_inductances(
        magnetic_points, magnetic_weights, magnetic_modes,
        (cfg.field_softening,), cfg.kernel_chunk_size,
    )[0]

    coupling = _assemble_port_coupling(
        sources, basis, transform, points, weights,
        coupling_form=cfg.coupling_form, current_modes=w,
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
