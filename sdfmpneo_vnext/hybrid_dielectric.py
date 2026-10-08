from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .hybrid_domain import validate_package_conductor_topology
from .dielectric_surface import DielectricSurfaceSolver
from .electric_tensor import PreparedTensorElectricTransmission
from .magnetic_surface import MagneticSurfaceSolver
from .em import MQSConfig
from .mixed import (
    DenseMixedConductorTeacher,
    MixedResult,
    _equilibrated_dense_solve,
)
from .prediction import StructuredPortPrediction
from .scene import (
    Scene,
    TensorElectricMaterial,
)


@dataclass(frozen=True)
class DielectricCoupledResult:
    mixed_result: MixedResult
    prediction: StructuredPortPrediction
    dielectric_dissipation_matrix: np.ndarray
    surface_density_transfer: np.ndarray
    effective_potential_matrix: np.ndarray
    raw_potential_reciprocity_defect: float
    surface_residual: float
    source_region_index: np.ndarray
    channel_labels: tuple[str, ...]
    magnetic_inductance_correction: np.ndarray | None = None
    magnetic_surface_residual: float = 0.0
    raw_magnetic_reciprocity_defect: float = 0.0
    tensor_electric_transmission: object | None = None

    @property
    def impedance(
        self,
    ) -> np.ndarray:
        """Canonical coupled port observable.

        ``mixed_result`` is the raw dense solver state used for field
        reconstruction.  Tensor energy calibration may replace the public port
        observable without changing those raw coefficients, so the coupled
        result must source its impedance from the structured prediction.
        """
        return np.asarray(
            self.prediction.impedance,
            dtype=complex,
        )

    @property
    def normalized_residual(
        self,
    ) -> float:
        return float(
            self.mixed_result.normalized_residual
        )

    @property
    def environment_dissipation_matrix(
        self,
    ) -> np.ndarray:
        """Aggregate non-conductor electric loss."""
        return np.asarray(
            self.dielectric_dissipation_matrix,
            dtype=complex,
        )

    @property
    def power_closure_error(
        self,
    ) -> float:
        return float(
            self.prediction.power_closure_error()
        )

    def coil_dissipation_matrices(
        self,
    ) -> np.ndarray:
        """Canonical conductor channels associated with ``prediction``."""
        indices = self.prediction.coil_channel_indices(
            self.impedance.shape[0]
        )
        return np.asarray(
            self.prediction.dissipation_channels[
                np.asarray(indices, dtype=int)
            ],
            dtype=complex,
        )

    def dissipation_channels(
        self,
    ) -> np.ndarray:
        return np.asarray(
            self.prediction.dissipation_channels,
            dtype=complex,
        )

    def channel_power(
        self,
        currents,
    ) -> np.ndarray:
        return (
            self.prediction.channel_power(
                currents
            )
        )

    def coil_power(
        self,
        currents,
    ) -> np.ndarray:
        return self.prediction.coil_power(
            currents
        )


class DielectricCoupledMixedTeacher:
    """Dense quasi-static conductor--electric-material correctness backend.

    Conductors use the canonical current--potential--charge mixed formulation.
    Piecewise homogeneous scalar packages are eliminated through a dielectric
    single-layer Schur response. Tensor-electric package regions or a tensor
    homogeneous infinite background use the mesh-free tensor Laplace MFS/Green
    response, producing the same effective nodal-potential operator. Magnetic
    permeability contrast is eliminated through a magnetic-scalar response and
    a local package energy correction to the current-mode inductance operator.
    """

    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        config: MQSConfig | None = None,
        *,
        charge_self_radius_factor: float = 0.75,
        surface_vertical_order: int = 16,
        surface_azimuthal_order: int = 32,
        magnetic_volume_axial_order: int = 8,
        magnetic_volume_radial_order: int = 6,
        magnetic_volume_azimuthal_order: int = 24,
        maximum_raw_reciprocity_defect: float = 0.15,
        maximum_raw_magnetic_reciprocity_defect: float = 0.08,
    ):
        if (
            not scene.packages
            and not isinstance(
                scene.medium,
                TensorElectricMaterial,
            )
        ):
            raise ValueError(
                "electric-coupled teacher requires package regions or a "
                "tensor-electric homogeneous background"
            )
        if maximum_raw_reciprocity_defect <= 0.0:
            raise ValueError(
                "maximum_raw_reciprocity_defect must be positive"
            )
        validate_package_conductor_topology(
            scene
        )

        self.scene = scene
        self.frequency_hz = float(
            frequency_hz
        )
        if (
            not np.isfinite(
                self.frequency_hz
            )
            or self.frequency_hz < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        self.conductive_dc = bool(
            self.frequency_hz == 0.0
            and (
                scene.medium.loss_conductivity(
                    0.0
                )
                > 0.0
                or any(
                    package.material.loss_conductivity(
                        0.0
                    )
                    > 0.0
                    for package
                    in scene.packages
                )
            )
        )
        self.config = (
            config
            or MQSConfig()
        )
        self.charge_self_radius_factor = float(
            charge_self_radius_factor
        )
        self.maximum_raw_reciprocity_defect = float(
            maximum_raw_reciprocity_defect
        )
        self.tensor_electric = bool(
            isinstance(
                scene.medium,
                TensorElectricMaterial,
            )
            or any(
                isinstance(
                    package.material,
                    TensorElectricMaterial,
                )
                for package in scene.packages
            )
        )
        self.surface_vertical_order = int(
            surface_vertical_order
        )
        self.surface_azimuthal_order = int(
            surface_azimuthal_order
        )

        self.maximum_raw_magnetic_reciprocity_defect = float(
            maximum_raw_magnetic_reciprocity_defect
        )
        if self.maximum_raw_magnetic_reciprocity_defect <= 0.0:
            raise ValueError(
                "maximum_raw_magnetic_reciprocity_defect must be positive"
            )
        self.magnetic_volume_axial_order = int(
            magnetic_volume_axial_order
        )
        self.magnetic_volume_radial_order = int(
            magnetic_volume_radial_order
        )
        self.magnetic_volume_azimuthal_order = int(
            magnetic_volume_azimuthal_order
        )
        if (
            self.magnetic_volume_axial_order < 2
            or self.magnetic_volume_radial_order < 2
            or self.magnetic_volume_azimuthal_order < 8
        ):
            raise ValueError(
                "magnetic package volume quadrature orders are too small"
            )

        conductor_scene = Scene(
            scene.coils,
            scene.medium,
            (),
        )
        self.conductor_teacher = DenseMixedConductorTeacher(
            conductor_scene,
            self.frequency_hz,
            self.config,
            charge_self_radius_factor=(
                self.charge_self_radius_factor
            ),
        )
        self.surface_solver = (
            None
            if self.tensor_electric
            else DielectricSurfaceSolver(
                scene.packages,
                scene.medium,
                self.frequency_hz,
                coefficient_mode=(
                    "conductivity"
                    if self.conductive_dc
                    else "permittivity"
                ),
                vertical_order=(
                    surface_vertical_order
                ),
                azimuthal_order=(
                    surface_azimuthal_order
                ),
            )
        )
        self.magnetic_surface_solver = (
            None
            if not scene.packages
            else MagneticSurfaceSolver(
                scene.packages,
                scene.medium,
                conductors=scene.coils,
                vertical_order=(
                    surface_vertical_order
                ),
                azimuthal_order=(
                    surface_azimuthal_order
                ),
            )
        )

    def _source_regions(
        self,
        positions: np.ndarray,
    ):
        n = len(
            positions
        )
        region = np.asarray(
            self.surface_solver.topology.deepest_containing(
                self.scene.packages,
                positions,
                tolerance=2e-12,
            ),
            dtype=int,
        )
        coefficient = np.full(
            n,
            self.surface_solver.background_permittivity,
            dtype=complex,
        )
        for package_index, package in enumerate(
            self.scene.packages
        ):
            mask = region == package_index
            if np.any(mask):
                coefficient[mask] = self.surface_solver._material_coefficient(
                    package.material
                )
        return region, coefficient

    def _direct_node_potential(
        self,
        positions,
        radii,
        source_permittivity,
    ):
        diff = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(diff, axis=2)
        self_distance = self.charge_self_radius_factor * np.sqrt(
            radii[:, None] * radii[None, :]
        )
        distance = distance.copy()
        np.fill_diagonal(distance, np.diag(self_distance))
        if np.any(distance <= 0.0):
            raise ValueError(
                "conductor charge nodes contain coincident points"
            )
        return 1.0 / (
            4.0
            * np.pi
            * distance
            * source_permittivity[None, :]
        )

    def _surface_incident_derivative(
        self,
        node_positions,
        node_radii,
        source_permittivity,
    ):
        diff = (
            self.surface_solver.positions[:, None, :]
            - node_positions[None, :, :]
        )
        soft = self.charge_self_radius_factor * node_radii[None, :]
        distance2 = np.sum(diff * diff, axis=2) + soft**2
        denominator = distance2**1.5
        normal_dot = np.einsum(
            "si,sji->sj",
            self.surface_solver.normals,
            diff,
        )
        return -normal_dot / (
            4.0
            * np.pi
            * denominator
            * source_permittivity[None, :]
        )

    def _induced_node_potential_matrix(
        self,
        node_positions,
        density_from_node_charge,
    ):
        diff = (
            node_positions[:, None, :]
            - self.surface_solver.positions[None, :, :]
        )
        distance = np.linalg.norm(diff, axis=2)
        geometry_points = np.concatenate(
            (
                np.asarray(node_positions, dtype=float),
                np.asarray(self.surface_solver.positions, dtype=float),
            ),
            axis=0,
        )
        geometry_center = np.mean(geometry_points, axis=0)
        scale = max(
            float(
                np.max(
                    np.linalg.norm(
                        geometry_points - geometry_center[None, :],
                        axis=1,
                    )
                )
            ),
            1.0,
        )
        if np.any(distance <= 1e-12 * scale):
            raise ValueError(
                "a conductor charge node lies on a dielectric interface; "
                "explicit conductor-interface contact physics is required"
            )
        evaluation = self.surface_solver.weights[None, :] / (
            4.0 * np.pi * distance
        )
        return evaluation @ density_from_node_charge

    def _effective_potential(
        self,
        node_positions,
        node_radii,
    ):
        source_region, source_permittivity = self._source_regions(
            node_positions
        )
        direct = self._direct_node_potential(
            node_positions,
            node_radii,
            source_permittivity,
        )
        incident_derivative = self._surface_incident_derivative(
            node_positions,
            node_radii,
            source_permittivity,
        )
        density_from_node_charge, surface_residuals = (
            self.surface_solver.solve_density_matrix(
                incident_derivative
            )
        )
        induced = self._induced_node_potential_matrix(
            node_positions,
            density_from_node_charge,
        )
        raw = direct + induced
        reciprocity_defect = float(
            np.linalg.norm(raw - raw.T)
            / max(np.linalg.norm(raw), 1e-30)
        )
        if reciprocity_defect > self.maximum_raw_reciprocity_defect:
            raise RuntimeError(
                "dielectric Schur potential failed the raw reciprocity "
                f"diagnostic: {reciprocity_defect:.3e}"
            )
        effective = 0.5 * (raw + raw.T)
        return (
            effective,
            density_from_node_charge,
            float(np.max(surface_residuals)),
            reciprocity_defect,
            source_region,
        )

    def _effective_conduction_potential(
        self,
        node_positions,
        node_radii,
    ):
        source_region, source_conductivity = self._source_regions(
            node_positions
        )
        source_conductivity = np.asarray(
            np.real(source_conductivity),
            dtype=float,
        )
        scale = max(float(np.max(source_conductivity)), 1e-30)
        conductive = source_conductivity > 1e-13 * scale
        n_node = len(node_positions)
        n_surface = len(self.surface_solver.weights)
        full_potential = np.zeros(
            (n_node, n_node),
            dtype=complex,
        )
        full_density = np.zeros(
            (n_surface, n_node),
            dtype=complex,
        )
        if not np.any(conductive):
            return (
                full_potential,
                full_density,
                0.0,
                0.0,
                source_region,
                conductive,
            )
        positions = np.asarray(node_positions, dtype=float)[conductive]
        radii = np.asarray(node_radii, dtype=float)[conductive]
        coefficient = source_conductivity[conductive].astype(complex)
        direct = self._direct_node_potential(
            positions,
            radii,
            coefficient,
        )
        incident_derivative = self._surface_incident_derivative(
            positions,
            radii,
            coefficient,
        )
        density, surface_residuals = self.surface_solver.solve_density_matrix(
            incident_derivative
        )
        induced = self._induced_node_potential_matrix(
            positions,
            density,
        )
        raw = direct + induced
        reciprocity_defect = float(
            np.linalg.norm(raw - raw.T)
            / max(np.linalg.norm(raw), 1e-30)
        )
        if reciprocity_defect > self.maximum_raw_reciprocity_defect:
            raise RuntimeError(
                "DC conduction Schur potential failed the raw reciprocity "
                f"diagnostic: {reciprocity_defect:.3e}"
            )
        effective = 0.5 * (raw + raw.T)
        indices = np.flatnonzero(conductive)
        full_potential[np.ix_(indices, indices)] = effective
        full_density[:, indices] = density
        return (
            full_potential,
            full_density,
            float(np.max(surface_residuals)),
            reciprocity_defect,
            source_region,
            conductive,
        )

    def solve(
        self,
    ) -> DielectricCoupledResult:
        (
            resistance,
            inductance,
            current_constraint,
            _,
        ) = self.conductor_teacher._mqs.assemble()
        if self.magnetic_surface_solver is None:
            magnetic_inductance_correction = np.zeros_like(
                inductance,
                dtype=float,
            )
            magnetic_surface_residual = 0.0
            magnetic_reciprocity_defect = 0.0
        else:
            (
                magnetic_inductance_correction,
                magnetic_surface_residual,
                magnetic_reciprocity_defect,
            ) = self.magnetic_surface_solver.inductance_correction(
                self.conductor_teacher._mqs,
                volume_axial_order=(
                    self.magnetic_volume_axial_order
                ),
                volume_radial_order=(
                    self.magnetic_volume_radial_order
                ),
                volume_azimuthal_order=(
                    self.magnetic_volume_azimuthal_order
                ),
                maximum_raw_reciprocity_defect=(
                    self.maximum_raw_magnetic_reciprocity_defect
                ),
            )
        inductance = inductance + magnetic_inductance_correction
        (
            divergence,
            port_injection,
            gauge,
            node_positions,
            node_radii,
        ) = self.conductor_teacher._topology(
            current_constraint
        )
        tensor_transmission = None
        if self.tensor_electric:
            tensor_transmission = PreparedTensorElectricTransmission(
                self.scene,
                self.frequency_hz,
                node_positions,
                self.charge_self_radius_factor * node_radii,
                conduction_dc=(
                    self.conductive_dc
                ),
                surface_vertical_order=(
                    self.surface_vertical_order
                ),
                surface_azimuthal_order=(
                    self.surface_azimuthal_order
                ),
                maximum_raw_reciprocity_defect=(
                    self.maximum_raw_reciprocity_defect
                ),
            )
            effective_potential = tensor_transmission.potential_matrix()
            density_from_node_charge = np.zeros(
                (0, len(node_positions)),
                dtype=complex,
            )
            surface_residual = float(
                tensor_transmission.interface_residual
            )
            reciprocity_defect = float(
                tensor_transmission.raw_reciprocity_defect
            )
            source_region = np.asarray(
                tensor_transmission.source_package_region,
                dtype=int,
            )
            conductive_node_mask = (
                np.ones(
                    len(node_positions),
                    dtype=bool,
                )
                if self.conductive_dc
                else None
            )
        elif self.conductive_dc:
            (
                effective_potential,
                density_from_node_charge,
                surface_residual,
                reciprocity_defect,
                source_region,
                conductive_node_mask,
            ) = self._effective_conduction_potential(
                node_positions,
                node_radii,
            )
        else:
            (
                effective_potential,
                density_from_node_charge,
                surface_residual,
                reciprocity_defect,
                source_region,
            ) = self._effective_potential(
                node_positions,
                node_radii,
            )
            conductive_node_mask = None

        current_operator = (
            resistance.astype(complex)
            + 1j
            * self.conductor_teacher.omega
            * inductance
        )
        reduced_divergence = gauge.T @ divergence
        reduced_port = gauge.T @ port_injection
        m = current_operator.shape[0]
        nr = reduced_divergence.shape[0]
        if self.conductive_dc:
            reduced_weight = np.sum(
                gauge[conductive_node_mask, :] ** 2,
                axis=0,
            )
            mixed_columns = (
                (reduced_weight > 1e-10)
                & (reduced_weight < 1.0 - 1e-10)
            )
            if np.any(mixed_columns):
                raise ValueError(
                    "a conductor reduced node subspace crosses a DC "
                    "conductive/insulating material boundary"
                )
            conductive_reduced_mask = reduced_weight > 0.5
            conductive_columns = np.flatnonzero(
                conductive_reduced_mask
            )
            nc = len(conductive_columns)
            selector = np.eye(nr, dtype=complex)[:, conductive_columns]
            if nc:
                conductive_gauge = gauge[:, conductive_columns]
                reduced_potential = (
                    conductive_gauge.T
                    @ effective_potential
                    @ conductive_gauge
                )
            else:
                reduced_potential = np.zeros((0, 0), dtype=complex)
            kkt = np.block(
                [
                    [
                        current_operator,
                        -reduced_divergence.T.astype(complex),
                        np.zeros((m, nc), dtype=complex),
                    ],
                    [
                        reduced_divergence.astype(complex),
                        np.zeros((nr, nr), dtype=complex),
                        selector,
                    ],
                    [
                        np.zeros((nc, m), dtype=complex),
                        selector.T,
                        -reduced_potential.astype(complex),
                    ],
                ]
            )
            rhs = np.vstack(
                (
                    np.zeros(
                        (m, port_injection.shape[1]),
                        dtype=complex,
                    ),
                    reduced_port.astype(complex),
                    np.zeros(
                        (nc, port_injection.shape[1]),
                        dtype=complex,
                    ),
                )
            )
        else:
            conductive_columns = None
            reduced_potential = gauge.T @ effective_potential @ gauge
            kkt = np.block(
                [
                    [
                        current_operator,
                        -reduced_divergence.T.astype(complex),
                        np.zeros((m, nr), dtype=complex),
                    ],
                    [
                        reduced_divergence.astype(complex),
                        np.zeros((nr, nr), dtype=complex),
                        1j
                        * self.conductor_teacher.omega
                        * np.eye(nr, dtype=complex),
                    ],
                    [
                        np.zeros((nr, m), dtype=complex),
                        np.eye(nr, dtype=complex),
                        -reduced_potential.astype(complex),
                    ],
                ]
            )
            rhs = np.vstack(
                (
                    np.zeros(
                        (m, port_injection.shape[1]),
                        dtype=complex,
                    ),
                    reduced_port.astype(complex),
                    np.zeros(
                        (nr, port_injection.shape[1]),
                        dtype=complex,
                    ),
                )
            )
        solution = _equilibrated_dense_solve(
            kkt,
            rhs,
        )
        current_coefficients = solution[:m]
        potential_reduced = solution[m : m + nr]
        state_reduced = solution[m + nr :]
        node_potential = gauge @ potential_reduced
        if self.conductive_dc:
            node_charge = np.zeros(
                (
                    gauge.shape[0],
                    state_reduced.shape[1],
                ),
                dtype=complex,
            )
            if len(conductive_columns):
                node_environment_current = (
                    gauge[:, conductive_columns]
                    @ state_reduced
                )
            else:
                node_environment_current = np.zeros(
                    (
                        gauge.shape[0],
                        state_reduced.shape[1],
                    ),
                    dtype=complex,
                )
            node_surface_state = node_environment_current
        else:
            node_charge = gauge @ state_reduced
            node_environment_current = None
            node_surface_state = node_charge
        impedance = port_injection.T @ node_potential

        residual = kkt @ solution - rhs
        normalized_residual = float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(rhs), 1.0)
        )
        segment_coils = np.asarray(
            [
                segment.coil
                for segment in self.conductor_teacher._mqs._segments
            ],
            dtype=int,
        )
        mode_segments = np.empty(
            self.conductor_teacher._mqs._n_modes,
            dtype=int,
        )
        for index, segment in enumerate(
            self.conductor_teacher._mqs._segments
        ):
            mode_segments[segment.mode_slice] = index

        mixed_result = MixedResult(
            impedance,
            current_coefficients,
            node_potential,
            node_charge,
            resistance,
            inductance,
            divergence,
            effective_potential,
            port_injection,
            normalized_residual,
            segment_coils,
            mode_segments,
            None,
            node_environment_current,
        )
        conductor_channels = mixed_result.coil_dissipation_matrices()
        if self.conductive_dc:
            dissipative_potential = 0.5 * (
                effective_potential
                + effective_potential.conj().T
            )
            dielectric_channel = (
                node_environment_current.conj().T
                @ dissipative_potential
                @ node_environment_current
            )
        else:
            imaginary_potential = (
                effective_potential
                - effective_potential.conj().T
            ) / (2j)
            dielectric_channel = (
                self.conductor_teacher.omega
                * node_charge.conj().T
                @ imaginary_potential
                @ node_charge
            )
        dielectric_channel = 0.5 * (
            dielectric_channel
            + dielectric_channel.conj().T
        )
        all_channels = np.concatenate(
            (
                conductor_channels,
                dielectric_channel[None, :, :],
            ),
            axis=0,
        )
        environment_label = (
            "electric_environment:aggregate"
            if (
                self.conductive_dc
                or self.scene.medium.loss_conductivity(
                    self.frequency_hz
                ) > 0.0
            )
            else "dielectric:aggregate"
        )
        prediction = StructuredPortPrediction(
            impedance,
            all_channels,
            tuple(
                f"coil:{index}"
                for index in range(len(self.scene.coils))
            )
            + (environment_label,),
        )
        surface_density_transfer = (
            density_from_node_charge @ node_surface_state
        )
        labels = tuple(
            f"conductor:{coil.name}"
            for coil in self.scene.coils
        ) + (environment_label,)
        return DielectricCoupledResult(
            mixed_result,
            prediction,
            dielectric_channel,
            surface_density_transfer,
            effective_potential,
            reciprocity_defect,
            surface_residual,
            source_region,
            labels,
            magnetic_inductance_correction,
            magnetic_surface_residual,
            magnetic_reciprocity_defect,
            tensor_transmission,
        )


class DielectricCoupledReferenceArtifact:
    """REFERENCE artifact for heterogeneous/tensor electric media."""

    def __init__(
        self,
        *,
        config: MQSConfig | None = None,
        surface_vertical_order: int = 16,
        surface_azimuthal_order: int = 32,
        magnetic_volume_axial_order: int = 8,
        magnetic_volume_radial_order: int = 6,
        magnetic_volume_azimuthal_order: int = 24,
        maximum_raw_magnetic_reciprocity_defect: float = 0.08,
    ):
        self.config = config or MQSConfig()
        self.surface_vertical_order = int(
            surface_vertical_order
        )
        self.surface_azimuthal_order = int(
            surface_azimuthal_order
        )
        self.magnetic_volume_axial_order = int(
            magnetic_volume_axial_order
        )
        self.magnetic_volume_radial_order = int(
            magnetic_volume_radial_order
        )
        self.magnetic_volume_azimuthal_order = int(
            magnetic_volume_azimuthal_order
        )
        self.maximum_raw_magnetic_reciprocity_defect = float(
            maximum_raw_magnetic_reciprocity_defect
        )

    def solve(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> DielectricCoupledResult:
        return DielectricCoupledMixedTeacher(
            scene,
            frequency_hz,
            self.config,
            surface_vertical_order=(
                self.surface_vertical_order
            ),
            surface_azimuthal_order=(
                self.surface_azimuthal_order
            ),
            magnetic_volume_axial_order=(
                self.magnetic_volume_axial_order
            ),
            magnetic_volume_radial_order=(
                self.magnetic_volume_radial_order
            ),
            magnetic_volume_azimuthal_order=(
                self.magnetic_volume_azimuthal_order
            ),
            maximum_raw_magnetic_reciprocity_defect=(
                self.maximum_raw_magnetic_reciprocity_defect
            ),
        ).solve()

    def predict_structured(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> StructuredPortPrediction:
        return self.solve(
            scene,
            frequency_hz,
        ).prediction

    def prepare_spatial(
        self,
        scene: Scene,
        frequency_hz: float,
        *,
        volume_axial_order: int = 8,
        volume_radial_order: int = 6,
        volume_azimuthal_order: int = 24,
        background_radial_order: int = 12,
        background_angular_order: int = 48,
        maximum_raw_closure_error: float = 0.25,
        normalized_closure_tolerance: float = 1e-6,
    ):
        from .hybrid_field import prepare_hybrid_reference_loss_field

        teacher = DielectricCoupledMixedTeacher(
            scene,
            frequency_hz,
            self.config,
            surface_vertical_order=(
                self.surface_vertical_order
            ),
            surface_azimuthal_order=(
                self.surface_azimuthal_order
            ),
            magnetic_volume_axial_order=(
                volume_axial_order
            ),
            magnetic_volume_radial_order=(
                volume_radial_order
            ),
            magnetic_volume_azimuthal_order=(
                volume_azimuthal_order
            ),
            maximum_raw_magnetic_reciprocity_defect=(
                self.maximum_raw_magnetic_reciprocity_defect
            ),
        )
        result = teacher.solve()
        return prepare_hybrid_reference_loss_field(
            teacher,
            result,
            volume_axial_order=(
                volume_axial_order
            ),
            volume_radial_order=(
                volume_radial_order
            ),
            volume_azimuthal_order=(
                volume_azimuthal_order
            ),
            background_radial_order=(
                background_radial_order
            ),
            background_angular_order=(
                background_angular_order
            ),
            maximum_raw_closure_error=(
                maximum_raw_closure_error
            ),
            normalized_closure_tolerance=(
                normalized_closure_tolerance
            ),
        )

    def predict(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> np.ndarray:
        return self.predict_structured(
            scene,
            frequency_hz,
        ).impedance
