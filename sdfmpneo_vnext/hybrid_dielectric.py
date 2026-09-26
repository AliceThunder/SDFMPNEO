from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dielectric_surface import DielectricSurfaceSolver
from .em import MQSConfig
from .mixed import (
    DenseMixedConductorTeacher,
    MixedResult,
    _equilibrated_dense_solve,
)
from .prediction import StructuredPortPrediction
from .scene import Scene


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

    @property
    def impedance(
        self,
    ) -> np.ndarray:
        return (
            self.mixed_result.impedance
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
        """Aggregate non-conductor electric loss.

        For a lossless background this is the historical package dielectric
        channel.  When the homogeneous background is lossy it contains the
        package plus background electric-environment dissipation until the
        spatial decomposition is requested explicitly.
        """
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

    def channel_power(
        self,
        currents,
    ) -> np.ndarray:
        return (
            self.prediction.channel_power(
                currents
            )
        )


class DielectricCoupledMixedTeacher:
    """Dense quasi-static conductor--dielectric correctness backend.

    Conductors use the canonical current--potential--charge mixed formulation.
    Piecewise homogeneous isotropic packages are eliminated through a
    dielectric single-layer Schur response, producing an effective nodal
    potential operator. The magnetic current block remains the homogeneous
    MQS conductor block; magnetic material contrast is intentionally rejected.
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
        maximum_raw_reciprocity_defect: float = 0.15,
    ):
        if not scene.packages:
            raise ValueError(
                "dielectric-coupled teacher requires at least one package"
            )
        if maximum_raw_reciprocity_defect <= 0.0:
            raise ValueError(
                "maximum_raw_reciprocity_defect must be positive"
            )
        for package in scene.packages:
            if not np.isclose(
                package.material.relative_permeability,
                scene.medium.relative_permeability,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise NotImplementedError(
                    "magnetic package contrast requires the magnetic SIE/VIE "
                    "extension and is not approximated by the dielectric solver"
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
        if (
            self.frequency_hz == 0.0
            and (
                scene.medium.conductivity
                > 0.0
                or any(
                    package.material.conductivity
                    > 0.0
                    for package
                    in scene.packages
                )
            )
        ):
            raise NotImplementedError(
                "conductive background/package media at DC require the static "
                "conduction interface formulation"
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

        conductor_scene = Scene(
            scene.coils,
            scene.medium,
            (),
        )
        self.conductor_teacher = (
            DenseMixedConductorTeacher(
                conductor_scene,
                self.frequency_hz,
                self.config,
                charge_self_radius_factor=(
                    self.charge_self_radius_factor
                ),
            )
        )
        self.surface_solver = (
            DielectricSurfaceSolver(
                scene.packages,
                scene.medium,
                self.frequency_hz,
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
        region = np.full(
            n,
            -1,
            dtype=int,
        )
        permittivity = np.full(
            n,
            self.surface_solver.background_permittivity,
            dtype=complex,
        )
        for package_index, package in enumerate(
            self.scene.packages
        ):
            inside = np.asarray(
                package.geometry.contains(
                    positions,
                    tolerance=2e-12,
                ),
                dtype=bool,
            )
            overlap = (
                inside
                & (
                    region
                    >= 0
                )
            )
            if np.any(
                overlap
            ):
                raise NotImplementedError(
                    "nested/overlapping dielectric packages containing the "
                    "same conductor charge node are not supported by the first "
                    "multi-domain SIE backend"
                )
            if np.any(
                inside
            ):
                region[
                    inside
                ] = package_index
                permittivity[
                    inside
                ] = (
                    package.material.complex_permittivity(
                        self.frequency_hz
                    )
                )
        return (
            region,
            permittivity,
        )

    def _direct_node_potential(
        self,
        positions,
        radii,
        source_permittivity,
    ):
        diff = (
            positions[
                :,
                None,
                :,
            ]
            - positions[
                None,
                :,
                :,
            ]
        )
        distance = np.linalg.norm(
            diff,
            axis=2,
        )
        self_distance = (
            self.charge_self_radius_factor
            * np.sqrt(
                radii[
                    :,
                    None,
                ]
                * radii[
                    None,
                    :,
                ]
            )
        )
        distance = distance.copy()
        np.fill_diagonal(
            distance,
            np.diag(
                self_distance
            ),
        )
        if np.any(
            distance <= 0.0
        ):
            raise ValueError(
                "conductor charge nodes contain coincident points"
            )
        return (
            1.0
            / (
                4.0
                * np.pi
                * distance
                * source_permittivity[
                    None,
                    :,
                ]
            )
        )

    def _surface_incident_derivative(
        self,
        node_positions,
        node_radii,
        source_permittivity,
    ):
        diff = (
            self.surface_solver.positions[
                :,
                None,
                :,
            ]
            - node_positions[
                None,
                :,
                :,
            ]
        )
        soft = (
            self.charge_self_radius_factor
            * node_radii[
                None,
                :,
            ]
        )
        distance2 = (
            np.sum(
                diff
                * diff,
                axis=2,
            )
            + soft**2
        )
        denominator = (
            distance2**1.5
        )
        normal_dot = np.einsum(
            "si,sji->sj",
            self.surface_solver.normals,
            diff,
        )
        return (
            -normal_dot
            / (
                4.0
                * np.pi
                * denominator
                * source_permittivity[
                    None,
                    :,
                ]
            )
        )

    def _induced_node_potential_matrix(
        self,
        node_positions,
        density_from_node_charge,
    ):
        diff = (
            node_positions[
                :,
                None,
                :,
            ]
            - self.surface_solver.positions[
                None,
                :,
                :,
            ]
        )
        distance = np.linalg.norm(
            diff,
            axis=2,
        )
        geometry_points = np.concatenate(
            (
                np.asarray(
                    node_positions,
                    dtype=float,
                ),
                np.asarray(
                    self.surface_solver.positions,
                    dtype=float,
                ),
            ),
            axis=0,
        )
        geometry_center = np.mean(
            geometry_points,
            axis=0,
        )
        scale = max(
            float(
                np.max(
                    np.linalg.norm(
                        geometry_points
                        - geometry_center[
                            None,
                            :
                        ],
                        axis=1,
                    )
                )
            ),
            1.0,
        )
        if np.any(
            distance
            <= 1e-12
            * scale
        ):
            raise ValueError(
                "a conductor charge node lies on a dielectric interface; "
                "explicit conductor-interface contact physics is required"
            )
        evaluation = (
            self.surface_solver.weights[
                None,
                :,
            ]
            / (
                4.0
                * np.pi
                * distance
            )
        )
        return (
            evaluation
            @ density_from_node_charge
        )

    def _effective_potential(
        self,
        node_positions,
        node_radii,
    ):
        (
            source_region,
            source_permittivity,
        ) = self._source_regions(
            node_positions
        )
        direct = (
            self._direct_node_potential(
                node_positions,
                node_radii,
                source_permittivity,
            )
        )
        incident_derivative = (
            self._surface_incident_derivative(
                node_positions,
                node_radii,
                source_permittivity,
            )
        )
        (
            density_from_node_charge,
            surface_residuals,
        ) = (
            self.surface_solver.solve_density_matrix(
                incident_derivative
            )
        )
        induced = (
            self._induced_node_potential_matrix(
                node_positions,
                density_from_node_charge,
            )
        )
        raw = (
            direct
            + induced
        )
        reciprocity_defect = float(
            np.linalg.norm(
                raw
                - raw.T
            )
            / max(
                np.linalg.norm(
                    raw
                ),
                1e-30,
            )
        )
        if (
            reciprocity_defect
            > self.maximum_raw_reciprocity_defect
        ):
            raise RuntimeError(
                "dielectric Schur potential failed the raw reciprocity "
                f"diagnostic: {reciprocity_defect:.3e}"
            )

        # Reciprocity is exact in the continuous passive isotropic problem.
        # Symmetrization removes only the Nyström/Galerkin mismatch after the
        # raw defect above has independently bounded that mismatch.
        effective = 0.5 * (
            raw
            + raw.T
        )
        return (
            effective,
            density_from_node_charge,
            float(
                np.max(
                    surface_residuals
                )
            ),
            reciprocity_defect,
            source_region,
        )

    def solve(
        self,
    ) -> DielectricCoupledResult:
        (
            resistance,
            inductance,
            current_constraint,
            _,
        ) = (
            self.conductor_teacher._mqs.assemble()
        )
        (
            divergence,
            port_injection,
            gauge,
            node_positions,
            node_radii,
        ) = (
            self.conductor_teacher._topology(
                current_constraint
            )
        )
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

        current_operator = (
            resistance.astype(
                complex
            )
            + 1j
            * self.conductor_teacher.omega
            * inductance
        )
        reduced_divergence = (
            gauge.T
            @ divergence
        )
        reduced_port = (
            gauge.T
            @ port_injection
        )
        reduced_potential = (
            gauge.T
            @ effective_potential
            @ gauge
        )

        m = current_operator.shape[
            0
        ]
        nr = reduced_divergence.shape[
            0
        ]
        kkt = np.block(
            [
                [
                    current_operator,
                    -reduced_divergence.T.astype(
                        complex
                    ),
                    np.zeros(
                        (
                            m,
                            nr,
                        ),
                        dtype=complex,
                    ),
                ],
                [
                    reduced_divergence.astype(
                        complex
                    ),
                    np.zeros(
                        (
                            nr,
                            nr,
                        ),
                        dtype=complex,
                    ),
                    1j
                    * self.conductor_teacher.omega
                    * np.eye(
                        nr,
                        dtype=complex,
                    ),
                ],
                [
                    np.zeros(
                        (
                            nr,
                            m,
                        ),
                        dtype=complex,
                    ),
                    np.eye(
                        nr,
                        dtype=complex,
                    ),
                    -reduced_potential.astype(
                        complex
                    ),
                ],
            ]
        )
        rhs = np.vstack(
            (
                np.zeros(
                    (
                        m,
                        port_injection.shape[
                            1
                        ],
                    ),
                    dtype=complex,
                ),
                reduced_port.astype(
                    complex
                ),
                np.zeros(
                    (
                        nr,
                        port_injection.shape[
                            1
                        ],
                    ),
                    dtype=complex,
                ),
            )
        )
        solution = (
            _equilibrated_dense_solve(
                kkt,
                rhs,
            )
        )
        current_coefficients = (
            solution[
                :m
            ]
        )
        potential_reduced = (
            solution[
                m : m + nr
            ]
        )
        charge_reduced = (
            solution[
                m + nr :
            ]
        )
        node_potential = (
            gauge
            @ potential_reduced
        )
        node_charge = (
            gauge
            @ charge_reduced
        )
        impedance = (
            port_injection.T
            @ node_potential
        )

        residual = (
            kkt
            @ solution
            - rhs
        )
        normalized_residual = float(
            np.linalg.norm(
                residual
            )
            / max(
                np.linalg.norm(
                    rhs
                ),
                1.0,
            )
        )

        segment_coils = np.asarray(
            [
                segment.coil
                for segment
                in self.conductor_teacher._mqs._segments
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
            mode_segments[
                segment.mode_slice
            ] = index

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
        )

        conductor_channels = (
            mixed_result.coil_dissipation_matrices()
        )
        imaginary_potential = (
            effective_potential
            - effective_potential.conj().T
        ) / (
            2j
        )
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
                dielectric_channel[
                    None,
                    :,
                    :,
                ],
            ),
            axis=0,
        )
        prediction = (
            StructuredPortPrediction(
                impedance,
                all_channels,
            )
        )

        surface_density_transfer = (
            density_from_node_charge
            @ node_charge
        )
        environment_label = (
            "electric_environment:aggregate"
            if self.scene.medium.conductivity
            > 0.0
            else "dielectric:aggregate"
        )
        labels = tuple(
            f"conductor:{coil.name}"
            for coil in self.scene.coils
        ) + (
            environment_label,
        )

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
        )



class DielectricCoupledReferenceArtifact:
    """REFERENCE artifact for the first piecewise-homogeneous dielectric extension."""

    def __init__(
        self,
        *,
        config: MQSConfig | None = None,
        surface_vertical_order: int = 16,
        surface_azimuthal_order: int = 32,
    ):
        self.config = (
            config
            or MQSConfig()
        )
        self.surface_vertical_order = int(
            surface_vertical_order
        )
        self.surface_azimuthal_order = int(
            surface_azimuthal_order
        )

    def solve(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> DielectricCoupledResult:
        return (
            DielectricCoupledMixedTeacher(
                scene,
                frequency_hz,
                self.config,
                surface_vertical_order=(
                    self.surface_vertical_order
                ),
                surface_azimuthal_order=(
                    self.surface_azimuthal_order
                ),
            ).solve()
        )

    def predict_structured(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> StructuredPortPrediction:
        return (
            self.solve(
                scene,
                frequency_hz,
            ).prediction
        )

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
        from .hybrid_field import (
            prepare_hybrid_reference_loss_field,
        )

        teacher = (
            DielectricCoupledMixedTeacher(
                scene,
                frequency_hz,
                self.config,
                surface_vertical_order=(
                    self.surface_vertical_order
                ),
                surface_azimuthal_order=(
                    self.surface_azimuthal_order
                ),
            )
        )
        result = teacher.solve()
        return (
            prepare_hybrid_reference_loss_field(
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
        )

    def predict(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> np.ndarray:
        return (
            self.predict_structured(
                scene,
                frequency_hz,
            ).impedance
        )
