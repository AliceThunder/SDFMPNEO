from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .basis import superellipse_section_quadrature
from .em import DenseMQSTeacher, MQSConfig
from .mixed import DenseMixedConductorTeacher
from .training_data import CANONICAL_REFERENCE_BACKEND


@dataclass(frozen=True)
class SpatialSurrogateAudit:
    samples: int
    mean_weighted_relative_error: float
    maximum_weighted_relative_error: float
    maximum_channel_closure_error: float
    maximum_probe_joule_error: float
    minimum_local_eigenvalue: float
    mean_offgrid_relative_error: float
    maximum_offgrid_relative_error: float
    maximum_offgrid_probe_joule_error: float
    maximum_cross_grid_closure_error: float
    passed: bool

    def to_dict(self):
        return {
            "samples": self.samples,
            "mean_weighted_relative_error": self.mean_weighted_relative_error,
            "maximum_weighted_relative_error": self.maximum_weighted_relative_error,
            "maximum_channel_closure_error": self.maximum_channel_closure_error,
            "maximum_probe_joule_error": self.maximum_probe_joule_error,
            "minimum_local_eigenvalue": self.minimum_local_eigenvalue,
            "mean_offgrid_relative_error": self.mean_offgrid_relative_error,
            "maximum_offgrid_relative_error": self.maximum_offgrid_relative_error,
            "maximum_offgrid_probe_joule_error": self.maximum_offgrid_probe_joule_error,
            "maximum_cross_grid_closure_error": self.maximum_cross_grid_closure_error,
            "passed": self.passed,
        }


def _prepare_spatial(
    spatial_artifact,
    scene,
    frequency_hz: float,
):
    if hasattr(
        spatial_artifact,
        "prepare",
    ):
        return spatial_artifact.prepare(
            scene,
            frequency_hz,
        )
    if hasattr(
        spatial_artifact,
        "prepare_spatial",
    ):
        return spatial_artifact.prepare_spatial(
            scene,
            frequency_hz,
        )
    raise TypeError(
        "spatial artifact must expose prepare or prepare_spatial"
    )


def _weighted_relative_error(
    predicted,
    target,
    weights,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=complex,
    )
    target = np.asarray(
        target,
        dtype=complex,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )
    numerator = float(
        np.sum(
            weights[:, None, None]
            * np.abs(
                predicted - target
            ) ** 2
        )
    )
    denominator = max(
        float(
            np.sum(
                weights[:, None, None]
                * np.abs(target) ** 2
            )
        ),
        1e-30,
    )
    return float(
        np.sqrt(
            numerator / denominator
        )
    )


def _probe_currents(
    n_ports: int,
):
    probes = [
        np.eye(
            n_ports,
            dtype=complex,
        )[:, port]
        for port in range(
            n_ports
        )
    ]
    probes.extend(
        [
            np.ones(
                n_ports,
                dtype=complex,
            ),
            (
                np.arange(
                    1,
                    n_ports + 1,
                    dtype=float,
                )
                + 0.3j
            ),
        ]
    )
    return tuple(
        probes
    )


def _probe_joule_error(
    predicted,
    target,
    weights,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=complex,
    )
    target = np.asarray(
        target,
        dtype=complex,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )
    maximum = 0.0
    for current in _probe_currents(
        int(
            target.shape[-1]
        )
    ):
        predicted_joule = (
            0.5
            * np.real(
                np.einsum(
                    "i,qij,j->q",
                    current.conj(),
                    predicted,
                    current,
                )
            )
        )
        target_joule = (
            0.5
            * np.real(
                np.einsum(
                    "i,qij,j->q",
                    current.conj(),
                    target,
                    current,
                )
            )
        )
        numerator = float(
            np.sum(
                weights
                * (
                    predicted_joule
                    - target_joule
                ) ** 2
            )
        )
        denominator = max(
            float(
                np.sum(
                    weights
                    * target_joule**2
                )
            ),
            1e-30,
        )
        maximum = max(
            maximum,
            float(
                np.sqrt(
                    numerator
                    / denominator
                )
            ),
        )
    return maximum


def _minimum_eigenvalue(
    matrices,
) -> float:
    matrices = np.asarray(
        matrices,
        dtype=complex,
    )
    hermitian = 0.5 * (
        matrices
        + matrices.conj().transpose(
            0,
            2,
            1,
        )
    )
    return float(
        np.min(
            np.linalg.eigvalsh(
                hermitian
            )
        )
    )


def _reference_teacher(
    sample,
):
    config = (
        sample.teacher_config
        if getattr(
            sample,
            "teacher_config",
            None,
        )
        is not None
        else MQSConfig()
    )
    backend = str(
        getattr(
            sample,
            "reference_backend",
            None,
        )
        or CANONICAL_REFERENCE_BACKEND
    ).lower()
    if backend == "mqs":
        teacher = DenseMQSTeacher(
            sample.scene,
            sample.frequency_hz,
            config,
        )
    elif backend == "mixed":
        teacher = (
            DenseMixedConductorTeacher(
                sample.scene,
                sample.frequency_hz,
                config,
            )
        )
    else:
        raise ValueError(
            "unsupported spatial reference backend"
        )
    return (
        teacher,
        teacher.solve(),
        config,
    )


def _offgrid_truth(
    sample,
):
    """Evaluate the same REFERENCE operator on a grid not stored in the dataset."""
    teacher, result, config = (
        _reference_teacher(
            sample
        )
    )
    base_teacher = getattr(
        teacher,
        "_mqs",
        teacher,
    )
    segments = (
        base_teacher._segments
    )
    coil_segments = {}
    for index, segment in enumerate(
        segments
    ):
        coil_segments.setdefault(
            int(
                segment.coil
            ),
            [],
        ).append(
            (
                index,
                segment,
            )
        )

    coil_index = []
    arc_fraction = []
    xy = []
    weights = []
    truth = []

    radial_order = max(
        2,
        int(
            config.radial_order
        )
        + 1,
    )
    angular_order = max(
        8,
        int(
            config.angular_order
        )
        + 5,
    )

    for coil, indexed_segments in (
        coil_segments.items()
    ):
        geometry = (
            sample.scene.coils[
                coil
            ].geometry
        )
        section = (
            superellipse_section_quadrature(
                geometry.conductor_width,
                geometry.conductor_thickness,
                geometry.cross_section_exponent,
                radial_order=(
                    radial_order
                ),
                angular_order=(
                    angular_order
                ),
            )
        )
        count = len(
            indexed_segments
        )
        for local_index, (
            _,
            segment,
        ) in enumerate(
            indexed_segments
        ):
            # Deliberately offset from the midpoint locations persisted in the
            # teacher dataset. The current modal teacher is piecewise
            # longitudinal, so this probes the same physical segment at an
            # unseen continuous-coordinate location.
            arc = (
                local_index
                + 0.37
            ) / count
            for local_xy, weight in zip(
                section.xy,
                section.weights,
            ):
                coil_index.append(
                    coil
                )
                arc_fraction.append(
                    arc
                )
                xy.append(
                    local_xy
                )
                weights.append(
                    float(
                        weight
                        * segment.length
                    )
                )
                truth.append(
                    teacher.local_dissipation_matrix(
                        result,
                        coil,
                        arc,
                        local_xy,
                    )
                )

    return (
        np.asarray(
            coil_index,
            dtype=int,
        ),
        np.asarray(
            arc_fraction,
            dtype=float,
        ),
        np.asarray(
            xy,
            dtype=float,
        ),
        np.asarray(
            weights,
            dtype=float,
        ),
        np.asarray(
            truth,
            dtype=complex,
        ),
    )


def _cross_grid_closure(
    prepared,
    coil_index,
    arc_fraction,
    xy,
    weights,
) -> float:
    predicted = np.asarray(
        prepared.local_dissipation_matrices(
            coil_index,
            arc_fraction,
            xy,
        ),
        dtype=complex,
    )
    channels = np.asarray(
        prepared.port_prediction.dissipation_channels,
        dtype=complex,
    )
    integrated = np.zeros_like(
        channels
    )
    for index, coil in enumerate(
        coil_index
    ):
        integrated[
            int(coil)
        ] += (
            weights[index]
            * predicted[index]
        )
    return float(
        np.linalg.norm(
            integrated
            - channels
        )
        / max(
            np.linalg.norm(
                channels
            ),
            1e-30,
        )
    )


def audit_spatial_surrogate(
    spatial_artifact,
    samples,
    *,
    mean_relative_error_limit: float = 0.10,
    maximum_relative_error_limit: float = 0.20,
    channel_closure_tolerance: float = 1e-5,
    maximum_probe_joule_error_limit: float = 0.20,
    mean_offgrid_relative_error_limit: float = 0.12,
    maximum_offgrid_relative_error_limit: float = 0.25,
    maximum_offgrid_probe_joule_error_limit: float = 0.25,
    cross_grid_closure_tolerance: float = 0.02,
    passivity_tolerance: float = 1e-10,
) -> SpatialSurrogateAudit:
    samples = tuple(
        samples
    )
    if not samples:
        raise ValueError(
            "spatial audit requires at least one sample"
        )
    positive_limits = (
        mean_relative_error_limit,
        maximum_relative_error_limit,
        maximum_probe_joule_error_limit,
        mean_offgrid_relative_error_limit,
        maximum_offgrid_relative_error_limit,
        maximum_offgrid_probe_joule_error_limit,
    )
    if (
        any(
            value <= 0.0
            for value
            in positive_limits
        )
        or channel_closure_tolerance
        < 0.0
        or cross_grid_closure_tolerance
        < 0.0
        or passivity_tolerance
        < 0.0
    ):
        raise ValueError(
            "invalid spatial audit tolerances"
        )

    errors = []
    offgrid_errors = []
    closure_errors = []
    cross_grid_errors = []
    maximum_probe_error = 0.0
    maximum_offgrid_probe_error = 0.0
    minimum_eigenvalue = np.inf

    for sample in samples:
        spatial = (
            sample.spatial_loss
        )
        if (
            spatial is None
            or sample.target_dissipation_channels
            is None
        ):
            raise ValueError(
                "spatial audit sample is missing physical field truth"
            )

        prepared = _prepare_spatial(
            spatial_artifact,
            sample.scene,
            sample.frequency_hz,
        )
        predicted = np.asarray(
            prepared.local_dissipation_matrices(
                spatial.coil_index,
                spatial.arc_fraction,
                spatial.xy,
            ),
            dtype=complex,
        )
        target = np.asarray(
            spatial.dissipation_matrix,
            dtype=complex,
        )
        weights = np.asarray(
            spatial.weights,
            dtype=float,
        )

        errors.append(
            _weighted_relative_error(
                predicted,
                target,
                weights,
            )
        )
        maximum_probe_error = max(
            maximum_probe_error,
            _probe_joule_error(
                predicted,
                target,
                weights,
            ),
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue,
            _minimum_eigenvalue(
                predicted
            ),
        )
        closure_errors.append(
            float(
                prepared.normalization_closure_error
            )
        )

        (
            offgrid_coil,
            offgrid_arc,
            offgrid_xy,
            offgrid_weights,
            offgrid_target,
        ) = _offgrid_truth(
            sample
        )
        offgrid_predicted = np.asarray(
            prepared.local_dissipation_matrices(
                offgrid_coil,
                offgrid_arc,
                offgrid_xy,
            ),
            dtype=complex,
        )
        offgrid_errors.append(
            _weighted_relative_error(
                offgrid_predicted,
                offgrid_target,
                offgrid_weights,
            )
        )
        maximum_offgrid_probe_error = max(
            maximum_offgrid_probe_error,
            _probe_joule_error(
                offgrid_predicted,
                offgrid_target,
                offgrid_weights,
            ),
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue,
            _minimum_eigenvalue(
                offgrid_predicted
            ),
        )
        cross_grid_errors.append(
            _cross_grid_closure(
                prepared,
                offgrid_coil,
                offgrid_arc,
                offgrid_xy,
                offgrid_weights,
            )
        )

    errors = np.asarray(
        errors,
        dtype=float,
    )
    offgrid_errors = np.asarray(
        offgrid_errors,
        dtype=float,
    )
    maximum_closure = float(
        np.max(
            closure_errors
        )
    )
    maximum_cross_grid = float(
        np.max(
            cross_grid_errors
        )
    )

    passed = bool(
        float(
            np.mean(
                errors
            )
        )
        <= mean_relative_error_limit
        and float(
            np.max(
                errors
            )
        )
        <= maximum_relative_error_limit
        and maximum_closure
        <= channel_closure_tolerance
        and maximum_probe_error
        <= maximum_probe_joule_error_limit
        and float(
            np.mean(
                offgrid_errors
            )
        )
        <= mean_offgrid_relative_error_limit
        and float(
            np.max(
                offgrid_errors
            )
        )
        <= maximum_offgrid_relative_error_limit
        and maximum_offgrid_probe_error
        <= maximum_offgrid_probe_joule_error_limit
        and maximum_cross_grid
        <= cross_grid_closure_tolerance
        and minimum_eigenvalue
        >= -passivity_tolerance
    )

    return SpatialSurrogateAudit(
        samples=len(
            samples
        ),
        mean_weighted_relative_error=float(
            np.mean(
                errors
            )
        ),
        maximum_weighted_relative_error=float(
            np.max(
                errors
            )
        ),
        maximum_channel_closure_error=(
            maximum_closure
        ),
        maximum_probe_joule_error=float(
            maximum_probe_error
        ),
        minimum_local_eigenvalue=float(
            minimum_eigenvalue
        ),
        mean_offgrid_relative_error=float(
            np.mean(
                offgrid_errors
            )
        ),
        maximum_offgrid_relative_error=float(
            np.max(
                offgrid_errors
            )
        ),
        maximum_offgrid_probe_joule_error=float(
            maximum_offgrid_probe_error
        ),
        maximum_cross_grid_closure_error=(
            maximum_cross_grid
        ),
        passed=passed,
    )
