from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .em import (
    DenseMQSTeacher,
    MQSConfig,
)
from .mixed import DenseMixedConductorTeacher


@dataclass(frozen=True)
class ConvergenceStep:
    config: MQSConfig
    impedance: np.ndarray
    relative_change: float | None


@dataclass(frozen=True)
class ConvergenceReport:
    steps: tuple[
        ConvergenceStep,
        ...,
    ]
    converged: bool
    tolerance: float


def impedance_convergence(
    scene,
    frequency_hz: float,
    configs,
    tolerance: float = 1e-3,
) -> ConvergenceReport:
    configs = tuple(configs)
    if len(configs) < 2:
        raise ValueError(
            "at least two configurations are required"
        )
    if tolerance <= 0:
        raise ValueError(
            "tolerance must be positive"
        )
    steps = []
    prev = None
    for cfg in configs:
        Z = DenseMQSTeacher(
            scene,
            frequency_hz,
            cfg,
        ).solve().impedance
        if prev is None:
            change = None
        else:
            change = float(
                np.linalg.norm(
                    Z - prev
                )
                / max(
                    np.linalg.norm(Z),
                    1e-30,
                )
            )
        steps.append(
            ConvergenceStep(
                cfg,
                Z,
                change,
            )
        )
        prev = Z
    return ConvergenceReport(
        tuple(steps),
        bool(
            steps[-1].relative_change
            <= tolerance
        ),
        tolerance,
    )



def mixed_impedance_convergence(
    scene,
    frequency_hz: float,
    configs,
    tolerance: float = 1e-3,
) -> ConvergenceReport:
    """Convergence report for the canonical current-potential-charge teacher."""
    configs = tuple(
        configs
    )
    if len(configs) < 2:
        raise ValueError(
            "at least two configurations are required"
        )
    if tolerance <= 0:
        raise ValueError(
            "tolerance must be positive"
        )
    steps = []
    previous = None
    for config in configs:
        impedance = (
            DenseMixedConductorTeacher(
                scene,
                frequency_hz,
                config,
            ).solve().impedance
        )
        if previous is None:
            change = None
        else:
            change = float(
                np.linalg.norm(
                    impedance
                    - previous
                )
                / max(
                    np.linalg.norm(
                        impedance
                    ),
                    1e-30,
                )
            )
        steps.append(
            ConvergenceStep(
                config,
                impedance,
                change,
            )
        )
        previous = impedance
    return ConvergenceReport(
        tuple(
            steps
        ),
        bool(
            steps[
                -1
            ].relative_change
            <= tolerance
        ),
        tolerance,
    )



@dataclass(frozen=True)
class ReferenceConvergenceDirection:
    name: str
    config: MQSConfig
    impedance_relative_change: float
    channel_relative_change: float
    local_loss_relative_change: float
    maximum_relative_change: float


@dataclass(frozen=True)
class ReferenceConvergenceReport:
    base_config: MQSConfig
    directions: tuple[
        ReferenceConvergenceDirection,
        ...,
    ]
    tolerance: float
    converged: bool

    @property
    def maximum_relative_change(
        self,
    ) -> float:
        return float(
            max(
                direction.maximum_relative_change
                for direction
                in self.directions
            )
        )


def _relative_observable_change(
    refined,
    base,
) -> float:
    refined = np.asarray(
        refined
    )
    base = np.asarray(
        base
    )
    return float(
        np.linalg.norm(
            refined
            - base
        )
        / max(
            np.linalg.norm(
                refined
            ),
            1e-30,
        )
    )


def _mixed_local_loss_probes(
    teacher,
    result,
):
    values = []
    for coil_index, coil in enumerate(
        teacher.scene.coils
    ):
        geometry = (
            coil.geometry
        )
        points = (
            (
                0.0,
                0.0,
            ),
            (
                0.35
                * geometry.conductor_width,
                0.0,
            ),
            (
                0.0,
                0.35
                * geometry.conductor_thickness,
            ),
        )
        for arc_fraction in (
            0.2,
            0.5,
            0.8,
        ):
            for xy in points:
                values.append(
                    teacher.local_dissipation_matrix(
                        result,
                        coil_index,
                        arc_fraction,
                        xy,
                    )
                )
    return np.asarray(
        values,
        dtype=complex,
    )


def _mixed_reference_observables(
    scene,
    frequency_hz: float,
    config: MQSConfig,
):
    teacher = (
        DenseMixedConductorTeacher(
            scene,
            frequency_hz,
            config,
        )
    )
    result = (
        teacher.solve()
    )
    return (
        result.impedance,
        result.coil_dissipation_matrices(),
        _mixed_local_loss_probes(
            teacher,
            result,
        ),
    )


def _longitudinal_refinement(
    config: MQSConfig,
) -> MQSConfig:
    return replace(
        config,
        segments_per_turn=max(
            config.segments_per_turn
            + 1,
            int(
                np.ceil(
                    1.5
                    * config.segments_per_turn
                )
            ),
        ),
        min_segments=max(
            config.min_segments
            + 1,
            int(
                np.ceil(
                    1.5
                    * config.min_segments
                )
            ),
        ),
    )


def _cross_section_refinement(
    config: MQSConfig,
) -> MQSConfig:
    options = {
        "section_degree": (
            config.section_degree
            + 1
        ),
    }
    if (
        config.section_basis_family
        == "adaptive"
    ):
        options[
            "skin_boundary_layers"
        ] = (
            config.skin_boundary_layers
            + 1
        )
        options[
            "skin_angular_order"
        ] = (
            config.skin_angular_order
            + 1
        )
    return replace(
        config,
        **options,
    )


def _quadrature_refinement(
    config: MQSConfig,
) -> MQSConfig:
    return replace(
        config,
        radial_order=(
            config.radial_order
            + 2
        ),
        angular_order=max(
            config.angular_order
            + 8,
            int(
                np.ceil(
                    1.5
                    * config.angular_order
                )
            ),
        ),
        line_order=(
            config.line_order
            + 1
        ),
    )


def mixed_reference_convergence(
    scene,
    frequency_hz: float,
    base_config: MQSConfig,
    *,
    tolerance: float = 1e-3,
    longitudinal_config: MQSConfig | None = None,
    cross_section_config: MQSConfig | None = None,
    quadrature_config: MQSConfig | None = None,
) -> ReferenceConvergenceReport:
    """Independent mixed-reference refinement in three discretization axes."""
    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be positive"
        )
    if not isinstance(
        base_config,
        MQSConfig,
    ):
        raise TypeError(
            "base_config must be MQSConfig"
        )

    base_impedance, base_channels, base_local = (
        _mixed_reference_observables(
            scene,
            frequency_hz,
            base_config,
        )
    )
    refinements = (
        (
            "longitudinal",
            (
                longitudinal_config
                if longitudinal_config
                is not None
                else _longitudinal_refinement(
                    base_config
                )
            ),
        ),
        (
            "cross_section",
            (
                cross_section_config
                if cross_section_config
                is not None
                else _cross_section_refinement(
                    base_config
                )
            ),
        ),
        (
            "quadrature",
            (
                quadrature_config
                if quadrature_config
                is not None
                else _quadrature_refinement(
                    base_config
                )
            ),
        ),
    )

    directions = []
    for name, config in refinements:
        (
            impedance,
            channels,
            local,
        ) = _mixed_reference_observables(
            scene,
            frequency_hz,
            config,
        )
        impedance_error = (
            _relative_observable_change(
                impedance,
                base_impedance,
            )
        )
        channel_error = (
            _relative_observable_change(
                channels,
                base_channels,
            )
        )
        local_error = (
            _relative_observable_change(
                local,
                base_local,
            )
        )
        maximum = float(
            max(
                impedance_error,
                channel_error,
                local_error,
            )
        )
        directions.append(
            ReferenceConvergenceDirection(
                name,
                config,
                impedance_error,
                channel_error,
                local_error,
                maximum,
            )
        )

    directions = tuple(
        directions
    )
    maximum = max(
        direction.maximum_relative_change
        for direction
        in directions
    )
    return ReferenceConvergenceReport(
        base_config=base_config,
        directions=directions,
        tolerance=float(
            tolerance
        ),
        converged=bool(
            maximum
            <= tolerance
        ),
    )
