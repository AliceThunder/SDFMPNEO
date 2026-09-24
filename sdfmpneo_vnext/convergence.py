from __future__ import annotations

from dataclasses import dataclass
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
