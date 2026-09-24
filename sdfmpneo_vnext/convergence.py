from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .em import (
    DenseMQSTeacher,
    MQSConfig,
)


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
