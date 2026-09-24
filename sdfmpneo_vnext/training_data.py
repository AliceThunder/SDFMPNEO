from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .analytic_baseline import analytic_port_baseline
from .em import DenseMQSTeacher, MQSConfig
from .features import EncodedScene, encode_scene_invariant
from .scene import Scene


@dataclass(frozen=True)
class TeacherSample:
    scene: Scene
    frequency_hz: float
    encoded: EncodedScene
    baseline_resistance: np.ndarray
    baseline_reactance: np.ndarray
    target_impedance: np.ndarray
    baseline_segments: int
    target_dissipation_channels: np.ndarray | None = None

    @staticmethod
    def generate(
        scene: Scene,
        frequency_hz: float,
        *,
        teacher_config: MQSConfig | None = None,
        baseline_segments: int = 96,
    ) -> "TeacherSample":
        encoded = encode_scene_invariant(
            scene,
            frequency_hz,
        )
        baseline = analytic_port_baseline(
            scene,
            frequency_hz,
            segments_per_coil=baseline_segments,
        )
        truth_result = DenseMQSTeacher(
            scene,
            frequency_hz,
            teacher_config or MQSConfig(),
        ).solve()
        truth = truth_result.impedance
        channels = (
            truth_result.coil_dissipation_matrices()
        )
        return TeacherSample(
            scene,
            float(frequency_hz),
            encoded,
            baseline.resistance,
            baseline.inductance
            * (
                2.0
                * np.pi
                * float(frequency_hz)
            ),
            truth,
            int(baseline_segments),
            channels,
        )
