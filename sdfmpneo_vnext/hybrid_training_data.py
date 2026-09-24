from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .analytic_baseline import analytic_port_baseline
from .em import MQSConfig
from .hybrid_dielectric import (
    DielectricCoupledMixedTeacher,
)
from .hybrid_features import (
    EncodedHybridScene,
    encode_hybrid_scene_invariant,
)
from .scene import Scene


HYBRID_REFERENCE_BACKEND = (
    "dielectric_mixed_sie"
)


@dataclass(frozen=True)
class HybridTeacherSample:
    scene: Scene
    frequency_hz: float
    encoded: EncodedHybridScene
    baseline_resistance: np.ndarray
    baseline_reactance: np.ndarray
    target_impedance: np.ndarray
    target_dissipation_channels: np.ndarray
    baseline_segments: int
    surface_vertical_order: int
    surface_azimuthal_order: int
    surface_residual: float
    raw_potential_reciprocity_defect: float
    power_closure_error: float
    reference_backend: str = (
        HYBRID_REFERENCE_BACKEND
    )

    @staticmethod
    def generate(
        scene: Scene,
        frequency_hz: float,
        *,
        teacher_config: MQSConfig | None = None,
        baseline_segments: int = 96,
        surface_vertical_order: int = 16,
        surface_azimuthal_order: int = 32,
    ) -> "HybridTeacherSample":
        if not scene.packages:
            raise ValueError(
                "hybrid teacher samples require at least one package"
            )
        encoded = (
            encode_hybrid_scene_invariant(
                scene,
                frequency_hz,
            )
        )
        conductor_scene = Scene(
            scene.coils,
            scene.medium,
            (),
        )
        baseline = (
            analytic_port_baseline(
                conductor_scene,
                frequency_hz,
                segments_per_coil=(
                    baseline_segments
                ),
            )
        )
        result = (
            DielectricCoupledMixedTeacher(
                scene,
                frequency_hz,
                teacher_config
                or MQSConfig(),
                surface_vertical_order=(
                    surface_vertical_order
                ),
                surface_azimuthal_order=(
                    surface_azimuthal_order
                ),
            ).solve()
        )
        return HybridTeacherSample(
            scene=scene,
            frequency_hz=float(
                frequency_hz
            ),
            encoded=encoded,
            baseline_resistance=(
                baseline.resistance
            ),
            baseline_reactance=(
                2.0
                * np.pi
                * float(
                    frequency_hz
                )
                * baseline.inductance
            ),
            target_impedance=(
                result.impedance
            ),
            target_dissipation_channels=(
                result.prediction.dissipation_channels
            ),
            baseline_segments=int(
                baseline_segments
            ),
            surface_vertical_order=int(
                surface_vertical_order
            ),
            surface_azimuthal_order=int(
                surface_azimuthal_order
            ),
            surface_residual=float(
                result.surface_residual
            ),
            raw_potential_reciprocity_defect=float(
                result.raw_potential_reciprocity_defect
            ),
            power_closure_error=float(
                result.power_closure_error
            ),
        )
