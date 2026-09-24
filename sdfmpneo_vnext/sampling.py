from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import RigidPose, SuperellipseSpiral, haar_rotation
from .scene import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    Scene,
)


@dataclass(frozen=True)
class MVPSceneSamplerConfig:
    outer_radius_range: tuple[float, float] = (0.02, 0.05)
    aspect_ratio_range: tuple[float, float] = (0.7, 1.3)
    turns_range: tuple[float, float] = (0.6, 1.6)
    pitch_range: tuple[float, float] = (8e-4, 3e-3)
    exponent_range: tuple[float, float] = (2.0, 5.0)
    width_range: tuple[float, float] = (5e-4, 2.5e-3)
    thickness_range: tuple[float, float] = (3e-4, 1.5e-3)
    separation_range: tuple[float, float] = (0.012, 0.06)
    conductivity_range: tuple[float, float] = (3.0e7, 6.0e7)
    frequency_range: tuple[float, float] = (20_000.0, 200_000.0)

    def __post_init__(self):
        for name in (
            "outer_radius_range",
            "aspect_ratio_range",
            "turns_range",
            "pitch_range",
            "exponent_range",
            "width_range",
            "thickness_range",
            "separation_range",
            "conductivity_range",
            "frequency_range",
        ):
            lo, hi = getattr(self, name)
            if not (
                np.isfinite(lo)
                and np.isfinite(hi)
                and 0.0 < lo < hi
            ):
                raise ValueError(
                    f"{name} must be a positive finite increasing pair"
                )


def _uniform(rng, bounds):
    return float(
        rng.uniform(bounds[0], bounds[1])
    )


def _log_uniform(rng, bounds):
    return float(
        np.exp(
            rng.uniform(
                np.log(bounds[0]),
                np.log(bounds[1]),
            )
        )
    )


def _sample_coil(
    rng: np.random.Generator,
    config: MVPSceneSamplerConfig,
    *,
    pose: RigidPose,
    name: str,
):
    radius = _uniform(
        rng,
        config.outer_radius_range,
    )
    aspect = _uniform(
        rng,
        config.aspect_ratio_range,
    )
    outer_a = radius * np.sqrt(aspect)
    outer_b = radius / np.sqrt(aspect)
    turns = _uniform(
        rng,
        config.turns_range,
    )
    width = _log_uniform(
        rng,
        config.width_range,
    )
    thickness = _log_uniform(
        rng,
        config.thickness_range,
    )
    max_pitch = min(
        config.pitch_range[1],
        0.55
        * min(outer_a, outer_b)
        / max(turns, 1e-6),
    )
    min_pitch = min(
        config.pitch_range[0],
        0.8 * max_pitch,
    )
    pitch = float(
        rng.uniform(
            min_pitch,
            max_pitch,
        )
    )
    exponent = _uniform(
        rng,
        config.exponent_range,
    )
    conductivity = _log_uniform(
        rng,
        config.conductivity_range,
    )
    geometry = SuperellipseSpiral(
        outer_a,
        outer_b,
        turns,
        pitch,
        pitch,
        exponent=exponent,
        conductor_width=width,
        conductor_thickness=thickness,
        cross_section_exponent=exponent,
        pose=pose,
    )
    material = ConductorMaterial(
        conductivity,
        resistance_temperature_coefficient=0.0035,
    )
    return CoilObject(
        geometry,
        material,
        name,
    )


def sample_two_coil_mvp_scene(
    rng: np.random.Generator,
    config: MVPSceneSamplerConfig | None = None,
):
    config = config or MVPSceneSamplerConfig()
    root = _sample_coil(
        rng,
        config,
        pose=RigidPose.identity(),
        name="tx",
    )
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    separation = _log_uniform(
        rng,
        config.separation_range,
    )
    relative_pose = RigidPose(
        haar_rotation(rng),
        separation * direction,
    )
    second = _sample_coil(
        rng,
        config,
        pose=relative_pose,
        name="rx",
    )
    scene = Scene(
        (root, second),
        HomogeneousMedium(),
    )
    frequency = _log_uniform(
        rng,
        config.frequency_range,
    )
    return scene, frequency
