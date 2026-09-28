from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .geometry import RigidPose, SuperellipseSpiral, haar_rotation
from .hybrid_domain import (
    package_domain_topology,
    scene_characteristic_center_scale,
)
from .package_geometry import SuperquadricPackageGeometry
from .scene import (
    CoilObject,
    ConductorMaterial,
    DebyeMaterial,
    EPS0,
    HomogeneousMedium,
    IsotropicMaterial,
    MultiDebyeMaterial,
    PackageObject,
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



@dataclass(frozen=True)
class HybridSceneSamplerConfig:
    conductor: MVPSceneSamplerConfig = MVPSceneSamplerConfig()
    package_margin_range: tuple[float, float] = (1.35, 2.0)
    package_half_z_range: tuple[float, float] = (0.004, 0.012)
    package_center_offset_fraction_range: tuple[float, float] = (0.0, 0.35)
    package_exponent_xy_range: tuple[float, float] = (2.0, 5.0)
    package_exponent_z_range: tuple[float, float] = (2.0, 5.0)
    package_count_range: tuple[int, int] = (1, 1)
    nested_package_probability: float = 0.0
    nested_package_scale_range: tuple[float, float] = (1.15, 1.45)
    free_inclusion_probability: float = 0.0
    free_inclusion_center_radius_fraction_range: tuple[float, float] = (0.65, 1.8)
    free_inclusion_half_extent_fraction_range: tuple[float, float] = (0.12, 0.45)
    dc_probability: float = 0.0
    dc_conductive_probability: float = 0.0
    relative_permittivity_range: tuple[float, float] = (1.5, 6.0)
    package_relative_permeability_range: tuple[float, float] = (1.0, 1.0)
    dielectric_conductivity_range: tuple[float, float] = (1e-7, 5e-3)
    lossless_probability: float = 0.20
    debye_package_probability: float = 0.0
    multi_debye_package_probability: float = 0.0
    package_debye_epsilon_infinite_range: tuple[float, float] = (1.5, 6.0)
    package_debye_delta_epsilon_range: tuple[float, float] = (0.5, 20.0)
    package_debye_relaxation_time_range: tuple[float, float] = (1e-8, 1e-4)
    background_relative_permittivity_range: tuple[float, float] = (1.0, 1.0)
    background_conductivity_range: tuple[float, float] = (1e-7, 5e-3)
    lossy_background_probability: float = 0.0
    debye_background_probability: float = 0.0
    multi_debye_background_probability: float = 0.0
    multi_debye_poles_range: tuple[int, int] = (2, 4)
    background_debye_epsilon_infinite_range: tuple[float, float] = (1.0, 6.0)
    background_debye_delta_epsilon_range: tuple[float, float] = (0.5, 30.0)
    background_debye_relaxation_time_range: tuple[float, float] = (1e-8, 1e-4)

    def __post_init__(self):
        for name in (
            "package_margin_range",
            "package_half_z_range",
            "package_exponent_xy_range",
            "package_exponent_z_range",
            "nested_package_scale_range",
            "free_inclusion_center_radius_fraction_range",
            "free_inclusion_half_extent_fraction_range",
            "relative_permittivity_range",
            "dielectric_conductivity_range",
            "package_debye_epsilon_infinite_range",
            "package_debye_delta_epsilon_range",
            "package_debye_relaxation_time_range",
            "background_conductivity_range",
            "background_debye_epsilon_infinite_range",
            "background_debye_delta_epsilon_range",
            "background_debye_relaxation_time_range",
        ):
            lo, hi = getattr(
                self,
                name,
            )
            if not (
                np.isfinite(
                    lo
                )
                and np.isfinite(
                    hi
                )
                and 0.0
                < lo
                < hi
            ):
                raise ValueError(
                    f"{name} must be a positive finite increasing pair"
                )
        mu_lo, mu_hi = (
            self.package_relative_permeability_range
        )
        if not (
            np.isfinite(
                mu_lo
            )
            and np.isfinite(
                mu_hi
            )
            and 0.0
            < mu_lo
            <= mu_hi
        ):
            raise ValueError(
                "package_relative_permeability_range must be a positive "
                "finite nondecreasing pair"
            )
        offset_lo, offset_hi = (
            self.package_center_offset_fraction_range
        )
        if not (
            np.isfinite(
                offset_lo
            )
            and np.isfinite(
                offset_hi
            )
            and 0.0
            <= offset_lo
            < offset_hi
        ):
            raise ValueError(
                "package_center_offset_fraction_range must be a finite "
                "nonnegative increasing pair"
            )
        bg_eps_lo, bg_eps_hi = (
            self.background_relative_permittivity_range
        )
        if not (
            np.isfinite(
                bg_eps_lo
            )
            and np.isfinite(
                bg_eps_hi
            )
            and 0.0
            < bg_eps_lo
            <= bg_eps_hi
        ):
            raise ValueError(
                "background_relative_permittivity_range must be a positive "
                "finite nondecreasing pair"
            )
        if not (
            0.0
            <= self.lossless_probability
            <= 1.0
        ):
            raise ValueError(
                "lossless_probability must lie in [0,1]"
            )
        package_lo, package_hi = (
            self.package_count_range
        )
        if (
            not isinstance(
                package_lo,
                (int, np.integer),
            )
            or not isinstance(
                package_hi,
                (int, np.integer),
            )
            or package_lo < 1
            or package_hi < package_lo
        ):
            raise ValueError(
                "package_count_range must be an integer range >= 1"
            )
        if not (
            0.0
            <= self.nested_package_probability
            <= 1.0
        ):
            raise ValueError(
                "nested_package_probability must lie in [0,1]"
            )
        if not (
            0.0
            <= self.free_inclusion_probability
            <= 1.0
        ):
            raise ValueError(
                "free_inclusion_probability must lie in [0,1]"
            )
        if not (
            0.0
            <= self.dc_probability
            <= 1.0
        ):
            raise ValueError(
                "dc_probability must lie in [0,1]"
            )
        if not (
            0.0
            <= self.dc_conductive_probability
            <= 1.0
        ):
            raise ValueError(
                "dc_conductive_probability must lie in [0,1]"
            )
        if not (
            0.0
            <= self.debye_package_probability
            <= 1.0
        ):
            raise ValueError(
                "debye_package_probability must lie in [0,1]"
            )
        if not (
            0.0
            <= self.lossy_background_probability
            <= 1.0
        ):
            raise ValueError(
                "lossy_background_probability must lie in [0,1]"
            )
        if not (
            0.0
            <= self.debye_background_probability
            <= 1.0
        ):
            raise ValueError(
                "debye_background_probability must lie in [0,1]"
            )
        for name in (
            "multi_debye_package_probability",
            "multi_debye_background_probability",
        ):
            value = float(
                getattr(
                    self,
                    name,
                )
            )
            if not (
                0.0
                <= value
                <= 1.0
            ):
                raise ValueError(
                    f"{name} must lie in [0,1]"
                )
        if (
            self.debye_package_probability
            + self.multi_debye_package_probability
            > 1.0
        ):
            raise ValueError(
                "package Debye probabilities must sum to <= 1"
            )
        if (
            self.debye_background_probability
            + self.multi_debye_background_probability
            > 1.0
        ):
            raise ValueError(
                "background Debye probabilities must sum to <= 1"
            )
        pole_lo, pole_hi = (
            self.multi_debye_poles_range
        )
        if (
            not isinstance(
                pole_lo,
                (int, np.integer),
            )
            or not isinstance(
                pole_hi,
                (int, np.integer),
            )
            or pole_lo < 2
            or pole_hi < pole_lo
        ):
            raise ValueError(
                "multi_debye_poles_range must be an integer range >= 2"
            )


    def geometry_domain_metadata(
        self,
    ):
        conductor = asdict(
            self.conductor
        )
        conductor = {
            key: (
                list(
                    value
                )
                if isinstance(
                    value,
                    tuple,
                )
                else value
            )
            for key, value
            in conductor.items()
        }
        if self.dc_probability > 0.0:
            conductor[
                "frequency_range"
            ][
                0
            ] = 0.0
        return {
            "n_coils": 2,
            "n_packages_range": [
                int(
                    self.package_count_range[
                        0
                    ]
                ),
                int(
                    self.package_count_range[
                        1
                    ]
                ),
            ],
            "conductor": conductor,
            "frequency_sampling": {
                "dc_probability": float(
                    self.dc_probability
                ),
                "dc_conductive_probability": float(
                    self.dc_conductive_probability
                ),
            },
            "coil_relative_pose": {
                "translation_direction": (
                    "isotropic_s2"
                ),
                "rotation": (
                    "haar_so3"
                ),
            },
            "package": {
                "margin_range": list(
                    self.package_margin_range
                ),
                "minimum_half_z_range": list(
                    self.package_half_z_range
                ),
                "center_offset_fraction_range": list(
                    self.package_center_offset_fraction_range
                ),
                "exponent_xy_range": list(
                    self.package_exponent_xy_range
                ),
                "exponent_z_range": list(
                    self.package_exponent_z_range
                ),
                "relative_rotation": (
                    "haar_so3"
                ),
                "nested_package_probability": float(
                    self.nested_package_probability
                ),
                "nested_package_scale_range": list(
                    self.nested_package_scale_range
                ),
                "nested_topology": (
                    "strict_chain_or_disjoint_roots"
                ),
                "free_inclusion_probability": float(
                    self.free_inclusion_probability
                ),
                "free_inclusion_center_radius_fraction_range": list(
                    self.free_inclusion_center_radius_fraction_range
                ),
                "free_inclusion_half_extent_fraction_range": list(
                    self.free_inclusion_half_extent_fraction_range
                ),
                "free_inclusion_topology": "disjoint_root",
                "enclosure_target_radius": (
                    0.90
                ),
                "enclosure_radius_range": [
                    float(
                        (
                            1.0
                            / self.package_margin_range[
                                1
                            ]
                        )
                        / (
                            self.nested_package_scale_range[
                                1
                            ]
                            ** max(
                                int(
                                    self.package_count_range[
                                        1
                                    ]
                                )
                                - 1,
                                0,
                            )
                        )
                    ),
                    0.90,
                ],
            },
        }

    def package_domain_metadata(
        self,
    ):
        debye_probability = float(
            self.debye_package_probability
        )
        multi_debye_probability = float(
            self.multi_debye_package_probability
        )
        dispersive_probability = (
            debye_probability
            + multi_debye_probability
        )
        constant_probability = (
            1.0
            - dispersive_probability
        )
        if dispersive_probability <= 0.0:
            epsilon_lower, epsilon_upper = (
                self.relative_permittivity_range
            )
        elif constant_probability <= 0.0:
            epsilon_lower = float(
                self.package_debye_epsilon_infinite_range[
                    0
                ]
            )
            epsilon_upper = float(
                self.package_debye_epsilon_infinite_range[
                    1
                ]
                + self.package_debye_delta_epsilon_range[
                    1
                ]
            )
        else:
            epsilon_lower = min(
                float(
                    self.relative_permittivity_range[
                        0
                    ]
                ),
                float(
                    self.package_debye_epsilon_infinite_range[
                        0
                    ]
                ),
            )
            epsilon_upper = max(
                float(
                    self.relative_permittivity_range[
                        1
                    ]
                ),
                float(
                    self.package_debye_epsilon_infinite_range[
                        1
                    ]
                    + self.package_debye_delta_epsilon_range[
                        1
                    ]
                ),
            )

        ohmic_upper = (
            float(
                self.dielectric_conductivity_range[
                    1
                ]
            )
            if (
                self.lossless_probability
                < 1.0
                or (
                    self.dc_probability
                    > 0.0
                    and self.dc_conductive_probability
                    > 0.0
                )
            )
            else 0.0
        )
        if dispersive_probability > 0.0:
            omega_max = (
                2.0
                * np.pi
                * float(
                    self.conductor.frequency_range[
                        1
                    ]
                )
            )
            debye_upper = (
                0.5
                * omega_max
                * EPS0
                * float(
                    self.package_debye_delta_epsilon_range[
                        1
                    ]
                )
            )
        else:
            debye_upper = 0.0

        return {
            "relative_permittivity_range": [
                float(
                    self.relative_permittivity_range[
                        0
                    ]
                ),
                float(
                    self.relative_permittivity_range[
                        1
                    ]
                ),
            ],
            "relative_permeability_range": [
                float(
                    self.package_relative_permeability_range[
                        0
                    ]
                ),
                float(
                    self.package_relative_permeability_range[
                        1
                    ]
                ),
            ],
            "conductivity_range": [
                float(
                    self.dielectric_conductivity_range[
                        0
                    ]
                ),
                float(
                    self.dielectric_conductivity_range[
                        1
                    ]
                ),
            ],
            "ohmic_lossless_probability": float(
                self.lossless_probability
            ),
            "debye_probability": (
                debye_probability
            ),
            "multi_debye_probability": (
                multi_debye_probability
            ),
            "multi_debye_poles_range": [
                int(
                    self.multi_debye_poles_range[
                        0
                    ]
                ),
                int(
                    self.multi_debye_poles_range[
                        1
                    ]
                ),
            ],
            "debye_epsilon_infinite_range": [
                float(
                    self.package_debye_epsilon_infinite_range[
                        0
                    ]
                ),
                float(
                    self.package_debye_epsilon_infinite_range[
                        1
                    ]
                ),
            ],
            "debye_delta_epsilon_range": [
                float(
                    self.package_debye_delta_epsilon_range[
                        0
                    ]
                ),
                float(
                    self.package_debye_delta_epsilon_range[
                        1
                    ]
                ),
            ],
            "debye_relaxation_time_range": [
                float(
                    self.package_debye_relaxation_time_range[
                        0
                    ]
                ),
                float(
                    self.package_debye_relaxation_time_range[
                        1
                    ]
                ),
            ],
            "effective_relative_permittivity_range": [
                float(
                    epsilon_lower
                ),
                float(
                    epsilon_upper
                ),
            ],
            "effective_loss_conductivity_range": [
                0.0,
                float(
                    ohmic_upper
                    + debye_upper
                ),
            ],
        }

    def background_domain_metadata(
        self,
    ):
        """Declared raw and frequency-effective homogeneous background domain."""
        debye_probability = float(
            self.debye_background_probability
        )
        multi_debye_probability = float(
            self.multi_debye_background_probability
        )
        dispersive_probability = (
            debye_probability
            + multi_debye_probability
        )
        constant_probability = (
            1.0
            - dispersive_probability
        )

        if dispersive_probability <= 0.0:
            epsilon_lower, epsilon_upper = (
                self.background_relative_permittivity_range
            )
        elif constant_probability <= 0.0:
            epsilon_lower = float(
                self.background_debye_epsilon_infinite_range[
                    0
                ]
            )
            epsilon_upper = float(
                self.background_debye_epsilon_infinite_range[
                    1
                ]
                + self.background_debye_delta_epsilon_range[
                    1
                ]
            )
        else:
            epsilon_lower = min(
                float(
                    self.background_relative_permittivity_range[
                        0
                    ]
                ),
                float(
                    self.background_debye_epsilon_infinite_range[
                        0
                    ]
                ),
            )
            epsilon_upper = max(
                float(
                    self.background_relative_permittivity_range[
                        1
                    ]
                ),
                float(
                    self.background_debye_epsilon_infinite_range[
                        1
                    ]
                    + self.background_debye_delta_epsilon_range[
                        1
                    ]
                ),
            )

        ohmic_upper = (
            float(
                self.background_conductivity_range[
                    1
                ]
            )
            if (
                self.lossy_background_probability
                > 0.0
                or (
                    self.dc_probability
                    > 0.0
                    and self.dc_conductive_probability
                    > 0.0
                )
            )
            else 0.0
        )
        if dispersive_probability > 0.0:
            omega_max = (
                2.0
                * np.pi
                * float(
                    self.conductor.frequency_range[
                        1
                    ]
                )
            )
            # For x = omega*tau, x/(1+x^2) <= 1/2.
            debye_upper = (
                0.5
                * omega_max
                * EPS0
                * float(
                    self.background_debye_delta_epsilon_range[
                        1
                    ]
                )
            )
        else:
            debye_upper = 0.0

        effective_loss_upper = (
            ohmic_upper
            + debye_upper
        )
        return {
            "relative_permittivity_range": [
                float(
                    self.background_relative_permittivity_range[
                        0
                    ]
                ),
                float(
                    self.background_relative_permittivity_range[
                        1
                    ]
                ),
            ],
            "conductivity_range": [
                float(
                    self.background_conductivity_range[
                        0
                    ]
                ),
                float(
                    self.background_conductivity_range[
                        1
                    ]
                ),
            ],
            "lossy_probability": float(
                self.lossy_background_probability
            ),
            "debye_probability": (
                debye_probability
            ),
            "multi_debye_probability": (
                multi_debye_probability
            ),
            "multi_debye_poles_range": [
                int(
                    self.multi_debye_poles_range[
                        0
                    ]
                ),
                int(
                    self.multi_debye_poles_range[
                        1
                    ]
                ),
            ],
            "debye_epsilon_infinite_range": [
                float(
                    self.background_debye_epsilon_infinite_range[
                        0
                    ]
                ),
                float(
                    self.background_debye_epsilon_infinite_range[
                        1
                    ]
                ),
            ],
            "debye_delta_epsilon_range": [
                float(
                    self.background_debye_delta_epsilon_range[
                        0
                    ]
                ),
                float(
                    self.background_debye_delta_epsilon_range[
                        1
                    ]
                ),
            ],
            "debye_relaxation_time_range": [
                float(
                    self.background_debye_relaxation_time_range[
                        0
                    ]
                ),
                float(
                    self.background_debye_relaxation_time_range[
                        1
                    ]
                ),
            ],
            "effective_relative_permittivity_range": [
                float(
                    epsilon_lower
                ),
                float(
                    epsilon_upper
                ),
            ],
            "effective_loss_conductivity_range": [
                0.0,
                float(
                    effective_loss_upper
                ),
            ],
        }


def _conductor_surface_samples(
    geometry,
    *,
    longitudinal_segments: int = 96,
    section_points: int = 20,
):
    return geometry.surface_samples(
        longitudinal_segments=(
            longitudinal_segments
        ),
        section_points=(
            section_points
        ),
    )


def _build_enclosing_package_geometry_candidate(
    rng: np.random.Generator,
    root,
    config,
):
    exponent_xy = _uniform(
        rng,
        config.package_exponent_xy_range,
    )
    exponent_z = _uniform(
        rng,
        config.package_exponent_z_range,
    )
    relative_rotation = haar_rotation(
        rng
    )
    direction = np.asarray(
        rng.normal(
            size=3
        ),
        dtype=float,
    )
    direction /= max(
        float(
            np.linalg.norm(
                direction
            )
        ),
        1e-30,
    )
    offset_fraction = _uniform(
        rng,
        config.package_center_offset_fraction_range,
    )
    offset_scale = max(
        float(
            root.outer_a
        ),
        float(
            root.outer_b
        ),
    )
    relative_pose = RigidPose(
        relative_rotation,
        direction
        * offset_fraction
        * offset_scale,
    )
    package_pose = root.pose.compose(
        relative_pose
    )

    conductor_points = _conductor_surface_samples(
        root
    )
    local = (
        (
            conductor_points
            - package_pose.translation[
                None,
                :
            ]
        )
        @ package_pose.rotation
    )
    bounds = np.max(
        np.abs(
            local
        ),
        axis=0,
    )
    margin = _uniform(
        rng,
        config.package_margin_range,
    )
    half_extents = (
        margin
        * np.maximum(
            bounds,
            1e-6,
        )
    )
    half_extents[
        2
    ] = max(
        float(
            half_extents[
                2
            ]
        ),
        _uniform(
            rng,
            config.package_half_z_range,
        ),
    )

    scaled = (
        np.abs(
            local
        )
        / half_extents[
            None,
            :
        ]
    )
    radial = (
        (
            scaled[
                :,
                0
            ] ** exponent_xy
            + scaled[
                :,
                1
            ] ** exponent_xy
        ) ** (
            exponent_z
            / exponent_xy
        )
        + scaled[
            :,
            2
        ] ** exponent_z
    ) ** (
        1.0
        / exponent_z
    )
    maximum_radius = float(
        np.max(
            radial
        )
    )
    target_radius = 0.90
    if maximum_radius > target_radius:
        half_extents = (
            half_extents
            * maximum_radius
            / target_radius
        )

    geometry = SuperquadricPackageGeometry(
        half_extents,
        exponent_xy=(
            exponent_xy
        ),
        exponent_z=(
            exponent_z
        ),
        pose=package_pose,
    )
    if not np.all(
        geometry.contains(
            conductor_points,
            tolerance=1e-10,
        )
    ):
        raise RuntimeError(
            "failed to construct an enclosing arbitrary-pose package"
        )
    return geometry


def _sample_enclosing_package_geometry(
    rng: np.random.Generator,
    root,
    config,
    *,
    other_geometries=(),
):
    other_geometries = tuple(
        other_geometries
    )
    for _ in range(
        32
    ):
        geometry = (
            _build_enclosing_package_geometry_candidate(
                rng,
                root,
                config,
            )
        )
        valid = True
        for other in other_geometries:
            try:
                geometry.classify_conductor(
                    other,
                    longitudinal_segments=64,
                    section_points=16,
                )
            except ValueError:
                valid = False
                break
        if valid:
            return geometry
    raise RuntimeError(
        "failed to sample a package pose without conductor-surface intersection"
    )


def _sample_multi_debye_material(
    rng: np.random.Generator,
    *,
    epsilon_infinite_range,
    delta_epsilon_range,
    relaxation_time_range,
    pole_count_range,
    conductivity: float,
    relative_permeability: float = 1.0,
):
    pole_lo, pole_hi = (
        pole_count_range
    )
    n_poles = int(
        rng.integers(
            int(
                pole_lo
            ),
            int(
                pole_hi
            )
            + 1,
        )
    )
    epsilon_infinite = _uniform(
        rng,
        epsilon_infinite_range,
    )
    total_delta = _uniform(
        rng,
        delta_epsilon_range,
    )
    fractions = rng.dirichlet(
        np.ones(
            n_poles,
            dtype=float,
        )
    )
    strengths = (
        total_delta
        * fractions
    )
    times = np.asarray(
        [
            _log_uniform(
                rng,
                relaxation_time_range,
            )
            for _ in range(
                n_poles
            )
        ],
        dtype=float,
    )
    order = np.argsort(
        times
    )
    return MultiDebyeMaterial(
        relative_permittivity_infinite=(
            epsilon_infinite
        ),
        relaxation_strengths=tuple(
            float(
                strengths[
                    index
                ]
            )
            for index in order
        ),
        relaxation_times=tuple(
            float(
                times[
                    index
                ]
            )
            for index in order
        ),
        relative_permeability=float(
            relative_permeability
        ),
        conductivity=float(
            conductivity
        ),
    )


def _sample_package_material(
    rng: np.random.Generator,
    config: HybridSceneSamplerConfig,
    *,
    force_conductive: bool | None = None,
):
    if force_conductive is True:
        conductivity = _log_uniform(
            rng,
            config.dielectric_conductivity_range,
        )
    elif force_conductive is False:
        conductivity = 0.0
    elif (
        rng.random()
        < config.lossless_probability
    ):
        conductivity = 0.0
    else:
        conductivity = _log_uniform(
            rng,
            config.dielectric_conductivity_range,
        )
    relative_permeability = _uniform(
        rng,
        config.package_relative_permeability_range,
    )
    draw = float(
        rng.random()
    )
    if (
        draw
        < config.multi_debye_package_probability
    ):
        return _sample_multi_debye_material(
            rng,
            epsilon_infinite_range=(
                config.package_debye_epsilon_infinite_range
            ),
            delta_epsilon_range=(
                config.package_debye_delta_epsilon_range
            ),
            relaxation_time_range=(
                config.package_debye_relaxation_time_range
            ),
            pole_count_range=(
                config.multi_debye_poles_range
            ),
            conductivity=conductivity,
            relative_permeability=(
                relative_permeability
            ),
        )
    if (
        draw
        < (
            config.multi_debye_package_probability
            + config.debye_package_probability
        )
    ):
        epsilon_infinite = _uniform(
            rng,
            config.package_debye_epsilon_infinite_range,
        )
        delta_epsilon = _uniform(
            rng,
            config.package_debye_delta_epsilon_range,
        )
        return DebyeMaterial(
            relative_permittivity_static=(
                epsilon_infinite
                + delta_epsilon
            ),
            relative_permittivity_infinite=(
                epsilon_infinite
            ),
            relaxation_time=_log_uniform(
                rng,
                config.package_debye_relaxation_time_range,
            ),
            relative_permeability=(
                relative_permeability
            ),
            conductivity=conductivity,
        )
    return IsotropicMaterial(
        relative_permittivity=_uniform(
            rng,
            config.relative_permittivity_range,
        ),
        relative_permeability=(
            relative_permeability
        ),
        conductivity=conductivity,
    )


def _package_probe_objects(
    geometries,
):
    probe_material = IsotropicMaterial(
        relative_permittivity=1.0,
    )
    return tuple(
        PackageObject(
            geometry,
            probe_material,
            f"probe{index}",
        )
        for index, geometry in enumerate(
            geometries
        )
    )


def _are_disjoint_package_roots(
    geometries,
) -> bool:
    if len(
        geometries
    ) <= 1:
        return True
    try:
        topology = package_domain_topology(
            _package_probe_objects(
                geometries
            )
        )
    except ValueError:
        return False
    return all(
        parent is None
        for parent in topology.parent
    )


def _sample_free_inclusion_geometry(
    rng: np.random.Generator,
    base_scene: Scene,
    config: HybridSceneSamplerConfig,
    existing_geometries=(),
):
    center, scale = (
        scene_characteristic_center_scale(
            base_scene
        )
    )
    for _ in range(
        96
    ):
        direction = np.asarray(
            rng.normal(
                size=3
            ),
            dtype=float,
        )
        direction /= max(
            float(
                np.linalg.norm(
                    direction
                )
            ),
            1e-30,
        )
        center_fraction = _uniform(
            rng,
            config.free_inclusion_center_radius_fraction_range,
        )
        half_extent_fraction = np.asarray(
            [
                _uniform(
                    rng,
                    config.free_inclusion_half_extent_fraction_range,
                )
                for _axis in range(
                    3
                )
            ],
            dtype=float,
        )
        geometry = SuperquadricPackageGeometry(
            scale
            * half_extent_fraction,
            exponent_xy=_uniform(
                rng,
                config.package_exponent_xy_range,
            ),
            exponent_z=_uniform(
                rng,
                config.package_exponent_z_range,
            ),
            pose=RigidPose(
                haar_rotation(
                    rng
                ),
                center
                + scale
                * center_fraction
                * direction,
            ),
        )
        valid = True
        for coil in base_scene.coils:
            try:
                classification = (
                    geometry.classify_conductor(
                        coil.geometry,
                        longitudinal_segments=64,
                        section_points=16,
                        tolerance=1e-10,
                    )
                )
            except ValueError:
                valid = False
                break
            if classification != "outside":
                valid = False
                break
        if not valid:
            continue
        candidates = tuple(
            existing_geometries
        ) + (
            geometry,
        )
        if _are_disjoint_package_roots(
            candidates
        ):
            return geometry
    raise RuntimeError(
        "failed to sample a free material inclusion disjoint from conductors "
        "and existing package roots"
    )


def _sample_disjoint_package_geometries(
    rng: np.random.Generator,
    base_scene: Scene,
    config: HybridSceneSamplerConfig,
    count: int,
):
    geometries = []
    used_enclosures = set()
    for index in range(
        count
    ):
        force_free = (
            len(
                used_enclosures
            )
            >= len(
                base_scene.coils
            )
        )
        sample_free = (
            force_free
            or (
                rng.random()
                < config.free_inclusion_probability
            )
        )
        if sample_free:
            geometries.append(
                _sample_free_inclusion_geometry(
                    rng,
                    base_scene,
                    config,
                    existing_geometries=(
                        geometries
                    ),
                )
            )
            continue

        available = [
            coil_index
            for coil_index in range(
                len(
                    base_scene.coils
                )
            )
            if coil_index
            not in used_enclosures
        ]
        target_index = (
            available[
                0
            ]
            if len(
                available
            )
            == 1
            else int(
                rng.choice(
                    available
                )
            )
        )
        target = base_scene.coils[
            target_index
        ].geometry
        other_conductors = tuple(
            coil.geometry
            for coil_index, coil in enumerate(
                base_scene.coils
            )
            if coil_index
            != target_index
        )
        accepted = None
        for _ in range(
            64
        ):
            candidate = (
                _sample_enclosing_package_geometry(
                    rng,
                    target,
                    config,
                    other_geometries=(
                        other_conductors
                    ),
                )
            )
            if _are_disjoint_package_roots(
                tuple(
                    geometries
                )
                + (
                    candidate,
                )
            ):
                accepted = candidate
                break
        if accepted is None:
            # Falling back to a free inclusion preserves the declared
            # disjoint-root topology without silently switching to nesting.
            accepted = _sample_free_inclusion_geometry(
                rng,
                base_scene,
                config,
                existing_geometries=(
                    geometries
                ),
            )
        geometries.append(
            accepted
        )
        used_enclosures.add(
            target_index
        )
    return tuple(
        geometries
    )


def _nested_package_geometries(
    rng: np.random.Generator,
    base_scene: Scene,
    config: HybridSceneSamplerConfig,
    count: int,
):
    root = base_scene.coils[
        0
    ].geometry
    for _ in range(
        32
    ):
        inner = _sample_enclosing_package_geometry(
            rng,
            root,
            config,
            other_geometries=tuple(
                coil.geometry
                for coil in base_scene.coils[
                    1:
                ]
            ),
        )
        geometries = [
            inner
        ]
        valid = True
        for _layer in range(
            1,
            count,
        ):
            previous = geometries[
                -1
            ]
            scale = _uniform(
                rng,
                config.nested_package_scale_range,
            )
            outer = SuperquadricPackageGeometry(
                previous.half_extents
                * scale,
                exponent_xy=(
                    previous.exponent_xy
                ),
                exponent_z=(
                    previous.exponent_z
                ),
                pose=previous.pose,
            )
            for coil in base_scene.coils:
                try:
                    outer.classify_conductor(
                        coil.geometry,
                        longitudinal_segments=64,
                        section_points=16,
                        tolerance=1e-10,
                    )
                except ValueError:
                    valid = False
                    break
            if not valid:
                break
            geometries.append(
                outer
            )
        if valid:
            return tuple(
                geometries
            )
    raise RuntimeError(
        "failed to sample a valid nested package chain"
    )


def _disjoint_two_package_geometries(
    rng: np.random.Generator,
    base_scene: Scene,
    config: HybridSceneSamplerConfig,
):
    if len(
        base_scene.coils
    ) < 2:
        raise RuntimeError(
            "disjoint two-package sampling requires two conductors"
        )
    for _ in range(
        48
    ):
        first = _sample_enclosing_package_geometry(
            rng,
            base_scene.coils[
                0
            ].geometry,
            config,
            other_geometries=(
                base_scene.coils[
                    1
                ].geometry,
            ),
        )
        second = _sample_enclosing_package_geometry(
            rng,
            base_scene.coils[
                1
            ].geometry,
            config,
            other_geometries=(
                base_scene.coils[
                    0
                ].geometry,
            ),
        )
        if (
            first.classify_conductor(
                base_scene.coils[
                    1
                ].geometry,
                longitudinal_segments=64,
                section_points=16,
                tolerance=1e-10,
            )
            != "outside"
            or second.classify_conductor(
                base_scene.coils[
                    0
                ].geometry,
                longitudinal_segments=64,
                section_points=16,
                tolerance=1e-10,
            )
            != "outside"
        ):
            continue
        probe_material = IsotropicMaterial(
            relative_permittivity=1.0,
        )
        probes = (
            PackageObject(
                first,
                probe_material,
                "first",
            ),
            PackageObject(
                second,
                probe_material,
                "second",
            ),
        )
        try:
            topology = package_domain_topology(
                probes
            )
        except ValueError:
            continue
        if all(
            parent is None
            for parent in topology.parent
        ):
            return (
                first,
                second,
            )
    raise RuntimeError(
        "failed to sample two disjoint package regions"
    )


def sample_hybrid_package_scene(
    rng: np.random.Generator,
    config: HybridSceneSamplerConfig | None = None,
):
    """Sample the first supported dielectric-package training domain.

    The package receives a Haar-distributed 3-D rotation and a random center
    offset relative to the primary coil. Its half extents are then enlarged
    from finite-conductor surface samples until the superquadric encloses the
    full primary conductor with margin. Common global SE(3) motion remains an
    exact symmetry and is tested separately rather than wasting teacher solves
    on duplicate scenes.
    """
    config = (
        config
        or HybridSceneSamplerConfig()
    )
    base_scene, frequency = (
        sample_two_coil_mvp_scene(
            rng,
            config.conductor,
        )
    )
    is_dc = bool(
        rng.random()
        < config.dc_probability
    )
    if is_dc:
        frequency = 0.0
    dc_conductive = bool(
        is_dc
        and (
            rng.random()
            < config.dc_conductive_probability
        )
    )
    if is_dc:
        background_conductivity = (
            _log_uniform(
                rng,
                config.background_conductivity_range,
            )
            if dc_conductive
            else 0.0
        )
    elif (
        rng.random()
        < config.lossy_background_probability
    ):
        background_conductivity = _log_uniform(
            rng,
            config.background_conductivity_range,
        )
    else:
        background_conductivity = 0.0

    background_model_draw = float(
        rng.random()
    )
    if (
        background_model_draw
        < config.multi_debye_background_probability
    ):
        background = _sample_multi_debye_material(
            rng,
            epsilon_infinite_range=(
                config.background_debye_epsilon_infinite_range
            ),
            delta_epsilon_range=(
                config.background_debye_delta_epsilon_range
            ),
            relaxation_time_range=(
                config.background_debye_relaxation_time_range
            ),
            pole_count_range=(
                config.multi_debye_poles_range
            ),
            conductivity=(
                background_conductivity
            ),
            relative_permeability=(
                base_scene.medium.relative_permeability
            ),
        )
    elif (
        background_model_draw
        < (
            config.multi_debye_background_probability
            + config.debye_background_probability
        )
    ):
        epsilon_infinite = _uniform(
            rng,
            config.background_debye_epsilon_infinite_range,
        )
        delta_epsilon = _uniform(
            rng,
            config.background_debye_delta_epsilon_range,
        )
        background = DebyeMaterial(
            relative_permittivity_static=(
                epsilon_infinite
                + delta_epsilon
            ),
            relative_permittivity_infinite=(
                epsilon_infinite
            ),
            relaxation_time=_log_uniform(
                rng,
                config.background_debye_relaxation_time_range,
            ),
            relative_permeability=(
                base_scene.medium.relative_permeability
            ),
            conductivity=(
                background_conductivity
            ),
        )
    else:
        background_relative_permittivity = _uniform(
            rng,
            config.background_relative_permittivity_range,
        )
        background = HomogeneousMedium(
            relative_permittivity=(
                background_relative_permittivity
            ),
            relative_permeability=(
                base_scene.medium.relative_permeability
            ),
            conductivity=(
                background_conductivity
            ),
        )

    package_lo, package_hi = (
        config.package_count_range
    )
    package_count = int(
        rng.integers(
            int(
                package_lo
            ),
            int(
                package_hi
            )
            + 1,
        )
    )
    nested = (
        package_count
        > 1
        and rng.random()
        < config.nested_package_probability
    )
    if nested:
        geometries = _nested_package_geometries(
            rng,
            base_scene,
            config,
            package_count,
        )
    else:
        geometries = (
            _sample_disjoint_package_geometries(
                rng,
                base_scene,
                config,
                package_count,
            )
        )

    packages = tuple(
        PackageObject(
            geometry,
            _sample_package_material(
                rng,
                config,
                force_conductive=(
                    False
                    if (
                        is_dc
                        and not dc_conductive
                    )
                    else None
                ),
            ),
            f"package{index}",
        )
        for index, geometry in enumerate(
            geometries
        )
    )
    package_domain_topology(
        packages
    )
    scene = Scene(
        base_scene.coils,
        background,
        packages,
    )
    return (
        scene,
        frequency,
    )
