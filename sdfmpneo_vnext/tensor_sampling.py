from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import haar_rotation
from .sampling import (
    HybridSceneSamplerConfig,
    sample_hybrid_package_scene,
)
from .scene import (
    PackageObject,
    Scene,
    TensorElectricMaterial,
)


def _log_uniform(
    rng: np.random.Generator,
    bounds,
) -> float:
    lower, upper = (
        float(
            bounds[
                0
            ]
        ),
        float(
            bounds[
                1
            ]
        ),
    )
    return float(
        np.exp(
            rng.uniform(
                np.log(
                    lower
                ),
                np.log(
                    upper
                ),
            )
        )
    )


def _sample_spd_tensor(
    rng: np.random.Generator,
    eigenvalue_range,
    *,
    allow_zero: bool,
    zero_probability: float,
):
    if (
        allow_zero
        and rng.random()
        < zero_probability
    ):
        return np.zeros(
            (
                3,
                3,
            ),
            dtype=float,
        )
    eigenvalues = np.asarray(
        [
            _log_uniform(
                rng,
                eigenvalue_range,
            )
            for _ in range(
                3
            )
        ],
        dtype=float,
    )
    rotation = haar_rotation(
        rng
    )
    tensor = (
        rotation
        @ np.diag(
            eigenvalues
        )
        @ rotation.T
    )
    return 0.5 * (
        tensor
        + tensor.T
    )


def _thermal_kwargs(
    material,
):
    return {
        "thermal_conductivity": getattr(
            material,
            "thermal_conductivity",
            None,
        ),
        "density": getattr(
            material,
            "density",
            None,
        ),
        "heat_capacity": getattr(
            material,
            "heat_capacity",
            None,
        ),
        "thermal_conductivity_tensor": getattr(
            material,
            "thermal_conductivity_tensor",
            None,
        ),
    }


def _tensor_material(
    rng: np.random.Generator,
    source,
    *,
    relative_permittivity_range,
    conductivity_range,
    lossless_probability: float,
):
    epsilon = _sample_spd_tensor(
        rng,
        relative_permittivity_range,
        allow_zero=False,
        zero_probability=0.0,
    )
    if (
        rng.random()
        < lossless_probability
    ):
        sigma = np.zeros(
            (
                3,
                3,
            ),
            dtype=float,
        )
    else:
        # TensorElectricMaterial currently requires epsilon and sigma to share
        # principal axes. Reuse epsilon's eigenvectors and draw independent
        # positive conductivity principal values.
        _, eigenvectors = np.linalg.eigh(
            epsilon
        )
        sigma_eigenvalues = np.asarray(
            [
                _log_uniform(
                    rng,
                    conductivity_range,
                )
                for _ in range(
                    3
                )
            ],
            dtype=float,
        )
        sigma = (
            eigenvectors
            @ np.diag(
                sigma_eigenvalues
            )
            @ eigenvectors.T
        )
    return TensorElectricMaterial(
        relative_permittivity_tensor=epsilon,
        conductivity_tensor=sigma,
        relative_permeability=float(
            source.relative_permeability
        ),
        **_thermal_kwargs(
            source
        ),
    )


@dataclass(frozen=True)
class TensorHybridSceneSamplerConfig:
    base: HybridSceneSamplerConfig = HybridSceneSamplerConfig()
    tensor_package_probability: float = 1.0
    tensor_background_probability: float = 0.0
    # Fraction of tensor scenes that contain no finite package at all. The
    # background is forced tensor-electric for these draws so the sample stays
    # inside the tensor-aware training family rather than collapsing to scalar.
    background_only_probability: float = 0.0
    tensor_relative_permittivity_range: tuple[float, float] = (
        1.5,
        10.0,
    )
    tensor_conductivity_range: tuple[float, float] = (
        1e-7,
        5e-3,
    )
    tensor_lossless_probability: float = 0.20
    maximum_scene_attempts: int = 64

    def __post_init__(
        self,
    ):
        for name in (
            "tensor_package_probability",
            "tensor_background_probability",
            "background_only_probability",
            "tensor_lossless_probability",
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
        for name in (
            "tensor_relative_permittivity_range",
            "tensor_conductivity_range",
        ):
            lower, upper = getattr(
                self,
                name,
            )
            if not (
                np.isfinite(
                    lower
                )
                and np.isfinite(
                    upper
                )
                and 0.0
                < lower
                < upper
            ):
                raise ValueError(
                    f"{name} must be a positive finite increasing pair"
                )
        if (
            not isinstance(
                self.maximum_scene_attempts,
                (int, np.integer),
            )
            or int(
                self.maximum_scene_attempts
            ) < 1
        ):
            raise ValueError(
                "maximum_scene_attempts must be a positive integer"
            )
        if (
            self.base.dc_probability
            > 0.0
            and (
                self.tensor_background_probability
                > 0.0
                or self.tensor_package_probability
                > 0.0
                or self.background_only_probability
                > 0.0
            )
        ):
            raise ValueError(
                "tensor FAST sampling currently requires dc_probability=0; "
                "exact-DC tensor training will be enabled after the partial-"
                "conduction tensor backend is promoted"
            )


def _sample_base_scene(
    rng: np.random.Generator,
    config: TensorHybridSceneSamplerConfig,
):
    last_error = None
    for _ in range(
        int(
            config.maximum_scene_attempts
        )
    ):
        try:
            return sample_hybrid_package_scene(
                rng,
                config.base,
            )
        except RuntimeError as exc:
            # Package/root generation is rejection sampling. A failed finite
            # draw budget is not a malformed tensor scene; continue from the
            # deterministic RNG stream. Preserve unrelated numerical failures.
            if not str(exc).startswith(
                "failed to sample"
            ):
                raise
            last_error = exc
    raise RuntimeError(
        "failed to sample a valid tensor hybrid scene after "
        f"{int(config.maximum_scene_attempts)} independent scene attempts"
    ) from last_error


def sample_tensor_hybrid_scene(
    rng: np.random.Generator,
    config: TensorHybridSceneSamplerConfig | None = None,
):
    config = config or TensorHybridSceneSamplerConfig()
    scene, frequency_hz = _sample_base_scene(
        rng,
        config,
    )

    if (
        rng.random()
        < config.background_only_probability
    ):
        medium = _tensor_material(
            rng,
            scene.medium,
            relative_permittivity_range=(
                config.tensor_relative_permittivity_range
            ),
            conductivity_range=(
                config.tensor_conductivity_range
            ),
            lossless_probability=(
                config.tensor_lossless_probability
            ),
        )
        return (
            Scene(
                scene.coils,
                medium,
                (),
            ),
            float(
                frequency_hz
            ),
        )

    packages = []
    tensor_package_count = 0
    for package in scene.packages:
        if (
            rng.random()
            < config.tensor_package_probability
        ):
            material = _tensor_material(
                rng,
                package.material,
                relative_permittivity_range=(
                    config.tensor_relative_permittivity_range
                ),
                conductivity_range=(
                    config.tensor_conductivity_range
                ),
                lossless_probability=(
                    config.tensor_lossless_probability
                ),
            )
            tensor_package_count += 1
        else:
            material = package.material
        packages.append(
            PackageObject(
                package.geometry,
                material,
                package.name,
            )
        )

    medium = scene.medium
    tensor_background = False
    if (
        rng.random()
        < config.tensor_background_probability
    ):
        medium = _tensor_material(
            rng,
            scene.medium,
            relative_permittivity_range=(
                config.tensor_relative_permittivity_range
            ),
            conductivity_range=(
                config.tensor_conductivity_range
            ),
            lossless_probability=(
                config.tensor_lossless_probability
            ),
        )
        tensor_background = True

    if (
        tensor_package_count == 0
        and not tensor_background
    ):
        # This sampler is explicitly for tensor-aware FAST artifacts. Keep at
        # least one tensor material in every generated scene so the training
        # set cannot silently collapse to the scalar regime.
        package = packages[
            0
        ]
        packages[
            0
        ] = PackageObject(
            package.geometry,
            _tensor_material(
                rng,
                package.material,
                relative_permittivity_range=(
                    config.tensor_relative_permittivity_range
                ),
                conductivity_range=(
                    config.tensor_conductivity_range
                ),
                lossless_probability=(
                    config.tensor_lossless_probability
                ),
            ),
            package.name,
        )

    return (
        Scene(
            scene.coils,
            medium,
            tuple(
                packages
            ),
        ),
        float(
            frequency_hz
        ),
    )
