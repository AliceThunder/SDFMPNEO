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
        float(bounds[0]),
        float(bounds[1]),
    )
    return float(
        np.exp(
            rng.uniform(
                np.log(lower),
                np.log(upper),
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
    if allow_zero and rng.random() < zero_probability:
        return np.zeros((3, 3), dtype=float)
    eigenvalues = np.asarray(
        [
            _log_uniform(rng, eigenvalue_range)
            for _ in range(3)
        ],
        dtype=float,
    )
    rotation = haar_rotation(rng)
    tensor = rotation @ np.diag(eigenvalues) @ rotation.T
    return 0.5 * (tensor + tensor.T)


def _thermal_kwargs(material):
    return {
        "thermal_conductivity": getattr(
            material,
            "thermal_conductivity",
            None,
        ),
        "density": getattr(material, "density", None),
        "heat_capacity": getattr(material, "heat_capacity", None),
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
    if rng.random() < lossless_probability:
        sigma = np.zeros((3, 3), dtype=float)
    else:
        # TensorElectricMaterial currently requires epsilon and sigma to share
        # principal axes. Reuse epsilon's eigenvectors and draw independent
        # positive conductivity principal values.
        _, eigenvectors = np.linalg.eigh(epsilon)
        sigma_eigenvalues = np.asarray(
            [
                _log_uniform(rng, conductivity_range)
                for _ in range(3)
            ],
            dtype=float,
        )
        sigma = (
            eigenvectors
            @ np.diag(sigma_eigenvalues)
            @ eigenvectors.T
        )
    return TensorElectricMaterial(
        relative_permittivity_tensor=epsilon,
        conductivity_tensor=sigma,
        relative_permeability=float(source.relative_permeability),
        **_thermal_kwargs(source),
    )


@dataclass(frozen=True)
class TensorHybridSceneSamplerConfig:
    base: HybridSceneSamplerConfig = HybridSceneSamplerConfig()
    tensor_package_probability: float = 1.0
    tensor_background_probability: float = 0.0
    tensor_relative_permittivity_range: tuple[float, float] = (
        1.5,
        10.0,
    )
    tensor_conductivity_range: tuple[float, float] = (
        1e-7,
        5e-3,
    )
    tensor_lossless_probability: float = 0.20

    def __post_init__(self):
        for name in (
            "tensor_package_probability",
            "tensor_background_probability",
            "tensor_lossless_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")

        for name in (
            "tensor_relative_permittivity_range",
            "tensor_conductivity_range",
        ):
            lower, upper = getattr(self, name)
            if not (
                np.isfinite(lower)
                and np.isfinite(upper)
                and 0.0 < lower < upper
            ):
                raise ValueError(
                    f"{name} must be a positive finite increasing pair"
                )

        if int(self.base.package_count_range[0]) < 1:
            raise ValueError(
                "tensor FAST sampling requires package_count_range minimum >= 1; "
                "the current tensor surrogate family is package-aware"
            )

        if (
            self.base.dc_probability > 0.0
            and (
                self.tensor_background_probability > 0.0
                or self.tensor_package_probability > 0.0
            )
        ):
            raise ValueError(
                "tensor FAST sampling currently requires dc_probability=0; "
                "exact-DC tensor training will be enabled after the partial-"
                "conduction tensor backend is promoted"
            )


def sample_tensor_hybrid_scene(
    rng: np.random.Generator,
    config: TensorHybridSceneSamplerConfig | None = None,
):
    config = config or TensorHybridSceneSamplerConfig()
    scene, frequency_hz = sample_hybrid_package_scene(
        rng,
        config.base,
    )

    packages = []
    tensor_package_count = 0
    for package in scene.packages:
        if rng.random() < config.tensor_package_probability:
            material = _tensor_material(
                rng,
                package.material,
                relative_permittivity_range=(
                    config.tensor_relative_permittivity_range
                ),
                conductivity_range=config.tensor_conductivity_range,
                lossless_probability=config.tensor_lossless_probability,
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
    if rng.random() < config.tensor_background_probability:
        medium = _tensor_material(
            rng,
            scene.medium,
            relative_permittivity_range=(
                config.tensor_relative_permittivity_range
            ),
            conductivity_range=config.tensor_conductivity_range,
            lossless_probability=config.tensor_lossless_probability,
        )
        tensor_background = True

    if tensor_package_count == 0 and not tensor_background:
        # This sampler is explicitly for tensor-aware FAST artifacts. Keep at
        # least one tensor material in every generated scene so the training
        # set cannot silently collapse to the scalar regime.
        package = packages[0]
        packages[0] = PackageObject(
            package.geometry,
            _tensor_material(
                rng,
                package.material,
                relative_permittivity_range=(
                    config.tensor_relative_permittivity_range
                ),
                conductivity_range=config.tensor_conductivity_range,
                lossless_probability=config.tensor_lossless_probability,
            ),
            package.name,
        )

    return (
        Scene(
            scene.coils,
            medium,
            tuple(packages),
        ),
        float(frequency_hz),
    )
