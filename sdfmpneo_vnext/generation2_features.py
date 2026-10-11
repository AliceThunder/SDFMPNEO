from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .features import EncodedScene
from .scene import EPS0, Scene
from .tensor_features import (
    TENSOR_COIL_FEATURE_DIM,
    TENSOR_PACKAGE_FEATURE_DIM,
    TENSOR_PAIR_FEATURE_DIM,
    TENSOR_CROSS_FEATURE_DIM,
    _material_tensors,
    encode_tensor_hybrid_scene_invariant,
)


GENERATION2_FEATURE_SCHEMA = 1
GENERATION2_COIL_FEATURE_DIM = TENSOR_COIL_FEATURE_DIM
GENERATION2_PACKAGE_PHYSICS_DIM = 6
GENERATION2_PAIR_PHYSICS_DIM = 2
GENERATION2_CROSS_PHYSICS_DIM = 4
GENERATION2_PACKAGE_FEATURE_DIM = (
    TENSOR_PACKAGE_FEATURE_DIM + GENERATION2_PACKAGE_PHYSICS_DIM
)
GENERATION2_PAIR_FEATURE_DIM = TENSOR_PAIR_FEATURE_DIM + GENERATION2_PAIR_PHYSICS_DIM
GENERATION2_CROSS_FEATURE_DIM = TENSOR_CROSS_FEATURE_DIM + GENERATION2_CROSS_PHYSICS_DIM


def _stable_spd_relative_logs(target, background) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    background = np.asarray(background, dtype=float)
    values, vectors = np.linalg.eigh(0.5 * (background + background.T))
    if float(np.min(values)) <= 0.0:
        raise ValueError("background permittivity tensor must be positive definite")
    inverse_sqrt = (
        vectors
        @ np.diag(1.0 / np.sqrt(values))
        @ vectors.T
    )
    relative = inverse_sqrt @ (0.5 * (target + target.T)) @ inverse_sqrt
    eigenvalues = np.linalg.eigvalsh(0.5 * (relative + relative.T))
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("relative permittivity contrast must be positive definite")
    return np.log(np.maximum(eigenvalues, 1e-30))


def _signed_log1p(value: float) -> float:
    value = float(value)
    return float(np.sign(value) * np.log1p(abs(value)))


def _package_physics_features(scene: Scene, frequency_hz: float, length_scale: float):
    background_epsilon, background_sigma = _material_tensors(
        scene.medium,
        frequency_hz,
    )
    omega = 2.0 * np.pi * float(frequency_hz)
    rows = []
    for package in scene.packages:
        package_epsilon, package_sigma = _material_tensors(
            package.material,
            frequency_hz,
        )
        log_relative = _stable_spd_relative_logs(
            package_epsilon,
            background_epsilon,
        )
        electric_bias = float(np.mean(log_relative))
        electric_anisotropy = float(np.std(log_relative))

        electric_scale = max(
            omega * EPS0 * float(np.linalg.norm(background_epsilon)),
            float(np.linalg.norm(background_sigma)),
            1e-18,
        )
        loss_contrast = float(
            np.linalg.norm(package_sigma - background_sigma) / electric_scale
        )
        mu_ratio = float(
            package.material.relative_permeability
            / scene.medium.relative_permeability
        )
        mu_contrast = _signed_log1p(mu_ratio - 1.0)
        volume = float(package.geometry.volume / max(length_scale**3, 1e-30))
        contrast_strength = float(
            np.sqrt(
                electric_bias**2
                + electric_anisotropy**2
                + np.log1p(loss_contrast) ** 2
                + mu_contrast**2
            )
        )
        polarizability_proxy = float(volume * np.tanh(contrast_strength))
        rows.append(
            [
                np.log1p(max(volume, 0.0)),
                electric_bias,
                electric_anisotropy,
                np.log1p(max(loss_contrast, 0.0)),
                mu_contrast,
                polarizability_proxy,
            ]
        )
    return (
        np.asarray(rows, dtype=float)
        if rows
        else np.zeros((0, GENERATION2_PACKAGE_PHYSICS_DIM), dtype=float)
    )


def _pair_physics_features(pair_features: np.ndarray) -> np.ndarray:
    pair_features = np.asarray(pair_features, dtype=float)
    n_source, n_target, _ = pair_features.shape
    result = np.zeros(
        (n_source, n_target, GENERATION2_PAIR_PHYSICS_DIM),
        dtype=float,
    )
    for source in range(n_source):
        for target in range(n_target):
            if source == target:
                continue
            distance = max(float(pair_features[source, target, 12]), 0.0)
            result[source, target, 0] = 1.0 / (1.0 + distance)
            result[source, target, 1] = 1.0 / (1.0 + distance**3)
    return result


def _cross_physics_features(
    cross_features: np.ndarray,
    package_physics: np.ndarray,
) -> np.ndarray:
    cross_features = np.asarray(cross_features, dtype=float)
    n_coils, n_packages, _ = cross_features.shape
    result = np.zeros(
        (n_coils, n_packages, GENERATION2_CROSS_PHYSICS_DIM),
        dtype=float,
    )
    for coil in range(n_coils):
        for package in range(n_packages):
            distance = max(float(cross_features[coil, package, 12]), 0.0)
            inverse = 1.0 / (1.0 + distance)
            inverse_cube = 1.0 / (1.0 + distance**3)
            volume = float(np.expm1(package_physics[package, 0]))
            proxy = float(package_physics[package, 5])
            result[coil, package] = (
                inverse,
                inverse_cube,
                volume * inverse_cube,
                proxy * inverse_cube,
            )
    return result


@dataclass(frozen=True)
class EncodedGeneration2Scene:
    coil: EncodedScene
    package_features: np.ndarray
    coil_package_features: np.ndarray
    package_pair_features: np.ndarray

    def __post_init__(self):
        package = np.asarray(self.package_features, dtype=float)
        cross = np.asarray(self.coil_package_features, dtype=float)
        pair = np.asarray(self.package_pair_features, dtype=float)
        n_coils = int(self.coil.node_features.shape[0])
        n_packages = int(package.shape[0])
        if self.coil.node_features.shape != (
            n_coils,
            GENERATION2_COIL_FEATURE_DIM,
        ):
            raise ValueError("generation-2 coil features have incompatible shape")
        if package.shape != (n_packages, GENERATION2_PACKAGE_FEATURE_DIM):
            raise ValueError("generation-2 package features have incompatible shape")
        if cross.shape != (
            n_coils,
            n_packages,
            GENERATION2_CROSS_FEATURE_DIM,
        ):
            raise ValueError("generation-2 coil-package features have incompatible shape")
        if pair.shape != (
            n_packages,
            n_packages,
            GENERATION2_PAIR_FEATURE_DIM,
        ):
            raise ValueError("generation-2 package-pair features have incompatible shape")
        if not (
            np.all(np.isfinite(self.coil.node_features))
            and np.all(np.isfinite(package))
            and np.all(np.isfinite(cross))
            and np.all(np.isfinite(pair))
        ):
            raise ValueError("generation-2 scene features must be finite")
        object.__setattr__(self, "package_features", package)
        object.__setattr__(self, "coil_package_features", cross)
        object.__setattr__(self, "package_pair_features", pair)

    @property
    def n_coils(self) -> int:
        return int(self.coil.node_features.shape[0])

    @property
    def n_packages(self) -> int:
        return int(self.package_features.shape[0])

    @property
    def length_scale(self) -> float:
        return float(self.coil.length_scale)


def encode_generation2_scene(
    scene: Scene,
    frequency_hz: float,
) -> EncodedGeneration2Scene:
    base = encode_tensor_hybrid_scene_invariant(scene, frequency_hz)
    package_physics = _package_physics_features(
        scene,
        frequency_hz,
        float(base.length_scale),
    )
    package_features = np.concatenate(
        (base.package_features, package_physics),
        axis=-1,
    )

    coil_pair_physics = _pair_physics_features(base.coil.pair_features)
    coil_pair = np.concatenate(
        (base.coil.pair_features, coil_pair_physics),
        axis=-1,
    )
    coil = EncodedScene(
        np.asarray(base.coil.node_features, dtype=float),
        coil_pair,
        float(base.length_scale),
    )

    package_pair_physics = _pair_physics_features(base.package_pair_features)
    package_pair = np.concatenate(
        (base.package_pair_features, package_pair_physics),
        axis=-1,
    )
    cross_physics = _cross_physics_features(
        base.coil_package_features,
        package_physics,
    )
    coil_package = np.concatenate(
        (base.coil_package_features, cross_physics),
        axis=-1,
    )

    return EncodedGeneration2Scene(
        coil,
        package_features,
        coil_package,
        package_pair,
    )


__all__ = [
    "GENERATION2_FEATURE_SCHEMA",
    "GENERATION2_COIL_FEATURE_DIM",
    "GENERATION2_PACKAGE_FEATURE_DIM",
    "GENERATION2_PAIR_FEATURE_DIM",
    "GENERATION2_CROSS_FEATURE_DIM",
    "EncodedGeneration2Scene",
    "encode_generation2_scene",
]
