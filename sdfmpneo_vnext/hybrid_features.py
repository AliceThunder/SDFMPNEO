from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .features import (
    EncodedScene,
    encode_scene_invariant,
)
from .scene import EPS0, Scene


@dataclass(frozen=True)
class EncodedHybridScene:
    coil: EncodedScene
    package_features: np.ndarray
    coil_package_features: np.ndarray
    package_pair_features: np.ndarray

    def __post_init__(self):
        package = np.asarray(
            self.package_features,
            dtype=float,
        )
        coil_package = np.asarray(
            self.coil_package_features,
            dtype=float,
        )
        package_pair = np.asarray(
            self.package_pair_features,
            dtype=float,
        )
        n_coils = (
            self.coil.node_features.shape[
                0
            ]
        )
        n_packages = (
            package.shape[0]
            if package.ndim == 2
            else -1
        )
        if (
            package.ndim != 2
            or package.shape[1]
            != 13
        ):
            raise ValueError(
                "package_features must have shape (n_packages,13)"
            )
        if (
            coil_package.shape
            != (
                n_coils,
                n_packages,
                15,
            )
        ):
            raise ValueError(
                "coil_package_features must have shape "
                "(n_coils,n_packages,15)"
            )
        if (
            package_pair.shape
            != (
                n_packages,
                n_packages,
                15,
            )
        ):
            raise ValueError(
                "package_pair_features must have shape "
                "(n_packages,n_packages,15)"
            )
        if (
            not np.all(
                np.isfinite(
                    package
                )
            )
            or not np.all(
                np.isfinite(
                    coil_package
                )
            )
            or not np.all(
                np.isfinite(
                    package_pair
                )
            )
        ):
            raise ValueError(
                "hybrid scene features must be finite"
            )
        object.__setattr__(
            self,
            "package_features",
            package,
        )
        object.__setattr__(
            self,
            "coil_package_features",
            coil_package,
        )
        object.__setattr__(
            self,
            "package_pair_features",
            package_pair,
        )

    @property
    def n_coils(
        self,
    ) -> int:
        return int(
            self.coil.node_features.shape[
                0
            ]
        )

    @property
    def n_packages(
        self,
    ) -> int:
        return int(
            self.package_features.shape[
                0
            ]
        )

    @property
    def length_scale(
        self,
    ) -> float:
        return float(
            self.coil.length_scale
        )


def _relative_pose_features(
    source_rotation,
    source_translation,
    target_rotation,
    target_translation,
    *,
    length_scale: float,
    source_size: float,
    target_size: float,
) -> np.ndarray:
    relative_translation = (
        source_rotation.T
        @ (
            target_translation
            - source_translation
        )
    ) / length_scale
    relative_rotation = (
        source_rotation.T
        @ target_rotation
    )
    distance = float(
        np.linalg.norm(
            relative_translation
        )
    )
    return np.concatenate(
        (
            relative_translation,
            relative_rotation.ravel(),
            np.array(
                [
                    distance,
                    target_size
                    / max(
                        source_size,
                        1e-12,
                    ),
                    (
                        source_size
                        + target_size
                    )
                    / max(
                        np.linalg.norm(
                            target_translation
                            - source_translation
                        ),
                        1e-12,
                    ),
                ],
                dtype=float,
            ),
        )
    )


def _package_features(
    scene: Scene,
    frequency_hz: float,
    length_scale: float,
) -> np.ndarray:
    rows = []
    omega = (
        2.0
        * np.pi
        * float(
            frequency_hz
        )
    )
    for package in scene.packages:
        geometry = (
            package.geometry
        )
        material = (
            package.material
        )
        epsilon_ratio = (
            material.relative_permittivity
            / scene.medium.relative_permittivity
        )
        permeability_ratio = (
            material.relative_permeability
            / scene.medium.relative_permeability
        )
        if frequency_hz > 0.0:
            loss_tangent_like = (
                material.conductivity
                / max(
                    omega
                    * EPS0
                    * material.relative_permittivity,
                    1e-30,
                )
            )
        else:
            loss_tangent_like = (
                0.0
                if material.conductivity
                == 0.0
                else 1e30
            )
        rows.append(
            [
                geometry.half_extents[
                    0
                ]
                / length_scale,
                geometry.half_extents[
                    1
                ]
                / length_scale,
                geometry.half_extents[
                    2
                ]
                / length_scale,
                geometry.exponent_xy,
                geometry.exponent_z,
                np.log(
                    material.relative_permittivity
                ),
                material.relative_permeability,
                np.log1p(
                    material.conductivity
                    / 1e-6
                ),
                np.log(
                    epsilon_ratio
                ),
                np.log(
                    permeability_ratio
                ),
                np.log1p(
                    loss_tangent_like
                ),
                geometry.volume
                / (
                    length_scale**3
                ),
                geometry.characteristic_length
                / length_scale,
            ]
        )
    if not rows:
        return np.zeros(
            (
                0,
                13,
            ),
            dtype=float,
        )
    return np.asarray(
        rows,
        dtype=float,
    )


def encode_hybrid_scene_invariant(
    scene: Scene,
    frequency_hz: float,
) -> EncodedHybridScene:
    if (
        not np.isfinite(
            frequency_hz
        )
        or frequency_hz < 0.0
    ):
        raise ValueError(
            "frequency_hz must be finite and nonnegative"
        )

    conductor_scene = Scene(
        scene.coils,
        scene.medium,
        (),
    )
    coil = encode_scene_invariant(
        conductor_scene,
        frequency_hz,
    )
    length_scale = (
        coil.length_scale
    )
    package = _package_features(
        scene,
        frequency_hz,
        length_scale,
    )

    n_coils = len(
        scene.coils
    )
    n_packages = len(
        scene.packages
    )
    coil_package = np.zeros(
        (
            n_coils,
            n_packages,
            15,
        ),
        dtype=float,
    )
    for coil_index, coil_object in enumerate(
        scene.coils
    ):
        coil_geometry = (
            coil_object.geometry
        )
        coil_size = float(
            np.sqrt(
                coil_geometry.outer_a
                * coil_geometry.outer_b
            )
        )
        for package_index, package_object in enumerate(
            scene.packages
        ):
            package_geometry = (
                package_object.geometry
            )
            coil_package[
                coil_index,
                package_index,
            ] = _relative_pose_features(
                coil_geometry.pose.rotation,
                coil_geometry.pose.translation,
                package_geometry.pose.rotation,
                package_geometry.pose.translation,
                length_scale=(
                    length_scale
                ),
                source_size=(
                    coil_size
                ),
                target_size=(
                    package_geometry.characteristic_length
                ),
            )

    package_pair = np.zeros(
        (
            n_packages,
            n_packages,
            15,
        ),
        dtype=float,
    )
    for i, source in enumerate(
        scene.packages
    ):
        source_geometry = (
            source.geometry
        )
        for j, target in enumerate(
            scene.packages
        ):
            target_geometry = (
                target.geometry
            )
            if i == j:
                package_pair[
                    i,
                    j,
                    3:12,
                ] = np.eye(
                    3
                ).ravel()
                package_pair[
                    i,
                    j,
                    13,
                ] = 1.0
                continue
            package_pair[
                i,
                j,
            ] = _relative_pose_features(
                source_geometry.pose.rotation,
                source_geometry.pose.translation,
                target_geometry.pose.rotation,
                target_geometry.pose.translation,
                length_scale=(
                    length_scale
                ),
                source_size=(
                    source_geometry.characteristic_length
                ),
                target_size=(
                    target_geometry.characteristic_length
                ),
            )

    return EncodedHybridScene(
        coil,
        package,
        coil_package,
        package_pair,
    )
