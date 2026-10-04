from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .features import EncodedScene, _skin_depth
from .hybrid_features import _relative_pose_features
from .scene import (
    MU0,
    Scene,
    TensorElectricMaterial,
)


TENSOR_COIL_FEATURE_DIM = 27
TENSOR_PACKAGE_FEATURE_DIM = 33
TENSOR_PAIR_FEATURE_DIM = 15
TENSOR_CROSS_FEATURE_DIM = 15


def _symmetric_components(
    matrix,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=float,
    )
    if matrix.shape != (
        3,
        3,
    ):
        raise ValueError(
            "tensor feature matrix must have shape (3,3)"
        )
    return np.asarray(
        [
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            np.sqrt(2.0) * matrix[0, 1],
            np.sqrt(2.0) * matrix[0, 2],
            np.sqrt(2.0) * matrix[1, 2],
        ],
        dtype=float,
    )


def _symmetric_matrix_log(
    matrix,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5
        * (
            matrix
            + matrix.T
        )
    )
    if np.min(
        eigenvalues
    ) <= 0.0:
        raise ValueError(
            "relative-permittivity tensor must be positive definite"
        )
    return (
        eigenvectors
        @ np.diag(
            np.log(
                eigenvalues
            )
        )
        @ eigenvectors.T
    )


def _conductivity_log1p(
    matrix,
    *,
    scale: float = 1e-6,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5
        * (
            matrix
            + matrix.T
        )
    )
    if np.min(
        eigenvalues
    ) < -1e-14:
        raise ValueError(
            "conductivity tensor must be positive semidefinite"
        )
    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )
    return (
        eigenvectors
        @ np.diag(
            np.log1p(
                eigenvalues
                / float(
                    scale
                )
            )
        )
        @ eigenvectors.T
    )


def _material_tensors(
    material,
    frequency_hz: float,
):
    if isinstance(
        material,
        TensorElectricMaterial,
    ):
        epsilon = np.asarray(
            material.relative_permittivity_tensor,
            dtype=float,
        )
        sigma = np.asarray(
            material.loss_conductivity_tensor(
                frequency_hz
            ),
            dtype=float,
        )
        return (
            epsilon,
            sigma,
        )

    relative = material.relative_permittivity_at(
        frequency_hz
    )
    epsilon_real = max(
        float(
            np.real(
                relative
            )
        ),
        1e-12,
    )
    sigma = max(
        float(
            material.loss_conductivity(
                frequency_hz
            )
        ),
        0.0,
    )
    return (
        epsilon_real
        * np.eye(
            3,
            dtype=float,
        ),
        sigma
        * np.eye(
            3,
            dtype=float,
        ),
    )


def _world_material_tensors(
    material,
    frequency_hz: float,
    *,
    local_rotation=None,
):
    epsilon, sigma = _material_tensors(
        material,
        frequency_hz,
    )
    if (
        isinstance(
            material,
            TensorElectricMaterial,
        )
        and local_rotation is not None
    ):
        rotation = np.asarray(
            local_rotation,
            dtype=float,
        )
        epsilon = (
            rotation
            @ epsilon
            @ rotation.T
        )
        sigma = (
            rotation
            @ sigma
            @ rotation.T
        )
    return (
        epsilon,
        sigma,
    )


def _tensor_block(
    epsilon,
    sigma,
) -> np.ndarray:
    return np.concatenate(
        (
            _symmetric_components(
                _symmetric_matrix_log(
                    epsilon
                )
            ),
            _symmetric_components(
                _conductivity_log1p(
                    sigma
                )
            ),
        )
    )


@dataclass(frozen=True)
class EncodedTensorHybridScene:
    coil: EncodedScene
    package_features: np.ndarray
    coil_package_features: np.ndarray
    package_pair_features: np.ndarray

    def __post_init__(
        self,
    ):
        package = np.asarray(
            self.package_features,
            dtype=float,
        )
        cross = np.asarray(
            self.coil_package_features,
            dtype=float,
        )
        pair = np.asarray(
            self.package_pair_features,
            dtype=float,
        )
        n_coils = self.coil.node_features.shape[
            0
        ]
        n_packages = package.shape[
            0
        ]
        if package.shape != (
            n_packages,
            TENSOR_PACKAGE_FEATURE_DIM,
        ):
            raise ValueError(
                "tensor package_features have incompatible shape"
            )
        if cross.shape != (
            n_coils,
            n_packages,
            TENSOR_CROSS_FEATURE_DIM,
        ):
            raise ValueError(
                "tensor coil-package features have incompatible shape"
            )
        if pair.shape != (
            n_packages,
            n_packages,
            TENSOR_PAIR_FEATURE_DIM,
        ):
            raise ValueError(
                "tensor package-pair features have incompatible shape"
            )
        if self.coil.node_features.shape[
            1
        ] != TENSOR_COIL_FEATURE_DIM:
            raise ValueError(
                "tensor coil features have incompatible width"
            )
        if (
            not np.all(
                np.isfinite(
                    self.coil.node_features
                )
            )
            or not np.all(
                np.isfinite(
                    package
                )
            )
            or not np.all(
                np.isfinite(
                    cross
                )
            )
            or not np.all(
                np.isfinite(
                    pair
                )
            )
        ):
            raise ValueError(
                "tensor hybrid features must be finite"
            )
        object.__setattr__(
            self,
            "package_features",
            package,
        )
        object.__setattr__(
            self,
            "coil_package_features",
            cross,
        )
        object.__setattr__(
            self,
            "package_pair_features",
            pair,
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


def encode_tensor_hybrid_scene_invariant(
    scene: Scene,
    frequency_hz: float,
) -> EncodedTensorHybridScene:
    if (
        not np.isfinite(
            frequency_hz
        )
        or frequency_hz < 0.0
    ):
        raise ValueError(
            "frequency_hz must be finite and nonnegative"
        )

    radii = np.asarray(
        [
            np.sqrt(
                coil.geometry.outer_a
                * coil.geometry.outer_b
            )
            for coil in scene.coils
        ],
        dtype=float,
    )
    length_scale = max(
        float(
            np.exp(
                np.mean(
                    np.log(
                        radii
                    )
                )
            )
        ),
        1e-12,
    )
    background_epsilon_world, background_sigma_world = (
        _world_material_tensors(
            scene.medium,
            frequency_hz,
        )
    )

    node_rows = []
    for coil in scene.coils:
        geometry = coil.geometry
        material = coil.material
        skin_depth = _skin_depth(
            frequency_hz,
            material.conductivity,
            MU0
            * material.relative_permeability,
        )
        skin_w = (
            0.0
            if not np.isfinite(
                skin_depth
            )
            else geometry.conductor_width
            / skin_depth
        )
        skin_h = (
            0.0
            if not np.isfinite(
                skin_depth
            )
            else geometry.conductor_thickness
            / skin_depth
        )
        rotation = np.asarray(
            geometry.pose.rotation,
            dtype=float,
        )
        background_epsilon_local = (
            rotation.T
            @ background_epsilon_world
            @ rotation
        )
        background_sigma_local = (
            rotation.T
            @ background_sigma_world
            @ rotation
        )
        node_rows.append(
            np.concatenate(
                (
                    np.asarray(
                        [
                            geometry.outer_a
                            / length_scale,
                            geometry.outer_b
                            / length_scale,
                            geometry.turns,
                            geometry.pitch_a
                            / length_scale,
                            geometry.pitch_b
                            / length_scale,
                            geometry.exponent,
                            geometry.conductor_width
                            / length_scale,
                            geometry.conductor_thickness
                            / length_scale,
                            geometry.cross_section_exponent,
                            np.log(
                                material.conductivity
                                / 1e7
                            ),
                            material.relative_permeability,
                            skin_w,
                            skin_h,
                        ],
                        dtype=float,
                    ),
                    _tensor_block(
                        background_epsilon_local,
                        background_sigma_local,
                    ),
                    np.asarray(
                        [
                            scene.medium.relative_permeability,
                            np.log1p(
                                frequency_hz
                                / 1e3
                            ),
                        ],
                        dtype=float,
                    ),
                )
            )
        )
    node_features = np.asarray(
        node_rows,
        dtype=float,
    )

    n_coils = len(
        scene.coils
    )
    coil_pair = np.zeros(
        (
            n_coils,
            n_coils,
            TENSOR_PAIR_FEATURE_DIM,
        ),
        dtype=float,
    )
    for i, source in enumerate(
        scene.coils
    ):
        source_geometry = source.geometry
        source_size = float(
            np.sqrt(
                source_geometry.outer_a
                * source_geometry.outer_b
            )
        )
        for j, target in enumerate(
            scene.coils
        ):
            target_geometry = target.geometry
            if i == j:
                coil_pair[
                    i,
                    j,
                    3:12,
                ] = np.eye(
                    3
                ).ravel()
                continue
            target_size = float(
                np.sqrt(
                    target_geometry.outer_a
                    * target_geometry.outer_b
                )
            )
            coil_pair[
                i,
                j,
            ] = _relative_pose_features(
                source_geometry.pose.rotation,
                source_geometry.pose.translation,
                target_geometry.pose.rotation,
                target_geometry.pose.translation,
                length_scale=length_scale,
                source_size=source_size,
                target_size=target_size,
            )
    coil_encoded = EncodedScene(
        node_features,
        coil_pair,
        length_scale,
    )

    package_rows = []
    for package in scene.packages:
        geometry = package.geometry
        package_epsilon_local, package_sigma_local = (
            _material_tensors(
                package.material,
                frequency_hz,
            )
        )
        rotation = np.asarray(
            geometry.pose.rotation,
            dtype=float,
        )
        background_epsilon_local = (
            rotation.T
            @ background_epsilon_world
            @ rotation
        )
        background_sigma_local = (
            rotation.T
            @ background_sigma_world
            @ rotation
        )
        package_rows.append(
            np.concatenate(
                (
                    np.asarray(
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
                            geometry.volume
                            / (
                                length_scale**3
                            ),
                            geometry.characteristic_length
                            / length_scale,
                            package.material.relative_permeability,
                            package.material.relative_permeability
                            / scene.medium.relative_permeability,
                        ],
                        dtype=float,
                    ),
                    _tensor_block(
                        package_epsilon_local,
                        package_sigma_local,
                    ),
                    _tensor_block(
                        background_epsilon_local,
                        background_sigma_local,
                    ),
                )
            )
        )
    package_features = (
        np.asarray(
            package_rows,
            dtype=float,
        )
        if package_rows
        else np.zeros(
            (
                0,
                TENSOR_PACKAGE_FEATURE_DIM,
            ),
            dtype=float,
        )
    )

    n_packages = len(
        scene.packages
    )
    coil_package = np.zeros(
        (
            n_coils,
            n_packages,
            TENSOR_CROSS_FEATURE_DIM,
        ),
        dtype=float,
    )
    for coil_index, coil in enumerate(
        scene.coils
    ):
        coil_geometry = coil.geometry
        coil_size = float(
            np.sqrt(
                coil_geometry.outer_a
                * coil_geometry.outer_b
            )
        )
        for package_index, package in enumerate(
            scene.packages
        ):
            package_geometry = package.geometry
            coil_package[
                coil_index,
                package_index,
            ] = _relative_pose_features(
                coil_geometry.pose.rotation,
                coil_geometry.pose.translation,
                package_geometry.pose.rotation,
                package_geometry.pose.translation,
                length_scale=length_scale,
                source_size=coil_size,
                target_size=(
                    package_geometry.characteristic_length
                ),
            )

    package_pair = np.zeros(
        (
            n_packages,
            n_packages,
            TENSOR_PAIR_FEATURE_DIM,
        ),
        dtype=float,
    )
    for i, source in enumerate(
        scene.packages
    ):
        source_geometry = source.geometry
        for j, target in enumerate(
            scene.packages
        ):
            if i == j:
                package_pair[
                    i,
                    j,
                    3:12,
                ] = np.eye(
                    3
                ).ravel()
                continue
            target_geometry = target.geometry
            package_pair[
                i,
                j,
            ] = _relative_pose_features(
                source_geometry.pose.rotation,
                source_geometry.pose.translation,
                target_geometry.pose.rotation,
                target_geometry.pose.translation,
                length_scale=length_scale,
                source_size=(
                    source_geometry.characteristic_length
                ),
                target_size=(
                    target_geometry.characteristic_length
                ),
            )

    return EncodedTensorHybridScene(
        coil_encoded,
        package_features,
        coil_package,
        package_pair,
    )
