from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .features import (
    EncodedScene,
    encode_scene_invariant,
)
from .scene import (
    HomogeneousMedium,
    Scene,
    TensorElectricMaterial,
)


def _symmetric_features(
    matrix,
    *,
    diagonal_scale: float | None = None,
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
    diagonal = np.clip(
        np.diag(
            matrix
        ),
        0.0,
        None,
    )
    if diagonal_scale is None:
        diagonal_features = np.log(
            np.maximum(
                diagonal,
                1e-12,
            )
        )
    else:
        diagonal_features = np.log1p(
            diagonal
            / float(
                diagonal_scale
            )
        )
    correlations = []
    for i, j in (
        (0, 1),
        (0, 2),
        (1, 2),
    ):
        denominator = np.sqrt(
            max(
                diagonal[
                    i
                ]
                * diagonal[
                    j
                ],
                1e-30,
            )
        )
        correlations.append(
            float(
                np.clip(
                    matrix[
                        i,
                        j
                    ]
                    / denominator,
                    -1.0,
                    1.0,
                )
            )
        )
    return np.concatenate(
        (
            diagonal_features,
            np.asarray(
                correlations,
                dtype=float,
            ),
        )
    )


def _effective_material_tensors(
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
        conductivity = np.asarray(
            material.loss_conductivity_tensor(
                frequency_hz
            ),
            dtype=float,
        )
        return (
            epsilon,
            conductivity,
        )
    relative = material.relative_permittivity_at(
        frequency_hz
    )
    epsilon = max(
        float(
            np.real(
                relative
            )
        ),
        1e-12,
    )
    conductivity = max(
        float(
            material.loss_conductivity(
                frequency_hz
            )
        ),
        0.0,
    )
    return (
        epsilon
        * np.eye(
            3
        ),
        conductivity
        * np.eye(
            3
        ),
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
            np.asarray(
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


@dataclass(frozen=True)
class EncodedTensorHybridScene:
    coil: EncodedScene
    package_features: np.ndarray
    coil_package_features: np.ndarray
    package_pair_features: np.ndarray
    physical_package_count: int

    def __post_init__(
        self,
    ):
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
        n_coils = int(
            self.coil.node_features.shape[
                0
            ]
        )
        n_materials = int(
            package.shape[
                0
            ]
        )
        if (
            package.ndim != 2
            or package.shape[
                1
            ] != 21
        ):
            raise ValueError(
                "tensor material features must have shape (n_materials,21)"
            )
        if coil_package.shape != (
            n_coils,
            n_materials,
            15,
        ):
            raise ValueError(
                "tensor coil-material features have incompatible shape"
            )
        if package_pair.shape != (
            n_materials,
            n_materials,
            15,
        ):
            raise ValueError(
                "tensor material-pair features have incompatible shape"
            )
        if (
            self.physical_package_count
            != n_materials
            - 1
        ):
            raise ValueError(
                "tensor material graph must contain exactly one background node"
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
            self.physical_package_count
        )

    @property
    def n_material_nodes(
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
    root = scene.coils[
        0
    ].geometry
    root_rotation = np.asarray(
        root.pose.rotation,
        dtype=float,
    )
    root_translation = np.asarray(
        root.pose.translation,
        dtype=float,
    )
    background_epsilon, background_sigma = (
        _effective_material_tensors(
            scene.medium,
            frequency_hz,
        )
    )
    background_epsilon_root = (
        root_rotation.T
        @ background_epsilon
        @ root_rotation
    )
    background_sigma_root = (
        root_rotation.T
        @ background_sigma
        @ root_rotation
    )
    epsilon_summary = float(
        np.linalg.det(
            background_epsilon
        ) ** (
            1.0
            / 3.0
        )
    )
    sigma_summary = float(
        np.trace(
            background_sigma
        )
        / 3.0
    )
    surrogate_medium = HomogeneousMedium(
        relative_permittivity=max(
            epsilon_summary,
            1e-12,
        ),
        relative_permeability=float(
            scene.medium.relative_permeability
        ),
        conductivity=max(
            sigma_summary,
            0.0,
        ),
    )
    coil = encode_scene_invariant(
        Scene(
            scene.coils,
            surrogate_medium,
            (),
        ),
        frequency_hz,
    )
    length_scale = float(
        coil.length_scale
    )

    material_rows = []
    material_rotations = []
    material_translations = []
    material_sizes = []

    def append_material(
        *,
        background: bool,
        half_extents,
        exponent_xy: float,
        exponent_z: float,
        volume: float,
        characteristic_length: float,
        relative_permeability: float,
        epsilon_tensor,
        conductivity_tensor,
        rotation,
        translation,
        size: float,
    ):
        material_rows.append(
            np.concatenate(
                (
                    np.asarray(
                        [
                            1.0
                            if background
                            else 0.0,
                            half_extents[
                                0
                            ]
                            / length_scale,
                            half_extents[
                                1
                            ]
                            / length_scale,
                            half_extents[
                                2
                            ]
                            / length_scale,
                            float(
                                exponent_xy
                            ),
                            float(
                                exponent_z
                            ),
                            np.log1p(
                                max(
                                    float(
                                        volume
                                    ),
                                    0.0,
                                )
                                / (
                                    length_scale**3
                                )
                            ),
                            np.log1p(
                                max(
                                    float(
                                        characteristic_length
                                    ),
                                    0.0,
                                )
                                / length_scale
                            ),
                            float(
                                relative_permeability
                            ),
                        ],
                        dtype=float,
                    ),
                    _symmetric_features(
                        epsilon_tensor
                    ),
                    _symmetric_features(
                        conductivity_tensor,
                        diagonal_scale=1e-6,
                    ),
                )
            )
        )
        material_rotations.append(
            np.asarray(
                rotation,
                dtype=float,
            )
        )
        material_translations.append(
            np.asarray(
                translation,
                dtype=float,
            )
        )
        material_sizes.append(
            float(
                size
            )
        )

    append_material(
        background=True,
        half_extents=np.zeros(
            3,
            dtype=float,
        ),
        exponent_xy=0.0,
        exponent_z=0.0,
        volume=0.0,
        characteristic_length=0.0,
        relative_permeability=(
            scene.medium.relative_permeability
        ),
        epsilon_tensor=(
            background_epsilon_root
        ),
        conductivity_tensor=(
            background_sigma_root
        ),
        rotation=root_rotation,
        translation=root_translation,
        size=length_scale,
    )

    for package in scene.packages:
        geometry = package.geometry
        epsilon, conductivity = (
            _effective_material_tensors(
                package.material,
                frequency_hz,
            )
        )
        if not isinstance(
            package.material,
            TensorElectricMaterial,
        ):
            epsilon_local = epsilon
            conductivity_local = conductivity
        else:
            epsilon_local = epsilon
            conductivity_local = conductivity
        append_material(
            background=False,
            half_extents=np.asarray(
                geometry.half_extents,
                dtype=float,
            ),
            exponent_xy=(
                geometry.exponent_xy
            ),
            exponent_z=(
                geometry.exponent_z
            ),
            volume=geometry.volume,
            characteristic_length=(
                geometry.characteristic_length
            ),
            relative_permeability=(
                package.material.relative_permeability
            ),
            epsilon_tensor=epsilon_local,
            conductivity_tensor=(
                conductivity_local
            ),
            rotation=(
                geometry.pose.rotation
            ),
            translation=(
                geometry.pose.translation
            ),
            size=(
                geometry.characteristic_length
            ),
        )

    package_features = np.asarray(
        material_rows,
        dtype=float,
    )
    n_coils = len(
        scene.coils
    )
    n_materials = len(
        material_rows
    )
    coil_material = np.zeros(
        (
            n_coils,
            n_materials,
            15,
        ),
        dtype=float,
    )
    for coil_index, coil_object in enumerate(
        scene.coils
    ):
        geometry = coil_object.geometry
        coil_size = float(
            np.sqrt(
                geometry.outer_a
                * geometry.outer_b
            )
        )
        for material_index in range(
            n_materials
        ):
            coil_material[
                coil_index,
                material_index,
            ] = _relative_pose_features(
                geometry.pose.rotation,
                geometry.pose.translation,
                material_rotations[
                    material_index
                ],
                material_translations[
                    material_index
                ],
                length_scale=length_scale,
                source_size=coil_size,
                target_size=(
                    material_sizes[
                        material_index
                    ]
                ),
            )

    material_pair = np.zeros(
        (
            n_materials,
            n_materials,
            15,
        ),
        dtype=float,
    )
    for i in range(
        n_materials
    ):
        for j in range(
            n_materials
        ):
            if i == j:
                material_pair[
                    i,
                    j,
                    3:12,
                ] = np.eye(
                    3
                ).ravel()
                continue
            material_pair[
                i,
                j,
            ] = _relative_pose_features(
                material_rotations[
                    i
                ],
                material_translations[
                    i
                ],
                material_rotations[
                    j
                ],
                material_translations[
                    j
                ],
                length_scale=length_scale,
                source_size=(
                    material_sizes[
                        i
                    ]
                ),
                target_size=(
                    material_sizes[
                        j
                    ]
                ),
            )

    return EncodedTensorHybridScene(
        coil=coil,
        package_features=(
            package_features
        ),
        coil_package_features=(
            coil_material
        ),
        package_pair_features=(
            material_pair
        ),
        physical_package_count=len(
            scene.packages
        ),
    )
