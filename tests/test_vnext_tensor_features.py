import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    PackageObject,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    TensorElectricMaterial,
    TENSOR_COIL_FEATURE_DIM,
    TENSOR_PACKAGE_FEATURE_DIM,
    encode_tensor_hybrid_scene_invariant,
    haar_rotation,
)


def _coil(
    pose=None,
):
    geometry = SuperellipseSpiral(
        0.018,
        0.015,
        0.65,
        0.001,
        0.001,
        conductor_width=8.0e-4,
        conductor_thickness=6.0e-4,
    )
    if pose is not None:
        geometry = geometry.transformed(
            pose
        )
    return CoilObject(
        geometry,
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )


def _package_geometry(
    pose=None,
):
    geometry = SuperquadricPackageGeometry(
        np.asarray(
            [
                0.029,
                0.026,
                0.006,
            ]
        ),
        exponent_xy=2.4,
        exponent_z=2.2,
    )
    if pose is not None:
        geometry = geometry.transformed(
            pose
        )
    return geometry


def test_tensor_feature_schema_preserves_full_local_tensor_information():
    epsilon = np.asarray(
        [
            [2.5, 0.4, 0.1],
            [0.4, 4.0, 0.2],
            [0.1, 0.2, 6.0],
        ],
        dtype=float,
    )
    sigma = np.asarray(
        [
            [1.5e-4, 2.0e-5, 1.0e-5],
            [2.0e-5, 3.0e-4, 3.0e-5],
            [1.0e-5, 3.0e-5, 5.0e-4],
        ],
        dtype=float,
    )
    scene = Scene(
        (
            _coil(),
        ),
        HomogeneousMedium(
            relative_permittivity=1.7,
            conductivity=8.0e-5,
        ),
        (
            PackageObject(
                _package_geometry(),
                TensorElectricMaterial(
                    relative_permittivity_tensor=epsilon,
                    conductivity_tensor=sigma,
                    relative_permeability=1.1,
                ),
                "tensor",
            ),
        ),
    )
    encoded = encode_tensor_hybrid_scene_invariant(
        scene,
        90_000.0,
    )

    assert encoded.coil.node_features.shape == (
        1,
        TENSOR_COIL_FEATURE_DIM,
    )
    assert encoded.package_features.shape == (
        1,
        TENSOR_PACKAGE_FEATURE_DIM,
    )
    assert np.all(
        np.isfinite(
            encoded.package_features
        )
    )
    # Off-diagonal tensor information must survive the encoding rather than
    # collapsing to a scalar mean.  The local epsilon block begins at index 9.
    epsilon_block = encoded.package_features[
        0,
        9:15,
    ]
    sigma_block = encoded.package_features[
        0,
        15:21,
    ]
    assert (
        np.linalg.norm(
            epsilon_block[
                3:
            ]
        )
        > 0.0
    )
    assert (
        np.linalg.norm(
            sigma_block[
                3:
            ]
        )
        > 0.0
    )


def test_tensor_hybrid_features_are_common_se3_invariant():
    background_epsilon = np.asarray(
        [
            [1.9, 0.2, 0.0],
            [0.2, 2.8, 0.1],
            [0.0, 0.1, 4.1],
        ],
        dtype=float,
    )
    background_sigma = np.asarray(
        [
            [8.0e-5, 1.0e-5, 0.0],
            [1.0e-5, 1.8e-4, 2.0e-5],
            [0.0, 2.0e-5, 3.5e-4],
        ],
        dtype=float,
    )
    package_epsilon = np.diag(
        [
            2.2,
            4.0,
            7.0,
        ]
    )
    package_sigma = np.diag(
        [
            5.0e-5,
            1.5e-4,
            4.0e-4,
        ]
    )
    package_material = TensorElectricMaterial(
        relative_permittivity_tensor=(
            package_epsilon
        ),
        conductivity_tensor=(
            package_sigma
        ),
        relative_permeability=1.15,
    )
    scene = Scene(
        (
            _coil(),
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=(
                background_epsilon
            ),
            conductivity_tensor=(
                background_sigma
            ),
        ),
        (
            PackageObject(
                _package_geometry(),
                package_material,
                "tensor",
            ),
        ),
    )
    reference = encode_tensor_hybrid_scene_invariant(
        scene,
        120_000.0,
    )

    rng = np.random.default_rng(
        55103
    )
    rotation = haar_rotation(
        rng
    )
    common = RigidPose(
        rotation,
        np.asarray(
            [
                0.12,
                -0.07,
                0.21,
            ]
        ),
    )
    moved = Scene(
        (
            _coil(
                common
            ),
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=(
                rotation
                @ background_epsilon
                @ rotation.T
            ),
            conductivity_tensor=(
                rotation
                @ background_sigma
                @ rotation.T
            ),
        ),
        (
            PackageObject(
                _package_geometry(
                    common
                ),
                package_material,
                "tensor",
            ),
        ),
    )
    actual = encode_tensor_hybrid_scene_invariant(
        moved,
        120_000.0,
    )

    assert np.allclose(
        actual.coil.node_features,
        reference.coil.node_features,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.allclose(
        actual.coil.pair_features,
        reference.coil.pair_features,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.allclose(
        actual.package_features,
        reference.package_features,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.allclose(
        actual.coil_package_features,
        reference.coil_package_features,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.allclose(
        actual.package_pair_features,
        reference.package_pair_features,
        rtol=2e-12,
        atol=2e-12,
    )


def test_tensor_encoder_accepts_scalar_materials_as_isotropic_tensors():
    scene = Scene(
        (
            _coil(),
        ),
        HomogeneousMedium(
            relative_permittivity=2.0,
            conductivity=1.0e-4,
        ),
        (
            PackageObject(
                _package_geometry(),
                IsotropicMaterial(
                    relative_permittivity=4.0,
                    conductivity=3.0e-4,
                ),
                "scalar",
            ),
        ),
    )
    encoded = encode_tensor_hybrid_scene_invariant(
        scene,
        80_000.0,
    )
    epsilon_block = encoded.package_features[
        0,
        9:15,
    ]
    sigma_block = encoded.package_features[
        0,
        15:21,
    ]
    assert np.allclose(
        epsilon_block[
            3:
        ],
        0.0,
        atol=1e-14,
    )
    assert np.allclose(
        sigma_block[
            3:
        ],
        0.0,
        atol=1e-14,
    )
