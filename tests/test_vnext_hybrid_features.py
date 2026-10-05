import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DebyeMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    PackageObject,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    encode_hybrid_scene_invariant,
    haar_rotation,
)


def _scene():
    coil = CoilObject(
        SuperellipseSpiral(
            0.022,
            0.018,
            0.8,
            0.0012,
            0.0010,
            exponent=3.0,
            conductor_width=9e-4,
            conductor_thickness=7e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )
    first = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.032, 0.025, 0.008]
            ),
            exponent_xy=3.0,
            exponent_z=4.0,
            pose=RigidPose.from_axis_angle(
                (0.0, 1.0, 0.0),
                0.2,
                translation=(
                    0.003,
                    -0.002,
                    0.001,
                ),
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=3.2,
            conductivity=0.002,
        ),
        "a",
    )
    second = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.014, 0.012, 0.010]
            ),
            exponent_xy=2.5,
            exponent_z=3.0,
            pose=RigidPose.from_axis_angle(
                (1.0, 0.0, 0.0),
                -0.35,
                translation=(
                    0.04,
                    0.01,
                    0.015,
                ),
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=2.1,
            conductivity=0.0,
        ),
        "b",
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
        (first, second),
    )


def test_hybrid_encoding_is_common_se3_invariant():
    scene = _scene()
    reference = (
        encode_hybrid_scene_invariant(
            scene,
            85_000.0,
        )
    )
    rng = np.random.default_rng(
        401
    )
    pose = RigidPose(
        haar_rotation(rng),
        np.array(
            [0.2, -0.1, 0.35]
        ),
    )
    moved = Scene(
        tuple(
            CoilObject(
                coil.geometry.transformed(
                    pose
                ),
                coil.material,
                coil.name,
            )
            for coil in scene.coils
        ),
        scene.medium,
        tuple(
            PackageObject(
                package.geometry.transformed(
                    pose
                ),
                package.material,
                package.name,
            )
            for package in scene.packages
        ),
    )
    encoded = (
        encode_hybrid_scene_invariant(
            moved,
            85_000.0,
        )
    )
    assert np.allclose(
        encoded.coil.node_features,
        reference.coil.node_features,
        rtol=0,
        atol=2e-13,
    )
    assert np.allclose(
        encoded.coil.pair_features,
        reference.coil.pair_features,
        rtol=0,
        atol=2e-13,
    )
    assert np.allclose(
        encoded.package_features,
        reference.package_features,
        rtol=0,
        atol=2e-13,
    )
    assert np.allclose(
        encoded.coil_package_features,
        reference.coil_package_features,
        rtol=0,
        atol=4e-13,
    )
    assert np.allclose(
        encoded.package_pair_features,
        reference.package_pair_features,
        rtol=0,
        atol=4e-13,
    )


def test_hybrid_encoding_is_package_permutation_equivariant():
    scene = _scene()
    reference = (
        encode_hybrid_scene_invariant(
            scene,
            85_000.0,
        )
    )
    swapped = Scene(
        scene.coils,
        scene.medium,
        (
            scene.packages[1],
            scene.packages[0],
        ),
    )
    encoded = (
        encode_hybrid_scene_invariant(
            swapped,
            85_000.0,
        )
    )
    permutation = np.array(
        [1, 0],
        dtype=int,
    )
    assert np.allclose(
        encoded.coil.node_features,
        reference.coil.node_features,
    )
    assert np.allclose(
        encoded.package_features,
        reference.package_features[
            permutation
        ],
    )
    assert np.allclose(
        encoded.coil_package_features,
        reference.coil_package_features[
            :,
            permutation,
        ],
    )
    assert np.allclose(
        encoded.package_pair_features,
        reference.package_pair_features[
            np.ix_(
                permutation,
                permutation,
                np.arange(
                    reference.package_pair_features.shape[
                        2
                    ]
                ),
            )
        ],
    )


def test_hybrid_encoding_without_packages_preserves_conductor_encoding():
    scene = Scene(
        _scene().coils,
        HomogeneousMedium(),
    )
    encoded = (
        encode_hybrid_scene_invariant(
            scene,
            85_000.0,
        )
    )
    assert encoded.n_packages == 0
    assert encoded.package_features.shape == (
        0,
        13,
    )
    assert encoded.coil_package_features.shape == (
        1,
        0,
        15,
    )
    assert encoded.package_pair_features.shape == (
        0,
        0,
        15,
    )


def test_hybrid_encoding_tracks_frequency_effective_debye_background():
    base = _scene()
    medium = DebyeMaterial(
        relative_permittivity_static=24.0,
        relative_permittivity_infinite=4.0,
        relaxation_time=3.0e-6,
        conductivity=0.0,
    )
    scene = Scene(
        base.coils,
        medium,
        base.packages,
    )
    low = encode_hybrid_scene_invariant(
        scene,
        10_000.0,
    )
    high = encode_hybrid_scene_invariant(
        scene,
        500_000.0,
    )

    # Background features keep the historical dimensionality but now carry
    # effective Re(epsilon_r) and effective loss conductivity at frequency.
    low_node = low.coil.node_features[
        0
    ]
    high_node = high.coil.node_features[
        0
    ]
    assert np.all(
        np.isfinite(
            low_node
        )
    )
    assert np.all(
        np.isfinite(
            high_node
        )
    )
    assert not np.isclose(
        low_node[
            13
        ],
        high_node[
            13
        ],
    )
    assert not np.isclose(
        low_node[
            15
        ],
        high_node[
            15
        ],
    )

    # Package/background electric contrast must use the same frequency-aware
    # background epsilon rather than the Debye static metadata value.
    assert not np.allclose(
        low.package_features[
            :,
            8
        ],
        high.package_features[
            :,
            8
        ],
    )
