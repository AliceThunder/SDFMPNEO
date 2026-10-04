import numpy as np

from sdfmpneo_vnext import (
    AnisotropicThermalMedium,
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    HomogeneousThermalMedium,
    RadialTensorThermalMaterialProfile,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    compile_graded_superquadric_regions,
    scene_thermal_package_media,
)


def _rotation_z(
    angle,
):
    cosine = np.cos(
        angle
    )
    sine = np.sin(
        angle
    )
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _coil():
    return CoilObject(
        SuperellipseSpiral(
            0.014,
            0.012,
            0.55,
            0.001,
            0.001,
            conductor_width=7.0e-4,
            conductor_thickness=5.0e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )


def _profile():
    return RadialTensorThermalMaterialProfile(
        normalized_radius=(
            0.0,
            1.0,
        ),
        relative_permittivity=(
            2.0,
            4.0,
        ),
        relative_permeability=(
            1.0,
            1.0,
        ),
        conductivity=(
            0.0,
            2.0e-4,
        ),
        thermal_conductivity_tensors=(
            np.diag(
                [
                    0.16,
                    0.36,
                    0.64,
                ]
            ),
            np.diag(
                [
                    0.64,
                    1.44,
                    2.56,
                ]
            ),
        ),
        density=(
            950.0,
            1250.0,
        ),
        heat_capacity=(
            1200.0,
            1800.0,
        ),
    )


def test_radial_tensor_thermal_profile_log_euclidean_midpoint_is_spd():
    profile = _profile()
    tensor = profile.thermal_conductivity_tensor_at(
        0.5
    )

    expected = np.diag(
        [
            np.sqrt(
                0.16
                * 0.64
            ),
            np.sqrt(
                0.36
                * 1.44
            ),
            np.sqrt(
                0.64
                * 2.56
            ),
        ]
    )
    assert np.allclose(
        tensor,
        expected,
        rtol=2e-13,
        atol=2e-14,
    )
    assert np.allclose(
        tensor,
        tensor.T,
        atol=1e-14,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                tensor
            )
        )
        > 0.0
    )

    material = profile.material_at(
        0.5
    )
    assert np.allclose(
        material.thermal_conductivity_tensor,
        expected,
    )
    assert np.isclose(
        material.relative_permittivity,
        3.0,
    )
    assert np.isclose(
        material.conductivity,
        1.0e-4,
    )


def test_graded_tensor_thermal_layers_rotate_local_tensor_into_world_frame():
    rotation = _rotation_z(
        0.47
    )
    pose = RigidPose(
        rotation,
        np.asarray(
            [
                0.03,
                -0.02,
                0.01,
            ]
        ),
    )
    outer = SuperquadricPackageGeometry(
        np.asarray(
            [
                0.032,
                0.028,
                0.007,
            ]
        ),
        exponent_xy=2.4,
        exponent_z=2.2,
        pose=pose,
    )
    coil = _coil()
    moved_coil = CoilObject(
        coil.geometry.transformed(
            pose
        ),
        coil.material,
        coil.name,
    )
    layers = compile_graded_superquadric_regions(
        outer,
        _profile(),
        shell_count=3,
        enclosed_coils=(
            moved_coil,
        ),
        clearance_fraction=0.02,
    )
    scene = Scene(
        (
            moved_coil,
        ),
        HomogeneousMedium(),
        layers,
    )
    background = HomogeneousThermalMedium(
        conductivity=0.55,
        density=1000.0,
        heat_capacity=4200.0,
        ambient_temperature=293.15,
    )
    resolved = dict(
        scene_thermal_package_media(
            scene,
            background,
        )
    )

    assert len(
        resolved
    ) == len(
        layers
    )
    for index, package in enumerate(
        layers
    ):
        medium = resolved[
            index
        ]
        assert isinstance(
            medium,
            AnisotropicThermalMedium,
        )
        local = np.asarray(
            package.material.thermal_conductivity_tensor,
            dtype=float,
        )
        expected = (
            rotation
            @ local
            @ rotation.T
        )
        assert np.allclose(
            medium.conductivity_tensor,
            expected,
            rtol=2e-12,
            atol=2e-13,
        )
        assert np.isclose(
            medium.density,
            package.material.density,
        )
        assert np.isclose(
            medium.heat_capacity,
            package.material.heat_capacity,
        )
