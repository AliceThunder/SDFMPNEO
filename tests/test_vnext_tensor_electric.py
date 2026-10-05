import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DielectricCoupledReferenceArtifact,
    HomogeneousMedium,
    IsotropicMaterial,
    MQSConfig,
    PackageObject,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    TensorElectricMaterial,
    haar_rotation,
    scene_from_dict,
    scene_to_dict,
)


CFG = MQSConfig(
    segments_per_turn=6,
    min_segments=8,
    section_degree=0,
    radial_order=3,
    angular_order=12,
    line_order=2,
)


def _coil(
    pose=None,
):
    geometry = SuperellipseSpiral(
        0.018,
        0.015,
        0.70,
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
        exponent_xy=2.3,
        exponent_z=2.1,
    )
    if pose is not None:
        geometry = geometry.transformed(
            pose
        )
    return geometry


def _artifact():
    return DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
        magnetic_volume_axial_order=3,
        magnetic_volume_radial_order=2,
        magnetic_volume_azimuthal_order=8,
        maximum_raw_magnetic_reciprocity_defect=0.20,
    )


def test_tensor_electric_isotropic_limit_matches_scalar_package_reference():
    epsilon_r = 4.0
    sigma = 2.0e-4
    geometry = _package_geometry()
    scalar_scene = Scene(
        (
            _coil(),
        ),
        HomogeneousMedium(),
        (
            PackageObject(
                geometry,
                IsotropicMaterial(
                    relative_permittivity=epsilon_r,
                    conductivity=sigma,
                ),
                "scalar",
            ),
        ),
    )
    tensor_scene = Scene(
        scalar_scene.coils,
        scalar_scene.medium,
        (
            PackageObject(
                geometry,
                TensorElectricMaterial(
                    relative_permittivity_tensor=(
                        epsilon_r
                        * np.eye(
                            3
                        )
                    ),
                    conductivity_tensor=(
                        sigma
                        * np.eye(
                            3
                        )
                    ),
                ),
                "tensor",
            ),
        ),
    )
    artifact = _artifact()
    frequency = 95_000.0
    scalar = artifact.solve(
        scalar_scene,
        frequency,
    )
    tensor = artifact.solve(
        tensor_scene,
        frequency,
    )

    assert (
        tensor.tensor_electric_transmission
        is not None
    )
    assert (
        tensor.surface_residual
        < 2e-5
    )
    assert (
        tensor.raw_potential_reciprocity_defect
        < 0.15
    )
    assert np.allclose(
        tensor.impedance,
        scalar.impedance,
        rtol=8e-3,
        atol=8e-7,
    )
    assert np.allclose(
        tensor.dielectric_dissipation_matrix,
        scalar.dielectric_dissipation_matrix,
        rtol=1.2e-2,
        atol=2e-8,
    )


def test_tensor_electric_package_is_common_se3_invariant():
    epsilon_local = np.diag(
        [
            2.2,
            4.1,
            7.0,
        ]
    )
    sigma_local = np.diag(
        [
            6.0e-5,
            1.7e-4,
            4.2e-4,
        ]
    )
    material = TensorElectricMaterial(
        relative_permittivity_tensor=(
            epsilon_local
        ),
        conductivity_tensor=(
            sigma_local
        ),
    )
    scene = Scene(
        (
            _coil(),
        ),
        HomogeneousMedium(),
        (
            PackageObject(
                _package_geometry(),
                material,
                "tensor",
            ),
        ),
    )
    artifact = _artifact()
    frequency = 120_000.0
    reference = artifact.solve(
        scene,
        frequency,
    )

    rng = np.random.default_rng(
        14021
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.asarray(
            [
                0.14,
                -0.10,
                0.22,
            ]
        ),
    )
    moved_scene = Scene(
        (
            _coil(
                common
            ),
        ),
        scene.medium,
        (
            PackageObject(
                _package_geometry(
                    common
                ),
                material,
                "tensor",
            ),
        ),
    )
    actual = artifact.solve(
        moved_scene,
        frequency,
    )

    assert np.allclose(
        actual.impedance,
        reference.impedance,
        rtol=3e-5,
        atol=3e-7,
    )
    assert np.allclose(
        actual.dielectric_dissipation_matrix,
        reference.dielectric_dissipation_matrix,
        rtol=5e-5,
        atol=5e-9,
    )


def test_tensor_electric_spatial_package_loss_is_psd_and_closes():
    material = TensorElectricMaterial(
        relative_permittivity_tensor=np.diag(
            [
                2.5,
                3.8,
                5.6,
            ]
        ),
        conductivity_tensor=np.diag(
            [
                8.0e-5,
                2.0e-4,
                5.0e-4,
            ]
        ),
    )
    scene = Scene(
        (
            _coil(),
        ),
        HomogeneousMedium(),
        (
            PackageObject(
                _package_geometry(),
                material,
                "tensor",
            ),
        ),
    )
    spatial = _artifact().prepare_spatial(
        scene,
        110_000.0,
        volume_axial_order=4,
        volume_radial_order=3,
        volume_azimuthal_order=12,
        maximum_raw_closure_error=5.0,
        normalized_closure_tolerance=3e-5,
    )
    point = np.asarray(
        [
            0.0,
            0.0,
            0.004,
        ]
    )
    local = spatial.raw_package_dissipation_matrices(
        0,
        point,
    )
    assert np.all(
        np.isfinite(
            local
        )
    )
    assert np.allclose(
        local,
        local.conj().T,
        atol=1e-12,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                local
            )
        )
        >= -1e-12
    )
    assert (
        spatial.normalization_closure_error
        < 3e-5
    )


def test_tensor_electric_package_round_trips_through_scene_serialization():
    epsilon = np.diag(
        [
            2.0,
            3.5,
            6.0,
        ]
    )
    sigma = np.diag(
        [
            5.0e-5,
            1.0e-4,
            4.0e-4,
        ]
    )
    scene = Scene(
        (
            _coil(),
        ),
        HomogeneousMedium(),
        (
            PackageObject(
                _package_geometry(),
                TensorElectricMaterial(
                    relative_permittivity_tensor=(
                        epsilon
                    ),
                    conductivity_tensor=(
                        sigma
                    ),
                    relative_permeability=1.2,
                ),
                "tensor",
            ),
        ),
    )
    restored = scene_from_dict(
        scene_to_dict(
            scene
        )
    )
    material = restored.packages[
        0
    ].material
    assert isinstance(
        material,
        TensorElectricMaterial,
    )
    assert np.allclose(
        material.relative_permittivity_tensor,
        epsilon,
    )
    assert np.allclose(
        material.conductivity_tensor,
        sigma,
    )
    assert np.isclose(
        material.relative_permeability,
        1.2,
    )
