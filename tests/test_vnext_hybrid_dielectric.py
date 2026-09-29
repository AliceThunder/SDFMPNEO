import numpy as np
import pytest

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DebyeMaterial,
    DenseMixedConductorTeacher,
    DielectricCoupledReferenceArtifact,
    HomogeneousMedium,
    HomogeneousThermalMedium,
    HybridSceneSamplerConfig,
    IsotropicMaterial,
    MQSConfig,
    PackageObject,
    RadialIsotropicMaterialProfile,
    RigidPose,
    Scene,
    StructuredPortPrediction,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    TabulatedMaterial,
    compile_graded_superquadric_regions,
    graded_material_convergence,
    graded_electrothermal_convergence,
    haar_rotation,
    sample_hybrid_package_scene,
)


CFG = MQSConfig(
    segments_per_turn=8,
    min_segments=10,
    section_degree=0,
    radial_order=3,
    angular_order=12,
    line_order=2,
)


def _coil():
    return CoilObject(
        SuperellipseSpiral(
            0.015,
            0.013,
            0.65,
            0.001,
            0.001,
            exponent=3.0,
            conductor_width=8e-4,
            conductor_thickness=6e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )


def _package(material):
    return PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.025, 0.022, 0.006]
            ),
            exponent_xy=2.0,
            exponent_z=2.0,
        ),
        material,
        "package",
    )


def _coupled(material):
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        (_package(material),),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
    )
    return (
        scene,
        artifact.solve(
            scene,
            80_000.0,
        ),
    )


def test_invisible_package_reduces_to_conductor_only_mixed_reference():
    scene, coupled = _coupled(
        IsotropicMaterial(
            relative_permittivity=1.0,
        )
    )
    bare_scene = Scene(
        scene.coils,
        scene.medium,
    )
    bare = DenseMixedConductorTeacher(
        bare_scene,
        80_000.0,
        CFG,
    ).solve()
    assert np.allclose(
        coupled.impedance,
        bare.impedance,
        rtol=2e-10,
        atol=2e-11,
    )
    assert np.allclose(
        coupled.surface_density_transfer,
        0.0,
        rtol=0,
        atol=0,
    )
    assert (
        coupled.power_closure_error
        < 1e-10
    )
    assert np.allclose(
        coupled.magnetic_inductance_correction,
        0.0,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        coupled.magnetic_surface_residual
        == 0.0
    )


def test_lossless_dielectric_changes_reactive_response_without_dielectric_loss():
    scene, coupled = _coupled(
        IsotropicMaterial(
            relative_permittivity=3.2,
        )
    )
    bare = DenseMixedConductorTeacher(
        Scene(
            scene.coils,
            scene.medium,
        ),
        80_000.0,
        CFG,
    ).solve()
    assert (
        abs(
            coupled.impedance[0, 0].imag
            - bare.impedance[0, 0].imag
        )
        > 1e-10
    )
    assert np.allclose(
        coupled.dielectric_dissipation_matrix,
        0.0,
        atol=2e-11,
        rtol=0,
    )
    assert (
        coupled.prediction.reciprocity_defect()
        < 1e-12
    )
    assert (
        coupled.power_closure_error
        < 2e-8
    )


def test_lossy_dielectric_has_independent_psd_loss_channel_and_power_closure():
    _, coupled = _coupled(
        IsotropicMaterial(
            relative_permittivity=3.0,
            conductivity=0.005,
        )
    )
    dielectric = (
        coupled.dielectric_dissipation_matrix
    )
    assert np.allclose(
        dielectric,
        dielectric.conj().T,
        atol=1e-11,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                dielectric
            )
        )
        >= -1e-10
    )
    powers = (
        coupled.channel_power(
            np.array(
                [1.0 + 0.2j]
            )
        )
    )
    assert powers[-1] > 0.0
    assert (
        coupled.power_closure_error
        < 2e-6
    )
    assert (
        coupled.normalized_residual
        < 1e-9
    )


def test_coupled_dielectric_response_is_common_se3_invariant():
    material = IsotropicMaterial(
        relative_permittivity=2.8,
        conductivity=0.002,
    )
    scene, result = _coupled(
        material
    )
    rng = np.random.default_rng(
        311
    )
    pose = RigidPose(
        haar_rotation(rng),
        np.array(
            [0.11, -0.07, 0.18]
        ),
    )
    moved = Scene(
        (
            CoilObject(
                scene.coils[
                    0
                ].geometry.transformed(
                    pose
                ),
                scene.coils[
                    0
                ].material,
                "coil",
            ),
        ),
        scene.medium,
        (
            PackageObject(
                scene.packages[
                    0
                ].geometry.transformed(
                    pose
                ),
                material,
                "package",
            ),
        ),
    )
    moved_result = (
        DielectricCoupledReferenceArtifact(
            config=CFG,
            surface_vertical_order=8,
            surface_azimuthal_order=16,
        ).solve(
            moved,
            80_000.0,
        )
    )
    assert np.allclose(
        moved_result.impedance,
        result.impedance,
        rtol=2e-9,
        atol=2e-10,
    )
    assert np.allclose(
        moved_result.dielectric_dissipation_matrix,
        result.dielectric_dissipation_matrix,
        rtol=2e-8,
        atol=2e-10,
    )


def test_structured_prediction_allows_more_loss_channels_than_ports():
    prediction = StructuredPortPrediction(
        np.array(
            [[2.0 + 3.0j]]
        ),
        np.array(
            [
                [[1.2 + 0j]],
                [[0.8 + 0j]],
            ]
        ),
    )
    assert prediction.n_channels == 2
    assert (
        prediction.power_closure_error()
        < 1e-14
    )
    assert np.allclose(
        prediction.channel_power(
            np.array(
                [2.0 + 0j]
            )
        ),
        [2.4, 1.6],
    )
    with pytest.raises(
        ValueError,
        match="channel_power",
    ):
        prediction.coil_power(
            np.array(
                [1.0 + 0j]
            )
        )



def test_lossy_dielectric_reference_spatial_field_is_psd_and_energy_closed():
    material = IsotropicMaterial(
        relative_permittivity=3.0,
        conductivity=0.005,
    )
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        (_package(material),),
    )
    artifact = (
        DielectricCoupledReferenceArtifact(
            config=CFG,
            surface_vertical_order=8,
            surface_azimuthal_order=16,
        )
    )
    spatial = artifact.prepare_spatial(
        scene,
        80_000.0,
        volume_axial_order=4,
        volume_radial_order=3,
        volume_azimuthal_order=12,
        maximum_raw_closure_error=5.0,
        normalized_closure_tolerance=1e-6,
    )
    matrix = (
        spatial.package_local_dissipation_matrix(
            0,
            np.zeros(3),
        )
    )
    assert np.allclose(
        matrix,
        matrix.conj().T,
        atol=1e-10,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                matrix
            )
        )
        >= -1e-10
    )
    assert (
        spatial.package_local_joule_density(
            0,
            np.zeros(3),
            np.array(
                [1.0 + 0.2j]
            ),
        )
        > 0.0
    )
    assert (
        spatial.normalized_dielectric_closure_error
        < 1e-6
    )
    assert np.allclose(
        np.sum(
            spatial.package_integrated_channels,
            axis=0,
        ),
        spatial.result.dielectric_dissipation_matrix,
        rtol=1e-5,
        atol=1e-10,
    )


def test_lossy_background_invisible_package_reduces_to_bare_lossy_mixed_reference():
    background = HomogeneousMedium(
        relative_permittivity=3.0,
        relative_permeability=1.0,
        conductivity=1e-4,
    )
    material = IsotropicMaterial(
        relative_permittivity=3.0,
        relative_permeability=1.0,
        conductivity=1e-4,
    )
    scene = Scene(
        (_coil(),),
        background,
        (_package(material),),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
    )
    coupled = artifact.solve(
        scene,
        80_000.0,
    )
    bare = DenseMixedConductorTeacher(
        Scene(
            scene.coils,
            background,
        ),
        80_000.0,
        CFG,
    ).solve()
    assert np.allclose(
        coupled.impedance,
        bare.impedance,
        rtol=3e-9,
        atol=3e-10,
    )
    assert np.allclose(
        coupled.surface_density_transfer,
        0.0,
        rtol=0.0,
        atol=1e-13,
    )
    assert (
        coupled.channel_labels[-1]
        == "electric_environment:aggregate"
    )
    assert (
        coupled.power_closure_error
        < 3e-6
    )


def test_lossy_background_with_dielectric_package_has_joint_spatial_environment_loss():
    background = HomogeneousMedium(
        relative_permittivity=2.2,
        relative_permeability=1.0,
        conductivity=1e-4,
    )
    material = IsotropicMaterial(
        relative_permittivity=4.0,
        conductivity=0.003,
    )
    scene = Scene(
        (_coil(),),
        background,
        (_package(material),),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
    )
    result = artifact.solve(
        scene,
        80_000.0,
    )
    environment = (
        result.environment_dissipation_matrix
    )
    assert np.allclose(
        environment,
        environment.conj().T,
        atol=2e-10,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                environment
            )
        )
        >= -2e-9
    )
    assert (
        result.channel_labels[-1]
        == "electric_environment:aggregate"
    )
    assert (
        result.power_closure_error
        < 5e-6
    )
    assert (
        result.normalized_residual
        < 1e-9
    )

    spatial = artifact.prepare_spatial(
        scene,
        80_000.0,
        volume_axial_order=4,
        volume_radial_order=3,
        volume_azimuthal_order=12,
        background_radial_order=10,
        background_angular_order=32,
        maximum_raw_closure_error=5.0,
        normalized_closure_tolerance=1e-6,
    )
    assert (
        spatial.background_channel_index
        == spatial.environment_channel_index
        == spatial.dielectric_channel_index
        == 1
    )
    assert (
        spatial.normalized_dielectric_closure_error
        < 1e-6
    )
    combined = (
        np.sum(
            spatial.package_integrated_channels,
            axis=0,
        )
        + spatial.background_integrated_channel
    )
    assert np.allclose(
        combined,
        environment,
        rtol=2e-5,
        atol=2e-10,
    )

    package_matrix = (
        spatial.package_local_dissipation_matrix(
            0,
            np.zeros(3),
        )
    )
    background_matrix = (
        spatial.background_dissipation_matrices(
            np.array(
                [0.0, 0.0, 0.03]
            )
        )
    )
    for matrix in (
        package_matrix,
        background_matrix,
    ):
        assert np.allclose(
            matrix,
            matrix.conj().T,
            atol=2e-10,
        )
        assert (
            np.min(
                np.linalg.eigvalsh(
                    matrix
                )
            )
            >= -2e-9
        )
    currents = np.array(
        [1.0 + 0.2j]
    )
    assert (
        spatial.package_local_joule_density(
            0,
            np.zeros(3),
            currents,
        )
        > 0.0
    )
    assert (
        spatial.background_joule_density(
            np.array(
                [0.0, 0.0, 0.03]
            ),
            currents,
        )
        > 0.0
    )


def test_debye_background_with_identical_debye_package_is_electromagnetically_invisible():
    frequency = 80_000.0
    medium = DebyeMaterial(
        relative_permittivity_static=18.0,
        relative_permittivity_infinite=4.0,
        relaxation_time=2.0e-6,
        relative_permeability=1.0,
        conductivity=0.0,
    )
    assert (
        medium.loss_conductivity(
            frequency
        )
        > 0.0
    )
    scene = Scene(
        (
            _coil(),
        ),
        medium,
        (
            _package(
                medium
            ),
        ),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
    )
    coupled = artifact.solve(
        scene,
        frequency,
    )
    bare = DenseMixedConductorTeacher(
        Scene(
            scene.coils,
            medium,
        ),
        frequency,
        CFG,
    ).solve()
    assert np.allclose(
        coupled.impedance,
        bare.impedance,
        rtol=5e-9,
        atol=5e-10,
    )
    assert np.allclose(
        coupled.surface_density_transfer,
        0.0,
        rtol=0.0,
        atol=2e-13,
    )
    assert (
        coupled.channel_labels[
            -1
        ]
        == "electric_environment:aggregate"
    )
    assert (
        coupled.power_closure_error
        < 5e-6
    )


def test_reference_rejects_package_surface_intersecting_finite_conductor():
    coil = _coil()
    crossing = PackageObject(
        SuperquadricPackageGeometry(
            np.asarray(
                [
                    coil.geometry.outer_a,
                    coil.geometry.outer_b,
                    0.5
                    * coil.geometry.conductor_thickness,
                ]
            ),
            exponent_xy=2.0,
            exponent_z=2.0,
        ),
        IsotropicMaterial(
            relative_permittivity=3.0,
        ),
        "crossing",
    )
    scene = Scene(
        (
            coil,
        ),
        HomogeneousMedium(),
        (
            crossing,
        ),
    )
    with pytest.raises(
        ValueError,
        match="package surface intersects",
    ):
        DielectricCoupledReferenceArtifact(
            config=CFG,
            surface_vertical_order=8,
            surface_azimuthal_order=16,
        ).solve(
            scene,
            80_000.0,
        )


def test_reference_supports_arbitrary_relative_3d_package_pose_from_training_sampler():
    config = HybridSceneSamplerConfig(
        package_center_offset_fraction_range=(
            0.12,
            0.24,
        ),
        lossless_probability=0.0,
    )
    scene, frequency = sample_hybrid_package_scene(
        np.random.default_rng(
            1061
        ),
        config,
    )
    root = scene.coils[
        0
    ].geometry
    package = scene.packages[
        0
    ].geometry
    relative_rotation = (
        root.pose.rotation.T
        @ package.pose.rotation
    )
    assert (
        np.linalg.norm(
            relative_rotation[
                2,
                :2
            ]
        )
        > 1e-2
    )
    result = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
    ).solve(
        scene,
        frequency,
    )
    assert (
        result.prediction.reciprocity_defect()
        < 2e-10
    )
    assert (
        result.power_closure_error
        < 1e-5
    )
    assert (
        result.normalized_residual
        < 1e-8
    )
    for channel in result.prediction.dissipation_channels:
        assert (
            np.min(
                np.linalg.eigvalsh(
                    channel
                )
            )
            >= -2e-8
        )


def test_tabulated_background_with_identical_tabulated_package_is_invisible():
    frequency = 80_000.0
    material = TabulatedMaterial(
        frequencies_hz=(
            20_000.0,
            80_000.0,
            300_000.0,
        ),
        relative_permittivity_real=(
            11.0,
            7.0,
            4.5,
        ),
        loss_conductivity_values=(
            2.0e-5,
            2.0e-4,
            1.2e-4,
        ),
    )
    scene = Scene(
        (
            _coil(),
        ),
        material,
        (
            _package(
                material
            ),
        ),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
    )
    coupled = artifact.solve(
        scene,
        frequency,
    )
    bare = DenseMixedConductorTeacher(
        Scene(
            scene.coils,
            material,
        ),
        frequency,
        CFG,
    ).solve()
    assert np.allclose(
        coupled.impedance,
        bare.impedance,
        rtol=5e-9,
        atol=5e-10,
    )
    assert np.allclose(
        coupled.surface_density_transfer,
        0.0,
        rtol=0.0,
        atol=2e-13,
    )
    assert (
        coupled.power_closure_error
        < 5e-6
    )


def test_magnetic_package_increases_inductive_response_and_preserves_power_structure():
    frequency = 80_000.0
    base_scene, base = _coupled(
        IsotropicMaterial(
            relative_permittivity=1.0,
            relative_permeability=1.0,
        )
    )
    magnetic_scene = Scene(
        base_scene.coils,
        base_scene.medium,
        (
            _package(
                IsotropicMaterial(
                    relative_permittivity=1.0,
                    relative_permeability=4.0,
                )
            ),
        ),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
        magnetic_volume_axial_order=4,
        magnetic_volume_radial_order=3,
        magnetic_volume_azimuthal_order=12,
    )
    magnetic = artifact.solve(
        magnetic_scene,
        frequency,
    )
    assert (
        np.linalg.norm(
            magnetic.magnetic_inductance_correction
        )
        > 0.0
    )
    assert (
        magnetic.impedance[
            0,
            0,
        ].imag
        > base.impedance[
            0,
            0,
        ].imag
    )
    assert (
        magnetic.magnetic_surface_residual
        < 1e-9
    )
    assert (
        magnetic.raw_magnetic_reciprocity_defect
        < 0.08
    )
    assert (
        magnetic.power_closure_error
        < 1e-6
    )


def test_magnetic_package_reference_is_common_se3_invariant():
    frequency = 65_000.0
    material = IsotropicMaterial(
        relative_permittivity=1.0,
        relative_permeability=3.0,
    )
    coil = _coil()
    package = _package(
        material
    )
    scene = Scene(
        (
            coil,
        ),
        HomogeneousMedium(),
        (
            package,
        ),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
        magnetic_volume_axial_order=3,
        magnetic_volume_radial_order=3,
        magnetic_volume_azimuthal_order=12,
    )
    reference = artifact.solve(
        scene,
        frequency,
    )

    rng = np.random.default_rng(
        1701
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.asarray(
            [
                0.17,
                -0.08,
                0.23,
            ]
        ),
    )
    moved_scene = Scene(
        (
            CoilObject(
                coil.geometry.transformed(
                    common
                ),
                coil.material,
                coil.name,
            ),
        ),
        scene.medium,
        (
            PackageObject(
                package.geometry.transformed(
                    common
                ),
                package.material,
                package.name,
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
        rtol=3e-6,
        atol=3e-8,
    )
    assert np.allclose(
        actual.magnetic_inductance_correction,
        reference.magnetic_inductance_correction,
        rtol=3e-6,
        atol=3e-10,
    )


def _nested_packages(
    outer_material,
    inner_material,
    *,
    pose=None,
):
    outer_geometry = SuperquadricPackageGeometry(
        np.asarray(
            [0.030, 0.027, 0.008]
        ),
        exponent_xy=2.4,
        exponent_z=2.2,
    )
    inner_geometry = SuperquadricPackageGeometry(
        np.asarray(
            [0.021, 0.019, 0.0045]
        ),
        exponent_xy=2.2,
        exponent_z=2.0,
    )
    if pose is not None:
        outer_geometry = outer_geometry.transformed(
            pose
        )
        inner_geometry = inner_geometry.transformed(
            pose
        )
    return (
        PackageObject(
            outer_geometry,
            outer_material,
            "outer",
        ),
        PackageObject(
            inner_geometry,
            inner_material,
            "inner",
        ),
    )


def _nested_artifact():
    return DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
        magnetic_volume_axial_order=3,
        magnetic_volume_radial_order=2,
        magnetic_volume_azimuthal_order=8,
        maximum_raw_magnetic_reciprocity_defect=0.20,
    )


def test_nested_invisible_packages_reduce_to_bare_reference():
    packages = _nested_packages(
        IsotropicMaterial(
            relative_permittivity=1.0,
        ),
        IsotropicMaterial(
            relative_permittivity=1.0,
        ),
    )
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        packages,
    )
    coupled = _nested_artifact().solve(
        scene,
        80_000.0,
    )
    bare = DenseMixedConductorTeacher(
        Scene(
            scene.coils,
            scene.medium,
        ),
        80_000.0,
        CFG,
    ).solve()
    assert np.allclose(
        coupled.impedance,
        bare.impedance,
        rtol=5e-9,
        atol=5e-10,
    )
    assert np.allclose(
        coupled.surface_density_transfer,
        0.0,
        atol=5e-13,
        rtol=0.0,
    )
    assert np.allclose(
        coupled.magnetic_inductance_correction,
        0.0,
        atol=5e-13,
        rtol=0.0,
    )


def test_nested_dielectric_inner_layer_changes_reactive_response_and_closes_power():
    outer = IsotropicMaterial(
        relative_permittivity=2.0,
    )
    packages_base = _nested_packages(
        outer,
        IsotropicMaterial(
            relative_permittivity=2.0,
        ),
    )
    packages_inner = _nested_packages(
        outer,
        IsotropicMaterial(
            relative_permittivity=5.0,
        ),
    )
    artifact = _nested_artifact()
    base = artifact.solve(
        Scene(
            (_coil(),),
            HomogeneousMedium(),
            packages_base,
        ),
        80_000.0,
    )
    nested = artifact.solve(
        Scene(
            (_coil(),),
            HomogeneousMedium(),
            packages_inner,
        ),
        80_000.0,
    )
    assert (
        abs(
            nested.impedance[
                0,
                0,
            ].imag
            - base.impedance[
                0,
                0,
            ].imag
        )
        > 1e-11
    )
    assert (
        nested.power_closure_error
        < 5e-7
    )
    assert (
        nested.raw_potential_reciprocity_defect
        < 0.15
    )


def test_nested_magnetic_layers_change_inductive_response_and_are_se3_invariant():
    outer_material = IsotropicMaterial(
        relative_permittivity=1.0,
        relative_permeability=2.0,
    )
    inner_material = IsotropicMaterial(
        relative_permittivity=1.0,
        relative_permeability=4.0,
    )
    packages = _nested_packages(
        outer_material,
        inner_material,
    )
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        packages,
    )
    artifact = _nested_artifact()
    reference = artifact.solve(
        scene,
        70_000.0,
    )
    assert (
        np.linalg.norm(
            reference.magnetic_inductance_correction
        )
        > 0.0
    )
    assert (
        reference.magnetic_surface_residual
        < 1e-7
    )
    assert (
        reference.raw_magnetic_reciprocity_defect
        < 0.20
    )

    rng = np.random.default_rng(
        1703
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.asarray(
            [0.16, -0.12, 0.23]
        ),
    )
    moved_coil = CoilObject(
        scene.coils[
            0
        ].geometry.transformed(
            common
        ),
        scene.coils[
            0
        ].material,
        "coil",
    )
    moved = Scene(
        (
            moved_coil,
        ),
        scene.medium,
        _nested_packages(
            outer_material,
            inner_material,
            pose=common,
        ),
    )
    actual = artifact.solve(
        moved,
        70_000.0,
    )
    assert np.allclose(
        actual.impedance,
        reference.impedance,
        rtol=2e-5,
        atol=2e-7,
    )
    assert np.allclose(
        actual.magnetic_inductance_correction,
        reference.magnetic_inductance_correction,
        rtol=2e-5,
        atol=2e-9,
    )


def test_partially_overlapping_packages_are_rejected():
    first = PackageObject(
        SuperquadricPackageGeometry(
            np.asarray(
                [0.024, 0.022, 0.007]
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=2.0,
        ),
        "first",
    )
    second = PackageObject(
        SuperquadricPackageGeometry(
            np.asarray(
                [0.024, 0.022, 0.007]
            ),
            pose=RigidPose(
                np.eye(
                    3
                ),
                np.asarray(
                    [0.025, 0.0, 0.0]
                ),
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=3.0,
        ),
        "second",
    )
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        (
            first,
            second,
        ),
    )
    with pytest.raises(
        ValueError,
        match="partially overlap|touch/intersect",
    ):
        _nested_artifact().solve(
            scene,
            80_000.0,
        )


def test_dc_package_with_background_matched_conductivity_is_conduction_invisible():
    frequency = 0.0
    conductivity = 2.0e-3
    medium = HomogeneousMedium(
        relative_permittivity=5.0,
        relative_permeability=1.0,
        conductivity=conductivity,
    )
    package = _package(
        IsotropicMaterial(
            relative_permittivity=18.0,
            relative_permeability=1.0,
            conductivity=conductivity,
        )
    )
    scene = Scene(
        (_coil(),),
        medium,
        (package,),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
    )
    coupled = artifact.solve(
        scene,
        frequency,
    )
    bare = DenseMixedConductorTeacher(
        Scene(
            scene.coils,
            medium,
        ),
        frequency,
        CFG,
    ).solve()

    assert (
        coupled.mixed_result.node_environment_current
        is not None
    )
    assert np.allclose(
        coupled.mixed_result.node_charge,
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    assert np.allclose(
        coupled.surface_density_transfer,
        0.0,
        atol=2e-12,
        rtol=0.0,
    )
    assert np.allclose(
        coupled.impedance,
        bare.impedance,
        rtol=2e-7,
        atol=2e-9,
    )
    assert (
        coupled.power_closure_error
        < 2e-7
    )
    assert (
        coupled.channel_labels[
            -1
        ]
        == "electric_environment:aggregate"
    )


def test_dc_conductive_package_contrast_changes_resistive_response_and_closes_power():
    frequency = 0.0
    background = HomogeneousMedium(
        relative_permittivity=2.0,
        relative_permeability=1.0,
        conductivity=1.5e-3,
    )
    matched_scene = Scene(
        (_coil(),),
        background,
        (
            _package(
                IsotropicMaterial(
                    relative_permittivity=3.0,
                    conductivity=1.5e-3,
                )
            ),
        ),
    )
    contrast_scene = Scene(
        (_coil(),),
        background,
        (
            _package(
                IsotropicMaterial(
                    relative_permittivity=3.0,
                    conductivity=6.0e-3,
                )
            ),
        ),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
    )
    matched = artifact.solve(
        matched_scene,
        frequency,
    )
    contrast = artifact.solve(
        contrast_scene,
        frequency,
    )

    assert (
        abs(
            contrast.impedance[
                0,
                0,
            ].real
            - matched.impedance[
                0,
                0,
            ].real
        )
        > 1e-9
    )
    assert (
        abs(
            contrast.impedance[
                0,
                0,
            ].imag
        )
        < 1e-10
    )
    environment = (
        contrast.dielectric_dissipation_matrix
    )
    assert np.allclose(
        environment,
        environment.conj().T,
        atol=1e-10,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                environment
            )
        )
        >= -2e-8
    )
    assert (
        np.linalg.norm(
            environment
        )
        > 0.0
    )
    assert (
        contrast.power_closure_error
        < 5e-6
    )


def test_dc_conductor_inside_insulating_package_has_no_environment_leakage():
    coil = _coil()
    scene = Scene(
        (coil,),
        HomogeneousMedium(
            conductivity=2.0e-3,
        ),
        (
            _package(
                IsotropicMaterial(
                    relative_permittivity=4.0,
                    conductivity=0.0,
                )
            ),
        ),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
    )
    result = artifact.solve(
        scene,
        0.0,
    )
    bare = DenseMixedConductorTeacher(
        Scene(
            (coil,),
            HomogeneousMedium(),
        ),
        0.0,
        CFG,
    ).solve()

    assert (
        result.mixed_result.node_environment_current
        is not None
    )
    assert np.allclose(
        result.mixed_result.node_environment_current,
        0.0,
        atol=2e-12,
        rtol=0.0,
    )
    assert np.allclose(
        result.surface_density_transfer,
        0.0,
        atol=2e-12,
        rtol=0.0,
    )
    assert np.allclose(
        result.impedance,
        bare.impedance,
        rtol=3e-7,
        atol=3e-9,
    )
    assert np.allclose(
        result.dielectric_dissipation_matrix,
        0.0,
        atol=3e-10,
        rtol=0.0,
    )
    assert (
        result.power_closure_error
        < 3e-7
    )


def test_dc_mixed_insulated_and_exposed_coils_use_partial_environment_current_state():
    insulated = _coil()
    exposed = CoilObject(
        insulated.geometry.transformed(
            RigidPose(
                np.eye(
                    3
                ),
                np.asarray(
                    [0.060, 0.0, 0.0]
                ),
            )
        ),
        insulated.material,
        "exposed",
    )
    scene = Scene(
        (
            insulated,
            exposed,
        ),
        HomogeneousMedium(
            conductivity=2.0e-3,
        ),
        (
            _package(
                IsotropicMaterial(
                    relative_permittivity=4.0,
                    conductivity=0.0,
                )
            ),
        ),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
    )
    result = artifact.solve(
        scene,
        0.0,
    )
    environment = (
        result.mixed_result.node_environment_current
    )
    assert environment is not None

    first_port_rows = np.flatnonzero(
        np.abs(
            result.mixed_result.port_injection[
                :,
                0,
            ]
        )
        > 0.0
    )
    first_slice = slice(
        int(
            first_port_rows[
                0
            ]
        ),
        int(
            first_port_rows[
                -1
            ]
        )
        + 1,
    )
    assert np.allclose(
        environment[
            first_slice
        ],
        0.0,
        atol=3e-12,
        rtol=0.0,
    )
    assert (
        np.linalg.norm(
            environment[
                first_slice.stop:
            ]
        )
        > 0.0
    )
    assert np.allclose(
        result.impedance.imag,
        0.0,
        atol=1e-10,
        rtol=0.0,
    )
    assert (
        np.linalg.norm(
            result.dielectric_dissipation_matrix
        )
        > 0.0
    )
    assert (
        result.power_closure_error
        < 8e-6
    )


def test_graded_package_compiler_keeps_all_internal_interfaces_off_conductor():
    coil = _coil()
    outer = _package(
        IsotropicMaterial(
            relative_permittivity=3.0,
        )
    ).geometry
    profile = RadialIsotropicMaterialProfile(
        normalized_radius=(
            0.0,
            1.0,
        ),
        relative_permittivity=(
            2.0,
            5.0,
        ),
        relative_permeability=(
            1.0,
            1.0,
        ),
        conductivity=(
            0.0,
            0.0,
        ),
    )
    layers = compile_graded_superquadric_regions(
        outer,
        profile,
        shell_count=5,
        enclosed_coils=(
            coil,
        ),
        clearance_fraction=0.04,
    )

    assert len(
        layers
    ) == 5
    assert np.allclose(
        layers[
            -1
        ].geometry.half_extents,
        outer.half_extents,
        rtol=0.0,
        atol=1e-14,
    )
    scales = np.asarray(
        [
            layer.geometry.half_extents[
                0
            ]
            / outer.half_extents[
                0
            ]
            for layer in layers
        ],
        dtype=float,
    )
    assert np.all(
        np.diff(
            scales
        )
        > 0.0
    )
    for layer in layers:
        assert (
            layer.geometry.classify_conductor(
                coil.geometry,
                longitudinal_segments=72,
                section_points=20,
                tolerance=1e-10,
            )
            == "inside"
        )


def test_constant_graded_profile_reduces_to_single_uniform_package():
    coil = _coil()
    material = IsotropicMaterial(
        relative_permittivity=3.5,
        relative_permeability=1.8,
        conductivity=1.0e-4,
    )
    outer = _package(
        material
    ).geometry
    profile = RadialIsotropicMaterialProfile(
        normalized_radius=(
            0.0,
            1.0,
        ),
        relative_permittivity=(
            material.relative_permittivity,
            material.relative_permittivity,
        ),
        relative_permeability=(
            material.relative_permeability,
            material.relative_permeability,
        ),
        conductivity=(
            material.conductivity,
            material.conductivity,
        ),
    )
    layered = compile_graded_superquadric_regions(
        outer,
        profile,
        shell_count=4,
        enclosed_coils=(
            coil,
        ),
    )
    artifact = DielectricCoupledReferenceArtifact(
        config=CFG,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
        magnetic_volume_axial_order=3,
        magnetic_volume_radial_order=2,
        magnetic_volume_azimuthal_order=8,
        maximum_raw_magnetic_reciprocity_defect=0.20,
    )
    frequency = 75_000.0
    layered_result = artifact.solve(
        Scene(
            (
                coil,
            ),
            HomogeneousMedium(),
            layered,
        ),
        frequency,
    )
    uniform_result = artifact.solve(
        Scene(
            (
                coil,
            ),
            HomogeneousMedium(),
            (
                PackageObject(
                    outer,
                    material,
                    "uniform",
                ),
            ),
        ),
        frequency,
    )

    assert np.allclose(
        layered_result.impedance,
        uniform_result.impedance,
        rtol=3e-5,
        atol=3e-7,
    )
    assert np.allclose(
        layered_result.dissipation_channels,
        uniform_result.dissipation_channels,
        rtol=4e-5,
        atol=4e-8,
    )
    assert (
        layered_result.power_closure_error
        < 8e-6
    )


def test_graded_package_callable_accepts_dispersive_material_responses():
    coil = _coil()
    outer = _package(
        IsotropicMaterial(
            relative_permittivity=3.0,
        )
    ).geometry

    def profile(radius):
        return DebyeMaterial(
            relative_permittivity_static=(
                4.0
                + 8.0
                * radius
            ),
            relative_permittivity_infinite=(
                2.0
                + radius
            ),
            relaxation_time=(
                1.0e-6
                * (
                    1.0
                    + radius
                )
            ),
            conductivity=(
                2.0e-4
                * radius
            ),
        )

    layers = compile_graded_superquadric_regions(
        outer,
        profile,
        shell_count=4,
        enclosed_coils=(
            coil,
        ),
    )
    assert all(
        isinstance(
            layer.material,
            DebyeMaterial,
        )
        for layer in layers
    )
    frequency = 120_000.0
    epsilon = np.asarray(
        [
            np.real(
                layer.material.relative_permittivity_at(
                    frequency
                )
            )
            for layer in layers
        ],
        dtype=float,
    )
    loss = np.asarray(
        [
            layer.material.loss_conductivity(
                frequency
            )
            for layer in layers
        ],
        dtype=float,
    )
    assert np.all(
        np.isfinite(
            epsilon
        )
    )
    assert np.all(
        loss
        >= 0.0
    )
    assert (
        np.ptp(
            epsilon
        )
        > 0.0
    )


def test_constant_graded_profile_convergence_report_converges_under_shell_refinement():
    coil = _coil()
    material = IsotropicMaterial(
        relative_permittivity=3.2,
        relative_permeability=1.4,
        conductivity=7.0e-5,
    )
    outer = _package(
        material
    ).geometry
    profile = RadialIsotropicMaterialProfile(
        normalized_radius=(
            0.0,
            1.0,
        ),
        relative_permittivity=(
            material.relative_permittivity,
            material.relative_permittivity,
        ),
        relative_permeability=(
            material.relative_permeability,
            material.relative_permeability,
        ),
        conductivity=(
            material.conductivity,
            material.conductivity,
        ),
    )
    report = graded_material_convergence(
        Scene(
            (
                coil,
            ),
            HomogeneousMedium(),
        ),
        70_000.0,
        outer,
        profile,
        shell_counts=(
            2,
            4,
        ),
        config=CFG,
        tolerance=2e-4,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
        magnetic_volume_axial_order=3,
        magnetic_volume_radial_order=2,
        magnetic_volume_azimuthal_order=8,
        maximum_raw_magnetic_reciprocity_defect=0.20,
    )
    assert report.converged
    assert len(
        report.steps
    ) == 2
    assert (
        report.steps[
            0
        ].maximum_relative_change
        is None
    )
    assert (
        report.steps[
            1
        ].maximum_relative_change
        is not None
    )
    assert (
        report.maximum_relative_change
        <= report.tolerance
    )
    assert len(
        report.final_packages
    ) == 4
    assert np.allclose(
        report.final_result.impedance,
        report.steps[
            -1
        ].impedance,
    )


def test_constant_graded_electrothermal_profile_converges_in_temperature_and_ports():
    coil = _coil()
    material = IsotropicMaterial(
        relative_permittivity=3.4,
        relative_permeability=1.0,
        conductivity=2.0e-4,
        thermal_conductivity=0.24,
        density=1180.0,
        heat_capacity=1350.0,
    )
    outer = _package(
        material
    ).geometry
    profile = RadialIsotropicMaterialProfile(
        normalized_radius=(
            0.0,
            1.0,
        ),
        relative_permittivity=(
            material.relative_permittivity,
            material.relative_permittivity,
        ),
        relative_permeability=(
            material.relative_permeability,
            material.relative_permeability,
        ),
        conductivity=(
            material.conductivity,
            material.conductivity,
        ),
        thermal_conductivity=(
            material.thermal_conductivity,
            material.thermal_conductivity,
        ),
        density=(
            material.density,
            material.density,
        ),
        heat_capacity=(
            material.heat_capacity,
            material.heat_capacity,
        ),
    )
    report = graded_electrothermal_convergence(
        Scene(
            (
                coil,
            ),
            HomogeneousMedium(),
        ),
        75_000.0,
        outer,
        profile,
        HomogeneousThermalMedium(
            conductivity=0.55,
            density=1000.0,
            heat_capacity=4200.0,
            ambient_temperature=293.15,
        ),
        np.asarray(
            [
                0.0,
                0.0,
                0.020,
            ]
        ),
        np.asarray(
            [
                0.5,
                2.0,
            ]
        ),
        np.asarray(
            [
                1.0 + 0.0j,
            ]
        ),
        shell_counts=(
            2,
            4,
        ),
        config=CFG,
        tolerance=8e-4,
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        magnetic_volume_axial_order=3,
        magnetic_volume_radial_order=2,
        magnetic_volume_azimuthal_order=8,
        thermal_longitudinal_segments=8,
        thermal_radial_order=3,
        thermal_angular_order=8,
        thermal_package_axial_order=3,
        thermal_package_radial_order=2,
        thermal_package_azimuthal_order=8,
        thermal_interface_vertical_order=4,
        thermal_interface_azimuthal_order=8,
        thermal_stehfest_order=6,
        thermal_interface_residual_tolerance=2e-3,
    )
    assert report.converged
    assert len(
        report.steps
    ) == 2
    assert (
        report.maximum_relative_change
        <= report.tolerance
    )
    assert np.all(
        report.steps[
            -1
        ].temperature_rise
        >= -1e-9
    )
    assert (
        np.max(
            report.steps[
                -1
            ].temperature_rise
        )
        > 0.0
    )
    assert len(
        report.final_packages
    ) == 4
