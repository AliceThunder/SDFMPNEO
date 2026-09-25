import numpy as np
import pytest

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DenseMixedConductorTeacher,
    DielectricCoupledReferenceArtifact,
    HomogeneousMedium,
    IsotropicMaterial,
    MQSConfig,
    PackageObject,
    RigidPose,
    Scene,
    StructuredPortPrediction,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    haar_rotation,
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
