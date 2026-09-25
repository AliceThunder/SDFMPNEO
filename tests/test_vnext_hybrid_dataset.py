import numpy as np
import pytest

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    HybridTeacherSample,
    ImmutableHybridTeacherDataset,
    PackageSpatialLossSamples,
    SpatialLossSamples,
    IsotropicMaterial,
    MQSConfig,
    PackageObject,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    analytic_port_baseline,
    encode_hybrid_scene_invariant,
)


def _scene(
    epsilon_r=2.5,
):
    coil = CoilObject(
        SuperellipseSpiral(
            0.015,
            0.013,
            0.65,
            0.001,
            0.001,
            conductor_width=8e-4,
            conductor_thickness=6e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )
    package = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.025, 0.022, 0.006]
            ),
            exponent_xy=2.0,
            exponent_z=2.0,
        ),
        IsotropicMaterial(
            relative_permittivity=(
                epsilon_r
            ),
        ),
        "package",
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
        (package,),
    )


def _manual_sample():
    scene = _scene()
    frequency = 70_000.0
    encoded = (
        encode_hybrid_scene_invariant(
            scene,
            frequency,
        )
    )
    baseline = (
        analytic_port_baseline(
            Scene(
                scene.coils,
                scene.medium,
            ),
            frequency,
            segments_per_coil=32,
        )
    )
    target = (
        baseline.resistance
        + 1j
        * 2.0
        * np.pi
        * frequency
        * baseline.inductance
    )
    channels = np.array(
        [
            [
                [
                    0.9
                    * target[
                        0,
                        0,
                    ].real
                ]
            ],
            [
                [
                    0.1
                    * target[
                        0,
                        0,
                    ].real
                ]
            ],
        ],
        dtype=complex,
    )
    conductor_spatial = SpatialLossSamples(
        np.asarray(
            [0],
            dtype=int,
        ),
        np.asarray(
            [0.5],
            dtype=float,
        ),
        np.asarray(
            [[0.0, 0.0]],
            dtype=float,
        ),
        np.asarray(
            [1.0],
            dtype=float,
        ),
        np.asarray(
            [channels[0]],
            dtype=complex,
        ),
    )
    package_spatial = PackageSpatialLossSamples(
        np.asarray(
            [0],
            dtype=int,
        ),
        np.asarray(
            [[0.0, 0.0, 0.0]],
            dtype=float,
        ),
        np.asarray(
            [1.0],
            dtype=float,
        ),
        np.asarray(
            [channels[1]],
            dtype=complex,
        ),
    )
    return HybridTeacherSample(
        scene=scene,
        frequency_hz=frequency,
        encoded=encoded,
        baseline_resistance=(
            baseline.resistance
        ),
        baseline_reactance=(
            target.imag
        ),
        target_impedance=(
            target
        ),
        target_dissipation_channels=(
            channels
        ),
        baseline_segments=32,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
        surface_residual=0.0,
        raw_potential_reciprocity_defect=0.0,
        power_closure_error=0.0,
        conductor_spatial_loss=(
            conductor_spatial
        ),
        package_spatial_loss=(
            package_spatial
        ),
        package_volume_axial_order=2,
        package_volume_radial_order=2,
        package_volume_azimuthal_order=8,
    )


def _config():
    return MQSConfig(
        segments_per_turn=8,
        min_segments=10,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )


def test_hybrid_dataset_round_trip_is_content_addressed(tmp_path):
    dataset = (
        ImmutableHybridTeacherDataset.create(
            tmp_path
            / "hybrid"
        )
    )
    sample = _manual_sample()
    first = dataset.add_sample(
        sample,
        teacher_config=_config(),
        split="validation",
    )
    second = dataset.add_sample(
        sample,
        teacher_config=_config(),
        split="validation",
    )
    assert (
        first.sample_id
        == second.sample_id
    )
    assert len(
        dataset.records
    ) == 1

    loaded = dataset.load_sample(
        first.sample_id
    )
    assert np.allclose(
        loaded.encoded.package_features,
        sample.encoded.package_features,
    )
    assert np.allclose(
        loaded.encoded.coil_package_features,
        sample.encoded.coil_package_features,
    )
    assert np.allclose(
        loaded.target_impedance,
        sample.target_impedance,
    )
    assert np.allclose(
        loaded.target_dissipation_channels,
        sample.target_dissipation_channels,
    )
    assert loaded.has_spatial_truth
    assert np.allclose(
        loaded.conductor_spatial_loss.dissipation_matrix,
        sample.conductor_spatial_loss.dissipation_matrix,
    )
    assert np.allclose(
        loaded.package_spatial_loss.local_position,
        sample.package_spatial_loss.local_position,
    )
    assert np.allclose(
        loaded.package_spatial_loss.dissipation_matrix,
        sample.package_spatial_loss.dissipation_matrix,
    )


def test_hybrid_active_learning_can_only_append_to_train(tmp_path):
    dataset = (
        ImmutableHybridTeacherDataset.create(
            tmp_path
            / "hybrid"
        )
    )
    with pytest.raises(
        ValueError,
        match="train",
    ):
        dataset.add_sample(
            _manual_sample(),
            teacher_config=_config(),
            source="active",
            split="release",
        )


def test_hybrid_teacher_generation_uses_coupled_reference_and_extra_loss_channel():
    sample = HybridTeacherSample.generate(
        _scene(
            epsilon_r=1.0
        ),
        80_000.0,
        teacher_config=_config(),
        baseline_segments=24,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
        include_spatial_truth=True,
        package_volume_axial_order=3,
        package_volume_radial_order=2,
        package_volume_azimuthal_order=8,
        maximum_raw_spatial_closure_error=5.0,
    )
    assert sample.target_impedance.shape == (
        1,
        1,
    )
    assert (
        sample.target_dissipation_channels.shape
        == (
            2,
            1,
            1,
        )
    )
    assert (
        sample.power_closure_error
        < 1e-8
    )
    assert (
        sample.surface_residual
        < 1e-10
    )
    assert sample.has_spatial_truth
    conductor_integrated = (
        sample.conductor_spatial_loss.integrated_channels(
            len(
                sample.scene.coils
            )
        )
    )
    package_integrated = (
        sample.package_spatial_loss.integrated_packages(
            len(
                sample.scene.packages
            )
        )
    )
    assert np.allclose(
        conductor_integrated,
        sample.target_dissipation_channels[
            : len(
                sample.scene.coils
            )
        ],
        rtol=2e-6,
        atol=2e-10,
    )
    assert np.allclose(
        np.sum(
            package_integrated,
            axis=0,
        ),
        sample.target_dissipation_channels[
            len(
                sample.scene.coils
            )
        ],
        rtol=2e-6,
        atol=2e-10,
    )
