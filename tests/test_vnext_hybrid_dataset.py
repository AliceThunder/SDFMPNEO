import numpy as np
import pytest

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DebyeMaterial,
    HomogeneousMedium,
    HybridSceneSamplerConfig,
    HybridTeacherSample,
    ImmutableHybridTeacherDataset,
    PackageSpatialLossSamples,
    SpatialLossSamples,
    IsotropicMaterial,
    MQSConfig,
    MultiDebyeMaterial,
    PackageObject,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    analytic_port_baseline,
    encode_hybrid_scene_invariant,
    sample_hybrid_package_scene,
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


def test_hybrid_sampler_default_background_domain_remains_lossless():
    rng = np.random.default_rng(
        701
    )
    for _ in range(
        4
    ):
        scene, frequency = (
            sample_hybrid_package_scene(
                rng
            )
        )
        assert (
            scene.medium.conductivity
            == 0.0
        )
        assert (
            scene.medium.relative_permittivity
            == 1.0
        )
        assert (
            frequency
            > 0.0
        )


def test_hybrid_sampler_can_opt_in_lossy_background_domain():
    rng = np.random.default_rng(
        703
    )
    config = HybridSceneSamplerConfig(
        background_relative_permittivity_range=(
            2.0,
            3.0,
        ),
        background_conductivity_range=(
            1.0e-4,
            2.0e-4,
        ),
        lossy_background_probability=1.0,
    )
    for _ in range(
        4
    ):
        scene, _ = (
            sample_hybrid_package_scene(
                rng,
                config,
            )
        )
        assert (
            2.0
            <= scene.medium.relative_permittivity
            <= 3.0
        )
        assert (
            1.0e-4
            <= scene.medium.conductivity
            <= 2.0e-4
        )


def test_hybrid_lossy_background_spatial_truth_closes_and_round_trips(tmp_path):
    base = _scene(
        epsilon_r=3.0
    )
    scene = Scene(
        base.coils,
        HomogeneousMedium(
            relative_permittivity=2.2,
            relative_permeability=1.0,
            conductivity=1.0e-4,
        ),
        base.packages,
    )
    sample = HybridTeacherSample.generate(
        scene,
        80_000.0,
        teacher_config=_config(),
        baseline_segments=24,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
        include_spatial_truth=True,
        package_volume_axial_order=3,
        package_volume_radial_order=2,
        package_volume_azimuthal_order=8,
        background_radial_order=8,
        background_angular_order=24,
        maximum_raw_spatial_closure_error=5.0,
    )
    assert (
        sample.target_dissipation_channels.shape
        == (
            2,
            1,
            1,
        )
    )
    assert sample.has_spatial_truth
    assert (
        sample.background_spatial_loss
        is not None
    )
    assert (
        sample.background_spatial_loss.integrated()[
            0,
            0,
        ].real
        > 0.0
    )

    package = (
        sample.package_spatial_loss.integrated_packages(
            len(
                sample.scene.packages
            )
        )
    )
    environment = (
        np.sum(
            package,
            axis=0,
        )
        + sample.background_spatial_loss.integrated()
    )
    assert np.allclose(
        environment,
        sample.target_dissipation_channels[
            len(
                sample.scene.coils
            )
        ],
        rtol=2e-5,
        atol=2e-10,
    )

    dataset = ImmutableHybridTeacherDataset.create(
        tmp_path
        / "hybrid-lossy"
    )
    record = dataset.add_sample(
        sample,
        teacher_config=_config(),
        split="validation",
    )
    loaded = dataset.load_sample(
        record.sample_id
    )
    assert loaded.has_spatial_truth
    assert (
        loaded.background_radial_order
        == 8
    )
    assert (
        loaded.background_angular_order
        == 24
    )
    assert np.allclose(
        loaded.background_spatial_loss.root_local_position,
        sample.background_spatial_loss.root_local_position,
    )
    assert np.allclose(
        loaded.background_spatial_loss.weights,
        sample.background_spatial_loss.weights,
    )
    assert np.allclose(
        loaded.background_spatial_loss.dissipation_matrix,
        sample.background_spatial_loss.dissipation_matrix,
    )


def test_hybrid_dataset_background_domain_metadata_includes_lossless_branch(tmp_path):
    metadata = {
        "background": {
            "relative_permittivity_range": [
                2.0,
                3.0,
            ],
            "conductivity_range": [
                1.0e-4,
                2.0e-3,
            ],
            "lossy_probability": 0.6,
        }
    }
    dataset = ImmutableHybridTeacherDataset.create(
        tmp_path
        / "hybrid-domain",
        domain_metadata=(
            metadata
        ),
    )
    assert (
        dataset.domain_metadata
        == metadata
    )
    assert (
        dataset.background_conductivity_domain
        == (
            0.0,
            2.0e-3,
        )
    )
    assert (
        dataset.background_permittivity_domain
        == (
            2.0,
            3.0,
        )
    )


def test_hybrid_sampler_generates_debye_background_and_declares_effective_domain(tmp_path):
    config = HybridSceneSamplerConfig(
        background_relative_permittivity_range=(
            1.0,
            2.0,
        ),
        background_conductivity_range=(
            1.0e-6,
            2.0e-4,
        ),
        lossy_background_probability=0.0,
        debye_package_probability=1.0,
        package_debye_epsilon_infinite_range=(
            2.0,
            3.0,
        ),
        package_debye_delta_epsilon_range=(
            4.0,
            5.0,
        ),
        package_debye_relaxation_time_range=(
            1.0e-6,
            2.0e-6,
        ),
        debye_background_probability=1.0,
        background_debye_epsilon_infinite_range=(
            2.0,
            3.0,
        ),
        background_debye_delta_epsilon_range=(
            5.0,
            6.0,
        ),
        background_debye_relaxation_time_range=(
            1.0e-6,
            2.0e-6,
        ),
    )
    rng = np.random.default_rng(
        931
    )
    scene, frequency = sample_hybrid_package_scene(
        rng,
        config,
    )
    assert isinstance(
        scene.medium,
        DebyeMaterial,
    )
    assert isinstance(
        scene.packages[
            0
        ].material,
        DebyeMaterial,
    )
    assert (
        scene.medium.conductivity
        == 0.0
    )
    assert (
        scene.medium.loss_conductivity(
            frequency
        )
        > 0.0
    )

    package_domain = (
        config.package_domain_metadata()
    )
    assert (
        package_domain[
            "debye_probability"
        ]
        == 1.0
    )
    assert (
        package_domain[
            "effective_relative_permittivity_range"
        ]
        == [
            2.0,
            8.0,
        ]
    )
    assert (
        package_domain[
            "effective_loss_conductivity_range"
        ][
            0
        ]
        == 0.0
    )
    assert (
        package_domain[
            "effective_loss_conductivity_range"
        ][
            1
        ]
        > 0.0
    )

    dataset = ImmutableHybridTeacherDataset.create(
        tmp_path
        / "hybrid-effective-domain",
        domain_metadata={
            "background": (
                config.background_domain_metadata()
            ),
            "package": package_domain,
        },
    )
    assert (
        dataset.package_permittivity_domain
        == (
            2.0,
            8.0,
        )
    )
    assert (
        dataset.package_loss_conductivity_domain[
            0
        ]
        == 0.0
    )
    assert (
        dataset.package_loss_conductivity_domain[
            1
        ]
        > 0.0
    )

    domain = config.background_domain_metadata()
    assert domain[
        "debye_probability"
    ] == 1.0
    assert domain[
        "effective_relative_permittivity_range"
    ] == [
        2.0,
        9.0,
    ]
    assert (
        domain[
            "effective_loss_conductivity_range"
        ][
            0
        ]
        == 0.0
    )
    assert (
        domain[
            "effective_loss_conductivity_range"
        ][
            1
        ]
        > 0.0
    )


def test_hybrid_sampler_generates_passive_multi_debye_media(tmp_path):
    config = HybridSceneSamplerConfig(
        lossless_probability=1.0,
        lossy_background_probability=0.0,
        debye_package_probability=0.0,
        multi_debye_package_probability=1.0,
        package_relative_permeability_range=(
            1.7,
            2.3,
        ),
        package_debye_epsilon_infinite_range=(
            2.0,
            3.0,
        ),
        package_debye_delta_epsilon_range=(
            4.0,
            5.0,
        ),
        package_debye_relaxation_time_range=(
            1.0e-7,
            1.0e-5,
        ),
        debye_background_probability=0.0,
        multi_debye_background_probability=1.0,
        multi_debye_poles_range=(
            3,
            3,
        ),
        background_debye_epsilon_infinite_range=(
            1.5,
            2.5,
        ),
        background_debye_delta_epsilon_range=(
            5.0,
            6.0,
        ),
        background_debye_relaxation_time_range=(
            1.0e-7,
            1.0e-5,
        ),
    )
    scene, frequency = sample_hybrid_package_scene(
        np.random.default_rng(
            947
        ),
        config,
    )
    assert isinstance(
        scene.medium,
        MultiDebyeMaterial,
    )
    assert isinstance(
        scene.packages[
            0
        ].material,
        MultiDebyeMaterial,
    )
    assert (
        1.7
        <= scene.packages[
            0
        ].material.relative_permeability
        <= 2.3
    )
    for material, delta_range in (
        (
            scene.medium,
            config.background_debye_delta_epsilon_range,
        ),
        (
            scene.packages[
                0
            ].material,
            config.package_debye_delta_epsilon_range,
        ),
    ):
        assert len(
            material.relaxation_strengths
        ) == 3
        assert len(
            material.relaxation_times
        ) == 3
        assert all(
            value > 0.0
            for value
            in material.relaxation_strengths
        )
        assert all(
            value > 0.0
            for value
            in material.relaxation_times
        )
        total_delta = sum(
            material.relaxation_strengths
        )
        assert (
            delta_range[
                0
            ]
            <= total_delta
            <= delta_range[
                1
            ]
        )
        assert (
            material.loss_conductivity(
                frequency
            )
            > 0.0
        )

    background_domain = (
        config.background_domain_metadata()
    )
    package_domain = (
        config.package_domain_metadata()
    )
    assert (
        background_domain[
            "multi_debye_probability"
        ]
        == 1.0
    )
    assert (
        package_domain[
            "multi_debye_probability"
        ]
        == 1.0
    )
    assert (
        package_domain[
            "relative_permeability_range"
        ]
        == [
            1.7,
            2.3,
        ]
    )
    assert (
        background_domain[
            "multi_debye_poles_range"
        ]
        == [
            3,
            3,
        ]
    )
    background_epsilon = float(
        np.real(
            scene.medium.relative_permittivity_at(
                frequency
            )
        )
    )
    package_epsilon = float(
        np.real(
            scene.packages[
                0
            ].material.relative_permittivity_at(
                frequency
            )
        )
    )
    assert (
        background_domain[
            "effective_relative_permittivity_range"
        ][
            0
        ]
        <= background_epsilon
        <= background_domain[
            "effective_relative_permittivity_range"
        ][
            1
        ]
    )
    assert (
        package_domain[
            "effective_relative_permittivity_range"
        ][
            0
        ]
        <= package_epsilon
        <= package_domain[
            "effective_relative_permittivity_range"
        ][
            1
        ]
    )

    dataset = ImmutableHybridTeacherDataset.create(
        tmp_path
        / "hybrid-magnetic-domain",
        domain_metadata={
            "background": (
                background_domain
            ),
            "package": (
                package_domain
            ),
        },
    )
    assert (
        dataset.package_permeability_domain
        == (
            1.7,
            2.3,
        )
    )


def test_hybrid_sampler_uses_arbitrary_3d_enclosing_package_pose(tmp_path):
    config = HybridSceneSamplerConfig(
        package_center_offset_fraction_range=(
            0.24,
            0.26,
        ),
    )
    scene, _ = sample_hybrid_package_scene(
        np.random.default_rng(
            977
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
    tilt = float(
        np.linalg.norm(
            relative_rotation[
                2,
                :2
            ]
        )
    )
    assert tilt > 1e-2

    offset_root = (
        (
            package.pose.translation
            - root.pose.translation
        )
        @ root.pose.rotation
    )
    offset_scale = max(
        root.outer_a,
        root.outer_b,
    )
    assert (
        np.linalg.norm(
            offset_root
        )
        > 0.20
        * offset_scale
    )

    conductor_surface = root.surface_samples(
        longitudinal_segments=96,
        section_points=20,
    )
    assert np.all(
        package.contains(
            conductor_surface
        )
    )
    for coil in scene.coils:
        assert (
            package.classify_conductor(
                coil.geometry
            )
            in (
                "inside",
                "outside",
            )
        )

    geometry_domain = (
        config.geometry_domain_metadata()
    )
    domain_dataset = ImmutableHybridTeacherDataset.create(
        tmp_path
        / "geometry-domain-round-trip",
        domain_metadata={
            "geometry": geometry_domain,
        },
    )
    assert (
        domain_dataset.geometry_domain
        == geometry_domain
    )
    assert (
        geometry_domain[
            "package"
        ][
            "relative_rotation"
        ]
        == "haar_so3"
    )
    assert (
        geometry_domain[
            "package"
        ][
            "center_offset_fraction_range"
        ]
        == [
            0.24,
            0.26,
        ]
    )


def test_hybrid_sampler_generates_three_strictly_nested_packages():
    config = HybridSceneSamplerConfig(
        package_count_range=(
            3,
            3,
        ),
        nested_package_probability=1.0,
        nested_package_scale_range=(
            1.15,
            1.25,
        ),
    )
    scene, _ = sample_hybrid_package_scene(
        np.random.default_rng(
            2401
        ),
        config,
    )
    assert len(
        scene.packages
    ) == 3
    for inner_index in range(
        2
    ):
        inner = scene.packages[
            inner_index
        ].geometry
        outer = scene.packages[
            inner_index
            + 1
        ].geometry
        assert np.all(
            outer.contains(
                inner.surface_points(
                    vertical_order=9,
                    azimuthal_order=24,
                ),
                tolerance=1e-10,
            )
        )
    metadata = config.geometry_domain_metadata()
    assert metadata[
        "n_packages_range"
    ] == [
        3,
        3,
    ]
    assert (
        metadata[
            "package"
        ][
            "nested_topology"
        ]
        == "strict_chain_or_disjoint_roots"
    )


def test_nested_package_teacher_spatial_truth_uses_unique_material_regions():
    sampler = HybridSceneSamplerConfig(
        package_count_range=(
            2,
            2,
        ),
        nested_package_probability=1.0,
        nested_package_scale_range=(
            1.18,
            1.22,
        ),
        lossless_probability=0.0,
    )
    scene, frequency = sample_hybrid_package_scene(
        np.random.default_rng(
            2417
        ),
        sampler,
    )
    sample = HybridTeacherSample.generate(
        scene,
        frequency,
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
    assert sample.target_dissipation_channels.shape == (
        len(
            scene.coils
        )
        + 1,
        len(
            scene.coils
        ),
        len(
            scene.coils
        ),
    )
    package_integrated = (
        sample.package_spatial_loss.integrated_packages(
            len(
                scene.packages
            )
        )
    )
    assert np.allclose(
        np.sum(
            package_integrated,
            axis=0,
        ),
        sample.target_dissipation_channels[
            len(
                scene.coils
            )
        ],
        rtol=3e-5,
        atol=3e-9,
    )

    outer_mask = (
        sample.package_spatial_loss.package_index
        == 1
    )
    outer_local = (
        sample.package_spatial_loss.local_position[
            outer_mask
        ]
    )
    outer_world = (
        scene.packages[
            1
        ].geometry.local_to_world(
            outer_local
        )
    )
    assert not np.any(
        scene.packages[
            0
        ].geometry.contains(
            outer_world,
            tolerance=2e-12,
        )
    )


def test_hybrid_sampler_teacher_and_dataset_support_true_conductive_dc(tmp_path):
    sampler = HybridSceneSamplerConfig(
        dc_probability=1.0,
        dc_conductive_probability=1.0,
        package_count_range=(
            1,
            1,
        ),
        background_conductivity_range=(
            8.0e-4,
            2.0e-3,
        ),
        dielectric_conductivity_range=(
            1.0e-3,
            4.0e-3,
        ),
        lossless_probability=1.0,
        lossy_background_probability=0.0,
    )
    scene, frequency = sample_hybrid_package_scene(
        np.random.default_rng(
            3209
        ),
        sampler,
    )
    assert (
        frequency
        == 0.0
    )
    assert (
        scene.medium.loss_conductivity(
            0.0
        )
        > 0.0
    )
    assert all(
        package.material.loss_conductivity(
            0.0
        )
        > 0.0
        for package in scene.packages
    )

    geometry_domain = (
        sampler.geometry_domain_metadata()
    )
    assert (
        geometry_domain[
            "conductor"
        ][
            "frequency_range"
        ][
            0
        ]
        == 0.0
    )
    assert (
        geometry_domain[
            "frequency_sampling"
        ][
            "dc_probability"
        ]
        == 1.0
    )
    assert (
        sampler.background_domain_metadata()[
            "effective_loss_conductivity_range"
        ][
            1
        ]
        > 0.0
    )
    assert (
        sampler.package_domain_metadata()[
            "effective_loss_conductivity_range"
        ][
            1
        ]
        > 0.0
    )

    sample = HybridTeacherSample.generate(
        scene,
        frequency,
        teacher_config=_config(),
        baseline_segments=24,
        surface_vertical_order=6,
        surface_azimuthal_order=12,
        include_spatial_truth=True,
        package_volume_axial_order=3,
        package_volume_radial_order=2,
        package_volume_azimuthal_order=8,
        background_radial_order=8,
        background_angular_order=24,
        maximum_raw_spatial_closure_error=5.0,
    )
    assert (
        sample.frequency_hz
        == 0.0
    )
    assert (
        sample.background_spatial_loss
        is not None
    )
    assert (
        sample.package_spatial_loss
        is not None
    )
    assert np.all(
        np.isfinite(
            sample.target_impedance
        )
    )
    assert (
        np.linalg.norm(
            sample.target_impedance.imag
        )
        < 1e-9
    )
    assert (
        sample.power_closure_error
        < 1e-5
    )

    domain_metadata = {
        "geometry": (
            geometry_domain
        ),
        "background": (
            sampler.background_domain_metadata()
        ),
        "package": (
            sampler.package_domain_metadata()
        ),
    }
    dataset = ImmutableHybridTeacherDataset.create(
        tmp_path
        / "hybrid-dc",
        split_seed=37,
        domain_metadata=domain_metadata,
    )
    record = dataset.add_sample(
        sample,
        teacher_config=_config(),
        split="train",
    )
    loaded = dataset.load_sample(
        record.sample_id
    )
    assert (
        loaded.frequency_hz
        == 0.0
    )
    assert np.allclose(
        loaded.target_impedance,
        sample.target_impedance,
    )
