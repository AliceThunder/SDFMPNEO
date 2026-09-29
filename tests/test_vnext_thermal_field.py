import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    AnisotropicThermalMedium,
    CoilObject,
    ConductorMaterial,
    ContinuousThermalGreenArtifact,
    HomogeneousMedium,
    HomogeneousThermalMedium,
    MixedReferenceArtifact,
    MQSConfig,
    PreparedThermalGreenField,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    ThermalSourceQuadrature,
    UniformLossFieldDecoder,
    build_thermal_source_quadrature,
    haar_rotation,
)


def _scene(pose=None):
    copper = ConductorMaterial(
        5.8e7
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.020,
            0.75,
            0.001,
            0.001,
            exponent=3.0,
            conductor_width=1.0e-3,
            conductor_thickness=0.8e-3,
            pose=(
                pose
                or RigidPose.identity()
            ),
        ),
        copper,
        "coil",
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
    )


def _medium():
    return HomogeneousThermalMedium(
        conductivity=0.6,
        density=1000.0,
        heat_capacity=4200.0,
        ambient_temperature=293.15,
    )


def _artifact():
    return ContinuousThermalGreenArtifact(
        UniformLossFieldDecoder(
            AnalyticBaselineArtifact(
                segments_per_coil=32,
            ),
            length_segments=32,
        ),
        _medium(),
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
    )


def test_thermal_source_quadrature_closes_port_loss_channels():
    scene = _scene()
    spatial = UniformLossFieldDecoder(
        AnalyticBaselineArtifact(
            segments_per_coil=32,
        ),
        length_segments=32,
    ).prepare(
        scene,
        40_000.0,
    )
    source = build_thermal_source_quadrature(
        scene,
        spatial,
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
    )
    assert source.normalization_closure_error < 1e-10
    assert np.allclose(
        source.integrated_channels(),
        spatial.port_prediction.dissipation_channels,
        rtol=2e-9,
        atol=2e-12,
    )
    currents = np.array(
        [2.0 - 0.3j]
    )
    assert np.allclose(
        source.coil_power(currents),
        spatial.port_prediction.coil_power(
            currents
        ),
        rtol=2e-9,
        atol=2e-12,
    )


def test_continuous_green_temperature_is_common_se3_invariant():
    scene = _scene()
    field = _artifact().prepare(
        scene,
        40_000.0,
    )
    query = np.array(
        [0.006, -0.004, 0.030]
    )
    currents = np.array(
        [2.0 + 0.1j]
    )
    temperature = field.temperature_step(
        query,
        8.0,
        currents,
    )

    rng = np.random.default_rng(
        27
    )
    common = RigidPose(
        haar_rotation(rng),
        np.array(
            [0.3, -0.2, 0.5]
        ),
    )
    moved = _scene(
        common
    )
    moved_field = _artifact().prepare(
        moved,
        40_000.0,
    )
    moved_temperature = (
        moved_field.temperature_step(
            common.apply(
                query
            ),
            8.0,
            currents,
        )
    )
    assert np.isclose(
        temperature,
        moved_temperature,
        rtol=2e-10,
        atol=2e-10,
    )


def test_continuous_green_time_history_matches_step_and_steady_limit():
    field = _artifact().prepare(
        _scene(),
        40_000.0,
    )
    query = np.array(
        [0.0, 0.0, 0.035]
    )
    currents = np.array(
        [1.8 + 0.0j]
    )
    times = np.array(
        [0.5, 2.0, 10.0]
    )
    step = np.asarray(
        [
            field.temperature_step(
                query,
                float(time),
                currents,
            )
            for time in times
        ]
    )
    history = field.temperature_history(
        query,
        np.array(
            [0.0, 10.0]
        ),
        currents[
            None,
            :
        ],
        times,
    )
    assert np.allclose(
        history,
        step,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.all(
        np.diff(
            step
        )
        > 0.0
    )
    steady = field.steady_temperature(
        query,
        currents,
    )
    assert steady > step[-1]


def test_lossy_background_has_continuous_psd_heat_channel_and_thermal_coupling():
    base = _scene()
    scene = Scene(
        base.coils,
        HomogeneousMedium(
            relative_permittivity=3.0,
            relative_permeability=1.0,
            conductivity=1e-4,
        ),
    )
    reference = MixedReferenceArtifact(
        config=MQSConfig(
            segments_per_turn=8,
            min_segments=8,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        )
    )
    spatial = reference.prepare_spatial(
        scene,
        40_000.0,
    )
    assert spatial.background_channel_index == 1
    assert (
        spatial.normalized_background_closure_error
        < 1e-7
    )
    query = np.array(
        [0.0, 0.0, 0.035]
    )
    matrix = spatial.background_dissipation_matrices(
        query
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
    currents = np.array(
        [1.8 + 0.2j]
    )
    assert (
        spatial.background_joule_density(
            query,
            currents,
        )
        >= -1e-12
    )

    source = build_thermal_source_quadrature(
        scene,
        spatial,
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
    )
    assert source.n_channels == 2
    assert (
        source.normalization_closure_error
        < 1e-8
    )
    assert np.allclose(
        source.integrated_channels(),
        spatial.port_prediction.dissipation_channels,
        rtol=2e-7,
        atol=2e-10,
    )
    channel_power = source.channel_power(
        currents
    )
    assert channel_power.shape == (2,)
    assert np.all(
        channel_power
        >= -1e-12
    )

    thermal = ContinuousThermalGreenArtifact(
        reference,
        _medium(),
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
    ).prepare(
        scene,
        40_000.0,
    )
    assert thermal.source.n_channels == 2
    temperature = thermal.temperature_step(
        query,
        2.0,
        currents,
    )
    assert np.isfinite(
        temperature
    )
    assert (
        temperature
        > thermal.medium.ambient_temperature
    )


def test_lossy_background_reference_spatial_closes_continuous_environment_channel():
    base = _scene()
    scene = Scene(
        base.coils,
        HomogeneousMedium(
            relative_permittivity=3.0,
            relative_permeability=1.0,
            conductivity=1e-4,
        ),
    )
    artifact = MixedReferenceArtifact(
        config=MQSConfig(
            segments_per_turn=8,
            min_segments=8,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        ),
        background_radial_order=10,
        background_angular_order=32,
    )
    spatial = artifact.prepare_spatial(
        scene,
        40_000.0,
    )
    assert spatial.background_channel_index == 1
    assert (
        spatial.normalized_background_closure_error
        < 1e-8
    )

    points, weights = spatial.background_quadrature(
        radial_order=10,
        angular_order=32,
    )
    matrices = spatial.background_dissipation_matrices(
        points
    )
    assert np.all(
        np.isfinite(
            points
        )
    )
    assert np.all(
        weights > 0.0
    )
    for matrix in matrices[:: max(1, len(matrices) // 17)]:
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

    integrated = np.sum(
        weights[
            :,
            None,
            None,
        ]
        * matrices,
        axis=0,
    )
    target = spatial.port_prediction.dissipation_channels[
        spatial.background_channel_index
    ]
    assert np.allclose(
        integrated,
        target,
        rtol=2e-7,
        atol=2e-10,
    )


def test_lossy_background_reference_thermal_field_includes_environment_heat_channel():
    base = _scene()
    scene = Scene(
        base.coils,
        HomogeneousMedium(
            relative_permittivity=2.5,
            relative_permeability=1.0,
            conductivity=8e-5,
        ),
    )
    artifact = MixedReferenceArtifact(
        config=MQSConfig(
            segments_per_turn=8,
            min_segments=8,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        ),
        background_radial_order=10,
        background_angular_order=32,
    )
    prepared = ContinuousThermalGreenArtifact(
        artifact,
        _medium(),
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
    ).prepare(
        scene,
        40_000.0,
    )
    channels = prepared.source.integrated_channels()
    assert channels.shape == (
        2,
        1,
        1,
    )
    assert (
        prepared.source.normalization_closure_error
        < 1e-8
    )
    target = artifact.predict_structured(
        scene,
        40_000.0,
    ).dissipation_channels
    assert np.allclose(
        channels,
        target,
        rtol=2e-7,
        atol=2e-10,
    )
    assert (
        channels[
            1,
            0,
            0,
        ].real
        > 0.0
    )
    temperature = prepared.temperature_step(
        np.array(
            [0.0, 0.0, 0.03]
        ),
        5.0,
        np.array(
            [1.5 + 0.0j]
        ),
    )
    assert np.isfinite(
        temperature
    )
    assert (
        temperature
        > prepared.medium.ambient_temperature
    )


def test_lossy_background_domain_excludes_finite_cross_section_conductor():
    base = _scene()
    scene = Scene(
        base.coils,
        HomogeneousMedium(
            relative_permittivity=3.0,
            conductivity=1e-4,
        ),
    )
    artifact = MixedReferenceArtifact(
        config=MQSConfig(
            segments_per_turn=8,
            min_segments=8,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        )
    )
    spatial = artifact.prepare_spatial(
        scene,
        40_000.0,
    )
    phi = (
        0.45
        * 2.0
        * np.pi
        * scene.coils[
            0
        ].geometry.turns
    )
    centerline = scene.coils[
        0
    ].geometry.centerline(
        np.asarray(
            [phi]
        )
    )
    exterior = spatial._background_domain_mask(
        centerline
    )
    assert exterior.shape == (1,)
    assert not bool(
        exterior[
            0
        ]
    )


def test_lossy_background_spatial_heat_is_common_se3_invariant():
    base = _scene()
    lossy = HomogeneousMedium(
        relative_permittivity=3.0,
        relative_permeability=1.0,
        conductivity=1e-4,
    )
    scene = Scene(
        base.coils,
        lossy,
    )
    config = MQSConfig(
        segments_per_turn=8,
        min_segments=8,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )
    reference = MixedReferenceArtifact(
        config=config,
        background_radial_order=10,
        background_angular_order=32,
    )
    spatial = reference.prepare_spatial(
        scene,
        40_000.0,
    )
    query = np.array(
        [0.006, -0.004, 0.030]
    )
    currents = np.array(
        [1.7 - 0.2j]
    )
    density = spatial.background_joule_density(
        query,
        currents,
    )

    rng = np.random.default_rng(
        918
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.array(
            [0.21, -0.13, 0.37]
        ),
    )
    moved_coils = tuple(
        type(coil)(
            coil.geometry.transformed(
                common
            ),
            coil.material,
            coil.name,
        )
        for coil in scene.coils
    )
    moved_scene = Scene(
        moved_coils,
        lossy,
    )
    moved_spatial = MixedReferenceArtifact(
        config=config,
        background_radial_order=10,
        background_angular_order=32,
    ).prepare_spatial(
        moved_scene,
        40_000.0,
    )
    moved_density = (
        moved_spatial.background_joule_density(
            common.apply(
                query
            ),
            currents,
        )
    )
    assert np.isclose(
        density,
        moved_density,
        rtol=3e-7,
        atol=1e-12,
    )
    assert np.isclose(
        spatial.raw_background_closure_error,
        moved_spatial.raw_background_closure_error,
        rtol=3e-7,
        atol=1e-12,
    )


def test_conductive_background_dc_spatial_loss_and_thermal_source_close():
    base = _scene()
    scene = Scene(
        base.coils,
        HomogeneousMedium(
            relative_permittivity=2.5,
            relative_permeability=1.0,
            conductivity=1.5e-3,
        ),
    )
    artifact = MixedReferenceArtifact(
        config=MQSConfig(
            segments_per_turn=8,
            min_segments=8,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        ),
        background_radial_order=10,
        background_angular_order=32,
    )
    spatial = artifact.prepare_spatial(
        scene,
        0.0,
    )
    assert (
        spatial.result.node_environment_current
        is not None
    )
    assert (
        spatial.background_channel_index
        == 1
    )
    points, weights = spatial.background_quadrature(
        radial_order=10,
        angular_order=32,
    )
    matrices = spatial.background_dissipation_matrices(
        points
    )
    assert np.all(
        np.isfinite(
            matrices
        )
    )
    integrated = np.sum(
        weights[
            :,
            None,
            None,
        ]
        * matrices,
        axis=0,
    )
    target = spatial.port_prediction.dissipation_channels[
        spatial.background_channel_index
    ]
    assert np.allclose(
        integrated,
        target,
        rtol=3e-7,
        atol=3e-10,
    )

    prepared = ContinuousThermalGreenArtifact(
        artifact,
        _medium(),
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
        background_radial_order=10,
        background_angular_order=32,
    ).prepare(
        scene,
        0.0,
    )
    assert (
        prepared.source.normalization_closure_error
        < 1e-8
    )
    channels = prepared.source.integrated_channels()
    assert np.allclose(
        channels,
        spatial.port_prediction.dissipation_channels,
        rtol=3e-7,
        atol=3e-10,
    )
    temperature = prepared.temperature_step(
        np.asarray(
            [0.0, 0.0, 0.03]
        ),
        4.0,
        np.asarray(
            [1.25 + 0.0j]
        ),
    )
    assert np.isfinite(
        temperature
    )
    assert (
        temperature
        > prepared.medium.ambient_temperature
    )


def _single_thermal_source(
    position=(0.0, 0.0, 0.0),
):
    return ThermalSourceQuadrature(
        positions=np.asarray(
            [
                position
            ],
            dtype=float,
        ),
        volume_weights=np.asarray(
            [
                1.0
            ],
            dtype=float,
        ),
        coil_index=np.asarray(
            [
                0
            ],
            dtype=int,
        ),
        arc_fraction=np.asarray(
            [
                0.0
            ],
            dtype=float,
        ),
        xy=np.zeros(
            (
                1,
                2,
            ),
            dtype=float,
        ),
        dissipation_matrices=np.asarray(
            [
                [
                    [
                        1.0
                    ]
                ]
            ],
            dtype=complex,
        ),
        effective_radius=np.asarray(
            [
                2.0e-4
            ],
            dtype=float,
        ),
        normalization_closure_error=0.0,
        normalization_correction=0.0,
    )


def test_anisotropic_thermal_green_reduces_to_isotropic_limit():
    source = _single_thermal_source()
    isotropic = PreparedThermalGreenField(
        source,
        HomogeneousThermalMedium(
            conductivity=0.6,
            density=1000.0,
            heat_capacity=4200.0,
        ),
    )
    anisotropic = PreparedThermalGreenField(
        source,
        AnisotropicThermalMedium(
            conductivity_tensor=(
                0.6
                * np.eye(
                    3
                )
            ),
            density=1000.0,
            heat_capacity=4200.0,
        ),
    )
    points = np.asarray(
        [
            [0.020, 0.0, 0.0],
            [0.0, -0.015, 0.010],
        ]
    )
    assert np.allclose(
        anisotropic.steady_response_matrix(
            points
        ),
        isotropic.steady_response_matrix(
            points
        ),
        rtol=2e-12,
        atol=2e-14,
    )
    assert np.allclose(
        anisotropic.step_response_matrix(
            points,
            3.0,
        ),
        isotropic.step_response_matrix(
            points,
            3.0,
        ),
        rtol=2e-12,
        atol=2e-14,
    )


def test_anisotropic_thermal_green_resolves_directional_conductivity():
    field = PreparedThermalGreenField(
        _single_thermal_source(),
        AnisotropicThermalMedium(
            conductivity_tensor=np.diag(
                [
                    2.4,
                    0.6,
                    0.6,
                ]
            ),
            density=1000.0,
            heat_capacity=4200.0,
        ),
    )
    response = field.steady_response_matrix(
        np.asarray(
            [
                [0.030, 0.0, 0.0],
                [0.0, 0.030, 0.0],
            ]
        )
    )
    assert (
        response[
            0,
            0,
            0
        ].real
        > response[
            1,
            0,
            0
        ].real
    )


def test_anisotropic_thermal_green_is_common_rotation_covariant():
    source = _single_thermal_source(
        (
            0.004,
            -0.003,
            0.002,
        )
    )
    tensor = np.asarray(
        [
            [1.4, 0.18, 0.0],
            [0.18, 0.8, 0.07],
            [0.0, 0.07, 0.5],
        ],
        dtype=float,
    )
    field = PreparedThermalGreenField(
        source,
        AnisotropicThermalMedium(
            conductivity_tensor=tensor,
            density=980.0,
            heat_capacity=3600.0,
        ),
    )
    query = np.asarray(
        [
            0.026,
            -0.011,
            0.019,
        ]
    )
    steady = field.steady_response_matrix(
        query
    )
    transient = field.step_response_matrix(
        query,
        2.5,
    )

    rng = np.random.default_rng(
        741
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.asarray(
            [
                0.11,
                -0.07,
                0.16,
            ]
        ),
    )
    moved_source = ThermalSourceQuadrature(
        positions=common.apply(
            source.positions
        ),
        volume_weights=source.volume_weights,
        coil_index=source.coil_index,
        arc_fraction=source.arc_fraction,
        xy=source.xy,
        dissipation_matrices=(
            source.dissipation_matrices
        ),
        effective_radius=(
            source.effective_radius
        ),
        normalization_closure_error=(
            source.normalization_closure_error
        ),
        normalization_correction=(
            source.normalization_correction
        ),
    )
    rotation = common.rotation
    moved_field = PreparedThermalGreenField(
        moved_source,
        AnisotropicThermalMedium(
            conductivity_tensor=(
                rotation
                @ tensor
                @ rotation.T
            ),
            density=980.0,
            heat_capacity=3600.0,
        ),
    )
    moved_query = common.apply(
        query
    )
    assert np.allclose(
        moved_field.steady_response_matrix(
            moved_query
        ),
        steady,
        rtol=2e-11,
        atol=2e-13,
    )
    assert np.allclose(
        moved_field.step_response_matrix(
            moved_query,
            2.5,
        ),
        transient,
        rtol=2e-11,
        atol=2e-13,
    )
