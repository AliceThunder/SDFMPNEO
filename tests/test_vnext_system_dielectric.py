import numpy as np
import pytest

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    DielectricCoupledResult,
    HomogeneousMedium,
    HomogeneousThermalMedium,
    IsotropicMaterial,
    MeshfreeVNextSystem,
    MQSConfig,
    PackageObject,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    build_thermal_source_quadrature,
    haar_rotation,
)


def _scene():
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
            relative_permittivity=1.0,
        ),
        "invisible",
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
        (package,),
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


def _system():
    return MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        reference_config=_config(),
        dielectric_surface_vertical_order=8,
        dielectric_surface_azimuthal_order=16,
    )


def test_system_reference_dispatches_package_scene_to_coupled_sie():
    system = _system()
    scene = _scene()
    prediction = system.reference_ports(
        scene,
        80_000.0,
    )
    assert prediction.impedance.shape == (
        1,
        1,
    )
    assert prediction.n_channels == 2
    assert (
        prediction.power_closure_error()
        < 1e-8
    )

    result = system.reference_result(
        scene,
        80_000.0,
    )
    assert isinstance(
        result,
        DielectricCoupledResult,
    )


def test_system_package_scene_has_discrete_certificate_and_reference_spatial_field():
    system = _system()
    scene = _scene()

    certified = system.certified_ports(
        scene,
        80_000.0,
        algebraic_tolerance=1e-8,
        surface_tolerance=1e-8,
        fast_domain_correction_limit=1.0,
    )
    assert (
        certified.status
        == "DISCRETE_CERTIFIED"
    )
    assert certified.algebraic_certified
    assert not certified.discretization_certified
    assert certified.used_reference_fallback
    assert (
        certified.operator_backend
        == "dense_dielectric_reference"
    )

    spatial = system.reference_spatial(
        scene,
        80_000.0,
    )
    assert (
        spatial.normalization_closure_error
        < 1e-6
    )
    assert np.allclose(
        spatial.package_local_dissipation_matrix(
            0,
            np.zeros(3),
        ),
        0.0,
        atol=1e-14,
    )


def test_package_reference_continuous_thermal_field_keeps_dielectric_channel():
    system = _system()
    scene = _scene()
    prepared = (
        system.reference_continuous_thermal_field(
            scene,
            80_000.0,
            HomogeneousThermalMedium(
                conductivity=0.4,
                density=1200.0,
                heat_capacity=1000.0,
            ),
            longitudinal_segments=8,
            radial_order=3,
            angular_order=12,
        )
    )
    channels = (
        prepared.source.integrated_channels()
    )
    assert channels.shape == (
        2,
        1,
        1,
    )
    assert (
        prepared.source.normalization_closure_error
        < 1e-6
    )
    assert np.allclose(
        channels[1],
        0.0,
        atol=1e-14,
    )


def test_package_fast_spatial_does_not_fall_back_to_conductor_only_decoder():
    system = _system()
    with pytest.raises(
        NotImplementedError,
        match="package-aware FAST spatial",
    ):
        system.fast_spatial(
            _scene(),
            80_000.0,
        )



def test_package_fast_ports_reject_conductor_only_artifact():
    system = _system()
    with pytest.raises(
        NotImplementedError,
        match="package-aware FAST port artifact",
    ):
        system.fast_ports(
            _scene(),
            80_000.0,
        )


def test_package_and_lossy_background_reference_continuous_thermal_field_closes_environment_channel():
    base = _scene()
    package = PackageObject(
        base.packages[
            0
        ].geometry,
        IsotropicMaterial(
            relative_permittivity=4.0,
            conductivity=0.003,
        ),
        "lossy-package",
    )
    scene = Scene(
        base.coils,
        HomogeneousMedium(
            relative_permittivity=2.2,
            relative_permeability=1.0,
            conductivity=1e-4,
        ),
        (
            package,
        ),
    )
    system = _system()
    prepared = (
        system.reference_continuous_thermal_field(
            scene,
            80_000.0,
            HomogeneousThermalMedium(
                conductivity=0.4,
                density=1200.0,
                heat_capacity=1000.0,
            ),
            longitudinal_segments=8,
            radial_order=3,
            angular_order=12,
            background_radial_order=10,
            background_angular_order=32,
            spatial_prepare_options={
                "volume_axial_order": 4,
                "volume_radial_order": 3,
                "volume_azimuthal_order": 12,
                "background_radial_order": 10,
                "background_angular_order": 32,
                "maximum_raw_closure_error": 5.0,
                "normalized_closure_tolerance": 1e-6,
            },
        )
    )
    channels = (
        prepared.source.integrated_channels()
    )
    target = system.reference_ports(
        scene,
        80_000.0,
    ).dissipation_channels
    assert channels.shape == (
        2,
        1,
        1,
    )
    assert target.shape == (
        2,
        1,
        1,
    )
    assert (
        prepared.source.normalization_closure_error
        < 1e-6
    )
    assert np.allclose(
        channels,
        target,
        rtol=2e-5,
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
        3.0,
        np.array(
            [1.2 + 0.1j]
        ),
    )
    assert np.isfinite(
        temperature
    )
    assert (
        temperature
        > prepared.medium.ambient_temperature
    )


def test_lossy_package_background_quadrature_excludes_objects_and_is_se3_invariant():
    base = _scene()
    package = PackageObject(
        base.packages[
            0
        ].geometry,
        IsotropicMaterial(
            relative_permittivity=3.0,
            conductivity=0.002,
        ),
        "lossy-package",
    )
    medium = HomogeneousMedium(
        relative_permittivity=2.2,
        relative_permeability=1.0,
        conductivity=1e-4,
    )
    scene = Scene(
        base.coils,
        medium,
        (
            package,
        ),
    )
    system = _system()
    spatial = system.reference_spatial(
        scene,
        80_000.0,
    )
    package_center = (
        package.geometry.pose.translation
    )
    phi = (
        0.35
        * 2.0
        * np.pi
        * scene.coils[
            0
        ].geometry.turns
    )
    conductor_center = (
        scene.coils[
            0
        ].geometry.centerline(
            np.asarray(
                [phi]
            )
        )[
            0
        ]
    )
    mask = spatial._background_domain_mask(
        np.vstack(
            (
                package_center,
                conductor_center,
            )
        )
    )
    assert np.array_equal(
        mask,
        np.array(
            [False, False]
        ),
    )

    points, weights = spatial.background_quadrature(
        radial_order=8,
        angular_order=24,
    )
    assert len(points) == len(weights)
    assert np.all(
        weights > 0.0
    )
    assert np.all(
        spatial._background_domain_mask(
            points
        )
    )

    currents = np.array(
        [1.1 - 0.2j]
    )
    query = np.array(
        [0.0, 0.0, 0.035]
    )
    density = spatial.background_joule_density(
        query,
        currents,
    )

    rng = np.random.default_rng(
        719
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.array(
            [0.17, -0.11, 0.29]
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
    old_geometry = package.geometry
    moved_package = PackageObject(
        SuperquadricPackageGeometry(
            old_geometry.half_extents,
            exponent_xy=(
                old_geometry.exponent_xy
            ),
            exponent_z=(
                old_geometry.exponent_z
            ),
            pose=common.compose(
                old_geometry.pose
            ),
        ),
        package.material,
        package.name,
    )
    moved_scene = Scene(
        moved_coils,
        medium,
        (
            moved_package,
        ),
    )
    moved_spatial = system.reference_spatial(
        moved_scene,
        80_000.0,
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
        rtol=5e-6,
        atol=1e-12,
    )


def test_package_and_background_exact_dc_conduction_closes_spatial_and_thermal_fields():
    base = _scene()
    package = PackageObject(
        base.packages[
            0
        ].geometry,
        IsotropicMaterial(
            relative_permittivity=9.0,
            relative_permeability=1.0,
            conductivity=5.0e-3,
        ),
        "dc-conductive-package",
    )
    scene = Scene(
        base.coils,
        HomogeneousMedium(
            relative_permittivity=2.0,
            relative_permeability=1.0,
            conductivity=1.5e-3,
        ),
        (
            package,
        ),
    )
    system = _system()
    spatial = system.reference_spatial(
        scene,
        0.0,
        volume_axial_order=4,
        volume_radial_order=3,
        volume_azimuthal_order=12,
        background_radial_order=10,
        background_angular_order=32,
        maximum_raw_closure_error=5.0,
        normalized_closure_tolerance=2e-6,
    )
    assert (
        spatial.result.mixed_result.node_environment_current
        is not None
    )
    target = spatial.port_prediction.dissipation_channels[
        spatial.environment_channel_index
    ]
    resolved = (
        np.sum(
            spatial.package_integrated_channels,
            axis=0,
        )
        + spatial.background_integrated_channel
    )
    assert np.allclose(
        resolved,
        target,
        rtol=3e-6,
        atol=3e-9,
    )
    assert (
        spatial.normalization_closure_error
        < 3e-6
    )
    assert (
        np.linalg.norm(
            spatial.package_integrated_channels[
                0
            ]
        )
        > 0.0
    )
    assert (
        np.linalg.norm(
            spatial.background_integrated_channel
        )
        > 0.0
    )

    prepared = system.reference_continuous_thermal_field(
        scene,
        0.0,
        HomogeneousThermalMedium(
            conductivity=0.45,
            density=1150.0,
            heat_capacity=1300.0,
        ),
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
        package_axial_order=4,
        package_radial_order=3,
        package_azimuthal_order=12,
        background_radial_order=10,
        background_angular_order=32,
        spatial_prepare_options={
            "volume_axial_order": 4,
            "volume_radial_order": 3,
            "volume_azimuthal_order": 12,
            "background_radial_order": 10,
            "background_angular_order": 32,
            "maximum_raw_closure_error": 5.0,
            "normalized_closure_tolerance": 2e-6,
        },
    )
    channels = prepared.source.integrated_channels()
    port_channels = system.reference_ports(
        scene,
        0.0,
    ).dissipation_channels
    assert np.allclose(
        channels,
        port_channels,
        rtol=3e-6,
        atol=3e-9,
    )
    temperature = prepared.temperature_step(
        np.asarray(
            [0.0, 0.0, 0.03]
        ),
        2.5,
        np.asarray(
            [1.1 + 0.0j]
        ),
    )
    assert np.isfinite(
        temperature
    )
    assert (
        temperature
        > prepared.medium.ambient_temperature
    )


def test_package_spatial_loss_excludes_finite_conductor_volume():
    base = _scene()
    package = PackageObject(
        base.packages[
            0
        ].geometry,
        IsotropicMaterial(
            relative_permittivity=4.0,
            conductivity=2.0e-3,
        ),
        "lossy-package",
    )
    scene = Scene(
        base.coils,
        base.medium,
        (
            package,
        ),
    )
    spatial = _system().reference_spatial(
        scene,
        90_000.0,
        volume_axial_order=4,
        volume_radial_order=3,
        volume_azimuthal_order=12,
        maximum_raw_closure_error=5.0,
    )
    centerline = scene.coils[
        0
    ].geometry.sample_centerline(
        9
    )[
        4
    ]
    assert bool(
        spatial.teacher.conductor_teacher._mqs.points_in_conductors(
            centerline
        )
    )
    conductor_loss = spatial.raw_package_dissipation_matrices(
        0,
        centerline,
    )
    assert np.allclose(
        conductor_loss,
        0.0,
        atol=0.0,
        rtol=0.0,
    )

    material_point = np.asarray(
        [
            0.0,
            0.0,
            0.004,
        ]
    )
    assert not bool(
        spatial.teacher.conductor_teacher._mqs.points_in_conductors(
            material_point
        )
    )
    assert bool(
        package.geometry.contains(
            material_point
        )
    )
    material_loss = spatial.raw_package_dissipation_matrices(
        0,
        material_point,
    )
    assert np.all(
        np.isfinite(
            material_loss
        )
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                0.5
                * (
                    material_loss
                    + material_loss.conj().T
                )
            )
        )
        >= -1e-12
    )


class _PublicSpatialProxy:
    def __init__(
        self,
        base,
    ):
        self._base = base
        self.frequency_hz = base.frequency_hz
        self.port_prediction = base.port_prediction
        self.dielectric_channel_index = getattr(
            base,
            "dielectric_channel_index",
            len(
                base.scene.coils
            ),
        )

    def local_dissipation_matrices(
        self,
        coil_indices,
        arc_fractions,
        xy_points,
    ):
        return np.asarray(
            [
                self._base.local_dissipation_matrix(
                    int(
                        coil
                    ),
                    float(
                        arc
                    ),
                    xy,
                )
                for coil, arc, xy
                in zip(
                    coil_indices,
                    arc_fractions,
                    xy_points,
                )
            ],
            dtype=complex,
        )

    def package_dissipation_matrices(
        self,
        *args,
        **kwargs,
    ):
        return self._base.package_dissipation_matrices(
            *args,
            **kwargs,
        )


def test_package_thermal_source_uses_public_spatial_api_without_reference_teacher():
    base = _scene()
    package = PackageObject(
        base.packages[
            0
        ].geometry,
        IsotropicMaterial(
            relative_permittivity=4.0,
            conductivity=2.0e-3,
        ),
        "lossy-package",
    )
    scene = Scene(
        base.coils,
        base.medium,
        (
            package,
        ),
    )
    spatial = _system().reference_spatial(
        scene,
        90_000.0,
        volume_axial_order=4,
        volume_radial_order=3,
        volume_azimuthal_order=12,
        maximum_raw_closure_error=5.0,
    )
    proxy = _PublicSpatialProxy(
        spatial
    )
    assert not hasattr(
        proxy,
        "teacher",
    )
    source = build_thermal_source_quadrature(
        scene,
        proxy,
        longitudinal_segments=8,
        radial_order=3,
        angular_order=12,
        package_axial_order=4,
        package_radial_order=3,
        package_azimuthal_order=12,
    )
    assert np.all(
        np.isfinite(
            source.dissipation_matrices
        )
    )
    assert (
        source.normalization_closure_error
        < 1e-7
    )
    assert np.allclose(
        source.integrated_channels(),
        spatial.port_prediction.dissipation_channels,
        rtol=2e-6,
        atol=2e-9,
    )
