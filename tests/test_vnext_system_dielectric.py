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
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
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
