import numpy as np
import pytest

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    DielectricCoupledResult,
    HomogeneousMedium,
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


def test_system_reference_dispatches_package_scene_to_coupled_sie():
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        reference_config=_config(),
        dielectric_surface_vertical_order=8,
        dielectric_surface_azimuthal_order=16,
    )
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


def test_system_refuses_to_mislabel_package_scene_as_certified_or_spatial():
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        reference_config=_config(),
        dielectric_surface_vertical_order=8,
        dielectric_surface_azimuthal_order=16,
    )
    scene = _scene()
    with pytest.raises(
        NotImplementedError,
        match="CERTIFIED",
    ):
        system.certified_ports(
            scene,
            80_000.0,
        )
    with pytest.raises(
        NotImplementedError,
        match="dielectric heat",
    ):
        system.reference_spatial(
            scene,
            80_000.0,
        )
