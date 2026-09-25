import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    MQSConfig,
    PackageObject,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    certify_dielectric_ports,
    hybrid_reference_convergence,
)


def _scene():
    coil = CoilObject(
        SuperellipseSpiral(
            0.014,
            0.012,
            0.6,
            0.001,
            0.001,
            conductor_width=7e-4,
            conductor_thickness=5e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )
    package = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.022, 0.019, 0.005]
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
        segments_per_turn=6,
        min_segments=8,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )


def test_hybrid_reference_convergence_refines_conductor_and_surface_axes():
    report = hybrid_reference_convergence(
        _scene(),
        60_000.0,
        _config(),
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        tolerance=0.5,
        surface_residual_tolerance=1e-7,
    )
    assert {
        direction.name
        for direction in report.directions
    } == {
        "longitudinal",
        "cross_section",
        "quadrature",
        "dielectric_surface",
    }
    assert np.isfinite(
        report.maximum_relative_change
    )
    assert np.isfinite(
        report.maximum_surface_residual
    )
    assert report.converged


def test_hybrid_full_certificate_requires_independent_convergence_evidence():
    scene = _scene()
    artifact = AnalyticBaselineArtifact(
        segments_per_coil=24,
    )
    discrete = certify_dielectric_ports(
        scene,
        60_000.0,
        artifact,
        config=_config(),
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        algebraic_tolerance=1e-8,
        surface_tolerance=1e-7,
        fast_domain_correction_limit=1.0,
    )
    assert (
        discrete.status
        == "DISCRETE_CERTIFIED"
    )
    assert discrete.algebraic_certified
    assert not discrete.certified
    assert discrete.used_reference_fallback

    report = hybrid_reference_convergence(
        scene,
        60_000.0,
        _config(),
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        tolerance=0.5,
        surface_residual_tolerance=1e-7,
    )
    certified = certify_dielectric_ports(
        scene,
        60_000.0,
        artifact,
        config=_config(),
        convergence_report=report,
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        algebraic_tolerance=1e-8,
        surface_tolerance=1e-7,
        fast_domain_correction_limit=1.0,
    )
    assert certified.certified
    assert certified.discretization_certified
    assert (
        certified.status
        == "CERTIFIED"
    )
    assert (
        certified.operator_backend
        == "dense_dielectric_reference"
    )
