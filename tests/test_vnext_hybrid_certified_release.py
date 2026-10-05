from types import SimpleNamespace
import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    DielectricCoupledReferenceArtifact,
    HomogeneousMedium,
    IsotropicMaterial,
    MQSConfig,
    PackageObject,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    audit_certified_release,
)


def _config(segments):
    return MQSConfig(
        segments_per_turn=segments,
        min_segments=segments,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
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


def test_hybrid_release_audit_dispatches_to_dielectric_certification():
    scene = _scene()
    frequency = 60_000.0
    truth = (
        DielectricCoupledReferenceArtifact(
            config=_config(7),
            surface_vertical_order=4,
            surface_azimuthal_order=8,
        ).predict_structured(
            scene,
            frequency,
        ).impedance
    )
    sample = SimpleNamespace(
        scene=scene,
        frequency_hz=frequency,
        target_impedance=truth,
        surface_vertical_order=4,
        surface_azimuthal_order=8,
    )
    report = audit_certified_release(
        AnalyticBaselineArtifact(
            segments_per_coil=20,
        ),
        (sample,),
        coarse_config=_config(6),
        fine_config=_config(7),
        convergence_tolerance=1.0,
        algebraic_tolerance=1e-6,
        correction_rtol=1e-9,
        correction_maxiter=80,
        fast_domain_correction_limit=10.0,
        truth_consistency_tolerance=1.0,
        operator_backend="dense",
    )
    assert report.samples == 1
    assert (
        report.certified_samples
        == 1
    )
    assert (
        report.fast_domain_valid_samples
        == 1
    )
    assert (
        report.maximum_dielectric_surface_change
        >= 0.0
    )
    assert (
        report.operator_backend
        == "dense_dielectric_reference"
    )
    assert report.passed
