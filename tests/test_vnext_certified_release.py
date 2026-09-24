import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    EncodedScene,
    HomogeneousMedium,
    MQSConfig,
    Scene,
    SuperellipseSpiral,
    TeacherSample,
    analytic_port_baseline,
    audit_certified_release,
    encode_scene_invariant,
)


def _sample():
    scene = Scene(
        (
            CoilObject(
                SuperellipseSpiral(
                    0.024,
                    0.020,
                    0.75,
                    0.001,
                    0.001,
                    exponent=3.0,
                    conductor_width=1e-3,
                    conductor_thickness=8e-4,
                ),
                ConductorMaterial(5.8e7),
            ),
        ),
        HomogeneousMedium(),
    )
    frequency = 8_000.0
    encoded = encode_scene_invariant(
        scene,
        frequency,
    )
    baseline = analytic_port_baseline(
        scene,
        frequency,
        segments_per_coil=20,
    )
    target = baseline.impedance
    return TeacherSample(
        scene,
        frequency,
        encoded,
        baseline.resistance,
        target.imag,
        target,
        20,
    )


def _cfg(segments):
    return MQSConfig(
        segments_per_turn=segments,
        min_segments=segments,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )


def test_certified_release_audit_requires_physical_and_fast_domain_pass():
    report = audit_certified_release(
        AnalyticBaselineArtifact(
            segments_per_coil=20,
        ),
        (_sample(),),
        coarse_config=_cfg(6),
        fine_config=_cfg(7),
        convergence_tolerance=1.0,
        algebraic_tolerance=1e-6,
        correction_rtol=1e-9,
        correction_maxiter=120,
        fast_domain_correction_limit=10.0,
        truth_consistency_tolerance=10.0,
        operator_backend="dense",
    )
    assert report.samples == 1
    assert report.certified_samples == 1
    assert report.fast_domain_valid_samples == 1
    assert report.maximum_final_residual <= 1e-6
    assert report.maximum_discretization_change <= 1.0
    assert report.maximum_longitudinal_change <= 1.0
    assert report.maximum_cross_section_change <= 1.0
    assert report.maximum_quadrature_change <= 1.0
    assert report.passed
