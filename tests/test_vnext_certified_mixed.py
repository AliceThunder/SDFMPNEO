import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    certify_mixed_ports,
    mixed_impedance_convergence,
)


def _scene():
    copper = ConductorMaterial(5.8e7)
    first = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.021,
            0.75,
            0.0012,
            0.0012,
            exponent=3.0,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ),
        copper,
    )
    second = CoilObject(
        SuperellipseSpiral(
            0.020,
            0.018,
            0.70,
            0.0010,
            0.0010,
            exponent=3.5,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
        ).transformed(
            RigidPose(
                np.eye(3),
                np.array([0.004, 0.0, 0.018]),
            )
        ),
        copper,
    )
    return Scene(
        (first, second),
        HomogeneousMedium(),
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


def test_mixed_certification_reduces_physical_residual_without_fallback():
    result = certify_mixed_ports(
        _scene(),
        15_000.0,
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        config=_cfg(8),
        algebraic_tolerance=1e-7,
        correction_rtol=1e-9,
        correction_restart=30,
        correction_maxiter=120,
        allow_reference_fallback=False,
    )
    assert result.algebraic_certified
    assert not result.used_reference_fallback
    assert result.final_residual <= 1e-7
    assert (
        result.final_residual
        <= result.initial_residual
        + 1e-14
    )
    assert result.status in (
        "DISCRETE_CERTIFIED",
        "CERTIFIED",
    )


def test_mixed_convergence_report_controls_full_certification_status():
    scene = _scene()
    report = mixed_impedance_convergence(
        scene,
        12_000.0,
        (
            _cfg(7),
            _cfg(8),
        ),
        tolerance=1.0,
    )
    assert report.converged
    result = certify_mixed_ports(
        scene,
        12_000.0,
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        ),
        config=_cfg(8),
        convergence_report=report,
        algebraic_tolerance=1e-7,
        correction_rtol=1e-9,
        correction_restart=30,
        correction_maxiter=120,
        allow_reference_fallback=False,
    )
    assert result.status == "CERTIFIED"
    assert result.certified
