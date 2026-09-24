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
        "CORRECTED_OUT_OF_FAST_DOMAIN",
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
        fast_domain_correction_limit=1.0,
    )
    assert result.status == "CERTIFIED"
    assert result.certified



def test_matrix_free_mixed_certification_matches_dense_corrected_ports():
    scene = _scene()
    config = _cfg(7)
    frequency = 8_000.0
    artifact = AnalyticBaselineArtifact(
        segments_per_coil=20,
    )

    dense = certify_mixed_ports(
        scene,
        frequency,
        artifact,
        config=config,
        algebraic_tolerance=1e-6,
        correction_rtol=1e-9,
        correction_restart=30,
        correction_maxiter=160,
        allow_reference_fallback=False,
        operator_backend="dense",
    )
    matrix_free = certify_mixed_ports(
        scene,
        frequency,
        artifact,
        config=config,
        algebraic_tolerance=1e-6,
        correction_rtol=1e-9,
        correction_restart=30,
        correction_maxiter=160,
        allow_reference_fallback=False,
        operator_backend="matrix_free",
        matrix_free_chunk_size=37,
    )

    assert dense.algebraic_certified
    assert matrix_free.algebraic_certified
    assert dense.operator_backend == "dense"
    assert matrix_free.operator_backend == "matrix_free"
    assert matrix_free.result.inductance_matrix is None
    assert matrix_free.port_certificate.certified
    assert matrix_free.final_residual <= 1e-6
    assert np.allclose(
        matrix_free.impedance,
        dense.impedance,
        rtol=5e-6,
        atol=5e-8,
    )
