import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    DenseMQSTeacher,
    HomogeneousMedium,
    MQSConfig,
    Scene,
    SuperellipseSpiral,
    certify_mqs_ports,
)


def _scene():
    copper = ConductorMaterial(
        5.8e7
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.021,
            0.8,
            0.001,
            0.001,
            exponent=3.0,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ),
        copper,
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
    )


def _config():
    return MQSConfig(
        segments_per_turn=10,
        min_segments=10,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )


def test_dc_uniform_fast_lift_passes_discrete_kkt_without_correction():
    artifact = AnalyticBaselineArtifact(
        segments_per_coil=10,
    )
    result = certify_mqs_ports(
        _scene(),
        0.0,
        artifact,
        config=_config(),
        algebraic_tolerance=1e-10,
        allow_reference_fallback=False,
    )
    assert result.status == "DISCRETE_CERTIFIED"
    assert result.algebraic_certified
    assert not result.discretization_certified
    assert not result.used_reference_fallback
    assert result.initial_residual < 1e-10
    assert result.final_residual < 1e-10
    assert result.correction_iterations == (0,)


def test_convergence_evidence_is_required_for_full_certified_status():
    class _Converged:
        converged = True

    result = certify_mqs_ports(
        _scene(),
        0.0,
        AnalyticBaselineArtifact(
            segments_per_coil=10,
        ),
        config=_config(),
        convergence_report=_Converged(),
        algebraic_tolerance=1e-10,
        allow_reference_fallback=False,
        fast_domain_correction_limit=1.0,
    )
    assert result.status == "CERTIFIED"
    assert result.certified
    assert result.discretization_certified


def test_ac_correction_returns_the_discrete_teacher_solution():
    scene = _scene()
    config = _config()
    frequency = 20_000.0
    certified = certify_mqs_ports(
        scene,
        frequency,
        AnalyticBaselineArtifact(
            segments_per_coil=10,
        ),
        config=config,
        algebraic_tolerance=1e-8,
        correction_rtol=1e-11,
        correction_maxiter=120,
        allow_reference_fallback=True,
    )
    reference = DenseMQSTeacher(
        scene,
        frequency,
        config,
    ).solve()
    assert certified.algebraic_certified
    assert np.allclose(
        certified.impedance,
        reference.impedance,
        rtol=2e-7,
        atol=2e-9,
    )
    assert certified.final_residual <= 1e-8
    assert (
        certified.initial_residual
        >= certified.final_residual
    )



def test_large_physical_correction_is_reported_outside_fast_domain():
    from sdfmpneo_vnext import StructuredPortPrediction

    class _BadFastArtifact:
        def __init__(self):
            self.base = AnalyticBaselineArtifact(
                segments_per_coil=10,
            )

        def predict_structured(self, scene, frequency_hz):
            prediction = self.base.predict_structured(
                scene,
                frequency_hz,
            )
            return StructuredPortPrediction(
                5.0 * prediction.impedance,
                5.0 * prediction.dissipation_channels,
            )

    class _Converged:
        converged = True

    result = certify_mqs_ports(
        _scene(),
        0.0,
        _BadFastArtifact(),
        config=_config(),
        convergence_report=_Converged(),
        algebraic_tolerance=1e-10,
        correction_rtol=1e-12,
        correction_maxiter=120,
        allow_reference_fallback=False,
        fast_domain_correction_limit=0.05,
    )
    assert result.algebraic_certified
    assert result.discretization_certified
    assert result.certified
    assert not result.fast_domain_valid
    assert (
        result.status
        == "CORRECTED_OUT_OF_FAST_DOMAIN"
    )
    assert result.relative_observable_correction > 0.05
