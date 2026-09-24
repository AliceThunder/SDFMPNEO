import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    CoilThermalProperties,
    ConductorMaterial,
    FastCurrentControlledEnvelope,
    FastVoltageControlledEnvelope,
    HomogeneousMedium,
    Scene,
    SuperellipseSpiral,
    build_lumped_coil_thermal_model,
)


def _scene():
    copper = ConductorMaterial(
        5.8e7,
        resistance_temperature_coefficient=0.00393,
        reference_temperature=293.15,
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.021,
            0.8,
            0.001,
            0.001,
            exponent=3.0,
            conductor_width=1.0e-3,
            conductor_thickness=0.8e-3,
        ),
        copper,
        "coil",
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
    )


def _thermal():
    return build_lumped_coil_thermal_model(
        (
            CoilThermalProperties(
                8.0,
                0.10,
            ),
        ),
        ambient_temperature=293.15,
    )


def test_analytic_structured_artifact_closes_dissipation():
    scene = _scene()
    artifact = AnalyticBaselineArtifact(
        segments_per_coil=48,
    )
    prediction = artifact.predict_structured(
        scene,
        40_000.0,
    )
    assert prediction.power_closure_error() < 1e-14
    assert prediction.reciprocity_defect() < 1e-14
    assert np.allclose(
        np.sum(
            prediction.dissipation_channels,
            axis=0,
        ),
        prediction.impedance.real,
        rtol=0,
        atol=1e-14,
    )


def test_fast_current_control_heats_without_maxwell_teacher():
    scene = _scene()
    envelope = FastCurrentControlledEnvelope(
        scene,
        40_000.0,
        _thermal(),
        AnalyticBaselineArtifact(
            segments_per_coil=48,
        ),
        coupling_tolerance=1e-9,
        max_coupling_iterations=20,
    )
    cold = envelope.step(
        np.zeros(1),
        np.array([2.0 + 0j]),
        0.0,
    )
    hot = envelope.step(
        np.zeros(1),
        np.array([2.0 + 0j]),
        20.0,
    )
    assert hot.converged
    assert hot.temperatures[0] > cold.temperatures[0]
    assert hot.impedance[0, 0].real > cold.impedance[0, 0].real
    assert hot.coil_power[0] > 0.0


def test_fast_voltage_control_reduces_current_as_resistance_rises():
    scene = _scene()
    envelope = FastVoltageControlledEnvelope(
        scene,
        40_000.0,
        _thermal(),
        AnalyticBaselineArtifact(
            segments_per_coil=48,
        ),
        external_impedance=np.array(
            [[0.05 + 0j]]
        ),
        coupling_tolerance=1e-9,
        max_coupling_iterations=20,
    )
    cold = envelope.step(
        np.zeros(1),
        np.array([1.0 + 0j]),
        0.0,
    )
    hot = envelope.step(
        np.zeros(1),
        np.array([1.0 + 0j]),
        30.0,
    )
    assert hot.converged
    assert hot.temperatures[0] > cold.temperatures[0]
    assert abs(hot.currents[0]) < abs(cold.currents[0])
