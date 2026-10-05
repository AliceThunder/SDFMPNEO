import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    CoilThermalProperties,
    ConductorMaterial,
    CurrentControlledEnvelope,
    HomogeneousMedium,
    MQSConfig,
    Scene,
    SuperellipseSpiral,
    build_lumped_coil_thermal_model,
)


def _single_scene(alpha=0.00393):
    copper = ConductorMaterial(
        5.8e7,
        resistance_temperature_coefficient=alpha,
        reference_temperature=293.15,
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.022,
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
        segments_per_turn=8,
        min_segments=10,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )


def test_temperature_dependent_conductivity_decreases_with_temperature():
    material = _single_scene().coils[0].material
    assert (
        material.conductivity_at(333.15)
        < material.conductivity
    )


def test_current_controlled_envelope_heats_and_increases_resistance():
    scene = _single_scene()
    thermal = build_lumped_coil_thermal_model(
        (
            CoilThermalProperties(
                8.0,
                0.12,
            ),
        ),
        ambient_temperature=293.15,
    )
    envelope = CurrentControlledEnvelope(
        scene,
        20_000.0,
        thermal,
        em_config=_config(),
        coupling_tolerance=1e-8,
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
    assert (
        hot.temperatures[0]
        > 293.15
    )
    assert (
        hot.impedance[0, 0].real
        > cold.impedance[0, 0].real
    )
    assert hot.coil_power[0] > 0.0


def test_lumped_thermal_builder_has_passive_pair_coupling():
    model = build_lumped_coil_thermal_model(
        (
            CoilThermalProperties(5.0, 0.1),
            CoilThermalProperties(7.0, 0.2),
        ),
        mutual_conductance=np.array(
            [
                [0.0, 0.05],
                [0.05, 0.0],
            ]
        ),
    )
    assert np.allclose(
        model.conductance,
        model.conductance.T,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                model.conductance
            )
        )
        > 0.0
    )
