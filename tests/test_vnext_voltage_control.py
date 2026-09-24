import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    CoilThermalProperties,
    ConductorMaterial,
    HomogeneousMedium,
    MQSConfig,
    Scene,
    SuperellipseSpiral,
    VoltageControlledEnvelope,
    build_lumped_coil_thermal_model,
)


def _scene(alpha=0.00393):
    copper = ConductorMaterial(
        5.8e7,
        resistance_temperature_coefficient=alpha,
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.023,
            0.020,
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


def _thermal():
    return build_lumped_coil_thermal_model(
        (CoilThermalProperties(7.0, 0.10),),
        ambient_temperature=293.15,
    )


def _cfg():
    return MQSConfig(
        segments_per_turn=8,
        min_segments=10,
        section_degree=0,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )


def test_voltage_control_changes_current_with_temperature():
    env = VoltageControlledEnvelope(
        _scene(),
        20_000.0,
        _thermal(),
        external_impedance=np.array([[0.05 + 0j]]),
        em_config=_cfg(),
        coupling_tolerance=1e-8,
        max_coupling_iterations=20,
    )
    cold = env.step(
        np.zeros(1),
        np.array([1.0 + 0j]),
        0.0,
    )
    hot = env.step(
        np.zeros(1),
        np.array([1.0 + 0j]),
        30.0,
    )
    assert hot.converged
    assert hot.temperatures[0] > cold.temperatures[0]
    assert (
        abs(hot.currents[0])
        < abs(cold.currents[0])
    )


def test_voltage_control_rejects_active_external_impedance():
    try:
        VoltageControlledEnvelope(
            _scene(),
            20_000.0,
            _thermal(),
            external_impedance=np.array([[-0.1 + 0j]]),
            em_config=_cfg(),
        )
    except ValueError as exc:
        assert "passive" in str(exc)
    else:
        raise AssertionError(
            "active external impedance must be rejected"
        )
