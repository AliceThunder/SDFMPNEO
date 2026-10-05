"""Minimal vNext mesh-free-first electrothermal MVP example."""

import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    CoilThermalProperties,
    ConductorMaterial,
    CurrentControlledEnvelope,
    HomogeneousMedium,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    build_lumped_coil_thermal_model,
)


def main():
    copper = ConductorMaterial(
        5.8e7,
        resistance_temperature_coefficient=0.00393,
    )
    tx = CoilObject(
        SuperellipseSpiral(
            0.04,
            0.035,
            1.5,
            0.003,
            0.003,
            exponent=4.0,
            conductor_width=2e-3,
            conductor_thickness=1e-3,
        ),
        copper,
        "tx",
    )
    rx = CoilObject(
        SuperellipseSpiral(
            0.032,
            0.028,
            1.25,
            0.0025,
            0.0025,
            exponent=3.5,
            conductor_width=2e-3,
            conductor_thickness=1e-3,
            pose=RigidPose.from_axis_angle(
                (1.0, 0.0, 0.0),
                np.deg2rad(20.0),
                translation=(0.008, 0.0, 0.025),
            ),
        ),
        copper,
        "rx",
    )
    scene = Scene(
        (tx, rx),
        HomogeneousMedium(),
    )
    thermal = build_lumped_coil_thermal_model(
        (
            CoilThermalProperties(12.0, 0.15),
            CoilThermalProperties(10.0, 0.13),
        ),
        mutual_conductance=np.array(
            [
                [0.0, 0.03],
                [0.03, 0.0],
            ]
        ),
    )
    envelope = CurrentControlledEnvelope(
        scene,
        85_000.0,
        thermal,
        em_config=MQSConfig(
            segments_per_turn=12,
            min_segments=18,
            section_degree=1,
            radial_order=4,
            angular_order=24,
            line_order=2,
        ),
    )
    currents = np.array(
        [4.0 + 0j, -1.5 + 0.3j]
    )
    trajectory = envelope.trajectory(
        np.zeros(2),
        currents,
        [0.0, 5.0, 20.0, 60.0],
    )
    for time, step in zip(
        [0.0, 5.0, 20.0, 60.0],
        trajectory,
    ):
        print(
            f"t={time:6.1f}s "
            f"T={step.temperatures} K "
            f"P={step.coil_power} W"
        )
    print("Z(85 kHz) =")
    print(trajectory[-1].impedance)


if __name__ == "__main__":
    main()
