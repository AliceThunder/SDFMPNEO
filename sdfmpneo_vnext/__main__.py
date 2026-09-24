from __future__ import annotations

import argparse
import numpy as np

from .certification import certify_port_result
from .electrothermal import (
    CoilThermalProperties,
    CurrentControlledEnvelope,
    build_lumped_coil_thermal_model,
)
from .em import DenseMQSTeacher, MQSConfig
from .geometry import RigidPose, SuperellipseSpiral
from .scene import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    Scene,
)


def _demo_scene() -> Scene:
    copper = ConductorMaterial(
        5.8e7,
        resistance_temperature_coefficient=0.00393,
    )
    tx = CoilObject(
        SuperellipseSpiral(
            0.035,
            0.030,
            1.0,
            0.002,
            0.002,
            exponent=4.0,
            conductor_width=1.5e-3,
            conductor_thickness=0.9e-3,
        ),
        copper,
        "tx",
    )
    rx = CoilObject(
        SuperellipseSpiral(
            0.028,
            0.024,
            1.0,
            0.0015,
            0.0015,
            exponent=3.5,
            conductor_width=1.4e-3,
            conductor_thickness=0.8e-3,
            pose=RigidPose.from_axis_angle(
                (1.0, 0.0, 0.0),
                np.deg2rad(15.0),
                translation=(0.006, 0.0, 0.024),
            ),
        ),
        copper,
        "rx",
    )
    return Scene(
        (tx, rx),
        HomogeneousMedium(),
    )


def self_check() -> int:
    scene = _demo_scene()
    cfg = MQSConfig(
        segments_per_turn=10,
        min_segments=12,
        section_degree=1,
        radial_order=3,
        angular_order=16,
        line_order=2,
    )
    teacher = DenseMQSTeacher(
        scene,
        85_000.0,
        cfg,
    )
    result = teacher.solve()
    cert = certify_port_result(
        result,
        residual_tolerance=1e-8,
        reciprocity_tolerance=1e-8,
        power_tolerance=1e-7,
    )
    thermal = build_lumped_coil_thermal_model(
        (
            CoilThermalProperties(10.0, 0.14),
            CoilThermalProperties(8.0, 0.12),
        ),
        mutual_conductance=np.array(
            [
                [0.0, 0.025],
                [0.025, 0.0],
            ]
        ),
    )
    envelope = CurrentControlledEnvelope(
        scene,
        85_000.0,
        thermal,
        em_config=cfg,
    )
    step = envelope.step(
        np.zeros(2),
        np.array([3.0 + 0j, -1.0 + 0.2j]),
        10.0,
    )

    np.set_printoptions(precision=6, suppress=False)
    print("SDF-MPNEO vNext MVP self-check")
    print("Z(85 kHz) [ohm]:")
    print(result.impedance)
    print(
        "certificate:",
        {
            "certified": cert.certified,
            "reciprocity_defect": cert.reciprocity_defect,
            "power_closure_error": cert.power_closure_error,
            "min_dissipation_eigenvalue": cert.minimum_dissipation_eigenvalue,
        },
    )
    print("T(10 s) [K]:", step.temperatures)
    print("coil power [W]:", step.coil_power)
    print(
        "thermal coupling:",
        {
            "converged": step.converged,
            "iterations": step.iterations,
            "residual": step.coupling_residual,
        },
    )
    return 0 if cert.certified and step.converged else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sdfmpneo_vnext",
        description="SDF-MPNEO vNext mesh-free-first MVP utilities",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the end-to-end dense MVP self-check",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
