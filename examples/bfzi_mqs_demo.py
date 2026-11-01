"""Run a small BFZI/seawater finite-domain MQS Galerkin example.

From repository root:
    python examples/bfzi_mqs_demo.py
The reported numbers are UNCONVERGED prototype results, not experimental
predictions or certified infinite-ocean accuracy.
"""
import numpy as np
from sdfmpneo_vnext.scene import ConductorMaterial

from sdfmpneo_bfzi import (
    BFZICoil, RigidPose, InsulatingBox, BFZIAssembly, BfziScene,
    Seawater, SeaMQSConfig, build_bfzi_source, solve_seawater_reference,
    BFZICurrentControlledEnvelope,
)


def build_demo_scene():
    coil = BFZICoil(
        turns=2, outer_half_size=0.075, corner_radius=0.045,
        pitch=0.005, alpha=0.5,
    )
    box = InsulatingBox((0.10, 0.10, 0.015))
    # This example's external return routing is explicit and is not an
    # automatic assumption about real charging-pad terminal connections.
    return_route = ((0.0, 0.0, 0.008), (0.09, 0.0, 0.008))
    copper = ConductorMaterial(
        5.8e7, resistance_temperature_coefficient=0.00393,
    )
    tx = BFZIAssembly(
        coil, RigidPose(np.eye(3), np.array([0.0, 0.0, -0.13])),
        box, return_route, material=copper,
    )
    rx = BFZIAssembly(
        coil, RigidPose(np.eye(3), np.array([0.0, 0.0, 0.13])),
        box, return_route, material=copper,
    )
    return BfziScene(tx, rx, Seawater(5.0), outer_radius=0.52)


def main():
    scene = build_demo_scene()
    sources = build_bfzi_source(scene, frequency_hz=200_000, order=4)
    result = solve_seawater_reference(
        sources, config=SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.025,
        ),
    )
    current = np.array([1.0, 0.3 + 0.2j])
    print("Air L (H):", sources.inductance_air)
    print("Seawater reaction Z (ohm):", result.impedance_seawater)
    print("Baseline port Z (ohm):", result.port_impedance)
    print("Sea power from port (W):", result.port_seawater_power(current))
    print("Sea power from same cubature (W):", result.quadrature_power(current))
    print("Finite conductor internal Z (ohm):", sources.conductor_internal_impedance())
    print("Conductor Joule power per coil (W):", result.port_conductor_power(current))
    print("Structured channel closure:", result.as_structured_prediction().power_closure_error())
    # Illustrative coil-only heat capacities/conductances, not calibrated
    # packaging/seawater temperature predictions.
    from sdfmpneo_vnext.electrothermal import (
        CoilThermalProperties, build_lumped_coil_thermal_model,
    )
    thermal = build_lumped_coil_thermal_model((
        CoilThermalProperties(20.0, 0.7),
        CoilThermalProperties(20.0, 0.7),
    ))
    step = BFZICurrentControlledEnvelope(result, thermal).step(
        np.zeros(2), current, 30.0,
    )
    print("Illustrative 30 s coil temperatures (K):", step.temperatures)
    print("Retained modes:", result.resistance.shape[0], "/", result.raw_basis_count)
    print("Note: finite-domain/basis/softening convergence not established")


if __name__ == "__main__":
    main()
