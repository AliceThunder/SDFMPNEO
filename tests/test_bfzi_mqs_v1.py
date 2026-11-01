"""Minimal BFZI end-to-end physical smoke tests, not a FEM accuracy claim."""
import numpy as np
import pytest

from sdfmpneo_bfzi import (
    BFZICoil, RigidPose, InsulatingBox, BFZIAssembly, Seawater,
    BfziScene, SeaMQSConfig, build_bfzi_source, solve_seawater_reference,
    compress_seawater_response,
)
from sdfmpneo_bfzi.vnext_bridge import BFZIVNextGeometry


def scene_pair():
    coil = BFZICoil(2, 0.075, 0.045, 0.005, alpha=0.5)
    box = InsulatingBox((0.10, 0.10, 0.015))
    route = ((0.0, 0.0, 0.008), (0.090, 0.0, 0.008))
    first = BFZIAssembly(
        coil, RigidPose(np.eye(3), np.array([0., 0., -.13])),
        box, return_waypoints=route,
    )
    second = BFZIAssembly(
        coil, RigidPose(np.eye(3), np.array([0., 0., .13])),
        box, return_waypoints=route,
    )
    return BfziScene(first, second, Seawater(5.0), outer_radius=0.52)


def test_port_source_closed_and_air_mutual():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    for assembly, line in zip(scene.assemblies, source.lines):
        # The physical spiral is oriented from outer START to inner END.
        # Its separate return starts at inner END and finishes at outer START.
        outer, inner = assembly.coil.endpoints()
        offset = 8 * assembly.coil.turns * 3
        first_return_segment = line.line_elements[offset:offset + 3].sum(axis=0)
        first_waypoint = np.asarray(assembly.return_waypoints[0])
        np.testing.assert_allclose(
            first_return_segment,
            assembly.pose.vector(first_waypoint - inner),
            atol=1e-12,
        )
        # A reversed return route would produce a nonzero total displacement.
        np.testing.assert_allclose(line.line_elements.sum(axis=0), 0., atol=1e-6)
    np.testing.assert_allclose(
        source.inductance_air, source.inductance_air.T, atol=1e-14,
    )
    assert np.all(np.isfinite(source.inductance_air))


def test_same_state_seawater_impedance_and_joule_heat():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    result = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=5, polar_order=7, azimuthal_order=10,
            field_softening=0.025,
        ),
    )
    z = result.impedance_seawater
    np.testing.assert_allclose(z, z.T, rtol=1e-9, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(0.5 * (z + z.conj().T))) >= -1e-12
    i = np.array([1.2 + 0.1j, -0.4 + 0.8j])
    np.testing.assert_allclose(
        [result.port_seawater_power(i), result.seawater_power(i)],
        [result.quadrature_power(i)] * 2, rtol=3e-8, atol=1e-12,
    )
    assert result.seawater_power(i) >= 0
    prediction = result.as_structured_prediction()
    assert prediction.power_closure_error() < 1e-8
    points = result.integration_points[:5]
    q = result.seawater_loss(points, i)
    assert q.shape == (5,) and np.all(q >= 0)
    current_field = result.current_density(points, i)
    assert current_field.shape == (5, 3)
    # In a true two-port response the current fields superpose, while
    # Joule heat is a quadratic field with cross terms.
    i_a = np.array([i[0], 0j])
    i_b = np.array([0j, i[1]])
    np.testing.assert_allclose(
        current_field,
        result.current_density(points, i_a) + result.current_density(points, i_b),
        rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        q,
        np.sum(np.abs(current_field)**2, axis=1) /
        (2 * scene.seawater.conductivity),
        rtol=1e-10, atol=1e-12,
    )
    # The deterministic projection preserves the same low-dimensional
    # port/volume power identity; it is not yet a trained FAST artifact.
    reduced = compress_seawater_response(
        result, rank=min(2, result.resistance.shape[0]),
    )
    np.testing.assert_allclose(
        reduced.port_seawater_power(i), reduced.quadrature_power(i),
        rtol=1e-7, atol=1e-12,
    )
    assert reduced.seawater_power(i) >= 0


def test_reject_overlapping_packages():
    scene = scene_pair()
    with pytest.raises(ValueError, match="overlap"):
        BfziScene(
            scene.transmitter,
            BFZIAssembly(
                scene.receiver.coil, scene.transmitter.pose,
                scene.receiver.enclosure, scene.receiver.return_waypoints,
            ),
            scene.seawater, scene.outer_radius,
        )


def test_vnext_bishop_frame_bridge():
    scene = scene_pair()
    a = scene.transmitter
    adapted = BFZIVNextGeometry(
        a.coil, a.pose, conductor_width=0.0016,
        conductor_thickness=0.0016,
    )
    polyline = adapted.polyline(80)
    assert polyline.points.shape == (81, 3)
    assert np.isfinite(polyline.normal1).all()
    assert polyline.total_length > 0


def test_box_wall_radial_cubature_preserves_sea_volume():
    scene = scene_pair()
    points, weights = scene.quadrature(5, 7, 10)
    plain_points, plain_weights = scene.quadrature(
        5, 7, 10, wall_layers=0,
    )
    assert len(points) > len(plain_points)
    np.testing.assert_allclose(
        weights.sum(), plain_weights.sum(), rtol=1e-12,
    )
    box_volumes = [
        8.0 * np.prod(assembly.enclosure.half_extents)
        for assembly in scene.assemblies
    ]
    exact_volume = (
        4.0 * np.pi * scene.outer_radius**3 / 3.0 - sum(box_volumes)
    )
    # Radial Gauss intervals terminate at actual OBB walls; the original
    # node-rejection rule missed ~0.4% of this same sea volume.
    assert abs(weights.sum() / exact_volume - 1.0) < 8e-4
    assert np.all(weights > 0.0)
    assert scene.in_seawater(points).all()


def test_finite_wire_skin_loss_and_lumped_thermal_coupling():
    from dataclasses import replace
    from sdfmpneo_vnext.scene import ConductorMaterial
    from sdfmpneo_vnext.electrothermal import (
        CoilThermalProperties, build_lumped_coil_thermal_model,
    )
    from sdfmpneo_bfzi import BFZICurrentControlledEnvelope

    original = scene_pair()
    warm_material = ConductorMaterial(
        5.8e7, resistance_temperature_coefficient=0.00393
    )
    tx = replace(original.transmitter, material=warm_material)
    rx = replace(original.receiver, material=warm_material)
    scene = replace(original, transmitter=tx, receiver=rx)
    source = build_bfzi_source(scene, 200_000., order=3)
    impedances = source.conductor_internal_impedance()
    lengths = np.array([
        np.linalg.norm(line.line_elements, axis=1).sum()
        for line in source.lines
    ])
    dc_resistance = lengths / (
        warm_material.conductivity * np.pi * tx.wire_radius**2
    )
    assert np.all(impedances.real > dc_resistance)
    assert np.all(impedances.imag > 0)
    assert np.all(
        source.conductor_internal_impedance([333.15, 333.15]).real
        > impedances.real
    )

    result = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=4, polar_order=6, azimuthal_order=8,
            field_softening=0.025,
        ),
    )
    currents = np.array([1.0 + 0.0j, 0.2 + 0.1j])
    predicted = result.as_structured_prediction()
    assert predicted.channel_labels == (
        "conductor:0", "conductor:1", "seawater",
    )
    assert predicted.power_closure_error() < 1e-8
    np.testing.assert_allclose(
        predicted.coil_power(currents),
        result.port_conductor_power(currents),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        predicted.channel_power(currents).sum(),
        result.port_total_power(currents),
        rtol=1e-8,
    )

    thermal = build_lumped_coil_thermal_model(
        (CoilThermalProperties(20.0, 0.7),
         CoilThermalProperties(20.0, 0.7)),
    )
    envelope = BFZICurrentControlledEnvelope(result, thermal)
    initial = np.zeros(2)
    step = envelope.step(initial, currents, 30.0)
    assert step.converged
    assert np.all(step.temperatures >= 293.15)
    assert np.all(step.coil_power >= 0.0)
    np.testing.assert_allclose(
        step.impedance, result.port_impedance_at(step.temperatures),
    )


def test_rotated_box_cubature_and_return_wire_clearance():
    from dataclasses import replace
    scene = scene_pair()
    angle = 0.37
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    rotated = replace(
        scene, receiver=replace(
            scene.receiver,
            pose=RigidPose(rotation, scene.receiver.pose.center),
        ),
    )
    points, weights = rotated.quadrature(4, 9, 14, wall_layers=2)
    assert len(points) > 0
    assert rotated.in_seawater(points).all()
    assert np.all(weights > 0.0)
    with pytest.raises(ValueError, match="return route"):
        replace(
            scene.transmitter,
            return_waypoints=((0., 0., 0.008), (0.11, 0., 0.008)),
        )
    with pytest.raises(ValueError, match="pitch"):
        replace(scene.transmitter, wire_radius=0.003)


def test_voltage_control_recomputes_current_at_warm_port_impedance():
    from dataclasses import replace
    from sdfmpneo_bfzi import BFZIVoltageControlledEnvelope
    from sdfmpneo_vnext.scene import ConductorMaterial
    from sdfmpneo_vnext.electrothermal import (
        CoilThermalProperties, build_lumped_coil_thermal_model,
    )

    initial_scene = scene_pair()
    copper = ConductorMaterial(
        5.8e7, resistance_temperature_coefficient=0.00393,
    )
    scene = replace(
        initial_scene,
        transmitter=replace(initial_scene.transmitter, material=copper),
        receiver=replace(initial_scene.receiver, material=copper),
    )
    source = build_bfzi_source(scene, 200_000., order=3)
    reference = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=4, polar_order=6, azimuthal_order=8,
            field_softening=0.025,
        ),
    )
    thermal = build_lumped_coil_thermal_model((
        CoilThermalProperties(20.0, 0.7),
        CoilThermalProperties(20.0, 0.7),
    ))
    external = np.diag([0.05 - 0.2j, 0.07 - 0.2j])
    envelope = BFZIVoltageControlledEnvelope(
        reference, thermal, external_impedance=external,
    )
    voltage = np.array([1.0 + 0.2j, 0.0j])
    step = envelope.step(np.zeros(2), voltage, 30.0)
    assert step.converged
    np.testing.assert_allclose(
        (step.impedance + external) @ step.currents, voltage,
        rtol=1e-10, atol=1e-10,
    )
    np.testing.assert_allclose(
        step.coil_power,
        reference.port_conductor_power(step.currents, step.temperatures),
    )
    with pytest.raises(ValueError, match="passive"):
        BFZIVoltageControlledEnvelope(
            reference, thermal, external_impedance=-np.eye(2),
        )


def test_round_wire_joule_profile_and_complete_path_heat_are_consistent():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    currents = np.array([1.1 + 0.2j, -0.4 + 0.5j])
    radius_nodes, radial_weights = np.polynomial.legendre.leggauss(24)
    fraction = 0.5 * (radius_nodes + 1)
    density = source.conductor_cross_section_loss(fraction, currents)
    assert density.shape == (24, 2)
    assert np.all(density >= 0.0)
    assert np.all(
        source.conductor_cross_section_loss(1.0, currents)
        > source.conductor_cross_section_loss(0.0, currents)
    )
    area_integral = np.array([
        np.pi * assembly.wire_radius**2
        * np.sum(radial_weights * fraction * density[:, port])
        for port, assembly in enumerate(scene.assemblies)
    ])
    lengths = np.array([
        np.linalg.norm(line.line_elements, axis=1).sum()
        for line in source.lines
    ])
    expected = 0.5 * source.conductor_internal_impedance().real * np.abs(currents)**2
    np.testing.assert_allclose(
        area_integral * lengths, expected, rtol=1e-8,
    )
    nodes = source.conductor_heat_nodes(currents)
    np.testing.assert_allclose(
        [watts.sum() for _, watts in nodes], expected, rtol=1e-12,
    )
