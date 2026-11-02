"""Minimal BFZI end-to-end physical smoke tests, not a FEM accuracy claim."""
import numpy as np
import pytest

from sdfmpneo_bfzi import (
    BFZICoil, RigidPose, InsulatingBox, BFZIAssembly, Seawater,
    BfziScene, SeaMQSConfig, build_bfzi_source, solve_seawater_reference,
    compress_seawater_response, BFZICurrentControlledEnvelope,
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
    volumes = source.conductor_volume_heat(
        currents, radial_order=12, angular_order=8,
    )
    np.testing.assert_allclose(
        [np.dot(volume_weights, density)
         for _, volume_weights, density in volumes],
        expected, rtol=1e-8,
    )
    for (positions, volume_weights, density), assembly, length in zip(
        volumes, scene.assemblies, lengths,
    ):
        assert positions.shape == (len(volume_weights), 3)
        assert np.all(volume_weights > 0)
        assert np.all(density >= 0)
        np.testing.assert_allclose(
            volume_weights.sum(), length * np.pi * assembly.wire_radius**2,
            rtol=1e-12,
        )


def test_nested_face_basis_refinement_keeps_original_current_modes():
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    coarse = build_sea_basis(scene, face_refinement=1)
    refined = build_sea_basis(scene, face_refinement=2)
    assert refined.size > coarse.size
    for original in coarse.modes:
        assert any(
            np.array_equal(original.center, candidate.center)
            and np.array_equal(original.direction, candidate.direction)
            and original.width == candidate.width
            for candidate in refined.modes
        )
    points, _ = scene.quadrature(3, 4, 6, wall_layers=1)
    fields = refined.evaluate(points[:5])
    assert fields.shape == (5, refined.size, 3)
    assert np.isfinite(fields).all()


def test_reference_refuses_basis_from_a_different_outer_domain():
    from dataclasses import replace
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    sources = build_bfzi_source(scene, 200_000., order=3)
    wrong_basis = build_sea_basis(replace(scene, outer_radius=0.65))
    with pytest.raises(ValueError, match="source scene"):
        solve_seawater_reference(sources, basis=wrong_basis)


def test_symmetric_green_tiles_are_independent_of_chunk_partition():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    base = dict(
        radial_order=4, polar_order=5, azimuthal_order=8,
        wall_layers=1, field_softening=0.025,
    )
    first = solve_seawater_reference(
        source, config=SeaMQSConfig(**base, kernel_chunk_size=17),
    )
    second = solve_seawater_reference(
        source, config=SeaMQSConfig(**base, kernel_chunk_size=29),
    )
    np.testing.assert_allclose(
        first.inductance, second.inductance, rtol=2e-12, atol=1e-14,
    )
    np.testing.assert_allclose(
        first.impedance_seawater, second.impedance_seawater,
        rtol=2e-12, atol=1e-14,
    )
    assert np.min(np.linalg.eigvalsh(first.inductance)) > -1e-12


def test_seawater_heat_nodes_drive_separate_thermal_rom_channel():
    from sdfmpneo_vnext.electrothermal import (
        CoilThermalProperties, build_lumped_coil_thermal_model,
    )

    scene = scene_pair()
    reference = solve_seawater_reference(
        build_bfzi_source(scene, 200_000., order=3),
        config=SeaMQSConfig(
            radial_order=4, polar_order=5, azimuthal_order=8,
            wall_layers=1, field_softening=0.025,
        ),
    )
    currents = np.array([1.3 + 0.2j, -0.3 + 0.4j])
    points, weights, heat = reference.seawater_heat_nodes(currents)
    assert len(points) == len(weights) == len(heat)
    assert scene.in_seawater(points).all()
    assert np.all(weights > 0)
    assert np.all(heat >= 0)
    np.testing.assert_allclose(
        weights @ heat, reference.port_seawater_power(currents),
        rtol=3e-9, atol=1e-12,
    )
    three_channels = reference.thermal_channel_power(currents)
    np.testing.assert_allclose(
        three_channels,
        reference.as_structured_prediction().channel_power(currents),
        rtol=3e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        three_channels[:2], reference.port_conductor_power(currents),
    )
    np.testing.assert_allclose(three_channels[2], weights @ heat, rtol=3e-9)

    thermal = build_lumped_coil_thermal_model(
        (CoilThermalProperties(20.0, 0.7),
         CoilThermalProperties(20.0, 0.7),
         CoilThermalProperties(100.0, 0.2)),
        mutual_conductance=np.array([
            [0.0, 0.0, 0.04],
            [0.0, 0.0, 0.04],
            [0.04, 0.04, 0.0],
        ]),
    )
    envelope = BFZICurrentControlledEnvelope(reference, thermal)
    step = envelope.step(np.zeros(3), currents, 30.0)
    assert step.converged
    assert step.state.shape == (3,)
    assert step.temperatures.shape == (3,)
    assert step.temperatures[2] >= thermal.reference_temperature
    np.testing.assert_allclose(
        step.coil_power,
        reference.port_conductor_power(currents, step.temperatures[:2]),
    )


def test_air_self_neumann_does_not_depend_on_coarse_port_gauss_nodes():
    scene = scene_pair()
    low = build_bfzi_source(
        scene, 200_000., order=3, air_panel_length_factor=4.0,
    )
    high = build_bfzi_source(
        scene, 200_000., order=5, air_panel_length_factor=4.0,
    )
    np.testing.assert_allclose(
        np.diag(low.inductance_air), np.diag(high.inductance_air),
        rtol=1e-13, atol=1e-16,
    )
    assert np.all(np.diag(low.inductance_air) > 0.0)
    # The mutual term may vary with source Gauss order, as intended.
    np.testing.assert_allclose(
        low.inductance_air, low.inductance_air.T, atol=1e-15,
    )


def test_analytical_straight_panel_self_integral_matches_high_order_quad():
    from scipy.integrate import quad

    length = 0.041
    radius = 0.0008
    analytic = 2 * (
        length * np.arcsinh(length / radius)
        - np.hypot(length, radius) + radius
    )
    # One-dimensional equivalent of the double Neumann panel integral.
    numeric = 2 * quad(
        lambda delta: (length - delta) / np.hypot(delta, radius),
        0.0, length, epsabs=1e-11,
    )[0]
    np.testing.assert_allclose(analytic, numeric, rtol=1e-12)


def test_box_shell_partition_is_positive_and_volume_consistent():
    from dataclasses import replace

    scene = scene_pair()
    points, weights = scene.box_shell_quadrature(
        8, 14, 24, shell_normal_order=3, shell_tangential_order=6,
    )
    assert np.all(weights > 0.0)
    assert scene.in_seawater(points).all()
    exact_volume = (
        4 * np.pi * scene.outer_radius**3 / 3
        - sum(8 * np.prod(a.enclosure.half_extents)
              for a in scene.assemblies)
    )
    # A smooth partition of unity allows positive overlap between the
    # shell and sphere cubatures: their weights must NOT be naively added
    # as unweighted disjoint shell volumes.
    assert abs(weights.sum() / exact_volume - 1) < 0.02

    angle = 0.29
    rotation = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)],
    ])
    rotated = replace(
        scene, receiver=replace(
            scene.receiver,
            pose=RigidPose(rotation, scene.receiver.pose.center),
        ),
    )
    rot_points, rot_weights = rotated.box_shell_quadrature(
        8, 14, 24, shell_normal_order=3, shell_tangential_order=6,
    )
    assert rotated.in_seawater(rot_points).all()
    assert np.all(rot_weights > 0.0)
    assert abs(rot_weights.sum() / exact_volume - 1) < 0.02


def test_box_shell_reference_retains_two_port_energy_identity():
    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000.0, order=3)
    reference = solve_seawater_reference(
        source, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            field_softening=0.025, quadrature_scheme="box_shell",
        ),
    )
    currents = np.array([1.0 + 0.2j, -0.3 + 0.5j])
    np.testing.assert_allclose(
        reference.port_seawater_power(currents),
        reference.quadrature_power(currents),
        rtol=1e-8, atol=1e-12,
    )
    assert np.all(reference.integration_weights > 0)
    assert scene.in_seawater(reference.integration_points).all()
    assert np.isfinite(reference.impedance_seawater).all()


def test_box_shell_tangential_gauss_and_global_modes_keep_geometry_consistent():
    from dataclasses import replace
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    # Changing the artificial outer sphere must not secretly rescale the
    # underlying global Gaussian trial functions.
    larger = replace(scene, outer_radius=0.70)
    global_a = build_sea_basis(scene).modes[-3:]
    global_b = build_sea_basis(larger).modes[-3:]
    np.testing.assert_allclose(
        [mode.width for mode in global_a],
        [mode.width for mode in global_b],
        rtol=0, atol=1e-14,
    )
    for order in (4, 5):
        points, weights = scene.box_shell_quadrature(
            5, 8, 12, shell_normal_order=3,
            shell_tangential_order=order,
        )
        assert scene.in_seawater(points).all()
        assert np.all(weights > 0.0)
    # The composite tangential rule now splits every physical edge and
    # face center, rather than moving a few global Gauss nodes past them.



def test_restricted_galerkin_subspace_matches_independent_direct_solve():
    """One Green assembly gives the exact same-cubature nested 15-mode solve."""
    from sdfmpneo_bfzi import (
        build_sea_basis, restrict_seawater_response,
    )

    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    config = SeaMQSConfig(
        radial_order=3, polar_order=4, azimuthal_order=6,
        shell_normal_order=2, shell_tangential_order=3,
        field_softening=0.025,
    )
    enriched_basis = build_sea_basis(scene, modes_per_face=2)
    coarse_basis = build_sea_basis(scene, modes_per_face=1)
    fine = solve_seawater_reference(
        source, basis=enriched_basis, config=config,
    )
    # Every first tangent in the 27-mode hierarchy is one of its
    # original face-center functions. The three global modes also match.
    kept = []
    for mode in coarse_basis.modes:
        matching = [
            i for i, candidate in enumerate(enriched_basis.modes)
            if np.array_equal(candidate.center, mode.center)
            and np.array_equal(candidate.direction, mode.direction)
            and candidate.width == mode.width
        ]
        assert len(matching) == 1
        kept.extend(matching)
    coarse_from_fine = restrict_seawater_response(fine, kept)
    direct = solve_seawater_reference(
        source, basis=coarse_basis, config=config,
    )
    np.testing.assert_allclose(
        coarse_from_fine.impedance_seawater, direct.impedance_seawater,
        rtol=2e-7, atol=1e-12,
    )
    currents = np.array([1.0 + 0.3j, -0.6 + 0.2j])
    np.testing.assert_allclose(
        coarse_from_fine.quadrature_power(currents),
        direct.quadrature_power(currents),
        rtol=2e-7, atol=1e-12,
    )
    probe = np.array([[0., 0., 0.], [0.14, 0.0, 0.0]])
    np.testing.assert_allclose(
        coarse_from_fine.current_density(probe, currents),
        direct.current_density(probe, currents),
        rtol=2e-7, atol=1e-11,
    )


def test_thin_face_refinement_is_strictly_nested_in_original_basis():
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    basic = build_sea_basis(scene, modes_per_face=2)
    patches = build_sea_basis(
        scene, modes_per_face=2, face_refinement=2,
        refine_thin_faces_only=True,
    )
    assert basic.size == 27
    assert patches.size == 59
    for mode in basic.modes:
        assert any(
            np.array_equal(mode.center, candidate.center)
            and np.array_equal(mode.direction, candidate.direction)
            and mode.width == candidate.width
            for candidate in patches.modes
        )



def test_source_guided_curl_modes_match_complete_port_vector_potential():
    """A spatially weighted closed-wire A has the expected solenoidal curl."""
    from sdfmpneo_bfzi import build_sea_basis

    scene = scene_pair()
    source = build_bfzi_source(scene, 200_000., order=3)
    base = build_sea_basis(scene, modes_per_face=2)
    guided = build_sea_basis(
        scene, modes_per_face=2, sources=source,
        include_source_guided=True,
    )
    assert guided.size == base.size + 4
    for i in range(base.size):
        np.testing.assert_allclose(
            guided.modes[i].center, base.modes[i].center
        )
        np.testing.assert_allclose(
            guided.modes[i].direction, base.modes[i].direction
        )
        assert guided.modes[i].width == base.modes[i].width
        assert guided.modes[i].source_port == -1
    assert [m.source_port for m in guided.modes[-4:]] == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="complete port sources"):
        build_sea_basis(scene, include_source_guided=True)

    probe = np.array([0.155, 0.025, 0.17])
    assert scene.in_seawater(probe)
    mode = guided.modes[-2]
    line = guided.source_lines[mode.source_port]

    def potential(x):
        outer = scene.outer_radius
        rel = x - scene.outer_center
        envelope = 1 - np.dot(rel, rel) / outer**2
        for assembly in scene.assemblies:
            dist2, _ = assembly.enclosure.squared_exterior_distance(
                x, assembly.pose,
            )
            d = np.sqrt(dist2)
            envelope *= d / (d + guided.wall_scale)
        offset = x - mode.center
        gaussian = np.exp(-np.dot(offset, offset) / mode.width**2)
        return envelope * gaussian * line.potential_kernel(x)

    h = 1e-5
    derivatives = []
    for axis in range(3):
        delta = np.eye(3)[axis] * h
        derivatives.append(
            (potential(probe + delta) - potential(probe - delta)) / (2 * h)
        )
    curl = np.array([
        derivatives[1][2] - derivatives[2][1],
        derivatives[2][0] - derivatives[0][2],
        derivatives[0][1] - derivatives[1][0],
    ])
    np.testing.assert_allclose(
        guided.evaluate(probe[None, :])[0, -2], curl,
        rtol=2e-5, atol=1e-7,
    )

    response = solve_seawater_reference(
        source, basis=guided, config=SeaMQSConfig(
            radial_order=3, polar_order=4, azimuthal_order=6,
            shell_normal_order=2, shell_tangential_order=3,
            field_softening=0.025,
        ),
    )
    currents = np.array([1.0 + 0.2j, -0.4 + 0.3j])
    assert np.isfinite(response.impedance_seawater).all()
    np.testing.assert_allclose(
        response.port_seawater_power(currents),
        response.quadrature_power(currents), rtol=2e-8, atol=1e-12,
    )
