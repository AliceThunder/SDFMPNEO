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
