"""Minimal BFZI end-to-end physical smoke tests, not a FEM accuracy claim."""
import numpy as np
import pytest

from sdfmpneo_bfzi import (
    BFZICoil, RigidPose, InsulatingBox, BFZIAssembly, Seawater,
    BfziScene, SeaMQSConfig, build_bfzi_source, solve_seawater_reference,
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
    for line in source.lines:
        # Exact closed-return circulation, up to Gauss rule error.
        np.testing.assert_allclose(line.line_elements.sum(axis=0), 0., atol=3e-5)
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
    assert result.current_density(points, i).shape == (5, 3)


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
