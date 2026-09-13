import numpy as np
import pytest

from sdfmpneo.unified_geometry import CoilGeometry, Pose, UnifiedUWPTGeometry, sample_geometry


def test_circle_and_rounded_square_share_one_geometry_interface():
    circle = CoilGeometry.from_mapping(
        "tx",
        dict(
            shape="circle", turns=1.5, outer_half_size=0.025, pitch=0.002,
            conductor_width=0.001, conductor_thickness=0.001,
            corner_radius=0.012, translation=[0.01, -0.02, 0.03], angles=[0.1, 0.2, -0.3],
        ),
    )
    square = CoilGeometry.from_mapping(
        "rx",
        dict(
            shape="rounded_square", turns=1.0, outer_half_size=0.025, pitch=0.002,
            conductor_width=0.001, conductor_thickness=0.001,
            corner_radius=0.012, translation=[0.0, 0.0, 0.04], angles=[0.0, 0.4, 0.2],
        ),
    )
    pc = circle.centerline(0.003)
    ps = square.centerline(0.003)
    assert pc.shape[1] == ps.shape[1] == 3
    assert len(pc) > 10 and len(ps) > 10
    assert np.all(np.isfinite(pc)) and np.all(np.isfinite(ps))
    local = np.array([[0.01, -0.02, 0.003], [-0.02, 0.01, -0.004]])
    assert np.allclose(circle.pose.inverse(circle.pose.apply(local)), local)


def test_custom_spline_is_supported_without_changing_model_semantics():
    coil = CoilGeometry(
        "custom", "spline", 0.001, 0.001, Pose(np.zeros(3), np.zeros(3)),
        control_points=np.array([
            [-0.02, 0.00, 0.0], [-0.01, 0.02, 0.0],
            [0.01, 0.02, 0.0], [0.02, 0.00, 0.0], [0.01, -0.02, 0.0],
        ]),
    )
    points = coil.centerline(0.004)
    assert len(points) >= 5
    assert np.all(np.isfinite(points))


def test_training_geometry_sampling_does_not_define_a_validity_box():
    base = {
        "transmitter": {
            "shape": "circle", "turns": 1.0, "outer_half_size": 0.02, "pitch": 0.002,
            "conductor_width": 0.001, "conductor_thickness": 0.001,
            "corner_radius": 0.01, "translation": [0, 0, 0], "angles": [0, 0, 0],
        },
        "receiver": {
            "shape": "circle", "turns": 1.0, "outer_half_size": 0.02, "pitch": 0.002,
            "conductor_width": 0.001, "conductor_thickness": 0.001,
            "corner_radius": 0.01, "translation": [0, 0, 0.03], "angles": [0, 0, 0],
        },
        "package_half_extent": [0.03, 0.03, 0.004],
    }
    sampled = sample_geometry(
        base,
        {
            "receiver": {
                "shape": {"choices": ["circle", "rounded_square"]},
                "translation": {"bounds": [[-0.02, 0.02], [-0.02, 0.02], [0.01, 0.06]]},
            }
        },
        np.random.default_rng(7),
    )
    geometry = UnifiedUWPTGeometry.from_mapping(sampled)
    assert geometry.n_ports == 2
    assert -0.02 <= geometry.coils[1].pose.translation[0] <= 0.02


def test_invalid_spiral_is_rejected_as_invalid_geometry_not_network_domain():
    with pytest.raises(ValueError, match="spiral collapses"):
        CoilGeometry.from_mapping(
            "bad",
            dict(
                shape="circle", turns=10.0, outer_half_size=0.01, pitch=0.002,
                conductor_width=0.001, conductor_thickness=0.001,
                translation=[0, 0, 0], angles=[0, 0, 0],
            ),
        )
