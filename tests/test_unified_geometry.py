import numpy as np
import pytest

from sdfmpneo.unified_geometry import (
    CoilGeometry,
    Pose,
    UnifiedUWPTGeometry,
    apply_geometry_family,
    geometry_family_coordinates,
    geometry_family_dimension,
    sample_geometry,
    sample_geometry_family,
)


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


def _production_family_fixture():
    base = {
        "transmitter": {
            "shape": "circle",
            "turns": 1.5,
            "outer_half_size": 0.025,
            "pitch": 0.002,
            "conductor_width": 0.0015,
            "conductor_thickness": 0.001,
            "corner_radius": 0.012,
            "translation": [0.0, 0.0, 0.0],
            "angles": [0.0, 0.0, 0.0],
        },
        "receiver": {
            "shape": "circle",
            "turns": 1.5,
            "outer_half_size": 0.025,
            "pitch": 0.002,
            "conductor_width": 0.0015,
            "conductor_thickness": 0.001,
            "corner_radius": 0.012,
            "translation": [0.0, 0.0, 0.035],
            "angles": [0.0, 0.0, 0.0],
        },
        "package_half_extent": [0.035, 0.035, 0.005],
    }
    family = {
        "schema": "scaled_uwpt_family_v1",
        "parameters": {
            "tx_planar_scale": {"bounds": [0.97, 1.03]},
            "rx_planar_scale": {"bounds": [0.97, 1.03]},
            "tx_thickness_scale": {"bounds": [0.95, 1.05]},
            "rx_thickness_scale": {"bounds": [0.95, 1.05]},
            "rx_offset_x": {"bounds": [-0.0002, 0.0002]},
            "rx_offset_y": {"bounds": [-0.0002, 0.0002]},
            "rx_gap": {"bounds": [0.0348, 0.0352]},
            "tx_package_scale": {"bounds": [0.98, 1.02]},
            "rx_package_scale": {"bounds": [0.98, 1.02]},
        },
    }
    return base, family


def test_scaled_production_family_roundtrip_and_linked_planar_dimensions():
    base, family = _production_family_fixture()
    assert geometry_family_dimension(family) == 9
    coordinates = {
        "tx_planar_scale": 1.02,
        "rx_planar_scale": 0.98,
        "tx_thickness_scale": 1.03,
        "rx_thickness_scale": 0.97,
        "rx_offset_x": 1.5e-4,
        "rx_offset_y": -1.0e-4,
        "rx_gap": 0.0351,
        "tx_package_scale": 1.01,
        "rx_package_scale": 0.99,
    }
    geometry = apply_geometry_family(
        base,
        family,
        coordinates,
    )
    assert np.isclose(
        geometry["transmitter"]["pitch"],
        base["transmitter"]["pitch"]
        * coordinates["tx_planar_scale"],
    )
    assert np.isclose(
        geometry["transmitter"]["conductor_width"],
        base["transmitter"]["conductor_width"]
        * coordinates["tx_planar_scale"],
    )
    assert np.isclose(
        geometry["receiver"]["translation"][0],
        coordinates["rx_offset_x"],
    )
    assert np.isclose(
        geometry["receiver"]["translation"][2],
        coordinates["rx_gap"],
    )
    recovered = geometry_family_coordinates(
        base,
        family,
        geometry,
    )
    for name, value in coordinates.items():
        assert np.isclose(recovered[name], value)


def test_scaled_production_family_sampling_stays_on_manifold():
    base, family = _production_family_fixture()
    rng = np.random.default_rng(41)
    for _ in range(20):
        geometry = sample_geometry_family(
            base,
            family,
            rng,
        )
        recovered = geometry_family_coordinates(
            base,
            family,
            geometry,
        )
        assert set(recovered) == set(
            family["parameters"]
        )
        UnifiedUWPTGeometry.from_mapping(geometry)


def test_scaled_production_family_rejects_independent_pitch_or_topology_change():
    base, family = _production_family_fixture()
    coordinates = {
        name: 0.5 * (
            spec["bounds"][0] + spec["bounds"][1]
        )
        for name, spec in family["parameters"].items()
    }
    geometry = apply_geometry_family(
        base,
        family,
        coordinates,
    )
    broken_pitch = {
        **geometry,
        "transmitter": dict(
            geometry["transmitter"]
        ),
    }
    broken_pitch["transmitter"]["pitch"] *= 1.01
    with pytest.raises(
        ValueError,
        match="does not follow planar_scale",
    ):
        geometry_family_coordinates(
            base,
            family,
            broken_pitch,
        )

    broken_shape = {
        **geometry,
        "receiver": dict(
            geometry["receiver"]
        ),
    }
    broken_shape["receiver"]["shape"] = "rounded_square"
    with pytest.raises(
        ValueError,
        match="shape is outside",
    ):
        geometry_family_coordinates(
            base,
            family,
            broken_shape,
        )
