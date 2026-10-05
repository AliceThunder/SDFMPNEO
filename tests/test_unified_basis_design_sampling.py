import numpy as np

import sdfmpneo.unified_runtime as runtime


def test_basis_design_geometries_use_maximin_distance_from_reference(monkeypatch):
    candidates = [
        {"x": 0.1},
        {"x": 0.2},
        {"x": 0.8},
        {"x": 0.9},
    ]

    monkeypatch.setattr(
        runtime,
        "_sample_geometries",
        lambda settings, n, rng, background: candidates[: int(n)],
    )
    monkeypatch.setattr(
        runtime,
        "_thermal_basis_design_features",
        lambda geometry: np.asarray([float(geometry["x"])], float),
    )

    settings = {
        "DEFAULT_GEOMETRY": {"x": 0.5},
        "TRAINING": {"basis_design_pool_multiplier": 2},
    }
    selected = runtime._basis_design_geometries(
        settings,
        2,
        np.random.default_rng(17),
        object(),
    )

    assert selected == [{"x": 0.1}, {"x": 0.9}]


def test_basis_design_geometry_count_is_exact(monkeypatch):
    candidates = [{"x": float(v)} for v in np.linspace(0.0, 1.0, 12)]

    monkeypatch.setattr(
        runtime,
        "_sample_geometries",
        lambda settings, n, rng, background: candidates[: int(n)],
    )
    monkeypatch.setattr(
        runtime,
        "_thermal_basis_design_features",
        lambda geometry: np.asarray(
            [float(geometry["x"]), float(geometry["x"]) ** 2],
            float,
        ),
    )

    settings = {
        "DEFAULT_GEOMETRY": {"x": 0.5},
        "TRAINING": {"basis_design_pool_multiplier": 3},
    }
    selected = runtime._basis_design_geometries(
        settings,
        4,
        np.random.default_rng(17),
        object(),
    )

    assert len(selected) == 4
    assert len({row["x"] for row in selected}) == 4


def test_geometry_sampling_rejects_adjacent_turn_overlap(monkeypatch):
    invalid = {
        "transmitter": {
            "shape": "circle",
            "turns": 1.5,
            "outer_half_size": 0.03,
            "pitch": 0.0012,
            "conductor_width": 0.0020,
            "conductor_thickness": 0.0010,
            "corner_radius": 0.012,
            "translation": [0.0, 0.0, 0.0],
            "angles": [0.0, 0.0, 0.0],
        },
        "receiver": {
            "shape": "circle",
            "turns": 1.5,
            "outer_half_size": 0.03,
            "pitch": 0.0012,
            "conductor_width": 0.0020,
            "conductor_thickness": 0.0010,
            "corner_radius": 0.012,
            "translation": [0.0, 0.0, 0.04],
            "angles": [0.0, 0.0, 0.0],
        },
        "package_half_extent": [0.04, 0.04, 0.006],
    }
    valid = {
        **invalid,
        "transmitter": {**invalid["transmitter"], "pitch": 0.0025},
        "receiver": {**invalid["receiver"], "pitch": 0.0025},
    }
    candidates = iter((invalid, valid))

    monkeypatch.setattr(
        runtime,
        "sample_geometry",
        lambda base, sampling, rng: next(candidates),
    )
    monkeypatch.setattr(
        runtime,
        "encode_geometry",
        lambda geometry: np.asarray([1.0], float),
    )

    class Background:
        @staticmethod
        def validate_geometry(geometry):
            return geometry

    selected = runtime._sample_geometries(
        {"DEFAULT_GEOMETRY": valid, "GEOMETRY_SAMPLING": {}},
        1,
        np.random.default_rng(17),
        Background(),
    )

    assert selected == [valid]



def test_thermal_basis_design_features_ignore_common_pose_but_keep_relative_pose():
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
            "shape": "rounded_square",
            "turns": 1.8,
            "outer_half_size": 0.03,
            "pitch": 0.0025,
            "conductor_width": 0.0012,
            "conductor_thickness": 0.0008,
            "corner_radius": 0.018,
            "translation": [0.0, 0.0, 0.04],
            "angles": [0.0, 0.0, 0.2],
        },
        "package_half_extent": [0.04, 0.035, 0.006],
    }

    shifted = {
        **base,
        "transmitter": {
            **base["transmitter"],
            "translation": [0.01, -0.02, 0.03],
        },
        "receiver": {
            **base["receiver"],
            "translation": [0.01, -0.02, 0.07],
        },
    }
    relative_changed = {
        **base,
        "receiver": {
            **base["receiver"],
            "translation": [0.01, 0.0, 0.04],
        },
    }

    f0 = runtime._thermal_basis_design_features(base)
    f_shifted = runtime._thermal_basis_design_features(shifted)
    f_relative = runtime._thermal_basis_design_features(relative_changed)

    assert np.allclose(f0, f_shifted)
    assert not np.allclose(f0, f_relative)


def test_thermal_basis_design_features_ignore_common_yaw():
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
            "translation": [0.0, 0.0, 0.04],
            "angles": [0.0, 0.0, 0.3],
        },
        "package_half_extent": [0.04, 0.04, 0.006],
    }
    rotated = {
        **base,
        "transmitter": {
            **base["transmitter"],
            "angles": [0.0, 0.0, 0.7],
        },
        "receiver": {
            **base["receiver"],
            "angles": [0.0, 0.0, 1.0],
        },
    }

    assert np.allclose(
        runtime._thermal_basis_design_features(base),
        runtime._thermal_basis_design_features(rotated),
        atol=1e-12,
    )



def test_hybrid_basis_design_unions_full_and_intrinsic_maximin(monkeypatch):
    candidates = [
        {"x": 0.0, "y": 0.5},
        {"x": 0.2, "y": 1.0},
        {"x": 0.4, "y": 0.0},
        {"x": 0.6, "y": 0.9},
        {"x": 0.8, "y": 0.1},
        {"x": 1.0, "y": 0.6},
        {"x": 0.1, "y": 0.2},
        {"x": 0.9, "y": 0.8},
    ]

    monkeypatch.setattr(
        runtime,
        "_sample_geometries",
        lambda settings, n, rng, background: candidates[: int(n)],
    )
    monkeypatch.setattr(
        runtime,
        "encode_geometry",
        lambda geometry: np.asarray([float(geometry["x"])], float),
    )
    monkeypatch.setattr(
        runtime,
        "_thermal_basis_design_features",
        lambda geometry: np.asarray([float(geometry["y"])], float),
    )

    settings = {
        "DEFAULT_GEOMETRY": {"x": 0.5, "y": 0.5},
        "TRAINING": {
            "basis_design_pool_multiplier": 2,
            "thermal_basis_design": "hybrid_full_intrinsic_union_v1",
        },
    }
    selected = runtime._basis_design_geometries(
        settings,
        4,
        np.random.default_rng(17),
        object(),
    )

    assert len(selected) == 4
    assert len({(row["x"], row["y"]) for row in selected}) == 4
    # Full-space and intrinsic-space extremes must both be represented.
    assert any(row["x"] in {0.0, 1.0} for row in selected)
    assert any(row["y"] in {0.0, 1.0} for row in selected)
