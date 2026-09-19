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
        "encode_geometry",
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
        "encode_geometry",
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
