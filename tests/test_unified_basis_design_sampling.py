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
