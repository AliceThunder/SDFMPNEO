from types import SimpleNamespace

import numpy as np
import pytest

from sdfmpneo import (
    CertifiedMultiChartGeometryFamily,
    MultiChartGeometryAnalyticEvolutionOperator,
)


class _FakeFamily:
    def __init__(self, n_modes, geometry_names, marker):
        self.n_modes = n_modes
        self.geometry_names = tuple(geometry_names)
        self.n_geometry = len(self.geometry_names)
        self.marker = marker

    def chart(self, geometry):
        field = SimpleNamespace(marker=self.marker, geometry=np.asarray(geometry, dtype=float))
        return SimpleNamespace(field=field)


class _FakeOperator:
    def __init__(self, n_modes, n_operating, marker):
        self.graph = SimpleNamespace(n_modes=n_modes)
        self.n_operating = n_operating
        self.physical_operating_names = tuple(f"u{k}" for k in range(n_operating))
        self.marker = marker

    def evaluate(self, t, *, geometry, a0, operating, stable=True):
        return SimpleNamespace(
            marker=self.marker,
            time=float(t),
            geometry=np.asarray(geometry),
            initial=np.asarray(a0),
            operating=np.asarray(operating),
            stable=bool(stable),
        )


def test_multi_chart_geometry_allows_different_full_mesh_families_behind_same_rank_interface():
    atlas = CertifiedMultiChartGeometryFamily(
        {
            "circle": _FakeFamily(3, ("radius", "offset"), "circle-mesh"),
            "rounded_square": _FakeFamily(3, ("half", "corner", "offset"), "square-mesh"),
        }
    )
    assert atlas.chart_names == ("circle", "rounded_square")
    assert atlas.n_modes == 3
    selected = atlas.select("rounded_square", np.array([1.0, 0.2, 0.1]))
    assert selected.field.marker == "square-mesh"
    assert np.array_equal(selected.geometry, np.array([1.0, 0.2, 0.1]))


def test_multi_chart_geometry_rejects_rank_mismatch():
    with pytest.raises(ValueError, match="same retained thermal rank"):
        CertifiedMultiChartGeometryFamily(
            {
                "a": _FakeFamily(2, ("g",), "a"),
                "b": _FakeFamily(3, ("g",), "b"),
            }
        )


def test_multi_chart_analytic_operator_dispatches_without_cross_chart_interpolation():
    operator = MultiChartGeometryAnalyticEvolutionOperator(
        {
            "circle": _FakeOperator(3, 2, "circle-op"),
            "rounded_square": _FakeOperator(3, 2, "square-op"),
        }
    )
    result = operator.evaluate(
        "circle",
        2.5,
        geometry=np.array([0.2]),
        a0=np.zeros(3),
        operating=np.array([1.0, 2.0]),
    )
    assert result.chart_name == "circle"
    assert result.prediction.marker == "circle-op"
