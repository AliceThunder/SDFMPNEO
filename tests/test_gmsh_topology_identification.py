import numpy as np
import pytest

from sdfmpneo.spatial.gmsh_pipeline import (
    _spherical_outer_boundary_surfaces,
    _terminal_surface_at_endpoint,
)


class _FakeModel:
    def __init__(self, boundaries, types):
        self.boundaries = boundaries
        self.types = types

    def getBoundary(self, dimtags, combined=False, oriented=False, recursive=False):
        del combined, oriented, recursive
        volume = int(dimtags[0][1])
        return [(2, tag) for tag in self.boundaries[volume]]

    def getType(self, dim, tag):
        assert dim == 2
        return self.types[int(tag)]


class _FakeOCC:
    def __init__(self, centers):
        self.centers = centers

    def getCenterOfMass(self, dim, tag):
        assert dim == 2
        return self.centers[int(tag)]


def test_outer_boundary_uses_cad_surface_type_not_radius_threshold():
    model = _FakeModel(
        {10: [1, 2, 3], 11: [3, 4]},
        {1: "Plane", 2: "BSpline surface", 3: "Sphere", 4: "Plane"},
    )
    assert _spherical_outer_boundary_surfaces(model, [10, 11]) == [3]


def test_outer_boundary_fails_closed_when_no_spherical_entity_exists():
    model = _FakeModel({10: [1, 2]}, {1: "Plane", 2: "BSpline surface"})
    with pytest.raises(RuntimeError, match="no spherical"):
        _spherical_outer_boundary_surfaces(model, [10])


def test_terminal_endpoint_resolution_rejects_floating_point_tie():
    model = _FakeModel({7: [1, 2]}, {1: "Plane", 2: "Plane"})
    occ = _FakeOCC({1: np.array([1.0, 0.0, 0.0]), 2: np.array([-1.0, 0.0, 0.0])})
    with pytest.raises(RuntimeError, match="not uniquely identified"):
        _terminal_surface_at_endpoint(model, occ, 7, np.zeros(3))
