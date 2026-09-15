"""Exact Cartesian-cell / rotated-package intersection fractions.

The package is an oriented rectangular box (OBB), while the Maxwell material
grid is Cartesian.  A fixed indicator quadrature aliases a thin rotated package
when its thickness is below the EM cell size.  Because both regions are convex
boxes, their intersection fraction can instead be evaluated geometrically:
collect corners lying in the opposite box plus both sets of edge/face
intersections, then take the convex-hull volume.  The represented package is
therefore independent of mesh resolution, rotation and sub-cell phase up to
floating-point geometry error.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial import ConvexHull, QhullError


_SIGNS = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)), float)
_EDGE_PAIRS = tuple(
    (i, j)
    for i in range(len(_SIGNS))
    for j in range(i + 1, len(_SIGNS))
    if int(np.count_nonzero(_SIGNS[i] != _SIGNS[j])) == 1
)


def _append_unique(points, point, tolerance):
    value = np.asarray(point, float).reshape(3)
    if not np.all(np.isfinite(value)):
        return
    for existing in points:
        if np.linalg.norm(existing - value) <= tolerance:
            return
    points.append(value)


def _inside_aabb(point, lo, hi, tolerance):
    value = np.asarray(point, float)
    return bool(np.all(value >= lo - tolerance) and np.all(value <= hi + tolerance))


def _inside_obb(package, half, point, tolerance):
    local = np.asarray(package.pose.inverse(np.asarray(point, float).reshape(1, 3)), float).reshape(3)
    return bool(np.all(np.abs(local) <= half + tolerance))


def _intersection_volume(package, half, obb_vertices, cell_lo, cell_hi):
    cell_lo = np.asarray(cell_lo, float).reshape(3)
    cell_hi = np.asarray(cell_hi, float).reshape(3)
    cell_center = 0.5 * (cell_lo + cell_hi)
    cell_half = 0.5 * (cell_hi - cell_lo)
    cell_vertices = cell_center + _SIGNS * cell_half
    scale = max(
        float(np.linalg.norm(cell_hi - cell_lo)),
        float(np.linalg.norm(2.0 * half)),
        1.0,
    )
    tolerance = 2.0e-12 * scale

    # Entire Cartesian cell is inside the OBB.
    if all(_inside_obb(package, half, point, tolerance) for point in cell_vertices):
        return float(np.prod(cell_hi - cell_lo))

    points = []
    for point in obb_vertices:
        if _inside_aabb(point, cell_lo, cell_hi, tolerance):
            _append_unique(points, point, tolerance)
    for point in cell_vertices:
        if _inside_obb(package, half, point, tolerance):
            _append_unique(points, point, tolerance)

    # OBB edges against the six Cartesian cell faces.
    for first, second in _EDGE_PAIRS:
        p0 = obb_vertices[first]
        p1 = obb_vertices[second]
        delta = p1 - p0
        for axis in range(3):
            denominator = float(delta[axis])
            if abs(denominator) <= tolerance:
                continue
            for plane in (cell_lo[axis], cell_hi[axis]):
                t = float((plane - p0[axis]) / denominator)
                if -tolerance <= t <= 1.0 + tolerance:
                    point = p0 + np.clip(t, 0.0, 1.0) * delta
                    if _inside_aabb(point, cell_lo, cell_hi, tolerance):
                        _append_unique(points, point, tolerance)

    # Cartesian cell edges against the six OBB faces.  Transform each edge to
    # package coordinates, where the OBB faces are simply q_axis=+/-half_axis.
    for first, second in _EDGE_PAIRS:
        p0 = cell_vertices[first]
        p1 = cell_vertices[second]
        local = np.asarray(package.pose.inverse(np.vstack((p0, p1))), float)
        q0 = local[0]
        qdelta = local[1] - local[0]
        world_delta = p1 - p0
        for axis in range(3):
            denominator = float(qdelta[axis])
            if abs(denominator) <= tolerance:
                continue
            for face in (-half[axis], half[axis]):
                t = float((face - q0[axis]) / denominator)
                if -tolerance <= t <= 1.0 + tolerance:
                    tc = float(np.clip(t, 0.0, 1.0))
                    q = q0 + tc * qdelta
                    if np.all(np.abs(q) <= half + tolerance):
                        point = p0 + tc * world_delta
                        if _inside_aabb(point, cell_lo, cell_hi, tolerance):
                            _append_unique(points, point, tolerance)

    if len(points) < 4:
        return 0.0
    values = np.asarray(points, float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered, tol=tolerance) < 3:
        return 0.0
    try:
        volume = float(ConvexHull(values).volume)
    except QhullError:
        # A genuinely three-dimensional box intersection should be hullable. If
        # Qhull encounters a near-degenerate sliver, its volume is below the
        # geometric tolerance and may safely be treated as zero.
        return 0.0
    cell_volume = float(np.prod(cell_hi - cell_lo))
    return float(np.clip(volume, 0.0, cell_volume))


def install(background_cls):
    if bool(getattr(background_cls, "_resolved_package_fraction_installed", False)):
        return background_cls

    def package_fraction(self, package):
        half = np.asarray(package.half_extent, float).reshape(3)
        if np.any(~np.isfinite(half)) or np.any(half <= 0.0):
            raise ValueError("package half extent must be finite and positive")
        obb_vertices = np.asarray(package.pose.apply(_SIGNS * half), float)
        world_lo = np.min(obb_vertices, axis=0)
        world_hi = np.max(obb_vertices, axis=0)
        index_sets = []
        for edges, lower, upper in zip((self.x, self.y, self.z), world_lo, world_hi):
            edge_values = np.asarray(edges, float)
            active = np.flatnonzero(
                (edge_values[:-1] < float(upper))
                & (edge_values[1:] > float(lower))
            )
            index_sets.append(active)

        out = np.zeros(self.n_cells, float)
        if any(len(active) == 0 for active in index_sets):
            return out
        for i, j, k in itertools.product(*index_sets):
            lo = np.array([self.x[i], self.y[j], self.z[k]], float)
            hi = np.array([self.x[i + 1], self.y[j + 1], self.z[k + 1]], float)
            volume = _intersection_volume(package, half, obb_vertices, lo, hi)
            if volume <= 0.0:
                continue
            out[self._cell_id(i, j, k)] = volume / float(self.cell_volumes[self._cell_id(i, j, k)])
        return np.clip(out, 0.0, 1.0)

    background_cls._package_fraction = package_fraction
    background_cls.package_fraction_model = "exact_obb_cartesian_cell_intersection_v1"
    background_cls._resolved_package_fraction_installed = True
    return background_cls


__all__ = ["install"]
