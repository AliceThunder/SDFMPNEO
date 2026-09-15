"""Mesh-stable subcell integration for thin rotated package material regions.

A production EM cell can be wider than the package thickness.  A fixed 3x3x3
indicator quadrature then aliases a rotated insulating package differently on
12-mm and 9-mm grids, which directly contaminates terminal self response.  The
physical OBB is unchanged; only its cell-volume integral is evaluated with a
physical subcell resolution tied to the package's thinnest dimension.
"""
from __future__ import annotations

from collections import defaultdict
import itertools
import numpy as np


_SUBCELLS_PER_MINIMUM_THICKNESS = 4.0
_MIN_SUBDIVISIONS = 2
_MAX_SUBDIVISIONS = 64
_CELL_CHUNK = 256


def _midpoint_offsets(nx, ny, nz):
    axes = [
        (np.arange(int(n), dtype=float) + 0.5) / float(n) - 0.5
        for n in (nx, ny, nz)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([value.reshape(-1) for value in mesh])


def install(background_cls):
    if bool(getattr(background_cls, "_resolved_package_fraction_installed", False)):
        return background_cls

    def package_fraction(self, package):
        half = np.asarray(package.half_extent, float).reshape(3)
        if np.any(~np.isfinite(half)) or np.any(half <= 0.0):
            raise ValueError("package half extent must be finite and positive")
        minimum_thickness = float(2.0 * np.min(half))
        target = minimum_thickness / _SUBCELLS_PER_MINIMUM_THICKNESS
        if target <= np.finfo(float).tiny:
            raise ValueError("package subcell integration target is invalid")

        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)), float)
        corners = np.asarray(package.pose.apply(signs * half), float)
        lo = np.min(corners, axis=0)
        hi = np.max(corners, axis=0)
        index_sets = []
        for edges, lower, upper in zip((self.x, self.y, self.z), lo, hi):
            active = np.flatnonzero(
                (np.asarray(edges[:-1], float) < float(upper))
                & (np.asarray(edges[1:], float) > float(lower))
            )
            index_sets.append(active)
        out = np.zeros(self.n_cells, float)
        if any(len(active) == 0 for active in index_sets):
            return out

        groups = defaultdict(list)
        for i, j, k in itertools.product(*index_sets):
            widths = np.array([self.dx[i], self.dy[j], self.dz[k]], float)
            subdivisions = tuple(
                max(_MIN_SUBDIVISIONS, int(np.ceil(float(width) / target)))
                for width in widths
            )
            if max(subdivisions) > _MAX_SUBDIVISIONS:
                raise RuntimeError(
                    "package material integration requires excessive subcell refinement; "
                    "enlarge the global fine-core coverage rather than accepting an aliased package"
                )
            groups[subdivisions].append((int(i), int(j), int(k)))

        for subdivisions, cells in groups.items():
            normalized = _midpoint_offsets(*subdivisions)
            samples_per_cell = int(normalized.shape[0])
            for start in range(0, len(cells), _CELL_CHUNK):
                chunk = cells[start : start + _CELL_CHUNK]
                centers = np.asarray(
                    [
                        [
                            0.5 * (self.x[i] + self.x[i + 1]),
                            0.5 * (self.y[j] + self.y[j + 1]),
                            0.5 * (self.z[k] + self.z[k + 1]),
                        ]
                        for i, j, k in chunk
                    ],
                    float,
                )
                widths = np.asarray(
                    [[self.dx[i], self.dy[j], self.dz[k]] for i, j, k in chunk],
                    float,
                )
                points = centers[:, None, :] + widths[:, None, :] * normalized[None, :, :]
                inside = np.asarray(
                    package.contains(points.reshape(-1, 3)), bool
                ).reshape(len(chunk), samples_per_cell)
                fractions = np.mean(inside, axis=1)
                for (i, j, k), fraction in zip(chunk, fractions):
                    out[self._cell_id(i, j, k)] = float(fraction)

        return np.clip(out, 0.0, 1.0)

    background_cls._package_fraction = package_fraction
    background_cls.package_fraction_model = "physical_obb_composite_midpoint_v1"
    background_cls.package_fraction_subcells_per_minimum_thickness = (
        _SUBCELLS_PER_MINIMUM_THICKNESS
    )
    background_cls._resolved_package_fraction_installed = True
    return background_cls


__all__ = ["install"]
