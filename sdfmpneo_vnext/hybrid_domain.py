from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .scene import Scene


@dataclass(frozen=True)
class PackageDomainTopology:
    parent: tuple[int | None, ...]
    depth: tuple[int, ...]

    def deepest_containing(
        self,
        packages,
        points,
        *,
        tolerance: float = 1e-10,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        scalar = points.ndim == 1
        points = np.atleast_2d(points)
        region = np.full(len(points), -1, dtype=int)
        best_depth = np.full(len(points), -1, dtype=int)
        for index, package in enumerate(packages):
            inside = np.asarray(
                package.geometry.contains(points, tolerance=tolerance),
                dtype=bool,
            )
            select = inside & (self.depth[index] > best_depth)
            region[select] = index
            best_depth[select] = self.depth[index]
        return region[0] if scalar else region


def package_domain_topology(
    packages,
    *,
    surface_vertical_order: int = 17,
    surface_azimuthal_order: int = 48,
    tolerance: float = 1e-9,
) -> PackageDomainTopology:
    packages = tuple(packages)
    n = len(packages)
    if n == 0:
        return PackageDomainTopology((), ())
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    surfaces = tuple(
        package.geometry.surface_points(
            vertical_order=surface_vertical_order,
            azimuthal_order=surface_azimuthal_order,
        )
        for package in packages
    )
    contains = np.zeros((n, n), dtype=bool)
    for outer in range(n):
        for inner in range(n):
            if outer == inner:
                continue
            level = np.asarray(
                packages[outer].geometry.implicit(surfaces[inner]),
                dtype=float,
            )
            if np.all(level <= -tolerance):
                contains[outer, inner] = True
            elif np.any(np.abs(level) < tolerance):
                raise ValueError(
                    "package surfaces touch/intersect; material regions must "
                    "be strictly nested or disjoint"
                )

    for left in range(n):
        for right in range(left + 1, n):
            if contains[left, right] or contains[right, left]:
                continue
            left_in_right = np.any(
                packages[right].geometry.contains(
                    surfaces[left], tolerance=tolerance
                )
            )
            right_in_left = np.any(
                packages[left].geometry.contains(
                    surfaces[right], tolerance=tolerance
                )
            )
            if left_in_right or right_in_left:
                raise ValueError(
                    "package volumes partially overlap; only strict nesting "
                    "or disjoint material regions are supported"
                )

    parent = []
    for inner in range(n):
        candidates = [outer for outer in range(n) if contains[outer, inner]]
        if not candidates:
            parent.append(None)
            continue
        direct = candidates[0]
        for candidate in candidates[1:]:
            if contains[direct, candidate]:
                direct = candidate
        parent.append(direct)

    depth = []
    for index in range(n):
        value = 0
        cursor = parent[index]
        seen = set()
        while cursor is not None:
            if cursor in seen:
                raise RuntimeError("cyclic package containment topology")
            seen.add(cursor)
            value += 1
            cursor = parent[cursor]
        depth.append(value)
    return PackageDomainTopology(tuple(parent), tuple(depth))


def _declared_range(
    mapping,
    name: str,
):
    values = np.asarray(
        mapping[
            name
        ],
        dtype=float,
    )
    if (
        values.shape
        != (
            2,
        )
        or np.any(
            ~np.isfinite(
                values
            )
        )
        or values[
            1
        ]
        < values[
            0
        ]
    ):
        raise ValueError(
            f"invalid declared geometry range: {name}"
        )
    return (
        float(
            values[
                0
            ]
        ),
        float(
            values[
                1
            ]
        ),
    )


def _within(
    value: float,
    bounds,
) -> bool:
    lower, upper = bounds
    tolerance = (
        1e-12
        * max(
            abs(
                lower
            ),
            abs(
                upper
            ),
            1.0,
        )
    )
    return bool(
        value
        >= lower
        - tolerance
        and value
        <= upper
        + tolerance
    )


def _enclosure_key(
    root,
    package,
):
    return (
        float(
            root.outer_a
        ),
        float(
            root.outer_b
        ),
        float(
            root.turns
        ),
        float(
            root.pitch_a
        ),
        float(
            root.pitch_b
        ),
        float(
            root.exponent
        ),
        float(
            root.conductor_width
        ),
        float(
            root.conductor_thickness
        ),
        float(
            root.cross_section_exponent
        ),
        tuple(
            np.asarray(
                root.pose.rotation,
                dtype=float,
            ).reshape(
                -1
            )
        ),
        tuple(
            np.asarray(
                root.pose.translation,
                dtype=float,
            )
        ),
        tuple(
            np.asarray(
                package.half_extents,
                dtype=float,
            )
        ),
        float(
            package.exponent_xy
        ),
        float(
            package.exponent_z
        ),
        tuple(
            np.asarray(
                package.pose.rotation,
                dtype=float,
            ).reshape(
                -1
            )
        ),
        tuple(
            np.asarray(
                package.pose.translation,
                dtype=float,
            )
        ),
    )


def validate_package_conductor_topology(
    scene: Scene,
    *,
    validated_pairs=None,
) -> None:
    """Reject package surfaces that cut through finite conductor volumes."""
    for package in scene.packages:
        for coil in scene.coils:
            key = _enclosure_key(
                coil.geometry,
                package.geometry,
            )
            if (
                validated_pairs
                is not None
                and key
                in validated_pairs
            ):
                continue
            package.geometry.classify_conductor(
                coil.geometry,
                longitudinal_segments=64,
                section_points=16,
                tolerance=1e-10,
            )
            if validated_pairs is not None:
                validated_pairs.add(
                    key
                )


def validate_hybrid_geometry_domain(
    scene: Scene,
    frequency_hz: float,
    domain,
    *,
    validated_enclosures=None,
) -> None:
    """Fail closed when a scene leaves the declared hybrid design domain."""
    if domain is None:
        return
    if not isinstance(
        domain,
        dict,
    ):
        raise TypeError(
            "geometry_domain must be a dictionary"
        )

    expected_coils = int(
        domain.get(
            "n_coils",
            len(
                scene.coils
            ),
        )
    )
    expected_packages = int(
        domain.get(
            "n_packages",
            len(
                scene.packages
            ),
        )
    )
    if len(
        scene.coils
    ) != expected_coils:
        raise ValueError(
            "coil count is outside the hybrid artifact geometry domain"
        )
    if len(
        scene.packages
    ) != expected_packages:
        raise ValueError(
            "package count is outside the hybrid artifact geometry domain"
        )

    conductor = domain.get(
        "conductor"
    )
    if not isinstance(
        conductor,
        dict,
    ):
        raise ValueError(
            "geometry domain is missing conductor metadata"
        )

    if not _within(
        float(
            frequency_hz
        ),
        _declared_range(
            conductor,
            "frequency_range",
        ),
    ):
        raise ValueError(
            "frequency is outside the hybrid artifact geometry domain"
        )

    for coil in scene.coils:
        geometry = (
            coil.geometry
        )
        radius = float(
            np.sqrt(
                geometry.outer_a
                * geometry.outer_b
            )
        )
        aspect = float(
            geometry.outer_a
            / geometry.outer_b
        )
        values = {
            "outer_radius_range": (
                radius
            ),
            "aspect_ratio_range": (
                aspect
            ),
            "turns_range": float(
                geometry.turns
            ),
            "pitch_range": float(
                geometry.pitch_a
            ),
            "exponent_range": float(
                geometry.exponent
            ),
            "width_range": float(
                geometry.conductor_width
            ),
            "thickness_range": float(
                geometry.conductor_thickness
            ),
            "conductivity_range": float(
                coil.material.conductivity
            ),
        }
        for name, value in values.items():
            if not _within(
                value,
                _declared_range(
                    conductor,
                    name,
                ),
            ):
                raise ValueError(
                    f"{name} value is outside the hybrid artifact geometry domain"
                )
        if not _within(
            float(
                geometry.pitch_b
            ),
            _declared_range(
                conductor,
                "pitch_range",
            ),
        ):
            raise ValueError(
                "pitch_range value is outside the hybrid artifact geometry domain"
            )
        if not _within(
            float(
                geometry.cross_section_exponent
            ),
            _declared_range(
                conductor,
                "exponent_range",
            ),
        ):
            raise ValueError(
                "cross-section exponent is outside the hybrid artifact "
                "geometry domain"
            )

    if len(
        scene.coils
    ) == 2:
        separation = float(
            np.linalg.norm(
                scene.coils[
                    1
                ].geometry.pose.translation
                - scene.coils[
                    0
                ].geometry.pose.translation
            )
        )
        if not _within(
            separation,
            _declared_range(
                conductor,
                "separation_range",
            ),
        ):
            raise ValueError(
                "coil separation is outside the hybrid artifact geometry domain"
            )

    package_domain = domain.get(
        "package"
    )
    if not isinstance(
        package_domain,
        dict,
    ):
        raise ValueError(
            "geometry domain is missing package metadata"
        )
    if (
        len(
            scene.packages
        )
        == 1
    ):
        package = scene.packages[
            0
        ].geometry
        root = scene.coils[
            0
        ].geometry
        if not _within(
            float(
                package.exponent_xy
            ),
            _declared_range(
                package_domain,
                "exponent_xy_range",
            ),
        ):
            raise ValueError(
                "package exponent_xy is outside the hybrid artifact geometry domain"
            )
        if not _within(
            float(
                package.exponent_z
            ),
            _declared_range(
                package_domain,
                "exponent_z_range",
            ),
        ):
            raise ValueError(
                "package exponent_z is outside the hybrid artifact geometry domain"
            )
        offset_fraction = float(
            np.linalg.norm(
                package.pose.translation
                - root.pose.translation
            )
            / max(
                root.outer_a,
                root.outer_b,
            )
        )
        if not _within(
            offset_fraction,
            _declared_range(
                package_domain,
                "center_offset_fraction_range",
            ),
        ):
            raise ValueError(
                "package center offset is outside the hybrid artifact geometry domain"
            )
        minimum_half_z = _declared_range(
            package_domain,
            "minimum_half_z_range",
        )[
            0
        ]
        if (
            float(
                package.half_extents[
                    2
                ]
            )
            < minimum_half_z
            - 1e-12
        ):
            raise ValueError(
                "package thickness is outside the hybrid artifact geometry domain"
            )

        enclosure_key = _enclosure_key(
            root,
            package,
        )
        if (
            validated_enclosures
            is None
            or enclosure_key
            not in validated_enclosures
        ):
            conductor_surface = root.surface_samples(
                longitudinal_segments=96,
                section_points=20,
            )
            implicit = np.asarray(
                package.implicit(
                    conductor_surface
                ),
                dtype=float,
            )
            if np.any(
                implicit
                > 1e-10
            ):
                raise ValueError(
                    "package does not enclose the finite primary conductor as "
                    "required by the hybrid artifact geometry domain"
                )
            enclosure_radius = float(
                np.max(
                    np.maximum(
                        implicit
                        + 1.0,
                        0.0,
                    ) ** (
                        1.0
                        / float(
                            package.exponent_z
                        )
                    )
                )
            )
            if not _within(
                enclosure_radius,
                _declared_range(
                    package_domain,
                    "enclosure_radius_range",
                ),
            ):
                raise ValueError(
                    "package enclosure scale is outside the hybrid artifact "
                    "geometry domain"
                )
            if validated_enclosures is not None:
                validated_enclosures.add(
                    enclosure_key
                )
