from __future__ import annotations

import numpy as np

from .scene import Scene


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


def validate_hybrid_geometry_domain(
    scene: Scene,
    frequency_hz: float,
    domain,
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
