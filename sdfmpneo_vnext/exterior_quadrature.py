from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .scene import Scene


@dataclass(frozen=True)
class SceneConductorSegment:
    coil: int
    midpoint: np.ndarray
    tangent: np.ndarray
    length: float
    n1: np.ndarray
    n2: np.ndarray


def scene_conductor_geometry(
    scene: Scene,
    *,
    segments_per_turn: int = 16,
):
    """Build mesh-free conductor sweep segments plus extent anchors.

    This helper is intentionally geometric only: it does not build any Maxwell
    basis or world-volume mesh. It is used by FAST spatial decoders that need
    the same exterior-domain semantics as the REFERENCE teacher.
    """
    if segments_per_turn < 4:
        raise ValueError(
            "segments_per_turn must be >= 4"
        )
    segments = []
    anchors = []
    radii = []
    for coil_index, coil in enumerate(
        scene.coils
    ):
        n_segments = max(
            4,
            int(
                np.ceil(
                    segments_per_turn
                    * coil.geometry.turns
                )
            ),
        )
        poly = coil.geometry.polyline(
            n_segments
        )
        anchors.append(
            poly.points
        )
        radii.append(
            np.full(
                len(
                    poly.points
                ),
                coil.geometry.equivalent_radius,
                dtype=float,
            )
        )
        for index in range(
            n_segments
        ):
            segments.append(
                SceneConductorSegment(
                    coil=coil_index,
                    midpoint=poly.midpoints[
                        index
                    ],
                    tangent=poly.tangents[
                        index
                    ],
                    length=float(
                        poly.lengths[
                            index
                        ]
                    ),
                    n1=poly.normal1[
                        index
                    ],
                    n2=poly.normal2[
                        index
                    ],
                )
            )
    return (
        tuple(
            segments
        ),
        np.concatenate(
            anchors,
            axis=0,
        ),
        np.concatenate(
            radii
        ),
    )


def fibonacci_directions(
    count: int,
) -> np.ndarray:
    if count < 8:
        raise ValueError(
            "background angular order must be >= 8"
        )
    index = np.arange(
        count,
        dtype=float,
    )
    z = (
        1.0
        - 2.0
        * (
            index
            + 0.5
        )
        / count
    )
    radius = np.sqrt(
        np.maximum(
            1.0
            - z * z,
            0.0,
        )
    )
    golden = (
        np.pi
        * (
            3.0
            - np.sqrt(
                5.0
            )
        )
    )
    angle = (
        golden
        * index
    )
    return np.column_stack(
        (
            radius
            * np.cos(
                angle
            ),
            radius
            * np.sin(
                angle
            ),
            z,
        )
    )


def homogeneous_background_domain_mask(
    scene: Scene,
    segments,
    points,
) -> np.ndarray:
    """Return the points that belong to the homogeneous background medium.

    Conductor finite cross-sections and package interiors are excluded.  The
    test is object-local and transported by the same segment frames / package
    poses as the physical model, so no world-volume grid is introduced.
    """
    points = np.asarray(
        points,
        dtype=float,
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            "points must have shape (n,3)"
        )

    exterior = np.ones(
        len(
            points
        ),
        dtype=bool,
    )
    for segment in segments:
        active = np.flatnonzero(
            exterior
        )
        if active.size == 0:
            break
        delta = (
            points[
                active
            ]
            - segment.midpoint[
                None,
                :
            ]
        )
        longitudinal = (
            delta
            @ segment.tangent
        )
        near = (
            np.abs(
                longitudinal
            )
            <= (
                0.5
                * segment.length
                + 1e-12
            )
        )
        if not np.any(
            near
        ):
            continue

        candidate = active[
            near
        ]
        transverse = (
            delta[
                near
            ]
            - longitudinal[
                near,
                None,
            ]
            * segment.tangent[
                None,
                :
            ]
        )
        geometry = scene.coils[
            segment.coil
        ].geometry
        half_width = (
            0.5
            * geometry.conductor_width
        )
        half_thickness = (
            0.5
            * geometry.conductor_thickness
        )
        exponent = float(
            geometry.cross_section_exponent
        )
        u = (
            transverse
            @ segment.n1
        ) / half_width
        v = (
            transverse
            @ segment.n2
        ) / half_thickness
        inside = (
            np.abs(
                u
            ) ** exponent
            + np.abs(
                v
            ) ** exponent
            <= (
                1.0
                + 1e-10
            )
        )
        exterior[
            candidate[
                inside
            ]
        ] = False

    for package in scene.packages:
        active = np.flatnonzero(
            exterior
        )
        if active.size == 0:
            break
        inside = np.asarray(
            package.geometry.contains(
                points[
                    active
                ],
                tolerance=2e-12,
            ),
            dtype=bool,
        )
        exterior[
            active[
                inside
            ]
        ] = False

    return exterior


def unbounded_background_quadrature(
    scene: Scene,
    segments,
    charge_positions,
    charge_radii,
    *,
    radial_order: int = 12,
    angular_order: int = 48,
):
    """Positive quadrature of the unbounded homogeneous background.

    The radial map r=s*x/(1-x) covers [0,infinity); angular directions are
    transported by an intrinsic coil frame. The origin and radial scale are
    built from Euclidean scene invariants, so common SE(3) motion changes only
    the world coordinates of the quadrature points, not its physical content.
    """
    if radial_order < 3:
        raise ValueError(
            "background radial order must be >= 3"
        )
    directions = fibonacci_directions(
        int(
            angular_order
        )
    )
    rotation = np.asarray(
        scene.coils[
            0
        ].geometry.pose.rotation,
        dtype=float,
    )
    directions = (
        directions
        @ rotation.T
    )

    positions = np.asarray(
        charge_positions,
        dtype=float,
    )
    radii = np.asarray(
        charge_radii,
        dtype=float,
    )
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or radii.shape != (
            len(
                positions
            ),
        )
    ):
        raise ValueError(
            "invalid charge geometry for background quadrature"
        )

    center = np.mean(
        positions,
        axis=0,
    )
    scale = max(
        float(
            np.max(
                np.linalg.norm(
                    positions
                    - center[
                        None,
                        :
                    ],
                    axis=1,
                )
                + radii
            )
        ),
        4.0
        * float(
            np.max(
                radii
            )
        ),
        1e-6,
    )
    for package in scene.packages:
        package_extent = (
            np.linalg.norm(
                package.geometry.pose.translation
                - center
            )
            + np.linalg.norm(
                package.geometry.half_extents
            )
        )
        scale = max(
            scale,
            float(
                package_extent
            ),
        )

    nodes, weights = (
        np.polynomial.legendre.leggauss(
            int(
                radial_order
            )
        )
    )
    unit = 0.5 * (
        nodes
        + 1.0
    )
    unit_weights = (
        0.5
        * weights
    )
    radius = (
        scale
        * unit
        / (
            1.0
            - unit
        )
    )
    derivative = (
        scale
        / (
            1.0
            - unit
        ) ** 2
    )
    points = (
        center[
            None,
            None,
            :
        ]
        + radius[
            :,
            None,
            None,
        ]
        * directions[
            None,
            :,
            :
        ]
    )
    volume_weights = (
        unit_weights[
            :,
            None
        ]
        * radius[
            :,
            None
        ] ** 2
        * derivative[
            :,
            None
        ]
        * (
            4.0
            * np.pi
            / int(
                angular_order
            )
        )
        * np.ones(
            (
                1,
                int(
                    angular_order
                ),
            ),
            dtype=float,
        )
    )
    points = points.reshape(
        -1,
        3,
    )
    volume_weights = (
        volume_weights.reshape(
            -1
        )
    )
    mask = homogeneous_background_domain_mask(
        scene,
        segments,
        points,
    )
    return (
        points[
            mask
        ],
        volume_weights[
            mask
        ],
    )
