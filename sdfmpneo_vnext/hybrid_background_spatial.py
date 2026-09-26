from __future__ import annotations

import math
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.hybrid_background_spatial requires the 'neural' extra: "
        "pip install 'sdfmpneo[neural]'"
    ) from exc

from .scene import Scene
from .spatial_neural import _mlp


def background_loss_gate(
    scene: Scene,
    frequency_hz: float,
) -> float:
    return float(
        scene.medium.loss_conductivity(
            frequency_hz
        )
        > 0.0
    )


def background_coordinate_features(
    scene: Scene,
    world_points,
    *,
    length_scale: float,
):
    """SE(3)-invariant point-to-object features for exterior loss decoding.

    Coil coordinates are expressed in each coil frame and normalized by the
    scene length scale. Package coordinates are expressed in each superquadric
    frame and normalized by its half extents. No world-coordinate feature is
    exposed to the network.
    """
    points = np.asarray(
        world_points,
        dtype=float,
    )
    points = np.atleast_2d(
        points
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.all(
            np.isfinite(
                points
            )
        )
    ):
        raise ValueError(
            "world_points must have shape (n,3) with finite values"
        )
    scale = float(
        length_scale
    )
    if (
        not np.isfinite(
            scale
        )
        or scale <= 0.0
    ):
        raise ValueError(
            "length_scale must be positive and finite"
        )

    n_points = len(
        points
    )
    coil_features = np.empty(
        (
            n_points,
            len(
                scene.coils
            ),
            5,
        ),
        dtype=float,
    )
    for coil_index, coil in enumerate(
        scene.coils
    ):
        pose = coil.geometry.pose
        local = (
            (
                points
                - pose.translation[
                    None,
                    :
                ]
            )
            @ pose.rotation
        ) / scale
        radius = np.linalg.norm(
            local,
            axis=1,
        )
        coil_features[
            :,
            coil_index,
            :3,
        ] = local
        coil_features[
            :,
            coil_index,
            3,
        ] = radius
        coil_features[
            :,
            coil_index,
            4,
        ] = 1.0 / (
            1.0
            + radius
        )

    package_features = np.empty(
        (
            n_points,
            len(
                scene.packages
            ),
            5,
        ),
        dtype=float,
    )
    for package_index, package in enumerate(
        scene.packages
    ):
        geometry = package.geometry
        local = (
            geometry.world_to_local(
                points
            )
            / geometry.half_extents[
                None,
                :
            ]
        )
        p = float(
            geometry.exponent_xy
        )
        q = float(
            geometry.exponent_z
        )
        rho = (
            (
                np.abs(
                    local[
                        :,
                        0,
                    ]
                ) ** p
                + np.abs(
                    local[
                        :,
                        1,
                    ]
                ) ** p
            ) ** (
                q / p
            )
            + np.abs(
                local[
                    :,
                    2,
                ]
            ) ** q
        ) ** (
            1.0 / q
        )
        package_features[
            :,
            package_index,
            :3,
        ] = local
        package_features[
            :,
            package_index,
            3,
        ] = rho
        package_features[
            :,
            package_index,
            4,
        ] = 1.0 / (
            1.0
            + np.abs(
                rho
                - 1.0
            )
        )

    return (
        coil_features,
        package_features,
    )


class BackgroundLossShapeNet(nn.Module):
    """Continuous PSD loss shape on the unbounded homogeneous background."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        field_hidden_dim: int = 64,
        factor_rank: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        if (
            hidden_dim < 1
            or field_hidden_dim < 4
            or factor_rank < 1
            or depth < 1
        ):
            raise ValueError(
                "invalid background spatial network dimensions"
            )
        self.hidden_dim = int(
            hidden_dim
        )
        self.field_hidden_dim = int(
            field_hidden_dim
        )
        self.factor_rank = int(
            factor_rank
        )
        self.depth = int(
            depth
        )
        self.package_message = _mlp(
            self.hidden_dim
            + 5,
            self.field_hidden_dim,
            self.hidden_dim,
            self.depth,
        )
        self.head = _mlp(
            2
            * self.hidden_dim
            + 5,
            self.field_hidden_dim,
            2
            * self.factor_rank,
            self.depth,
        )

    def raw_matrices(
        self,
        coil_latent,
        package_latent,
        coil_coordinate_features,
        package_coordinate_features,
    ):
        coil_coordinates = torch.as_tensor(
            coil_coordinate_features,
            dtype=coil_latent.dtype,
            device=coil_latent.device,
        )
        package_coordinates = torch.as_tensor(
            package_coordinate_features,
            dtype=coil_latent.dtype,
            device=coil_latent.device,
        )
        n_points = int(
            coil_coordinates.shape[
                0
            ]
        )
        n_ports = int(
            coil_latent.shape[
                0
            ]
        )
        if (
            coil_coordinates.shape
            != (
                n_points,
                n_ports,
                5,
            )
            or package_coordinates.shape[
                0
            ]
            != n_points
            or package_coordinates.shape[
                1
            ]
            != package_latent.shape[
                0
            ]
            or package_coordinates.shape[
                2
            ]
            != 5
        ):
            raise ValueError(
                "background coordinate features have incompatible shapes"
            )

        complex_dtype = (
            torch.complex64
            if coil_latent.dtype
            == torch.float32
            else torch.complex128
        )
        factors = []
        for point in range(
            n_points
        ):
            package_messages = []
            for package in range(
                int(
                    package_latent.shape[
                        0
                    ]
                )
            ):
                package_messages.append(
                    self.package_message(
                        torch.cat(
                            (
                                package_latent[
                                    package
                                ],
                                package_coordinates[
                                    point,
                                    package,
                                ],
                            ),
                            dim=-1,
                        )
                    )
                )
            if package_messages:
                package_context = (
                    torch.stack(
                        package_messages,
                        dim=0,
                    ).sum(
                        dim=0
                    )
                    / math.sqrt(
                        len(
                            package_messages
                        )
                    )
                )
            else:
                package_context = torch.zeros(
                    self.hidden_dim,
                    dtype=coil_latent.dtype,
                    device=coil_latent.device,
                )

            rows = []
            for port in range(
                n_ports
            ):
                raw = self.head(
                    torch.cat(
                        (
                            coil_latent[
                                port
                            ],
                            package_context,
                            coil_coordinates[
                                point,
                                port,
                            ],
                        ),
                        dim=-1,
                    )
                )
                rows.append(
                    raw[
                        : self.factor_rank
                    ].to(
                        complex_dtype
                    )
                    + 1j
                    * raw[
                        self.factor_rank :
                    ].to(
                        complex_dtype
                    )
                )
            factors.append(
                torch.stack(
                    rows,
                    dim=0,
                )
            )

        factors = torch.stack(
            factors,
            dim=0,
        )
        matrices = torch.einsum(
            "qpr,qsr->qps",
            factors.conj(),
            factors,
        )
        n = matrices.shape[
            -1
        ]
        trace_scale = torch.clamp(
            torch.real(
                torch.diagonal(
                    matrices,
                    dim1=-2,
                    dim2=-1,
                ).sum(
                    dim=-1
                )
            ),
            min=1e-12,
        )
        eye = torch.eye(
            n,
            dtype=complex_dtype,
            device=coil_latent.device,
        )
        matrices = (
            matrices
            + (
                1e-9
                * trace_scale[
                    :,
                    None,
                    None,
                ]
                / max(
                    n,
                    1,
                )
            )
            * eye[
                None,
                :,
                :,
            ]
        )
        return 0.5 * (
            matrices
            + matrices.conj().transpose(
                -1,
                -2,
            )
        )
