from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.hybrid_spatial_neural requires the 'neural' extra: "
        "pip install 'sdfmpneo[neural]'"
    ) from exc

from .exterior_quadrature import (
    homogeneous_background_domain_mask,
    scene_conductor_geometry,
    unbounded_background_quadrature,
)
from .hybrid_background_spatial import (
    BackgroundLossShapeNet,
    background_coordinate_features,
    background_loss_gate,
)
from .hybrid_features import encode_hybrid_scene_invariant
from .hybrid_training_data import HybridTeacherSample
from .scene import Scene
from .spatial_neural import (
    SpatialLossShapeNet,
    _coordinate_features,
    _mlp,
    _normalization_rule,
    _psd_sqrt,
    _stable_cholesky,
)


HYBRID_SPATIAL_ARTIFACT_SCHEMA = 2
SUPPORTED_HYBRID_SPATIAL_ARTIFACT_SCHEMAS = (
    1,
    HYBRID_SPATIAL_ARTIFACT_SCHEMA,
)


def _package_loss_gate(
    scene: Scene,
    frequency_hz: float,
    package_index,
):
    package_index = np.asarray(
        package_index,
        dtype=int,
    )
    return np.asarray(
        [
            1.0
            if scene.packages[
                int(index)
            ].material.loss_conductivity(
                frequency_hz
            )
            > 0.0
            else 0.0
            for index in package_index
        ],
        dtype=float,
    )


def _package_coordinate_features(
    scene: Scene,
    package_index,
    local_position,
):
    package_index = np.asarray(
        package_index,
        dtype=int,
    )
    local_position = np.asarray(
        local_position,
        dtype=float,
    )
    n = len(package_index)
    if local_position.shape != (n, 3):
        raise ValueError(
            "package local positions have incompatible shape"
        )
    out = np.empty(
        (n, 5),
        dtype=float,
    )
    for index in range(n):
        package = int(package_index[index])
        if not (
            0 <= package < len(scene.packages)
        ):
            raise IndexError(
                "package_index out of range"
            )
        geometry = scene.packages[
            package
        ].geometry
        xyz = (
            local_position[index]
            / geometry.half_extents
        )
        p = float(
            geometry.exponent_xy
        )
        q = float(
            geometry.exponent_z
        )
        rho = (
            (
                abs(xyz[0]) ** p
                + abs(xyz[1]) ** p
            ) ** (q / p)
            + abs(xyz[2]) ** q
        ) ** (1.0 / q)
        out[index] = (
            xyz[0],
            xyz[1],
            xyz[2],
            rho,
            rho**2,
        )
    return out


class PackageLossShapeNet(nn.Module):
    """Package-local continuous PSD dielectric-loss shape network."""

    def __init__(
        self,
        hidden_dim: int,
        cross_dim: int,
        *,
        field_hidden_dim: int = 64,
        factor_rank: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        if (
            hidden_dim < 1
            or cross_dim < 1
            or field_hidden_dim < 4
            or factor_rank < 1
            or depth < 1
        ):
            raise ValueError(
                "invalid package spatial network dimensions"
            )
        self.hidden_dim = int(
            hidden_dim
        )
        self.cross_dim = int(
            cross_dim
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
        self.head = _mlp(
            2 * self.hidden_dim
            + self.cross_dim
            + 5,
            self.field_hidden_dim,
            2 * self.factor_rank,
            self.depth,
        )

    def raw_matrices(
        self,
        coil_latent,
        package_latent,
        coil_package_features,
        package_index,
        coordinate_features,
    ):
        package_index = torch.as_tensor(
            package_index,
            dtype=torch.long,
            device=coil_latent.device,
        )
        coordinates = torch.as_tensor(
            coordinate_features,
            dtype=coil_latent.dtype,
            device=coil_latent.device,
        )
        n_points = int(
            package_index.numel()
        )
        n_ports = int(
            coil_latent.shape[0]
        )
        complex_dtype = (
            torch.complex64
            if coil_latent.dtype
            == torch.float32
            else torch.complex128
        )
        factors = []
        for point in range(n_points):
            package = int(
                package_index[
                    point
                ].item()
            )
            rows = []
            for port in range(n_ports):
                features = torch.cat(
                    (
                        package_latent[
                            package
                        ],
                        coil_latent[
                            port
                        ],
                        coil_package_features[
                            port,
                            package,
                        ],
                        coordinates[
                            point
                        ],
                    ),
                    dim=-1,
                )
                raw = self.head(
                    features
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
        n = matrices.shape[-1]
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
                / max(n, 1)
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


class HybridSpatialLossShapeNet(
    nn.Module
):
    def __init__(
        self,
        hidden_dim: int,
        coil_pair_dim: int,
        cross_dim: int,
        *,
        field_hidden_dim: int = 64,
        factor_rank: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        self.hidden_dim = int(
            hidden_dim
        )
        self.coil_pair_dim = int(
            coil_pair_dim
        )
        self.cross_dim = int(
            cross_dim
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
        self.conductor = (
            SpatialLossShapeNet(
                self.hidden_dim,
                self.coil_pair_dim,
                field_hidden_dim=(
                    self.field_hidden_dim
                ),
                factor_rank=(
                    self.factor_rank
                ),
                depth=self.depth,
            )
        )
        self.package = (
            PackageLossShapeNet(
                self.hidden_dim,
                self.cross_dim,
                field_hidden_dim=(
                    self.field_hidden_dim
                ),
                factor_rank=(
                    self.factor_rank
                ),
                depth=self.depth,
            )
        )
        self.background = (
            BackgroundLossShapeNet(
                self.hidden_dim,
                field_hidden_dim=(
                    self.field_hidden_dim
                ),
                factor_rank=(
                    self.factor_rank
                ),
                depth=self.depth,
            )
        )


def _conductor_transforms(
    raw,
    coil_index,
    weights,
    target_channels,
):
    coil_index = torch.as_tensor(
        coil_index,
        dtype=torch.long,
        device=raw.device,
    )
    weights = torch.as_tensor(
        weights,
        dtype=raw.real.dtype,
        device=raw.device,
    )
    target_channels = torch.as_tensor(
        target_channels,
        dtype=raw.dtype,
        device=raw.device,
    )
    transforms = []
    for coil in range(
        int(
            target_channels.shape[0]
        )
    ):
        mask = (
            coil_index == coil
        )
        if not bool(
            torch.any(mask)
        ):
            raise ValueError(
                "conductor normalization rule has no points for a coil"
            )
        integral = torch.sum(
            weights[
                mask,
                None,
                None,
            ]
            * raw[
                mask
            ],
            dim=0,
        )
        integral = 0.5 * (
            integral
            + integral.conj().T
        )
        identity = torch.eye(
            integral.shape[0],
            dtype=integral.dtype,
            device=integral.device,
        )
        inverse_chol = (
            torch.linalg.solve_triangular(
                _stable_cholesky(
                    integral
                ),
                identity,
                upper=False,
            )
        )
        transforms.append(
            _psd_sqrt(
                target_channels[
                    coil
                ],
                inverse=False,
            )
            @ inverse_chol
        )
    return tuple(
        transforms
    )


def _package_transform(
    raw,
    weights,
    target_channel,
):
    weights = torch.as_tensor(
        weights,
        dtype=raw.real.dtype,
        device=raw.device,
    )
    target = torch.as_tensor(
        target_channel,
        dtype=raw.dtype,
        device=raw.device,
    )
    target_norm = torch.linalg.norm(
        target
    )
    if float(
        target_norm.detach().cpu()
    ) <= 1e-18:
        return torch.zeros(
            target.shape,
            dtype=target.dtype,
            device=target.device,
        )
    integral = torch.sum(
        weights[
            :,
            None,
            None,
        ]
        * raw,
        dim=0,
    )
    integral = 0.5 * (
        integral
        + integral.conj().T
    )
    identity = torch.eye(
        integral.shape[0],
        dtype=integral.dtype,
        device=integral.device,
    )
    inverse_chol = (
        torch.linalg.solve_triangular(
            _stable_cholesky(
                integral
            ),
            identity,
            upper=False,
        )
    )
    return (
        _psd_sqrt(
            target,
            inverse=False,
        )
        @ inverse_chol
    )


def _environment_transform(
    raw_package,
    package_weights,
    raw_background,
    background_weights,
    target_channel,
):
    parts = [
        raw_package
    ]
    weight_parts = [
        np.asarray(
            package_weights,
            dtype=float,
        )
    ]
    if (
        raw_background is not None
        and int(
            raw_background.shape[
                0
            ]
        )
        > 0
    ):
        parts.append(
            raw_background
        )
        weight_parts.append(
            np.asarray(
                background_weights,
                dtype=float,
            )
        )
    raw = torch.cat(
        parts,
        dim=0,
    )
    weights = np.concatenate(
        weight_parts
    )
    return _package_transform(
        raw,
        weights,
        target_channel,
    )



def _apply_by_coil(
    raw,
    coil_index,
    transforms,
):
    coil_index = torch.as_tensor(
        coil_index,
        dtype=torch.long,
        device=raw.device,
    )
    out = torch.empty_like(
        raw
    )
    for coil, transform in enumerate(
        transforms
    ):
        mask = (
            coil_index == coil
        )
        if not bool(
            torch.any(mask)
        ):
            continue
        values = (
            transform[
                None,
                :,
                :,
            ]
            @ raw[
                mask
            ]
            @ transform.conj().T[
                None,
                :,
                :,
            ]
        )
        out[
            mask
        ] = 0.5 * (
            values
            + values.conj().transpose(
                -1,
                -2,
            )
        )
    return out


def _apply_transform(
    raw,
    transform,
):
    values = (
        transform[
            None,
            :,
            :,
        ]
        @ raw
        @ transform.conj().T[
            None,
            :,
            :,
        ]
    )
    return 0.5 * (
        values
        + values.conj().transpose(
            -1,
            -2,
        )
    )


def _package_normalization_rule(
    scene: Scene,
    *,
    axial_order: int,
    radial_order: int,
    azimuthal_order: int,
):
    if (
        axial_order < 2
        or radial_order < 2
        or azimuthal_order < 8
    ):
        raise ValueError(
            "invalid package normalization quadrature"
        )
    ids = []
    local = []
    weights = []
    for index, package in enumerate(
        scene.packages
    ):
        quadrature = (
            package.geometry.volume_quadrature(
                axial_order=axial_order,
                radial_order=radial_order,
                azimuthal_order=azimuthal_order,
            )
        )
        count = len(
            quadrature.weights
        )
        ids.append(
            np.full(
                count,
                index,
                dtype=int,
            )
        )
        local.append(
            quadrature.local_positions
        )
        weights.append(
            quadrature.weights
        )
    return (
        np.concatenate(ids),
        np.concatenate(
            local,
            axis=0,
        ),
        np.concatenate(weights),
    )


def _latent(
    port_artifact,
    scene: Scene,
    frequency_hz: float,
):
    encoded = (
        encode_hybrid_scene_invariant(
            scene,
            frequency_hz,
        )
    )
    (
        coil_node,
        coil_pair,
        package,
        coil_package,
        package_pair,
    ) = port_artifact.normalizer.normalize(
        encoded
    )
    model = port_artifact.model
    dtype = next(
        model.parameters()
    ).dtype
    device = next(
        model.parameters()
    ).device
    tensors = (
        torch.as_tensor(
            coil_node,
            dtype=dtype,
            device=device,
        ),
        torch.as_tensor(
            coil_pair,
            dtype=dtype,
            device=device,
        ),
        torch.as_tensor(
            package,
            dtype=dtype,
            device=device,
        ),
        torch.as_tensor(
            coil_package,
            dtype=dtype,
            device=device,
        ),
        torch.as_tensor(
            package_pair,
            dtype=dtype,
            device=device,
        ),
    )
    model.eval()
    with torch.no_grad():
        (
            coil_latent,
            package_latent,
        ) = model._latent(
            *tensors
        )
    return (
        coil_latent,
        package_latent,
        tensors[1],
        tensors[3],
        float(
            encoded.length_scale
        ),
    )


@dataclass(frozen=True)
class PreparedHybridSpatialLossField:
    scene: Scene
    frequency_hz: float
    port_prediction: object
    model: HybridSpatialLossShapeNet
    coil_latent: object
    package_latent: object
    coil_pair_features: object
    coil_package_features: object
    conductor_transforms: tuple
    package_transform: object
    background_segments: tuple
    background_anchor_positions: np.ndarray
    background_anchor_radii: np.ndarray
    length_scale: float
    normalization_closure_error: float
    device: str

    @property
    def dielectric_channel_index(
        self,
    ) -> int:
        return len(
            self.scene.coils
        )

    @property
    def background_channel_index(
        self,
    ) -> int | None:
        if (
            self.scene.medium.conductivity
            <= 0.0
        ):
            return None
        return self.dielectric_channel_index

    @property
    def environment_transform(
        self,
    ):
        return self.package_transform

    def _background_domain_mask(
        self,
        points,
    ) -> np.ndarray:
        return homogeneous_background_domain_mask(
            self.scene,
            self.background_segments,
            points,
        )

    def background_quadrature(
        self,
        *,
        radial_order: int = 12,
        angular_order: int = 48,
    ):
        return unbounded_background_quadrature(
            self.scene,
            self.background_segments,
            self.background_anchor_positions,
            self.background_anchor_radii,
            radial_order=radial_order,
            angular_order=angular_order,
        )

    def background_dissipation_matrices(
        self,
        points,
    ) -> np.ndarray:
        points = np.asarray(
            points,
            dtype=float,
        )
        scalar = (
            points.ndim == 1
        )
        points = np.atleast_2d(
            points
        )
        if (
            points.ndim != 2
            or points.shape[1] != 3
        ):
            raise ValueError(
                "points must have shape (3,) or (n,3)"
            )
        n_ports = int(
            self.port_prediction.impedance.shape[
                0
            ]
        )
        out = np.zeros(
            (
                len(
                    points
                ),
                n_ports,
                n_ports,
            ),
            dtype=complex,
        )
        if (
            background_loss_gate(
                self.scene,
                self.frequency_hz,
            )
            <= 0.0
        ):
            return (
                out[
                    0
                ]
                if scalar
                else out
            )
        exterior = (
            self._background_domain_mask(
                points
            )
        )
        if np.any(
            exterior
        ):
            (
                coil_coordinates,
                package_coordinates,
            ) = background_coordinate_features(
                self.scene,
                points[
                    exterior
                ],
                length_scale=(
                    self.length_scale
                ),
            )
            self.model.eval()
            with torch.no_grad():
                raw = (
                    self.model.background.raw_matrices(
                        self.coil_latent,
                        self.package_latent,
                        coil_coordinates,
                        package_coordinates,
                    )
                )
                values = _apply_transform(
                    raw,
                    self.environment_transform,
                )
            out[
                exterior
            ] = (
                values.detach()
                .cpu()
                .numpy()
            )
        return (
            out[
                0
            ]
            if scalar
            else out
        )

    def background_joule_density(
        self,
        points,
        currents,
    ):
        matrices = (
            self.background_dissipation_matrices(
                points
            )
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if currents.shape != (
            self.port_prediction.impedance.shape[
                0
            ],
        ):
            raise ValueError(
                "currents have wrong shape"
            )
        return 0.5 * np.real(
            np.einsum(
                "i,...ij,j->...",
                currents.conj(),
                matrices,
                currents,
            )
        )

    def local_dissipation_matrices(
        self,
        coil_index,
        arc_fraction,
        xy,
    ) -> np.ndarray:
        coil_index = np.asarray(
            coil_index,
            dtype=int,
        )
        arc_fraction = np.asarray(
            arc_fraction,
            dtype=float,
        )
        xy = np.asarray(
            xy,
            dtype=float,
        )
        n = len(
            coil_index
        )
        if (
            coil_index.ndim != 1
            or arc_fraction.shape
            != (n,)
            or xy.shape
            != (n, 2)
        ):
            raise ValueError(
                "conductor spatial query arrays have incompatible shapes"
            )
        if np.any(
            (coil_index < 0)
            | (
                coil_index
                >= len(
                    self.scene.coils
                )
            )
        ):
            raise IndexError(
                "coil_index out of range"
            )
        if np.any(
            (arc_fraction < 0.0)
            | (arc_fraction > 1.0)
        ):
            raise ValueError(
                "arc_fraction must lie in [0,1]"
            )

        inside = np.ones(
            n,
            dtype=bool,
        )
        for point in range(n):
            geometry = (
                self.scene.coils[
                    int(
                        coil_index[
                            point
                        ]
                    )
                ].geometry
            )
            xn = (
                abs(
                    xy[
                        point,
                        0,
                    ]
                )
                / (
                    0.5
                    * geometry.conductor_width
                )
            )
            yn = (
                abs(
                    xy[
                        point,
                        1,
                    ]
                )
                / (
                    0.5
                    * geometry.conductor_thickness
                )
            )
            inside[
                point
            ] = (
                xn
                ** geometry.cross_section_exponent
                + yn
                ** geometry.cross_section_exponent
                <= 1.0 + 1e-12
            )

        coordinates = (
            _coordinate_features(
                self.scene,
                coil_index,
                arc_fraction,
                xy,
            )
        )
        self.model.eval()
        with torch.no_grad():
            raw = (
                self.model.conductor.raw_matrices(
                    self.coil_latent,
                    self.coil_pair_features,
                    coil_index,
                    coordinates,
                )
            )
            values = _apply_by_coil(
                raw,
                coil_index,
                self.conductor_transforms,
            )
            if not np.all(
                inside
            ):
                mask = torch.as_tensor(
                    ~inside,
                    dtype=torch.bool,
                    device=raw.device,
                )
                values[
                    mask
                ] = 0.0
        return (
            values.detach()
            .cpu()
            .numpy()
        )

    def local_dissipation_matrix(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return self.local_dissipation_matrices(
            np.asarray(
                [coil_index],
                dtype=int,
            ),
            np.asarray(
                [arc_fraction],
                dtype=float,
            ),
            np.asarray(
                [xy],
                dtype=float,
            ),
        )[0]

    def package_local_dissipation_matrices(
        self,
        package_index,
        local_position,
    ) -> np.ndarray:
        package_index = np.asarray(
            package_index,
            dtype=int,
        )
        local_position = np.asarray(
            local_position,
            dtype=float,
        )
        n = len(
            package_index
        )
        if (
            package_index.ndim != 1
            or local_position.shape
            != (n, 3)
        ):
            raise ValueError(
                "package spatial query arrays have incompatible shapes"
            )
        if np.any(
            (package_index < 0)
            | (
                package_index
                >= len(
                    self.scene.packages
                )
            )
        ):
            raise IndexError(
                "package_index out of range"
            )
        inside = np.ones(
            n,
            dtype=bool,
        )
        for point in range(n):
            package = int(
                package_index[
                    point
                ]
            )
            inside[
                point
            ] = bool(
                self.scene.packages[
                    package
                ].geometry.implicit_local(
                    local_position[
                        point
                    ]
                )
                <= 1e-12
            )
        coordinates = (
            _package_coordinate_features(
                self.scene,
                package_index,
                local_position,
            )
        )
        self.model.eval()
        with torch.no_grad():
            raw = (
                self.model.package.raw_matrices(
                    self.coil_latent,
                    self.package_latent,
                    self.coil_package_features,
                    package_index,
                    coordinates,
                )
            )
            package_gate = torch.as_tensor(
                _package_loss_gate(
                    self.scene,
                    self.frequency_hz,
                    package_index,
                ),
                dtype=raw.real.dtype,
                device=raw.device,
            )
            raw = (
                raw
                * package_gate[
                    :,
                    None,
                    None,
                ]
            )
            values = _apply_transform(
                raw,
                self.package_transform,
            )
            if not np.all(
                inside
            ):
                mask = torch.as_tensor(
                    ~inside,
                    dtype=torch.bool,
                    device=raw.device,
                )
                values[
                    mask
                ] = 0.0
        return (
            values.detach()
            .cpu()
            .numpy()
        )

    def package_local_dissipation_matrix(
        self,
        package_index: int,
        local_position,
    ) -> np.ndarray:
        return (
            self.package_local_dissipation_matrices(
                np.asarray(
                    [package_index],
                    dtype=int,
                ),
                np.asarray(
                    [local_position],
                    dtype=float,
                ),
            )[0]
        )

    def package_dissipation_matrices(
        self,
        package_index: int,
        world_positions,
    ) -> np.ndarray:
        world_positions = np.asarray(
            world_positions,
            dtype=float,
        )
        if (
            world_positions.ndim != 2
            or world_positions.shape[1]
            != 3
        ):
            raise ValueError(
                "world_positions must have shape (n,3)"
            )
        geometry = (
            self.scene.packages[
                int(
                    package_index
                )
            ].geometry
        )
        local = geometry.world_to_local(
            world_positions
        )
        return (
            self.package_local_dissipation_matrices(
                np.full(
                    len(local),
                    int(
                        package_index
                    ),
                    dtype=int,
                ),
                local,
            )
        )

    def package_dissipation_matrix(
        self,
        package_index: int,
        world_position,
    ) -> np.ndarray:
        return self.package_dissipation_matrices(
            package_index,
            np.asarray(
                [world_position],
                dtype=float,
            ),
        )[0]

    def local_joule_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        matrix = (
            self.local_dissipation_matrix(
                coil_index,
                arc_fraction,
                xy,
            )
        )
        return float(
            0.5
            * np.real(
                np.vdot(
                    currents,
                    matrix
                    @ currents,
                )
            )
        )

    def package_local_joule_density(
        self,
        package_index: int,
        local_position,
        currents,
    ) -> float:
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        matrix = (
            self.package_local_dissipation_matrix(
                package_index,
                local_position,
            )
        )
        return float(
            0.5
            * np.real(
                np.vdot(
                    currents,
                    matrix
                    @ currents,
                )
            )
        )


class HybridSpatialLossArtifact:
    supports_packages = True
    supports_lossy_background = False

    def __init__(
        self,
        port_artifact,
        model: HybridSpatialLossShapeNet,
        *,
        conductor_longitudinal_points: int = 12,
        conductor_radial_order: int = 3,
        conductor_angular_order: int = 16,
        package_axial_order: int = 6,
        package_radial_order: int = 4,
        package_azimuthal_order: int = 16,
        background_segments_per_turn: int = 16,
        background_radial_order: int = 12,
        background_angular_order: int = 48,
        background_conductivity_range=None,
        device: str = "cpu",
    ):
        if not bool(
            getattr(
                port_artifact,
                "supports_packages",
                False,
            )
        ):
            raise TypeError(
                "hybrid spatial artifact requires a package-aware port artifact"
            )
        if not hasattr(
            port_artifact,
            "fingerprint",
        ):
            raise TypeError(
                "hybrid spatial artifact requires a fingerprinted port artifact"
            )
        self.port_artifact = (
            port_artifact
        )
        self.port_fingerprint = (
            port_artifact.fingerprint()
        )
        self.port_artifact.model.to(
            device
        )
        self.port_artifact.device = str(
            device
        )
        port_dtype = next(
            self.port_artifact.model.parameters()
        ).dtype
        self.model = model.to(
            device=device,
            dtype=port_dtype,
        )
        self.conductor_longitudinal_points = int(
            conductor_longitudinal_points
        )
        self.conductor_radial_order = int(
            conductor_radial_order
        )
        self.conductor_angular_order = int(
            conductor_angular_order
        )
        self.package_axial_order = int(
            package_axial_order
        )
        self.package_radial_order = int(
            package_radial_order
        )
        self.package_azimuthal_order = int(
            package_azimuthal_order
        )
        self.background_segments_per_turn = int(
            background_segments_per_turn
        )
        self.background_radial_order = int(
            background_radial_order
        )
        self.background_angular_order = int(
            background_angular_order
        )
        if (
            self.background_segments_per_turn
            < 4
            or self.background_radial_order
            < 3
            or self.background_angular_order
            < 8
        ):
            raise ValueError(
                "invalid background spatial normalization resolution"
            )
        if background_conductivity_range is None:
            self.background_conductivity_range = None
        else:
            values = np.asarray(
                background_conductivity_range,
                dtype=float,
            )
            if (
                values.shape != (
                    2,
                )
                or np.any(
                    ~np.isfinite(
                        values
                    )
                )
                or values[
                    0
                ] < 0.0
                or values[
                    1
                ] < values[
                    0
                ]
            ):
                raise ValueError(
                    "background_conductivity_range must be a finite "
                    "nonnegative increasing pair"
                )
            self.background_conductivity_range = (
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
        self.supports_lossy_background = bool(
            self.background_conductivity_range
            is not None
            and self.background_conductivity_range[
                1
            ]
            > 0.0
            and bool(
                getattr(
                    self.port_artifact,
                    "supports_lossy_background",
                    False,
                )
            )
        )
        if (
            self.background_conductivity_range
            is not None
            and self.background_conductivity_range[
                1
            ]
            > 0.0
            and not bool(
                getattr(
                    self.port_artifact,
                    "supports_lossy_background",
                    False,
                )
            )
        ):
            raise ValueError(
                "lossy-background spatial artifacts require a port artifact "
                "trained for lossy homogeneous backgrounds"
            )
        self.device = str(
            device
        )

    def prepare(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> PreparedHybridSpatialLossField:
        if not scene.packages:
            raise ValueError(
                "hybrid spatial artifact requires at least one package"
            )
        conductivity = float(
            scene.medium.conductivity
        )
        if conductivity > 0.0:
            if not self.supports_lossy_background:
                raise NotImplementedError(
                    "this hybrid spatial artifact has no trained continuous "
                    "background-loss decoder for lossy homogeneous media"
                )
            lower, upper = (
                self.background_conductivity_range
            )
            tolerance = (
                1e-12
                * max(
                    upper,
                    1.0,
                )
            )
            if (
                conductivity
                < lower
                - tolerance
                or conductivity
                > upper
                + tolerance
            ):
                raise ValueError(
                    "background conductivity is outside the hybrid spatial "
                    f"training domain [{lower:.6g}, {upper:.6g}] S/m"
                )
        prediction = (
            self.port_artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )
        n_coils = len(
            scene.coils
        )
        if (
            prediction.dissipation_channels.shape[
                0
            ]
            != n_coils + 1
        ):
            raise ValueError(
                "hybrid spatial artifact expects one conductor channel per coil "
                "plus one aggregate dielectric channel"
            )
        (
            coil_latent,
            package_latent,
            coil_pair,
            coil_package,
            length_scale,
        ) = _latent(
            self.port_artifact,
            scene,
            frequency_hz,
        )

        (
            conductor_ids,
            conductor_arc,
            conductor_xy,
            conductor_weights,
        ) = _normalization_rule(
            scene,
            longitudinal_points=(
                self.conductor_longitudinal_points
            ),
            radial_order=(
                self.conductor_radial_order
            ),
            angular_order=(
                self.conductor_angular_order
            ),
        )
        conductor_coordinates = (
            _coordinate_features(
                scene,
                conductor_ids,
                conductor_arc,
                conductor_xy,
            )
        )
        (
            package_ids,
            package_local,
            package_weights,
        ) = _package_normalization_rule(
            scene,
            axial_order=(
                self.package_axial_order
            ),
            radial_order=(
                self.package_radial_order
            ),
            azimuthal_order=(
                self.package_azimuthal_order
            ),
        )
        package_coordinates = (
            _package_coordinate_features(
                scene,
                package_ids,
                package_local,
            )
        )
        (
            background_segments,
            background_anchor_positions,
            background_anchor_radii,
        ) = scene_conductor_geometry(
            scene,
            segments_per_turn=(
                self.background_segments_per_turn
            ),
        )
        if conductivity > 0.0:
            (
                background_points,
                background_weights,
            ) = unbounded_background_quadrature(
                scene,
                background_segments,
                background_anchor_positions,
                background_anchor_radii,
                radial_order=(
                    self.background_radial_order
                ),
                angular_order=(
                    self.background_angular_order
                ),
            )
            (
                background_coil_coordinates,
                background_package_coordinates,
            ) = background_coordinate_features(
                scene,
                background_points,
                length_scale=(
                    length_scale
                ),
            )
        else:
            background_points = np.empty(
                (
                    0,
                    3,
                ),
                dtype=float,
            )
            background_weights = np.empty(
                0,
                dtype=float,
            )
            background_coil_coordinates = np.empty(
                (
                    0,
                    n_coils,
                    5,
                ),
                dtype=float,
            )
            background_package_coordinates = np.empty(
                (
                    0,
                    len(
                        scene.packages
                    ),
                    5,
                ),
                dtype=float,
            )

        self.model.eval()
        with torch.no_grad():
            raw_conductor = (
                self.model.conductor.raw_matrices(
                    coil_latent,
                    coil_pair,
                    conductor_ids,
                    conductor_coordinates,
                )
            )
            conductor_transforms = (
                _conductor_transforms(
                    raw_conductor,
                    conductor_ids,
                    conductor_weights,
                    prediction.dissipation_channels[
                        :n_coils
                    ],
                )
            )
            raw_package = (
                self.model.package.raw_matrices(
                    coil_latent,
                    package_latent,
                    coil_package,
                    package_ids,
                    package_coordinates,
                )
            )
            package_gate = torch.as_tensor(
                _package_loss_gate(
                    scene,
                    frequency_hz,
                    package_ids,
                ),
                dtype=raw_package.real.dtype,
                device=raw_package.device,
            )
            raw_package = (
                raw_package
                * package_gate[
                    :,
                    None,
                    None,
                ]
            )
            if len(
                background_points
            ):
                raw_background = (
                    self.model.background.raw_matrices(
                        coil_latent,
                        package_latent,
                        background_coil_coordinates,
                        background_package_coordinates,
                    )
                )
                raw_background = (
                    raw_background
                    * float(
                        background_loss_gate(
                            scene,
                            frequency_hz,
                        )
                    )
                )
            else:
                raw_background = torch.empty(
                    (
                        0,
                        n_coils,
                        n_coils,
                    ),
                    dtype=raw_package.dtype,
                    device=raw_package.device,
                )
            package_transform = (
                _environment_transform(
                    raw_package,
                    package_weights,
                    raw_background,
                    background_weights,
                    prediction.dissipation_channels[
                        n_coils
                    ],
                )
            )

            conductor_values = (
                _apply_by_coil(
                    raw_conductor,
                    conductor_ids,
                    conductor_transforms,
                )
            )
            package_values = (
                _apply_transform(
                    raw_package,
                    package_transform,
                )
            )
            if int(
                raw_background.shape[
                    0
                ]
            ):
                background_values = (
                    _apply_transform(
                        raw_background,
                        package_transform,
                    )
                )
            else:
                background_values = raw_background

        integrated = np.zeros_like(
            prediction.dissipation_channels,
            dtype=complex,
        )
        conductor_values_np = (
            conductor_values.detach()
            .cpu()
            .numpy()
        )
        for coil in range(
            n_coils
        ):
            mask = (
                conductor_ids == coil
            )
            integrated[
                coil
            ] = np.sum(
                conductor_weights[
                    mask,
                    None,
                    None,
                ]
                * conductor_values_np[
                    mask
                ],
                axis=0,
            )
        integrated[
            n_coils
        ] = np.sum(
            package_weights[
                :,
                None,
                None,
            ]
            * package_values.detach()
            .cpu()
            .numpy(),
            axis=0,
        )
        if len(
            background_weights
        ):
            integrated[
                n_coils
            ] += np.sum(
                background_weights[
                    :,
                    None,
                    None,
                ]
                * background_values.detach()
                .cpu()
                .numpy(),
                axis=0,
            )
        closure = float(
            np.linalg.norm(
                integrated
                - prediction.dissipation_channels
            )
            / max(
                np.linalg.norm(
                    prediction.dissipation_channels
                ),
                1e-30,
            )
        )
        return PreparedHybridSpatialLossField(
            scene,
            float(
                frequency_hz
            ),
            prediction,
            self.model,
            coil_latent,
            package_latent,
            coil_pair,
            coil_package,
            conductor_transforms,
            package_transform,
            background_segments,
            background_anchor_positions,
            background_anchor_radii,
            float(
                length_scale
            ),
            closure,
            self.device,
        )

    def save(
        self,
        path,
    ):
        payload = {
            "schema": (
                HYBRID_SPATIAL_ARTIFACT_SCHEMA
            ),
            "port_fingerprint": (
                self.port_fingerprint
            ),
            "model_config": {
                "hidden_dim": (
                    self.model.hidden_dim
                ),
                "coil_pair_dim": (
                    self.model.coil_pair_dim
                ),
                "cross_dim": (
                    self.model.cross_dim
                ),
                "field_hidden_dim": (
                    self.model.field_hidden_dim
                ),
                "factor_rank": (
                    self.model.factor_rank
                ),
                "depth": (
                    self.model.depth
                ),
            },
            "model_state": (
                self.model.state_dict()
            ),
            "conductor_longitudinal_points": (
                self.conductor_longitudinal_points
            ),
            "conductor_radial_order": (
                self.conductor_radial_order
            ),
            "conductor_angular_order": (
                self.conductor_angular_order
            ),
            "package_axial_order": (
                self.package_axial_order
            ),
            "package_radial_order": (
                self.package_radial_order
            ),
            "package_azimuthal_order": (
                self.package_azimuthal_order
            ),
            "background_segments_per_turn": (
                self.background_segments_per_turn
            ),
            "background_radial_order": (
                self.background_radial_order
            ),
            "background_angular_order": (
                self.background_angular_order
            ),
            "background_conductivity_range": (
                self.background_conductivity_range
            ),
        }
        torch.save(
            payload,
            Path(path),
        )

    @staticmethod
    def load(
        path,
        port_artifact,
        *,
        device: str = "cpu",
    ):
        try:
            payload = torch.load(
                Path(path),
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(
                Path(path),
                map_location=device,
            )
        if (
            payload.get("schema")
            not in SUPPORTED_HYBRID_SPATIAL_ARTIFACT_SCHEMAS
        ):
            raise ValueError(
                "unsupported hybrid spatial artifact schema"
            )
        if not hasattr(
            port_artifact,
            "fingerprint",
        ):
            raise TypeError(
                "hybrid spatial artifact requires a fingerprinted port artifact"
            )
        if (
            payload.get(
                "port_fingerprint"
            )
            != port_artifact.fingerprint()
        ):
            raise ValueError(
                "hybrid spatial port fingerprint mismatch"
            )
        model = (
            HybridSpatialLossShapeNet(
                **payload[
                    "model_config"
                ]
            )
        )
        if int(
            payload.get(
                "schema",
                -1,
            )
        ) == 1:
            incompatible = model.load_state_dict(
                payload[
                    "model_state"
                ],
                strict=False,
            )
            if (
                incompatible.unexpected_keys
                or any(
                    not key.startswith(
                        "background."
                    )
                    for key in incompatible.missing_keys
                )
            ):
                raise ValueError(
                    "legacy hybrid spatial artifact has incompatible model state"
                )
        else:
            model.load_state_dict(
                payload[
                    "model_state"
                ]
            )
        model.eval()
        return HybridSpatialLossArtifact(
            port_artifact,
            model,
            conductor_longitudinal_points=int(
                payload[
                    "conductor_longitudinal_points"
                ]
            ),
            conductor_radial_order=int(
                payload[
                    "conductor_radial_order"
                ]
            ),
            conductor_angular_order=int(
                payload[
                    "conductor_angular_order"
                ]
            ),
            package_axial_order=int(
                payload[
                    "package_axial_order"
                ]
            ),
            package_radial_order=int(
                payload[
                    "package_radial_order"
                ]
            ),
            package_azimuthal_order=int(
                payload[
                    "package_azimuthal_order"
                ]
            ),
            background_segments_per_turn=int(
                payload.get(
                    "background_segments_per_turn",
                    16,
                )
            ),
            background_radial_order=int(
                payload.get(
                    "background_radial_order",
                    12,
                )
            ),
            background_angular_order=int(
                payload.get(
                    "background_angular_order",
                    48,
                )
            ),
            background_conductivity_range=(
                payload.get(
                    "background_conductivity_range"
                )
            ),
            device=device,
        )


@dataclass(frozen=True)
class HybridSpatialTrainingReport:
    final_loss: float
    epochs: int
    best_epoch: int
    best_validation_error: float | None
    stopped_early: bool
    best_validation_shape_error: float | None = None


def _weighted_relative_loss(
    predicted,
    target,
    weights,
):
    weights = torch.as_tensor(
        weights,
        dtype=predicted.real.dtype,
        device=predicted.device,
    )
    target = torch.as_tensor(
        target,
        dtype=predicted.dtype,
        device=predicted.device,
    )
    numerator = torch.sum(
        weights[
            :,
            None,
            None,
        ]
        * torch.abs(
            predicted
            - target
        ) ** 2
    )
    denominator = (
        torch.sum(
            weights[
                :,
                None,
                None,
            ]
            * torch.abs(
                target
            ) ** 2
        )
        + 1e-18
    )
    return numerator / denominator


def _sample_loss(
    model,
    port_artifact,
    sample: HybridTeacherSample,
    *,
    device: str,
):
    if not sample.has_spatial_truth:
        raise ValueError(
            "hybrid spatial training requires conductor and package spatial truth"
        )
    if (
        sample.target_dissipation_channels.shape[
            0
        ]
        != len(
            sample.scene.coils
        )
        + 1
    ):
        raise ValueError(
            "hybrid spatial training requires one aggregate dielectric channel"
        )
    (
        coil_latent,
        package_latent,
        coil_pair,
        coil_package,
        length_scale,
    ) = _latent(
        port_artifact,
        sample.scene,
        sample.frequency_hz,
    )

    conductor = (
        sample.conductor_spatial_loss
    )
    conductor_coordinates = (
        _coordinate_features(
            sample.scene,
            conductor.coil_index,
            conductor.arc_fraction,
            conductor.xy,
        )
    )
    raw_conductor = (
        model.conductor.raw_matrices(
            coil_latent,
            coil_pair,
            conductor.coil_index,
            conductor_coordinates,
        )
    )
    conductor_transforms = (
        _conductor_transforms(
            raw_conductor,
            conductor.coil_index,
            conductor.weights,
            sample.target_dissipation_channels[
                : len(
                    sample.scene.coils
                )
            ],
        )
    )
    predicted_conductor = (
        _apply_by_coil(
            raw_conductor,
            conductor.coil_index,
            conductor_transforms,
        )
    )
    conductor_loss = (
        _weighted_relative_loss(
            predicted_conductor,
            conductor.dissipation_matrix,
            conductor.weights,
        )
    )

    package = (
        sample.package_spatial_loss
    )
    package_coordinates = (
        _package_coordinate_features(
            sample.scene,
            package.package_index,
            package.local_position,
        )
    )
    raw_package = (
        model.package.raw_matrices(
            coil_latent,
            package_latent,
            coil_package,
            package.package_index,
            package_coordinates,
        )
    )
    package_gate = torch.as_tensor(
        _package_loss_gate(
            sample.scene,
            sample.frequency_hz,
            package.package_index,
        ),
        dtype=raw_package.real.dtype,
        device=raw_package.device,
    )
    raw_package = (
        raw_package
        * package_gate[
            :,
            None,
            None,
        ]
    )
    background = (
        sample.background_spatial_loss
    )
    if background is not None:
        root_pose = (
            sample.scene.coils[
                0
            ].geometry.pose
        )
        background_world = root_pose.apply(
            background.root_local_position
        )
        (
            background_coil_coordinates,
            background_package_coordinates,
        ) = background_coordinate_features(
            sample.scene,
            background_world,
            length_scale=(
                length_scale
            ),
        )
        raw_background = (
            model.background.raw_matrices(
                coil_latent,
                package_latent,
                background_coil_coordinates,
                background_package_coordinates,
            )
        )
        raw_background = (
            raw_background
            * float(
                background_loss_gate(
                    sample.scene,
                    sample.frequency_hz,
                )
            )
        )
        background_weights = (
            background.weights
        )
    else:
        raw_background = torch.empty(
            (
                0,
                len(
                    sample.scene.coils
                ),
                len(
                    sample.scene.coils
                ),
            ),
            dtype=raw_package.dtype,
            device=raw_package.device,
        )
        background_weights = np.empty(
            0,
            dtype=float,
        )

    transform = (
        _environment_transform(
            raw_package,
            package.weights,
            raw_background,
            background_weights,
            sample.target_dissipation_channels[
                len(
                    sample.scene.coils
                )
            ],
        )
    )
    predicted_package = (
        _apply_transform(
            raw_package,
            transform,
        )
    )
    package_loss = (
        _weighted_relative_loss(
            predicted_package,
            package.dissipation_matrix,
            package.weights,
        )
    )
    background_loss = torch.zeros(
        (),
        dtype=package_loss.dtype,
        device=package_loss.device,
    )
    if background is not None:
        predicted_background = (
            _apply_transform(
                raw_background,
                transform,
            )
        )
        background_loss = (
            _weighted_relative_loss(
                predicted_background,
                background.dissipation_matrix,
                background.weights,
            )
        )
    return (
        conductor_loss
        + package_loss
        + background_loss
    )


def _weighted_relative_error_numpy(
    predicted,
    target,
    weights,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=complex,
    )
    target = np.asarray(
        target,
        dtype=complex,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )
    numerator = np.sum(
        weights[
            :,
            None,
            None,
        ]
        * np.abs(
            predicted
            - target
        ) ** 2
    )
    denominator = (
        np.sum(
            weights[
                :,
                None,
                None,
            ]
            * np.abs(
                target
            ) ** 2
        )
        + 1e-18
    )
    return float(
        numerator
        / denominator
    )


def _sample_end_to_end_error(
    model,
    port_artifact,
    sample: HybridTeacherSample,
    *,
    conductor_longitudinal_points: int,
    conductor_radial_order: int,
    conductor_angular_order: int,
    package_axial_order: int,
    package_radial_order: int,
    package_azimuthal_order: int,
    background_segments_per_turn: int,
    background_radial_order: int,
    background_angular_order: int,
    background_conductivity_range,
    device: str,
) -> float:
    if not sample.has_spatial_truth:
        raise ValueError(
            "end-to-end hybrid spatial validation requires spatial truth"
        )
    artifact = HybridSpatialLossArtifact(
        port_artifact,
        model,
        conductor_longitudinal_points=(
            conductor_longitudinal_points
        ),
        conductor_radial_order=(
            conductor_radial_order
        ),
        conductor_angular_order=(
            conductor_angular_order
        ),
        package_axial_order=(
            package_axial_order
        ),
        package_radial_order=(
            package_radial_order
        ),
        package_azimuthal_order=(
            package_azimuthal_order
        ),
        background_segments_per_turn=(
            background_segments_per_turn
        ),
        background_radial_order=(
            background_radial_order
        ),
        background_angular_order=(
            background_angular_order
        ),
        background_conductivity_range=(
            background_conductivity_range
        ),
        device=device,
    )
    prepared = artifact.prepare(
        sample.scene,
        sample.frequency_hz,
    )

    conductor = sample.conductor_spatial_loss
    conductor_predicted = (
        prepared.local_dissipation_matrices(
            conductor.coil_index,
            conductor.arc_fraction,
            conductor.xy,
        )
    )
    conductor_error = (
        _weighted_relative_error_numpy(
            conductor_predicted,
            conductor.dissipation_matrix,
            conductor.weights,
        )
    )

    package = sample.package_spatial_loss
    package_predicted = (
        prepared.package_local_dissipation_matrices(
            package.package_index,
            package.local_position,
        )
    )
    package_error = (
        _weighted_relative_error_numpy(
            package_predicted,
            package.dissipation_matrix,
            package.weights,
        )
    )

    background_error = 0.0
    background = (
        sample.background_spatial_loss
    )
    if background is not None:
        root_pose = (
            sample.scene.coils[
                0
            ].geometry.pose
        )
        background_world = root_pose.apply(
            background.root_local_position
        )
        background_predicted = (
            prepared.background_dissipation_matrices(
                background_world
            )
        )
        background_error = (
            _weighted_relative_error_numpy(
                background_predicted,
                background.dissipation_matrix,
                background.weights,
            )
        )
    return float(
        conductor_error
        + package_error
        + background_error
    )


def train_hybrid_spatial_loss_surrogate(
    port_artifact,
    samples,
    *,
    validation_samples=(),
    field_hidden_dim: int = 64,
    factor_rank: int = 4,
    depth: int = 2,
    epochs: int = 120,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    patience: int = 20,
    validation_interval: int = 1,
    min_improvement: float = 1e-5,
    seed: int = 37,
    conductor_longitudinal_points: int = 12,
    conductor_radial_order: int = 3,
    conductor_angular_order: int = 16,
    package_axial_order: int = 6,
    package_radial_order: int = 4,
    package_azimuthal_order: int = 16,
    background_segments_per_turn: int = 16,
    background_radial_order: int = 12,
    background_angular_order: int = 48,
    background_conductivity_range=None,
    device: str = "cpu",
):
    samples = tuple(
        samples
    )
    validation_samples = tuple(
        validation_samples
    )
    if not samples:
        raise ValueError(
            "at least one hybrid spatial training sample is required"
        )
    if (
        not bool(
            getattr(
                port_artifact,
                "supports_packages",
                False,
            )
        )
        or not hasattr(
            port_artifact,
            "fingerprint",
        )
    ):
        raise TypeError(
            "hybrid spatial training requires a fingerprinted package-aware port artifact"
        )
    if any(
        not sample.has_spatial_truth
        for sample in samples
    ):
        raise ValueError(
            "all hybrid spatial training samples require spatial truth"
        )
    if (
        epochs < 1
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or patience < 1
        or validation_interval < 1
        or min_improvement < 0.0
        or background_segments_per_turn < 4
        or background_radial_order < 3
        or background_angular_order < 8
    ):
        raise ValueError(
            "invalid hybrid spatial training configuration"
        )

    training_background_conductivity = np.asarray(
        [
            sample.scene.medium.conductivity
            for sample
            in samples
        ],
        dtype=float,
    )
    has_lossy_background_training = bool(
        np.any(
            training_background_conductivity
            > 0.0
        )
    )
    if has_lossy_background_training:
        if not bool(
            getattr(
                port_artifact,
                "supports_lossy_background",
                False,
            )
        ):
            raise ValueError(
                "lossy-background spatial training requires a port artifact "
                "trained for lossy homogeneous backgrounds"
            )
        candidate = (
            getattr(
                port_artifact,
                "background_conductivity_range",
                None,
            )
            if background_conductivity_range
            is None
            else background_conductivity_range
        )
        if candidate is None:
            raise ValueError(
                "lossy-background spatial training requires a declared "
                "background conductivity domain"
            )
        values = np.asarray(
            candidate,
            dtype=float,
        )
        if (
            values.shape != (
                2,
            )
            or np.any(
                ~np.isfinite(
                    values
                )
            )
            or values[
                0
            ] < 0.0
            or values[
                1
            ] < values[
                0
            ]
        ):
            raise ValueError(
                "background_conductivity_range must be a finite "
                "nonnegative increasing pair"
            )
        resolved_background_conductivity_range = (
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
        port_range = getattr(
            port_artifact,
            "background_conductivity_range",
            None,
        )
        if port_range is not None:
            port_lower, port_upper = (
                port_range
            )
            lower, upper = (
                resolved_background_conductivity_range
            )
            if (
                lower
                < port_lower
                or upper
                > port_upper
            ):
                raise ValueError(
                    "spatial background conductivity domain cannot exceed "
                    "the port artifact domain"
                )
    else:
        resolved_background_conductivity_range = None

    for sample in (
        samples
        + validation_samples
    ):
        conductivity = float(
            sample.scene.medium.conductivity
        )
        if resolved_background_conductivity_range is None:
            if conductivity > 0.0:
                raise ValueError(
                    "sample background conductivity lies outside the "
                    "hybrid spatial training domain"
                )
            continue
        lower, upper = (
            resolved_background_conductivity_range
        )
        if (
            conductivity
            < lower
            or conductivity
            > upper
        ):
            raise ValueError(
                "sample background conductivity lies outside the "
                "hybrid spatial training domain"
            )

    torch.manual_seed(
        seed
    )
    np.random.seed(
        seed
    )
    port_model = (
        port_artifact.model
    )
    for parameter in (
        port_model.parameters()
    ):
        parameter.requires_grad_(
            False
        )
    model = (
        HybridSpatialLossShapeNet(
            port_model.hidden_dim,
            port_model.coil_pair_dim,
            port_model.cross_dim,
            field_hidden_dim=(
                field_hidden_dim
            ),
            factor_rank=(
                factor_rank
            ),
            depth=depth,
        ).to(
            device=device,
            dtype=next(
                port_model.parameters()
            ).dtype,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=(
            weight_decay
        ),
    )

    final_loss = np.inf
    best_epoch = 0
    best_validation_error = None
    best_validation_shape_error = None
    best_state = None
    stale = 0
    stopped_early = False
    epochs_run = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        order = np.random.permutation(
            len(samples)
        )
        epoch_loss = 0.0
        for index in order:
            optimizer.zero_grad(
                set_to_none=True
            )
            loss = _sample_loss(
                model,
                port_artifact,
                samples[
                    int(index)
                ],
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                10.0,
            )
            optimizer.step()
            epoch_loss += float(
                loss.detach().cpu()
            )
        final_loss = (
            epoch_loss
            / len(samples)
        )
        epochs_run = epoch

        if validation_samples:
            if (
                epoch
                % validation_interval
                != 0
                and epoch
                != epochs
            ):
                continue
            model.eval()
            with torch.no_grad():
                shape_values = [
                    float(
                        _sample_loss(
                            model,
                            port_artifact,
                            sample,
                            device=device,
                        ).detach().cpu()
                    )
                    for sample
                    in validation_samples
                ]
            end_to_end_values = [
                _sample_end_to_end_error(
                    model,
                    port_artifact,
                    sample,
                    conductor_longitudinal_points=(
                        conductor_longitudinal_points
                    ),
                    conductor_radial_order=(
                        conductor_radial_order
                    ),
                    conductor_angular_order=(
                        conductor_angular_order
                    ),
                    package_axial_order=(
                        package_axial_order
                    ),
                    package_radial_order=(
                        package_radial_order
                    ),
                    package_azimuthal_order=(
                        package_azimuthal_order
                    ),
                    background_segments_per_turn=(
                        background_segments_per_turn
                    ),
                    background_radial_order=(
                        background_radial_order
                    ),
                    background_angular_order=(
                        background_angular_order
                    ),
                    background_conductivity_range=(
                        resolved_background_conductivity_range
                    ),
                    device=device,
                )
                for sample
                in validation_samples
            ]
            shape_score = float(
                np.mean(
                    shape_values
                )
            )
            score = float(
                np.mean(
                    end_to_end_values
                )
            )
            if (
                best_validation_error
                is None
                or score
                < best_validation_error
                - min_improvement
            ):
                best_validation_error = (
                    score
                )
                best_validation_shape_error = (
                    shape_score
                )
                best_epoch = epoch
                best_state = {
                    key: value.detach()
                    .cpu()
                    .clone()
                    for key, value
                    in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    stopped_early = True
                    break
        else:
            best_epoch = epoch
            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "hybrid spatial training completed without a selectable model state"
        )
    model.load_state_dict(
        best_state
    )
    model.eval()
    artifact = (
        HybridSpatialLossArtifact(
            port_artifact,
            model,
            conductor_longitudinal_points=(
                conductor_longitudinal_points
            ),
            conductor_radial_order=(
                conductor_radial_order
            ),
            conductor_angular_order=(
                conductor_angular_order
            ),
            package_axial_order=(
                package_axial_order
            ),
            package_radial_order=(
                package_radial_order
            ),
            package_azimuthal_order=(
                package_azimuthal_order
            ),
            background_segments_per_turn=(
                background_segments_per_turn
            ),
            background_radial_order=(
                background_radial_order
            ),
            background_angular_order=(
                background_angular_order
            ),
            background_conductivity_range=(
                resolved_background_conductivity_range
            ),
            device=device,
        )
    )
    return (
        artifact,
        HybridSpatialTrainingReport(
            float(
                final_loss
            ),
            int(
                epochs_run
            ),
            int(
                best_epoch
            ),
            (
                None
                if best_validation_error
                is None
                else float(
                    best_validation_error
                )
            ),
            bool(
                stopped_early
            ),
            (
                None
                if best_validation_shape_error
                is None
                else float(
                    best_validation_shape_error
                )
            ),
        ),
    )
