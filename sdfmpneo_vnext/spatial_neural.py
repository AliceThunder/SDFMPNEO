from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.spatial_neural requires the 'neural' extra: "
        "pip install 'sdfmpneo[neural]'"
    ) from exc

from .basis import superellipse_section_quadrature
from .features import encode_scene_invariant
from .scene import Scene
from .training_data import TeacherSample


SPATIAL_ARTIFACT_SCHEMA = 2


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
):
    layers = []
    width = input_dim
    for _ in range(
        depth
    ):
        layers.extend(
            [
                nn.Linear(
                    width,
                    hidden_dim,
                ),
                nn.SiLU(),
            ]
        )
        width = hidden_dim
    layers.append(
        nn.Linear(
            width,
            output_dim,
        )
    )
    return nn.Sequential(
        *layers
    )


def _psd_sqrt(
    matrix,
    *,
    inverse: bool,
):
    matrix = 0.5 * (
        matrix
        + matrix.conj().transpose(
            -1,
            -2,
        )
    )
    eigenvalues, eigenvectors = (
        torch.linalg.eigh(
            matrix
        )
    )
    scale = torch.clamp(
        torch.max(
            torch.abs(
                eigenvalues
            )
        ),
        min=1e-20,
    )
    if inverse:
        floor = (
            1e-10
            * scale
            + 1e-16
        )
        values = torch.rsqrt(
            torch.clamp(
                eigenvalues.real,
                min=floor,
            )
        )
    else:
        values = torch.sqrt(
            torch.clamp(
                eigenvalues.real,
                min=0.0,
            )
        )
    return (
        eigenvectors
        @ torch.diag(
            values.to(
                eigenvectors.dtype
            )
        )
        @ eigenvectors.conj().transpose(
            -1,
            -2,
        )
    )


def _stable_cholesky(
    matrix,
):
    """Differentiable Cholesky with adaptive roundoff-only stabilization."""
    matrix = 0.5 * (
        matrix
        + matrix.conj().T
    )
    chol, info = (
        torch.linalg.cholesky_ex(
            matrix,
            check_errors=False,
        )
    )
    if int(
        info.max().detach().cpu()
    ) == 0:
        return chol

    n = matrix.shape[-1]
    identity = torch.eye(
        n,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    scale = torch.clamp(
        torch.real(
            torch.trace(
                matrix
            )
        )
        / max(
            n,
            1,
        ),
        min=1e-30,
    )
    dtype_eps = torch.finfo(
        matrix.real.dtype
    ).eps
    for multiplier in (
        8.0,
        64.0,
        512.0,
        4096.0,
    ):
        jitter = (
            multiplier
            * dtype_eps
            * scale
        )
        chol, info = (
            torch.linalg.cholesky_ex(
                matrix
                + jitter
                * identity,
                check_errors=False,
            )
        )
        if int(
            info.max().detach().cpu()
        ) == 0:
            return chol
    raise RuntimeError(
        "spatial PSD normalization remained numerically singular "
        "after roundoff-scale stabilization"
    )


def _coordinate_features(
    scene: Scene,
    coil_index,
    arc_fraction,
    xy,
):
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
        arc_fraction.shape
        != (n,)
        or xy.shape
        != (
            n,
            2,
        )
    ):
        raise ValueError(
            "spatial coordinates have incompatible shapes"
        )
    out = np.empty(
        (
            n,
            5,
        ),
        dtype=float,
    )
    for index in range(
        n
    ):
        coil = int(
            coil_index[
                index
            ]
        )
        geometry = (
            scene.coils[
                coil
            ].geometry
        )
        xn = (
            xy[
                index,
                0,
            ]
            / (
                0.5
                * geometry.conductor_width
            )
        )
        yn = (
            xy[
                index,
                1,
            ]
            / (
                0.5
                * geometry.conductor_thickness
            )
        )
        m = (
            geometry.cross_section_exponent
        )
        rho = (
            abs(
                xn
            ) ** m
            + abs(
                yn
            ) ** m
        ) ** (
            1.0
            / m
        )
        out[
            index
        ] = (
            2.0
            * arc_fraction[
                index
            ]
            - 1.0,
            xn,
            yn,
            rho,
            rho**2,
        )
    return out


class SpatialLossShapeNet(
    nn.Module
):
    """Continuous intrinsic PSD field-shape network."""

    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int,
        *,
        field_hidden_dim: int = 64,
        factor_rank: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        if (
            hidden_dim < 1
            or pair_dim < 1
            or field_hidden_dim < 4
            or factor_rank < 1
            or depth < 1
        ):
            raise ValueError(
                "invalid spatial network dimensions"
            )
        self.hidden_dim = int(
            hidden_dim
        )
        self.pair_dim = int(
            pair_dim
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
            2
            * self.hidden_dim
            + self.pair_dim
            + 5,
            self.field_hidden_dim,
            2
            * self.factor_rank,
            self.depth,
        )

    def raw_matrices(
        self,
        latent,
        pair_features,
        coil_index,
        coordinate_features,
    ):
        coil_index = torch.as_tensor(
            coil_index,
            dtype=torch.long,
            device=latent.device,
        )
        coordinates = torch.as_tensor(
            coordinate_features,
            dtype=latent.dtype,
            device=latent.device,
        )
        n_points = int(
            coil_index.numel()
        )
        n_ports = int(
            latent.shape[0]
        )
        complex_dtype = (
            torch.complex64
            if latent.dtype
            == torch.float32
            else torch.complex128
        )
        factors = []
        for point in range(
            n_points
        ):
            coil = int(
                coil_index[
                    point
                ].item()
            )
            rows = []
            for port in range(
                n_ports
            ):
                features = torch.cat(
                    (
                        latent[
                            coil
                        ],
                        latent[
                            port
                        ],
                        pair_features[
                            coil,
                            port,
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
            device=latent.device,
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


def _normalize_by_coil(
    raw_matrices,
    coil_index,
    weights,
    target_channels,
):
    coil_index = torch.as_tensor(
        coil_index,
        dtype=torch.long,
        device=raw_matrices.device,
    )
    weights = torch.as_tensor(
        weights,
        dtype=raw_matrices.real.dtype,
        device=raw_matrices.device,
    )
    target_channels = torch.as_tensor(
        target_channels,
        dtype=raw_matrices.dtype,
        device=raw_matrices.device,
    )
    normalized = (
        torch.empty_like(
            raw_matrices
        )
    )
    n_coils = int(
        target_channels.shape[0]
    )
    for coil in range(
        n_coils
    ):
        mask = (
            coil_index
            == coil
        )
        if not bool(
            torch.any(
                mask
            )
        ):
            raise ValueError(
                "normalization quadrature has no samples for a coil"
            )
        raw_integral = torch.sum(
            weights[
                mask,
                None,
                None,
            ]
            * raw_matrices[
                mask
            ],
            dim=0,
        )
        raw_integral = 0.5 * (
            raw_integral
            + raw_integral.conj().T
        )
        identity = torch.eye(
            raw_integral.shape[0],
            dtype=raw_integral.dtype,
            device=raw_integral.device,
        )
        chol = _stable_cholesky(
            raw_integral
        )
        inverse_chol = (
            torch.linalg.solve_triangular(
                chol,
                identity,
                upper=False,
            )
        )
        transform = (
            _psd_sqrt(
                target_channels[
                    coil
                ],
                inverse=False,
            )
            @ inverse_chol
        )
        values = (
            transform[
                None,
                :,
                :
            ]
            @ raw_matrices[
                mask
            ]
            @ transform.conj().transpose(
                -1,
                -2,
            )[
                None,
                :,
                :
            ]
        )
        normalized[
            mask
        ] = 0.5 * (
            values
            + values.conj().transpose(
                -1,
                -2,
            )
        )
    return normalized


def _normalization_rule(
    scene: Scene,
    *,
    longitudinal_points: int,
    radial_order: int,
    angular_order: int,
):
    if (
        longitudinal_points < 2
        or radial_order < 2
        or angular_order < 8
    ):
        raise ValueError(
            "invalid spatial normalization quadrature"
        )
    coils = []
    arc = []
    xy = []
    weights = []
    for coil_index, coil in enumerate(
        scene.coils
    ):
        geometry = (
            coil.geometry
        )
        section = (
            superellipse_section_quadrature(
                geometry.conductor_width,
                geometry.conductor_thickness,
                geometry.cross_section_exponent,
                radial_order=(
                    radial_order
                ),
                angular_order=(
                    angular_order
                ),
            )
        )
        poly = geometry.polyline(
            longitudinal_points
        )
        for longitudinal in range(
            longitudinal_points
        ):
            count = len(
                section.weights
            )
            coils.append(
                np.full(
                    count,
                    coil_index,
                    dtype=int,
                )
            )
            arc.append(
                np.full(
                    count,
                    (
                        longitudinal
                        + 0.5
                    )
                    / longitudinal_points,
                    dtype=float,
                )
            )
            xy.append(
                section.xy
            )
            weights.append(
                section.weights
                * (
                    poly.total_length
                    / longitudinal_points
                )
            )
    return (
        np.concatenate(
            coils
        ),
        np.concatenate(
            arc
        ),
        np.concatenate(
            xy,
            axis=0,
        ),
        np.concatenate(
            weights
        ),
    )


@dataclass(frozen=True)
class PreparedSpatialLossField:
    scene: Scene
    frequency_hz: float
    port_prediction: object
    model: SpatialLossShapeNet
    latent: object
    pair_features: object
    transforms: tuple
    normalization_closure_error: float
    device: str

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
        if coil_index.ndim != 1:
            raise ValueError(
                "coil_index must be one-dimensional"
            )
        n_query = len(
            coil_index
        )
        if (
            arc_fraction.shape != (n_query,)
            or xy.shape != (n_query, 2)
        ):
            raise ValueError(
                "spatial query arrays have incompatible shapes"
            )
        if np.any(
            (arc_fraction < 0.0)
            | (arc_fraction > 1.0)
        ):
            raise ValueError(
                "arc_fraction must lie in [0,1]"
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

        inside = np.ones(
            n_query,
            dtype=bool,
        )
        for index in range(
            n_query
        ):
            coil = int(
                coil_index[index]
            )
            geometry = (
                self.scene.coils[
                    coil
                ].geometry
            )
            xn = (
                abs(
                    float(
                        xy[index, 0]
                    )
                )
                / (
                    0.5
                    * geometry.conductor_width
                )
            )
            yn = (
                abs(
                    float(
                        xy[index, 1]
                    )
                )
                / (
                    0.5
                    * geometry.conductor_thickness
                )
            )
            inside[index] = bool(
                xn
                ** geometry.cross_section_exponent
                + yn
                ** geometry.cross_section_exponent
                <= 1.0 + 1e-12
            )

        coordinates = _coordinate_features(
            self.scene,
            coil_index,
            arc_fraction,
            xy,
        )
        with torch.no_grad():
            raw = self.model.raw_matrices(
                self.latent,
                self.pair_features,
                coil_index,
                coordinates,
            )
            out = torch.zeros_like(
                raw
            )
            for coil in range(
                len(
                    self.scene.coils
                )
            ):
                mask_np = (
                    (coil_index == coil)
                    & inside
                )
                if not np.any(
                    mask_np
                ):
                    continue
                mask = torch.as_tensor(
                    mask_np,
                    dtype=torch.bool,
                    device=raw.device,
                )
                transform = self.transforms[
                    coil
                ]
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
        return (
            out.detach().cpu().numpy()
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

    def local_joule_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        matrix = (
            self.local_dissipation_matrix(
                coil_index,
                arc_fraction,
                xy,
            )
        )
        currents = np.asarray(
            currents,
            dtype=complex,
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


class NeuralSpatialLossArtifact:
    supports_packages = False
    def __init__(
        self,
        port_artifact,
        model: SpatialLossShapeNet,
        *,
        longitudinal_points: int = 12,
        radial_order: int = 3,
        angular_order: int = 16,
        device: str = "cpu",
    ):
        if not hasattr(
            port_artifact,
            "predict_structured",
        ):
            raise TypeError(
                "port_artifact must expose predict_structured"
            )
        if not hasattr(
            port_artifact,
            "model",
        ):
            raise TypeError(
                "spatial neural artifact requires a neural port artifact"
            )
        if not hasattr(
            port_artifact,
            "fingerprint",
        ):
            raise TypeError(
                "spatial neural artifact requires a fingerprinted port artifact"
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
        self.longitudinal_points = int(
            longitudinal_points
        )
        self.radial_order = int(
            radial_order
        )
        self.angular_order = int(
            angular_order
        )
        self.device = str(
            device
        )

    def _latent(
        self,
        scene,
        frequency_hz,
    ):
        encoded = (
            encode_scene_invariant(
                scene,
                frequency_hz,
            )
        )
        node, pair = (
            self.port_artifact.normalizer.normalize(
                encoded
            )
        )
        port_model = (
            self.port_artifact.model
        )
        dtype = next(
            port_model.parameters()
        ).dtype
        device = next(
            port_model.parameters()
        ).device
        port_model.eval()
        with torch.no_grad():
            node_tensor = torch.as_tensor(
                node,
                dtype=dtype,
                device=device,
            )
            pair_tensor = torch.as_tensor(
                pair,
                dtype=dtype,
                device=device,
            )
            latent = (
                port_model._updated_features(
                    node_tensor,
                    pair_tensor,
                )
            )
        return (
            latent,
            pair_tensor,
        )

    def prepare(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> PreparedSpatialLossField:
        port_prediction = (
            self.port_artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )
        latent, pair = self._latent(
            scene,
            frequency_hz,
        )
        (
            coil_index,
            arc,
            xy,
            weights,
        ) = _normalization_rule(
            scene,
            longitudinal_points=(
                self.longitudinal_points
            ),
            radial_order=(
                self.radial_order
            ),
            angular_order=(
                self.angular_order
            ),
        )
        coordinate_features = (
            _coordinate_features(
                scene,
                coil_index,
                arc,
                xy,
            )
        )
        dtype = latent.dtype
        device = latent.device
        self.model.eval()
        with torch.no_grad():
            raw = (
                self.model.raw_matrices(
                    latent,
                    pair,
                    coil_index,
                    coordinate_features,
                )
            )
            target_channels = torch.as_tensor(
                port_prediction.dissipation_channels,
                dtype=raw.dtype,
                device=device,
            )
            weight_tensor = torch.as_tensor(
                weights,
                dtype=dtype,
                device=device,
            )
            transforms = []
            coil_tensor = torch.as_tensor(
                coil_index,
                dtype=torch.long,
                device=device,
            )
            for coil in range(
                len(
                    scene.coils
                )
            ):
                mask = (
                    coil_tensor
                    == coil
                )
                raw_integral = torch.sum(
                    weight_tensor[
                        mask,
                        None,
                        None,
                    ]
                    * raw[
                        mask
                    ],
                    dim=0,
                )
                raw_integral = 0.5 * (
                    raw_integral
                    + raw_integral.conj().T
                )
                identity = torch.eye(
                    raw_integral.shape[0],
                    dtype=raw_integral.dtype,
                    device=raw_integral.device,
                )
                chol = _stable_cholesky(
            raw_integral
        )
                inverse_chol = (
                    torch.linalg.solve_triangular(
                        chol,
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
        normalized_integral = np.zeros_like(
            port_prediction.dissipation_channels,
            dtype=complex,
        )
        with torch.no_grad():
            for coil in range(
                len(
                    scene.coils
                )
            ):
                mask = (
                    coil_tensor
                    == coil
                )
                transform = transforms[
                    coil
                ]
                values = (
                    transform[
                        None,
                        :,
                        :
                    ]
                    @ raw[
                        mask
                    ]
                    @ transform.conj().T[
                        None,
                        :,
                        :
                    ]
                )
                integrated = torch.sum(
                    weight_tensor[
                        mask,
                        None,
                        None,
                    ]
                    * values,
                    dim=0,
                )
                normalized_integral[
                    coil
                ] = (
                    integrated
                    .detach()
                    .cpu()
                    .numpy()
                )
        closure_error = float(
            np.linalg.norm(
                normalized_integral
                - port_prediction.dissipation_channels
            )
            / max(
                np.linalg.norm(
                    port_prediction.dissipation_channels
                ),
                1e-30,
            )
        )
        return PreparedSpatialLossField(
            scene,
            float(
                frequency_hz
            ),
            port_prediction,
            self.model,
            latent,
            pair,
            tuple(
                transforms
            ),
            closure_error,
            self.device,
        )

    def local_dissipation_matrix(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ):
        return (
            self.prepare(
                scene,
                frequency_hz,
            ).local_dissipation_matrix(
                coil_index,
                arc_fraction,
                xy,
            )
        )

    def save(
        self,
        path,
    ):
        payload = {
            "schema": (
                SPATIAL_ARTIFACT_SCHEMA
            ),
            "model_config": {
                "hidden_dim": (
                    self.model.hidden_dim
                ),
                "pair_dim": (
                    self.model.pair_dim
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
            "longitudinal_points": (
                self.longitudinal_points
            ),
            "radial_order": (
                self.radial_order
            ),
            "angular_order": (
                self.angular_order
            ),
            "port_fingerprint": (
                self.port_fingerprint
            ),
        }
        torch.save(
            payload,
            Path(
                path
            ),
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
                Path(
                    path
                ),
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(
                Path(
                    path
                ),
                map_location=device,
            )
        if (
            payload.get(
                "schema"
            )
            != SPATIAL_ARTIFACT_SCHEMA
        ):
            raise ValueError(
                "unsupported vNext spatial artifact schema"
            )
        expected_port_fingerprint = str(
            payload.get(
                "port_fingerprint",
                "",
            )
        )
        actual_port_fingerprint = (
            port_artifact.fingerprint()
            if hasattr(
                port_artifact,
                "fingerprint",
            )
            else ""
        )
        if (
            not expected_port_fingerprint
            or expected_port_fingerprint
            != actual_port_fingerprint
        ):
            raise ValueError(
                "spatial artifact port fingerprint mismatch"
            )
        model = SpatialLossShapeNet(
            **payload[
                "model_config"
            ]
        )
        model.load_state_dict(
            payload[
                "model_state"
            ]
        )
        model.eval()
        return NeuralSpatialLossArtifact(
            port_artifact,
            model,
            longitudinal_points=int(
                payload[
                    "longitudinal_points"
                ]
            ),
            radial_order=int(
                payload[
                    "radial_order"
                ]
            ),
            angular_order=int(
                payload[
                    "angular_order"
                ]
            ),
            device=device,
        )


@dataclass(frozen=True)
class SpatialTrainingReport:
    final_loss: float
    epochs: int
    best_epoch: int
    best_validation_error: float | None
    stopped_early: bool


def _sample_spatial_loss(
    model,
    port_artifact,
    sample: TeacherSample,
    *,
    device: str,
):
    if (
        sample.spatial_loss
        is None
        or sample.target_dissipation_channels
        is None
    ):
        raise ValueError(
            "spatial training requires spatial truth and dissipation channels"
        )
    encoded = sample.encoded
    node, pair = (
        port_artifact.normalizer.normalize(
            encoded
        )
    )
    port_model = (
        port_artifact.model
    )
    port_dtype = next(
        port_model.parameters()
    ).dtype
    port_model.eval()
    with torch.no_grad():
        node_tensor = torch.as_tensor(
            node,
            dtype=port_dtype,
            device=device,
        )
        pair_tensor = torch.as_tensor(
            pair,
            dtype=port_dtype,
            device=device,
        )
        latent = (
            port_model._updated_features(
                node_tensor,
                pair_tensor,
            )
        )

    spatial = (
        sample.spatial_loss
    )
    coordinate_features = (
        _coordinate_features(
            sample.scene,
            spatial.coil_index,
            spatial.arc_fraction,
            spatial.xy,
        )
    )
    raw = model.raw_matrices(
        latent,
        pair_tensor,
        spatial.coil_index,
        coordinate_features,
    )
    target_channels = torch.as_tensor(
        sample.target_dissipation_channels,
        dtype=raw.dtype,
        device=device,
    )
    normalized = _normalize_by_coil(
        raw,
        spatial.coil_index,
        spatial.weights,
        target_channels,
    )
    target = torch.as_tensor(
        spatial.dissipation_matrix,
        dtype=raw.dtype,
        device=device,
    )
    weights = torch.as_tensor(
        spatial.weights,
        dtype=raw.real.dtype,
        device=device,
    )
    numerator = torch.sum(
        weights[
            :,
            None,
            None,
        ]
        * torch.abs(
            normalized
            - target
        ) ** 2
    )
    denominator = torch.sum(
        weights[
            :,
            None,
            None,
        ]
        * torch.abs(
            target
        ) ** 2
    ) + 1e-18
    return (
        numerator
        / denominator
    )


def train_spatial_loss_surrogate(
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
    seed: int = 23,
    longitudinal_points: int = 12,
    radial_order: int = 3,
    angular_order: int = 16,
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
            "at least one spatial training sample is required"
        )
    if (
        epochs < 1
        or learning_rate <= 0.0
        or patience < 1
        or validation_interval < 1
        or min_improvement < 0.0
    ):
        raise ValueError(
            "invalid spatial training configuration"
        )
    if any(
        sample.spatial_loss is None
        or sample.target_dissipation_channels is None
        for sample in samples
    ):
        raise ValueError(
            "all spatial training samples require spatial truth"
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
    hidden_dim = int(
        port_model.hidden_dim
    )
    pair_dim = int(
        port_model.pair_dim
    )

    model = SpatialLossShapeNet(
        hidden_dim,
        pair_dim,
        field_hidden_dim=(
            field_hidden_dim
        ),
        factor_rank=(
            factor_rank
        ),
        depth=depth,
    ).to(
        device
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
            len(
                samples
            )
        )
        epoch_loss = 0.0
        for index in order:
            optimizer.zero_grad(
                set_to_none=True
            )
            loss = (
                _sample_spatial_loss(
                    model,
                    port_artifact,
                    samples[
                        int(
                            index
                        )
                    ],
                    device=device,
                )
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
            / len(
                samples
            )
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
                values = [
                    float(
                        _sample_spatial_loss(
                            model,
                            port_artifact,
                            sample,
                            device=device,
                        ).detach().cpu()
                    )
                    for sample
                    in validation_samples
                ]
            score = float(
                np.mean(
                    values
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
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
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
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "spatial training completed without a selectable state"
        )
    model.load_state_dict(
        best_state
    )
    model.eval()
    return (
        NeuralSpatialLossArtifact(
            port_artifact,
            model,
            longitudinal_points=(
                longitudinal_points
            ),
            radial_order=(
                radial_order
            ),
            angular_order=(
                angular_order
            ),
            device=device,
        ),
        SpatialTrainingReport(
            final_loss=float(
                final_loss
            ),
            epochs=int(
                epochs_run
            ),
            best_epoch=int(
                best_epoch
            ),
            best_validation_error=(
                None
                if best_validation_error
                is None
                else float(
                    best_validation_error
                )
            ),
            stopped_early=bool(
                stopped_early
            ),
        ),
    )
