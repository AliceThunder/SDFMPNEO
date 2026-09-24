from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "sdfmpneo_vnext.spatial_neural requires the 'neural' extra: "
        "pip install 'sdfmpneo[neural]'"
    ) from exc

from .basis import superellipse_section_quadrature
from .features import EncodedScene, encode_scene_invariant
from .scene import Scene
from .training_data import TeacherSample


SPATIAL_ARTIFACT_SCHEMA = 1
LOCAL_FEATURE_DIM = 7


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
):
    layers = []
    width = int(input_dim)
    for _ in range(int(depth)):
        layers.extend(
            [
                nn.Linear(
                    width,
                    hidden_dim,
                ),
                nn.SiLU(),
            ]
        )
        width = int(hidden_dim)
    layers.append(
        nn.Linear(
            width,
            output_dim,
        )
    )
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class SpatialFeatureNormalizer:
    node_mean: np.ndarray
    node_scale: np.ndarray
    pair_mean: np.ndarray
    pair_scale: np.ndarray

    @staticmethod
    def fit(
        samples,
        *,
        floor: float = 1e-8,
    ) -> "SpatialFeatureNormalizer":
        samples = tuple(samples)
        if not samples:
            raise ValueError(
                "at least one teacher sample is required"
            )
        node = np.concatenate(
            [
                sample.encoded.node_features
                for sample in samples
            ],
            axis=0,
        )
        pair = np.concatenate(
            [
                sample.encoded.pair_features.reshape(
                    -1,
                    sample.encoded.pair_features.shape[-1],
                )
                for sample in samples
            ],
            axis=0,
        )
        return SpatialFeatureNormalizer(
            node.mean(axis=0),
            np.maximum(
                node.std(axis=0),
                floor,
            ),
            pair.mean(axis=0),
            np.maximum(
                pair.std(axis=0),
                floor,
            ),
        )

    def normalize(
        self,
        encoded: EncodedScene,
    ):
        return (
            (
                encoded.node_features
                - self.node_mean
            )
            / self.node_scale,
            (
                encoded.pair_features
                - self.pair_mean
            )
            / self.pair_scale,
        )

    def to_dict(self):
        return {
            "node_mean": self.node_mean,
            "node_scale": self.node_scale,
            "pair_mean": self.pair_mean,
            "pair_scale": self.pair_scale,
        }

    @staticmethod
    def from_dict(data):
        def _array(value):
            if hasattr(
                value,
                "detach",
            ):
                value = (
                    value.detach()
                    .cpu()
                    .numpy()
                )
            return np.asarray(
                value,
                dtype=float,
            )

        return SpatialFeatureNormalizer(
            _array(
                data["node_mean"]
            ),
            _array(
                data["node_scale"]
            ),
            _array(
                data["pair_mean"]
            ),
            _array(
                data["pair_scale"]
            ),
        )


def _local_features_numpy(
    scene: Scene,
    coil_index: int,
    arc_fraction,
    xy,
):
    arc = np.asarray(
        arc_fraction,
        dtype=float,
    )
    xy = np.asarray(
        xy,
        dtype=float,
    )
    scalar = (
        arc.ndim == 0
        and xy.ndim == 1
    )
    arc = np.atleast_1d(
        arc
    )
    xy = np.atleast_2d(
        xy
    )
    if xy.shape != (
        len(arc),
        2,
    ):
        raise ValueError(
            "arc_fraction and xy must describe the same number of points"
        )
    if np.any(
        (arc < 0.0)
        | (arc > 1.0)
    ):
        raise ValueError(
            "arc_fraction must lie in [0,1]"
        )
    if not (
        0
        <= coil_index
        < len(scene.coils)
    ):
        raise IndexError(
            "coil_index out of range"
        )

    geometry = (
        scene.coils[
            coil_index
        ].geometry
    )
    a = (
        0.5
        * geometry.conductor_width
    )
    b = (
        0.5
        * geometry.conductor_thickness
    )
    xhat = (
        xy[:, 0]
        / a
    )
    yhat = (
        xy[:, 1]
        / b
    )
    exponent = (
        geometry.cross_section_exponent
    )
    rho = (
        np.abs(xhat) ** exponent
        + np.abs(yhat) ** exponent
    ) ** (
        1.0 / exponent
    )
    inside = (
        rho
        <= 1.0 + 1e-12
    )
    features = np.stack(
        (
            2.0 * arc - 1.0,
            np.sin(
                2.0
                * np.pi
                * arc
            ),
            np.cos(
                2.0
                * np.pi
                * arc
            ),
            xhat,
            yhat,
            rho,
            1.0 - rho,
        ),
        axis=1,
    )
    if scalar:
        return (
            features[0],
            bool(
                inside[0]
            ),
        )
    return (
        features,
        inside,
    )


def _psd_sqrt(
    matrix,
):
    matrix = 0.5 * (
        matrix
        + matrix.conj().transpose(
            -1,
            -2,
        )
    )
    values, vectors = (
        torch.linalg.eigh(
            matrix
        )
    )
    values = torch.clamp(
        values.real,
        min=0.0,
    )
    return (
        vectors
        @ torch.diag_embed(
            torch.sqrt(
                values
            ).to(
                vectors.dtype
            )
        )
        @ vectors.conj().transpose(
            -1,
            -2,
        )
    )


def _normalize_raw_shapes(
    raw,
    weights,
):
    if raw.ndim != 3:
        raise ValueError(
            "raw field must have shape (n_points,n_ports,n_ports)"
        )
    weights = weights.to(
        dtype=raw.real.dtype,
        device=raw.device,
    )
    if weights.shape != (
        raw.shape[0],
    ):
        raise ValueError(
            "weights have wrong shape"
        )

    n_ports = raw.shape[1]
    eye = torch.eye(
        n_ports,
        dtype=raw.dtype,
        device=raw.device,
    )
    trace_scale = torch.mean(
        torch.real(
            torch.diagonal(
                raw,
                dim1=-2,
                dim2=-1,
            ).sum(
                dim=-1
            )
        )
    )
    jitter = (
        1e-8
        * torch.clamp(
            trace_scale
            / max(
                n_ports,
                1,
            ),
            min=0.0,
        )
        + 1e-12
    )
    raw = (
        raw
        + jitter
        * eye.unsqueeze(0)
    )

    integrated = torch.einsum(
        "q,qij->ij",
        weights,
        raw,
    )
    integrated = 0.5 * (
        integrated
        + integrated.conj().T
    )
    chol = torch.linalg.cholesky(
        integrated
    )
    inverse_chol = (
        torch.linalg.solve_triangular(
            chol,
            eye,
            upper=False,
        )
    )
    normalized = torch.einsum(
        "ab,qbc,dc->qad",
        inverse_chol,
        raw,
        inverse_chol.conj(),
    )
    normalized = 0.5 * (
        normalized
        + normalized.conj().transpose(
            -1,
            -2,
        )
    )
    return normalized


class SpatialLossShapeNet(nn.Module):
    """Scene-graph-conditioned continuous PSD loss-shape operator."""

    def __init__(
        self,
        node_dim: int = 17,
        pair_dim: int = 15,
        hidden_dim: int = 64,
        factor_rank: int = 2,
        depth: int = 2,
        local_dim: int = LOCAL_FEATURE_DIM,
    ):
        super().__init__()
        if (
            hidden_dim < 4
            or factor_rank < 1
            or depth < 1
            or local_dim < 1
        ):
            raise ValueError(
                "invalid spatial network dimensions"
            )
        self.node_dim = int(
            node_dim
        )
        self.pair_dim = int(
            pair_dim
        )
        self.hidden_dim = int(
            hidden_dim
        )
        self.factor_rank = int(
            factor_rank
        )
        self.depth = int(
            depth
        )
        self.local_dim = int(
            local_dim
        )

        self.node_encoder = _mlp(
            self.node_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.depth,
        )
        self.edge_encoder = _mlp(
            2 * self.hidden_dim
            + self.pair_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.depth,
        )
        self.node_update = _mlp(
            2 * self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.depth,
        )
        self.local_factor_head = _mlp(
            2 * self.hidden_dim
            + self.pair_dim
            + self.local_dim,
            self.hidden_dim,
            2 * self.factor_rank,
            self.depth,
        )

    def encode_scene(
        self,
        node_features,
        pair_features,
    ):
        h = self.node_encoder(
            node_features
        )
        n = h.shape[0]
        messages = []
        for i in range(n):
            incoming = []
            for j in range(n):
                if i == j:
                    continue
                edge = torch.cat(
                    (
                        h[i],
                        h[j],
                        pair_features[
                            i,
                            j,
                        ],
                    ),
                    dim=-1,
                )
                incoming.append(
                    self.edge_encoder(
                        edge
                    )
                )
            if incoming:
                aggregate = torch.stack(
                    incoming,
                    dim=0,
                ).sum(
                    dim=0
                ) / math.sqrt(
                    len(
                        incoming
                    )
                )
            else:
                aggregate = (
                    torch.zeros_like(
                        h[i]
                    )
                )
            messages.append(
                aggregate
            )
        return self.node_update(
            torch.cat(
                (
                    h,
                    torch.stack(
                        messages,
                        dim=0,
                    ),
                ),
                dim=-1,
            )
        )

    def raw_matrices(
        self,
        updated,
        pair_features,
        coil_index: int,
        local_features,
    ):
        local_features = torch.atleast_2d(
            local_features
        )
        n_points = (
            local_features.shape[0]
        )
        n_ports = (
            updated.shape[0]
        )
        rows = []
        for port in range(
            n_ports
        ):
            parts = (
                updated[
                    coil_index
                ].expand(
                    n_points,
                    -1,
                ),
                updated[
                    port
                ].expand(
                    n_points,
                    -1,
                ),
                pair_features[
                    coil_index,
                    port,
                ].expand(
                    n_points,
                    -1,
                ),
                local_features,
            )
            raw = (
                self.local_factor_head(
                    torch.cat(
                        parts,
                        dim=-1,
                    )
                )
            )
            real = raw[
                :,
                : self.factor_rank,
            ]
            imag = raw[
                :,
                self.factor_rank :,
            ]
            complex_dtype = (
                torch.complex64
                if real.dtype
                == torch.float32
                else torch.complex128
            )
            rows.append(
                real.to(
                    complex_dtype
                )
                + 1j
                * imag.to(
                    complex_dtype
                )
            )
        factor = torch.stack(
            rows,
            dim=1,
        )
        matrices = torch.einsum(
            "qir,qjr->qij",
            factor,
            factor.conj(),
        )
        return 0.5 * (
            matrices
            + matrices.conj().transpose(
                -1,
                -2,
            )
        )


@dataclass(frozen=True)
class SpatialTrainingReport:
    final_loss: float
    epochs: int
    samples: int
    best_epoch: int
    best_validation_error: float | None
    stopped_early: bool


class NeuralSpatialLossArtifact:
    """Continuous learned Joule field normalized to port dissipation channels."""

    def __init__(
        self,
        port_artifact,
        model: SpatialLossShapeNet,
        normalizer: SpatialFeatureNormalizer,
        *,
        integration_segments_per_turn: int = 12,
        integration_min_segments: int = 16,
        integration_radial_order: int = 4,
        integration_angular_order: int = 24,
        device: str = "cpu",
    ):
        if not hasattr(
            port_artifact,
            "predict_structured",
        ):
            raise TypeError(
                "port_artifact must expose predict_structured"
            )
        if (
            integration_segments_per_turn
            < 4
            or integration_min_segments
            < 4
            or integration_radial_order
            < 2
            or integration_angular_order
            < 8
        ):
            raise ValueError(
                "invalid spatial integration configuration"
            )
        self.port_artifact = (
            port_artifact
        )
        self.model = model
        self.normalizer = (
            normalizer
        )
        self.integration_segments_per_turn = int(
            integration_segments_per_turn
        )
        self.integration_min_segments = int(
            integration_min_segments
        )
        self.integration_radial_order = int(
            integration_radial_order
        )
        self.integration_angular_order = int(
            integration_angular_order
        )
        self.device = str(
            device
        )
        self.model.to(
            self.device
        )
        self.model.eval()

    def predict_structured(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        return (
            self.port_artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )

    def predict(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        return (
            self.predict_structured(
                scene,
                frequency_hz,
            ).impedance
        )

    def _scene_tensors(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        encoded = (
            encode_scene_invariant(
                scene,
                frequency_hz,
            )
        )
        node, pair = (
            self.normalizer.normalize(
                encoded
            )
        )
        dtype = next(
            self.model.parameters()
        ).dtype
        device = next(
            self.model.parameters()
        ).device
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
        with torch.no_grad():
            updated = (
                self.model.encode_scene(
                    node_tensor,
                    pair_tensor,
                )
            )
        return (
            updated,
            pair_tensor,
            dtype,
            device,
        )

    def _integration_rule(
        self,
        scene: Scene,
        coil_index: int,
    ):
        geometry = (
            scene.coils[
                coil_index
            ].geometry
        )
        n_segments = max(
            self.integration_min_segments,
            int(
                np.ceil(
                    self.integration_segments_per_turn
                    * geometry.turns
                )
            ),
        )
        polyline = geometry.polyline(
            n_segments
        )
        section = (
            superellipse_section_quadrature(
                geometry.conductor_width,
                geometry.conductor_thickness,
                geometry.cross_section_exponent,
                self.integration_radial_order,
                self.integration_angular_order,
            )
        )
        arc = []
        xy = []
        weights = []
        for segment_index, length in enumerate(
            polyline.lengths
        ):
            count = len(
                section.weights
            )
            arc.append(
                np.full(
                    count,
                    (
                        segment_index
                        + 0.5
                    )
                    / n_segments,
                    dtype=float,
                )
            )
            xy.append(
                section.xy
            )
            weights.append(
                section.weights
                * float(
                    length
                )
            )
        return (
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

    def _shape_normalizer(
        self,
        scene: Scene,
        coil_index: int,
        updated,
        pair_tensor,
        dtype,
        device,
    ):
        arc, xy, weights = (
            self._integration_rule(
                scene,
                coil_index,
            )
        )
        local, _ = (
            _local_features_numpy(
                scene,
                coil_index,
                arc,
                xy,
            )
        )
        local_tensor = torch.as_tensor(
            local,
            dtype=dtype,
            device=device,
        )
        weight_tensor = torch.as_tensor(
            weights,
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            raw = (
                self.model.raw_matrices(
                    updated,
                    pair_tensor,
                    coil_index,
                    local_tensor,
                )
            )
            normalized = (
                _normalize_raw_shapes(
                    raw,
                    weight_tensor,
                )
            )
        return (
            arc,
            xy,
            weights,
            normalized,
        )

    def local_dissipation_matrices(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction,
        xy,
    ) -> np.ndarray:
        local, inside = (
            _local_features_numpy(
                scene,
                coil_index,
                arc_fraction,
                xy,
            )
        )
        local = np.atleast_2d(
            local
        )
        inside = np.atleast_1d(
            inside
        )
        updated, pair, dtype, device = (
            self._scene_tensors(
                scene,
                frequency_hz,
            )
        )
        (
            _,
            _,
            weights,
            normalized_grid,
        ) = self._shape_normalizer(
            scene,
            coil_index,
            updated,
            pair,
            dtype,
            device,
        )

        # Recover the common normalization congruence from the integration
        # rule by applying the same raw normalization map to the query points.
        arc_grid, xy_grid, weights_grid = (
            self._integration_rule(
                scene,
                coil_index,
            )
        )
        local_grid, _ = (
            _local_features_numpy(
                scene,
                coil_index,
                arc_grid,
                xy_grid,
            )
        )
        with torch.no_grad():
            raw_grid = (
                self.model.raw_matrices(
                    updated,
                    pair,
                    coil_index,
                    torch.as_tensor(
                        local_grid,
                        dtype=dtype,
                        device=device,
                    ),
                )
            )
            # Build the same Cholesky congruence explicitly.
            n_ports = raw_grid.shape[1]
            eye = torch.eye(
                n_ports,
                dtype=raw_grid.dtype,
                device=device,
            )
            trace_scale = torch.mean(
                torch.real(
                    torch.diagonal(
                        raw_grid,
                        dim1=-2,
                        dim2=-1,
                    ).sum(
                        dim=-1
                    )
                )
            )
            jitter = (
                1e-8
                * torch.clamp(
                    trace_scale
                    / max(
                        n_ports,
                        1,
                    ),
                    min=0.0,
                )
                + 1e-12
            )
            raw_grid = (
                raw_grid
                + jitter
                * eye.unsqueeze(0)
            )
            weight_tensor = (
                torch.as_tensor(
                    weights_grid,
                    dtype=dtype,
                    device=device,
                )
            )
            integrated = (
                torch.einsum(
                    "q,qij->ij",
                    weight_tensor,
                    raw_grid,
                )
            )
            integrated = 0.5 * (
                integrated
                + integrated.conj().T
            )
            chol = (
                torch.linalg.cholesky(
                    integrated
                )
            )
            inverse_chol = (
                torch.linalg.solve_triangular(
                    chol,
                    eye,
                    upper=False,
                )
            )

            raw_query = (
                self.model.raw_matrices(
                    updated,
                    pair,
                    coil_index,
                    torch.as_tensor(
                        local,
                        dtype=dtype,
                        device=device,
                    ),
                )
            )
            raw_query = (
                raw_query
                + jitter
                * eye.unsqueeze(0)
            )
            shape = torch.einsum(
                "ab,qbc,dc->qad",
                inverse_chol,
                raw_query,
                inverse_chol.conj(),
            )
            shape = 0.5 * (
                shape
                + shape.conj().transpose(
                    -1,
                    -2,
                )
            )

        prediction = (
            self.port_artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )
        channel = torch.as_tensor(
            prediction.dissipation_channels[
                coil_index
            ],
            dtype=(
                torch.complex64
                if dtype
                == torch.float32
                else torch.complex128
            ),
            device=device,
        )
        with torch.no_grad():
            root = _psd_sqrt(
                channel
            )
            field = torch.einsum(
                "ab,qbc,dc->qad",
                root,
                shape,
                root.conj(),
            )
            field = 0.5 * (
                field
                + field.conj().transpose(
                    -1,
                    -2,
                )
            )
        out = (
            field.detach()
            .cpu()
            .numpy()
        )
        out[
            ~inside
        ] = 0.0
        return out

    def local_dissipation_matrix(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return self.local_dissipation_matrices(
            scene,
            frequency_hz,
            coil_index,
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
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        matrix = (
            self.local_dissipation_matrix(
                scene,
                frequency_hz,
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

    def integrated_matrix(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
    ) -> np.ndarray:
        arc, xy, weights = (
            self._integration_rule(
                scene,
                coil_index,
            )
        )
        matrices = (
            self.local_dissipation_matrices(
                scene,
                frequency_hz,
                coil_index,
                arc,
                xy,
            )
        )
        return np.einsum(
            "q,qij->ij",
            weights,
            matrices,
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
                "node_dim": (
                    self.model.node_dim
                ),
                "pair_dim": (
                    self.model.pair_dim
                ),
                "hidden_dim": (
                    self.model.hidden_dim
                ),
                "factor_rank": (
                    self.model.factor_rank
                ),
                "depth": (
                    self.model.depth
                ),
                "local_dim": (
                    self.model.local_dim
                ),
            },
            "model_state": (
                self.model.state_dict()
            ),
            "normalizer": (
                self.normalizer.to_dict()
            ),
            "integration": {
                "segments_per_turn": (
                    self.integration_segments_per_turn
                ),
                "min_segments": (
                    self.integration_min_segments
                ),
                "radial_order": (
                    self.integration_radial_order
                ),
                "angular_order": (
                    self.integration_angular_order
                ),
            },
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
        *,
        port_artifact,
        device: str = "cpu",
    ) -> "NeuralSpatialLossArtifact":
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
        integration = payload[
            "integration"
        ]
        return NeuralSpatialLossArtifact(
            port_artifact,
            model,
            SpatialFeatureNormalizer.from_dict(
                payload[
                    "normalizer"
                ]
            ),
            integration_segments_per_turn=int(
                integration[
                    "segments_per_turn"
                ]
            ),
            integration_min_segments=int(
                integration[
                    "min_segments"
                ]
            ),
            integration_radial_order=int(
                integration[
                    "radial_order"
                ]
            ),
            integration_angular_order=int(
                integration[
                    "angular_order"
                ]
            ),
            device=device,
        )


def _training_scene_tensors(
    model,
    normalizer,
    sample,
    device,
):
    node, pair = (
        normalizer.normalize(
            sample.encoded
        )
    )
    dtype = next(
        model.parameters()
    ).dtype
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
    return (
        model.encode_scene(
            node_tensor,
            pair_tensor,
        ),
        pair_tensor,
        dtype,
    )


def _sample_spatial_loss(
    model,
    normalizer,
    sample,
    *,
    device: str,
):
    spatial = sample.spatial_loss
    if (
        spatial is None
        or sample.target_dissipation_channels
        is None
    ):
        raise ValueError(
            "spatial training requires continuous loss truth and channel truth"
        )

    updated, pair, dtype = (
        _training_scene_tensors(
            model,
            normalizer,
            sample,
            device,
        )
    )
    complex_dtype = (
        torch.complex64
        if dtype
        == torch.float32
        else torch.complex128
    )
    total = torch.zeros(
        (),
        dtype=dtype,
        device=device,
    )
    used = 0

    for coil in range(
        len(
            sample.scene.coils
        )
    ):
        mask = (
            spatial.coil_index
            == coil
        )
        if not np.any(
            mask
        ):
            continue
        local, inside = (
            _local_features_numpy(
                sample.scene,
                coil,
                spatial.arc_fraction[
                    mask
                ],
                spatial.xy[
                    mask
                ],
            )
        )
        if not np.all(
            inside
        ):
            raise ValueError(
                "teacher spatial points must lie inside the conductor section"
            )
        weights = torch.as_tensor(
            spatial.weights[
                mask
            ],
            dtype=dtype,
            device=device,
        )
        raw = model.raw_matrices(
            updated,
            pair,
            coil,
            torch.as_tensor(
                local,
                dtype=dtype,
                device=device,
            ),
        )
        shape = (
            _normalize_raw_shapes(
                raw,
                weights,
            )
        )
        channel = torch.as_tensor(
            sample.target_dissipation_channels[
                coil
            ],
            dtype=complex_dtype,
            device=device,
        )
        root = _psd_sqrt(
            channel
        )
        predicted = torch.einsum(
            "ab,qbc,dc->qad",
            root,
            shape,
            root.conj(),
        )
        predicted = 0.5 * (
            predicted
            + predicted.conj().transpose(
                -1,
                -2,
            )
        )
        target = torch.as_tensor(
            spatial.dissipation_matrix[
                mask
            ],
            dtype=complex_dtype,
            device=device,
        )
        difference = torch.abs(
            predicted
            - target
        ) ** 2
        target_energy = torch.abs(
            target
        ) ** 2
        numerator = torch.sum(
            weights[
                :,
                None,
                None,
            ]
            * difference.real
        )
        denominator = torch.sum(
            weights[
                :,
                None,
                None,
            ]
            * target_energy.real
        )
        total = (
            total
            + numerator
            / torch.clamp(
                denominator,
                min=1e-20,
            )
        )
        used += 1

    if used == 0:
        raise ValueError(
            "spatial truth contains no coil samples"
        )
    return (
        total
        / used
    )


def _spatial_validation_error(
    model,
    normalizer,
    samples,
    *,
    device: str,
) -> float:
    samples = tuple(
        samples
    )
    if not samples:
        raise ValueError(
            "validation samples are empty"
        )
    model.eval()
    values = []
    with torch.no_grad():
        for sample in samples:
            value = (
                _sample_spatial_loss(
                    model,
                    normalizer,
                    sample,
                    device=device,
                )
            )
            values.append(
                float(
                    torch.sqrt(
                        torch.clamp(
                            value,
                            min=0.0,
                        )
                    ).cpu()
                )
            )
    return float(
        np.mean(
            values
        )
    )


def train_spatial_loss_surrogate(
    samples,
    *,
    validation_samples=(),
    hidden_dim: int = 64,
    factor_rank: int = 2,
    depth: int = 2,
    epochs: int = 120,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    patience: int = 20,
    validation_interval: int = 1,
    min_improvement: float = 1e-4,
    seed: int = 29,
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
            "at least one spatial teacher sample is required"
        )
    if (
        epochs < 1
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or patience < 1
        or validation_interval < 1
        or min_improvement < 0.0
    ):
        raise ValueError(
            "invalid spatial training configuration"
        )
    for sample in (
        samples
        + validation_samples
    ):
        if (
            sample.spatial_loss
            is None
            or sample.target_dissipation_channels
            is None
        ):
            raise ValueError(
                "all spatial train/validation samples require continuous loss truth"
            )

    torch.manual_seed(
        seed
    )
    np.random.seed(
        seed
    )

    normalizer = (
        SpatialFeatureNormalizer.fit(
            samples
        )
    )
    model = SpatialLossShapeNet(
        node_dim=(
            samples[
                0
            ].encoded.node_features.shape[
                -1
            ]
        ),
        pair_dim=(
            samples[
                0
            ].encoded.pair_features.shape[
                -1
            ]
        ),
        hidden_dim=(
            hidden_dim
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
        accumulated = 0.0
        for index in order:
            optimizer.zero_grad(
                set_to_none=True
            )
            loss = (
                _sample_spatial_loss(
                    model,
                    normalizer,
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
            accumulated += float(
                loss.detach().cpu()
            )

        final_loss = (
            accumulated
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
            score = (
                _spatial_validation_error(
                    model,
                    normalizer,
                    validation_samples,
                    device=device,
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
            "spatial training completed without a selectable model state"
        )
    model.load_state_dict(
        best_state
    )
    model.eval()

    return (
        model,
        normalizer,
        SpatialTrainingReport(
            final_loss=float(
                final_loss
            ),
            epochs=int(
                epochs_run
            ),
            samples=len(
                samples
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
