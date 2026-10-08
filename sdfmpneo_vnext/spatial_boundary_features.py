from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "boundary-aware spatial decoding requires the 'neural' extra"
    ) from exc

from . import hybrid_spatial_neural as _hybrid
from .spatial_neural import _mlp


CONDUCTOR_COORDINATE_FEATURE_DIM = 6


def boundary_aware_conductor_coordinates(scene, coil_index, arc_fraction, xy):
    """Conductor-local coordinates with explicit superellipse boundary radius."""
    coil_index = np.asarray(coil_index, dtype=int)
    arc_fraction = np.asarray(arc_fraction, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if coil_index.ndim != 1:
        raise ValueError("coil_index must be one-dimensional")
    n = len(coil_index)
    if arc_fraction.shape != (n,) or xy.shape != (n, 2):
        raise ValueError("conductor spatial coordinates have incompatible shapes")
    if np.any((coil_index < 0) | (coil_index >= len(scene.coils))):
        raise IndexError("coil_index out of range")
    if np.any((arc_fraction < 0.0) | (arc_fraction > 1.0)):
        raise ValueError("arc_fraction must lie in [0,1]")

    half_width = np.asarray(
        [0.5 * scene.coils[int(index)].geometry.conductor_width for index in coil_index],
        dtype=float,
    )
    half_thickness = np.asarray(
        [
            0.5 * scene.coils[int(index)].geometry.conductor_thickness
            for index in coil_index
        ],
        dtype=float,
    )
    exponent = np.asarray(
        [scene.coils[int(index)].geometry.cross_section_exponent for index in coil_index],
        dtype=float,
    )
    x = xy[:, 0] / half_width
    y = xy[:, 1] / half_thickness
    rho = (np.abs(x) ** exponent + np.abs(y) ** exponent) ** (1.0 / exponent)
    theta = 2.0 * np.pi * arc_fraction
    return np.column_stack((x, y, rho, rho**2, np.sin(theta), np.cos(theta)))


class BoundaryAwareConductorLossShapeNet(nn.Module):
    """PSD conductor decoder using explicit superellipse boundary coordinates."""

    def __init__(
        self,
        hidden_dim: int,
        coil_pair_dim: int,
        *,
        field_hidden_dim: int = 64,
        factor_rank: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        if (
            hidden_dim < 1
            or coil_pair_dim < 1
            or field_hidden_dim < 4
            or factor_rank < 1
            or depth < 1
        ):
            raise ValueError("invalid conductor spatial network dimensions")
        self.hidden_dim = int(hidden_dim)
        self.coil_pair_dim = int(coil_pair_dim)
        self.field_hidden_dim = int(field_hidden_dim)
        self.factor_rank = int(factor_rank)
        self.depth = int(depth)
        self.head = _mlp(
            2 * self.hidden_dim
            + self.coil_pair_dim
            + CONDUCTOR_COORDINATE_FEATURE_DIM,
            self.field_hidden_dim,
            2 * self.factor_rank,
            self.depth,
        )

    def raw_matrices(
        self,
        coil_latent,
        coil_pair_features,
        coil_index,
        coordinate_features,
    ):
        coil_index = torch.as_tensor(
            coil_index,
            dtype=torch.long,
            device=coil_latent.device,
        )
        coordinates = torch.as_tensor(
            coordinate_features,
            dtype=coil_latent.dtype,
            device=coil_latent.device,
        )
        n_points = int(coil_index.numel())
        n_ports = int(coil_latent.shape[0])
        if (
            coil_index.ndim != 1
            or coordinates.shape
            != (n_points, CONDUCTOR_COORDINATE_FEATURE_DIM)
        ):
            raise ValueError("conductor spatial query arrays have incompatible shapes")
        if n_points and (
            torch.any(coil_index < 0) or torch.any(coil_index >= n_ports)
        ):
            raise IndexError("coil_index out of range")

        source_latent = coil_latent[coil_index]
        pair_rows = coil_pair_features[coil_index, :, :]
        source_rows = source_latent[:, None, :].expand(n_points, n_ports, -1)
        port_rows = coil_latent[None, :, :].expand(n_points, n_ports, -1)
        coordinate_rows = coordinates[:, None, :].expand(n_points, n_ports, -1)
        raw = self.head(
            torch.cat(
                (source_rows, port_rows, pair_rows, coordinate_rows),
                dim=-1,
            )
        )
        real = raw[..., : self.factor_rank]
        imag = raw[..., self.factor_rank :]
        complex_dtype = (
            torch.complex64 if coil_latent.dtype == torch.float32 else torch.complex128
        )
        factors = real.to(complex_dtype) + 1j * imag.to(complex_dtype)
        matrices = torch.einsum("qpr,qsr->qps", factors.conj(), factors)
        n = matrices.shape[-1]
        trace_scale = torch.clamp(
            torch.real(torch.diagonal(matrices, dim1=-2, dim2=-1).sum(dim=-1)),
            min=1e-12,
        )
        eye = torch.eye(n, dtype=complex_dtype, device=coil_latent.device)
        matrices = matrices + (
            1e-9 * trace_scale[:, None, None] / max(n, 1)
        ) * eye[None, :, :]
        return 0.5 * (matrices + matrices.conj().transpose(-1, -2))


class BoundaryAwareHybridSpatialLossShapeNet(_hybrid.HybridSpatialLossShapeNet):
    """Tensor FAST spatial model with a boundary-aware conductor decoder."""

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
        super().__init__(
            hidden_dim,
            coil_pair_dim,
            cross_dim,
            field_hidden_dim=field_hidden_dim,
            factor_rank=factor_rank,
            depth=depth,
        )
        self.conductor = BoundaryAwareConductorLossShapeNet(
            self.hidden_dim,
            self.coil_pair_dim,
            field_hidden_dim=self.field_hidden_dim,
            factor_rank=self.factor_rank,
            depth=self.depth,
        )


class BoundaryAwarePreparedHybridSpatialLossField(
    _hybrid.PreparedHybridSpatialLossField
):
    """Prepared tensor field whose conductor queries use schema-2 coordinates."""

    def local_dissipation_matrices(self, coil_index, arc_fraction, xy) -> np.ndarray:
        coil_index = np.asarray(coil_index, dtype=int)
        arc_fraction = np.asarray(arc_fraction, dtype=float)
        xy = np.asarray(xy, dtype=float)
        n = len(coil_index)
        if (
            coil_index.ndim != 1
            or arc_fraction.shape != (n,)
            or xy.shape != (n, 2)
        ):
            raise ValueError("conductor spatial query arrays have incompatible shapes")
        if np.any((coil_index < 0) | (coil_index >= len(self.scene.coils))):
            raise IndexError("coil_index out of range")
        if np.any((arc_fraction < 0.0) | (arc_fraction > 1.0)):
            raise ValueError("arc_fraction must lie in [0,1]")

        inside = np.ones(n, dtype=bool)
        for point in range(n):
            geometry = self.scene.coils[int(coil_index[point])].geometry
            xn = abs(xy[point, 0]) / (0.5 * geometry.conductor_width)
            yn = abs(xy[point, 1]) / (0.5 * geometry.conductor_thickness)
            exponent = geometry.cross_section_exponent
            inside[point] = xn**exponent + yn**exponent <= 1.0 + 1e-12

        coordinates = boundary_aware_conductor_coordinates(
            self.scene,
            coil_index,
            arc_fraction,
            xy,
        )
        self.model.eval()
        with torch.no_grad():
            raw = self.model.conductor.raw_matrices(
                self.coil_latent,
                self.coil_pair_features,
                coil_index,
                coordinates,
            )
            values = _hybrid._apply_by_coil(
                raw,
                coil_index,
                self.conductor_transforms,
            )
            if not np.all(inside):
                mask = torch.as_tensor(~inside, dtype=torch.bool, device=raw.device)
                values[mask] = 0.0
        return values.detach().cpu().numpy()


def install_boundary_aware_conductor_features() -> None:
    """Compatibility no-op; tensor facades install the scoped classes explicitly."""
    return None
