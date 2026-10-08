from __future__ import annotations

from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.tensor_artifact_io requires the 'neural' extra"
    ) from exc

from .device import resolve_torch_device
from .hybrid_neural import HybridNormalizer, HybridPhysicsFactoredResidualNet
from .spatial_boundary_features import BoundaryAwareHybridSpatialLossShapeNet
from .tensor_neural import (
    TENSOR_HYBRID_ARTIFACT_SCHEMA,
    TensorHybridNeuralResidualArtifact,
)
from .tensor_spatial_neural import (
    TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA,
    TensorHybridSpatialLossArtifact,
    tensor_port_fingerprint,
)


def _state_dtype(state_dict):
    for value in state_dict.values():
        if torch.is_floating_point(value):
            return value.dtype
    return torch.get_default_dtype()


def load_tensor_port_artifact(path, *, device: str = "auto"):
    """Load a tensor port artifact without silently changing weight dtype."""
    resolved_device = resolve_torch_device(device)
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    if int(payload.get("schema", -1)) != TENSOR_HYBRID_ARTIFACT_SCHEMA:
        raise ValueError("unsupported tensor hybrid artifact schema")
    metadata = payload["model"]
    state = metadata["state_dict"]
    dtype = _state_dtype(state)
    if resolved_device == "mps" and dtype == torch.float64:
        raise ValueError(
            "this tensor artifact uses float64 and cannot be loaded on MPS; "
            "use cpu or retrain/save it in float32"
        )
    model = HybridPhysicsFactoredResidualNet(
        coil_dim=int(metadata["coil_dim"]),
        coil_pair_dim=int(metadata["coil_pair_dim"]),
        package_dim=int(metadata["package_dim"]),
        cross_dim=int(metadata["cross_dim"]),
        package_pair_dim=int(metadata["package_pair_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        factor_rank=int(metadata["factor_rank"]),
        depth=int(metadata["depth"]),
    ).to(dtype=dtype)
    model.load_state_dict(state)
    return TensorHybridNeuralResidualArtifact(
        model,
        HybridNormalizer.from_dict(payload["normalizer"]),
        baseline_segments=int(payload["baseline_segments"]),
        material_domain=payload["material_domain"],
        geometry_domain=payload.get("geometry_domain"),
        device=resolved_device,
    )


def load_tensor_spatial_artifact(
    path,
    port_artifact,
    *,
    device: str = "auto",
):
    """Load a tensor spatial artifact using the exact port-model dtype."""
    resolved_device = resolve_torch_device(device)
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    if int(payload.get("schema", -1)) != TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA:
        raise ValueError("unsupported tensor hybrid spatial artifact schema")
    if payload.get("port_fingerprint") != tensor_port_fingerprint(port_artifact):
        raise ValueError("tensor spatial port fingerprint mismatch")
    dtype = next(port_artifact.model.parameters()).dtype
    if resolved_device == "mps" and dtype == torch.float64:
        raise ValueError(
            "this tensor spatial artifact uses float64 and cannot be loaded on MPS"
        )
    model = BoundaryAwareHybridSpatialLossShapeNet(**payload["model_config"]).to(
        dtype=dtype
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return TensorHybridSpatialLossArtifact(
        port_artifact,
        model,
        conductor_longitudinal_points=int(payload["conductor_longitudinal_points"]),
        conductor_radial_order=int(payload["conductor_radial_order"]),
        conductor_angular_order=int(payload["conductor_angular_order"]),
        package_axial_order=int(payload["package_axial_order"]),
        package_radial_order=int(payload["package_radial_order"]),
        package_azimuthal_order=int(payload["package_azimuthal_order"]),
        background_segments_per_turn=int(payload.get("background_segments_per_turn", 16)),
        background_radial_order=int(payload.get("background_radial_order", 12)),
        background_angular_order=int(payload.get("background_angular_order", 48)),
        device=resolved_device,
    )
