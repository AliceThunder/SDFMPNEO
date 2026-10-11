from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("generation-2 objectives require the 'neural' extra") from exc


GENERATION2_PORT_TRAINING_CONTRACT = 1
GENERATION2_SPATIAL_TRAINING_CONTRACT = 1


def _real_dtype(value):
    return value.real.dtype if torch.is_complex(value) else value.dtype


def symmetric_relative_scene_loss(predicted, target, *, dims, floor: float = 0.0):
    """Per-scene symmetric squared relative error with a physical zero floor."""
    error = torch.mean(torch.abs(predicted - target) ** 2, dim=dims)
    energy = torch.mean(
        torch.abs(predicted) ** 2 + torch.abs(target) ** 2,
        dim=dims,
    )
    dtype = _real_dtype(predicted)
    minimum = max(float(floor), float(torch.finfo(dtype).tiny))
    return error / torch.clamp(energy, min=minimum)


@dataclass(frozen=True)
class Generation2PortLosses:
    composite: torch.Tensor
    resistance: torch.Tensor
    reactance: torch.Tensor
    channels: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.composite.detach().cpu()),
            "resistance_loss": float(self.resistance.detach().cpu()),
            "reactance_loss": float(self.reactance.detach().cpu()),
            "channel_loss": float(self.channels.detach().cpu()),
        }


def generation2_port_losses(
    resistance,
    reactance,
    channels,
    target_impedance,
    target_channels,
    *,
    resistance_weight: float = 1.0,
    reactance_weight: float = 1.0,
    channel_weight: float = 2.0,
    resistance_floor: float = 0.0,
    reactance_floor: float = 0.0,
) -> Generation2PortLosses:
    """Return separate Port observables and their optimization composite."""
    target_impedance = torch.as_tensor(
        target_impedance,
        dtype=(
            torch.complex64
            if resistance.dtype == torch.float32
            else torch.complex128
        ),
        device=resistance.device,
    )
    target_channels = torch.as_tensor(
        target_channels,
        dtype=target_impedance.dtype,
        device=resistance.device,
    )
    resistance_per_scene = symmetric_relative_scene_loss(
        resistance,
        target_impedance.real,
        dims=(-2, -1),
        floor=resistance_floor,
    )
    reactance_per_scene = symmetric_relative_scene_loss(
        reactance,
        target_impedance.imag,
        dims=(-2, -1),
        floor=reactance_floor,
    )
    channel_per_scene = symmetric_relative_scene_loss(
        channels,
        target_channels,
        dims=(-3, -2, -1),
    )
    resistance_loss = torch.mean(resistance_per_scene)
    reactance_loss = torch.mean(reactance_per_scene)
    channel_loss = torch.mean(channel_per_scene)
    composite = (
        float(resistance_weight) * resistance_loss
        + float(reactance_weight) * reactance_loss
        + float(channel_weight) * channel_loss
    )
    return Generation2PortLosses(
        composite,
        resistance_loss,
        reactance_loss,
        channel_loss,
    )


def weighted_spatial_error_energy(predicted, target, weights):
    """Physical weighted local error and symmetric energy for one region."""
    weights = torch.as_tensor(
        weights,
        dtype=_real_dtype(predicted),
        device=predicted.device,
    )
    target = torch.as_tensor(target, dtype=predicted.dtype, device=predicted.device)
    error = torch.sum(
        weights[:, None, None] * torch.abs(predicted - target) ** 2
    )
    energy = torch.sum(
        weights[:, None, None]
        * (torch.abs(predicted) ** 2 + torch.abs(target) ** 2)
    )
    return error, energy


def spatial_relative_from_error_energy(error, energy):
    tiny = torch.finfo(_real_dtype(error)).tiny
    return error / torch.clamp(energy, min=tiny)


def numpy_weighted_spatial_error_energy(predicted, target, weights):
    predicted = np.asarray(predicted, dtype=complex)
    target = np.asarray(target, dtype=complex)
    weights = np.asarray(weights, dtype=float)
    error = float(
        np.sum(weights[:, None, None] * np.abs(predicted - target) ** 2)
    )
    energy = float(
        np.sum(
            weights[:, None, None]
            * (np.abs(predicted) ** 2 + np.abs(target) ** 2)
        )
    )
    return error, energy


def numpy_relative_from_error_energy(error: float, energy: float) -> float:
    tiny = np.finfo(float).tiny
    if energy <= tiny:
        return 0.0 if error <= tiny else 1.0
    return float(error / energy)


__all__ = [
    "GENERATION2_PORT_TRAINING_CONTRACT",
    "GENERATION2_SPATIAL_TRAINING_CONTRACT",
    "Generation2PortLosses",
    "symmetric_relative_scene_loss",
    "generation2_port_losses",
    "weighted_spatial_error_energy",
    "spatial_relative_from_error_energy",
    "numpy_weighted_spatial_error_energy",
    "numpy_relative_from_error_energy",
]
