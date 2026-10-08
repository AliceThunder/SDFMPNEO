from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("balanced vNext objectives require the 'neural' extra") from exc

from . import performance as _performance


PORT_TRAINING_CONTRACT = 2
SPATIAL_TRAINING_CONTRACT = 3


def _real_dtype(value):
    return value.real.dtype if torch.is_complex(value) else value.dtype


def _symmetric_relative_mean(predicted, target, *, dims, floor=0.0):
    """Scale-robust squared error, bounded when one side approaches zero."""
    error = torch.mean(torch.abs(predicted - target) ** 2, dim=dims)
    energy = torch.mean(
        torch.abs(predicted) ** 2 + torch.abs(target) ** 2,
        dim=dims,
    )
    dtype = _real_dtype(predicted)
    minimum = max(float(floor), float(torch.finfo(dtype).tiny))
    return error / torch.clamp(energy, min=minimum)


def balanced_port_batch_loss(model, normalizer, batch, *, channel_loss_weight: float):
    """Per-scene relative port loss for broad geometry/material dynamic ranges.

    The network still decodes the exact same passive structured quantities.  Only
    the optimization metric changes: each scene is normalized by its own
    predicted/teacher energy instead of one dataset-wide scalar.  Small values
    therefore cannot explode the loss, while large scenes no longer dominate it.
    """
    resistance, reactance, channels = _performance.forward_structured_batch(
        model,
        *batch["normalized"],
        batch["baseline_resistance"],
        batch["baseline_reactance"],
        resistance_scale=normalizer.resistance_scale,
        reactance_scale=normalizer.reactance_scale,
        dielectric_loss_gate=batch["dielectric_loss_gate"],
        reactance_gate=batch["reactance_gate"],
    )
    target = batch["target_impedance"]

    resistance_floor = (max(float(normalizer.resistance_scale), 1e-30) * 1e-6) ** 2
    reactance_floor = (max(float(normalizer.reactance_scale), 1e-30) * 1e-6) ** 2
    resistance_loss = _symmetric_relative_mean(
        resistance,
        target.real,
        dims=(-2, -1),
        floor=resistance_floor,
    )
    reactance_loss = _symmetric_relative_mean(
        reactance,
        target.imag,
        dims=(-2, -1),
        floor=reactance_floor,
    )
    channel_loss = _symmetric_relative_mean(
        channels,
        batch["target_channels"],
        dims=(-3, -2, -1),
    )
    return torch.mean(
        resistance_loss
        + reactance_loss
        + float(channel_loss_weight) * channel_loss
    )


def balanced_spatial_relative_loss(predicted, target, weights):
    """Symmetric weighted field error for one physical spatial region.

    The historical target-only denominator can become arbitrarily small for a
    weakly lossy package/background and make that negligible region dominate the
    whole optimization.  The symmetric energy denominator remains scale
    invariant but stays finite when either prediction or truth approaches zero.
    """
    weights = torch.as_tensor(
        weights,
        dtype=_real_dtype(predicted),
        device=predicted.device,
    )
    error = torch.sum(
        weights[:, None, None] * torch.abs(predicted - target) ** 2
    )
    energy = torch.sum(
        weights[:, None, None]
        * (torch.abs(predicted) ** 2 + torch.abs(target) ** 2)
    )
    tiny = torch.finfo(_real_dtype(predicted)).tiny
    return error / torch.clamp(energy, min=tiny)


def _balanced_spatial_relative_error_numpy(predicted, target, weights) -> float:
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
    if energy <= np.finfo(float).tiny:
        return 0.0 if error <= np.finfo(float).tiny else 1.0
    return error / energy


def balanced_spatial_end_to_end_error(artifact, sample) -> float:
    """Inference-path spatial error using the same robust regional metric."""
    prepared = artifact.prepare(sample.scene, sample.frequency_hz)

    conductor = sample.conductor_spatial_loss
    conductor_error = _balanced_spatial_relative_error_numpy(
        prepared.local_dissipation_matrices(
            conductor.coil_index,
            conductor.arc_fraction,
            conductor.xy,
        ),
        conductor.dissipation_matrix,
        conductor.weights,
    )

    package = sample.package_spatial_loss
    package_error = _balanced_spatial_relative_error_numpy(
        prepared.package_local_dissipation_matrices(
            package.package_index,
            package.local_position,
        ),
        package.dissipation_matrix,
        package.weights,
    )

    background_error = 0.0
    if sample.background_spatial_loss is not None:
        background = sample.background_spatial_loss
        root_pose = sample.scene.coils[0].geometry.pose
        world = root_pose.apply(background.root_local_position)
        background_error = _balanced_spatial_relative_error_numpy(
            prepared.background_dissipation_matrices(world),
            background.dissipation_matrix,
            background.weights,
        )

    return float(conductor_error + package_error + background_error)


# The accelerated port trainer resolves this global at call time.  Installing
# the balanced objective here keeps the public accelerated API consistent with
# the controlled workflow without duplicating the training loop.
_performance._batch_loss = balanced_port_batch_loss
