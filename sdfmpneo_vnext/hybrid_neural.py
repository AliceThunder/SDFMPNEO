from __future__ import annotations

"""Public hybrid-neural API with empty-package graph support.

The large implementation remains in :mod:`_hybrid_neural_core`.  This facade
only specializes the two shared primitives that need a mathematically neutral
empty-set definition so tensor-background-only scenes can use the same model
without introducing a fictitious package node.
"""

import numpy as np

from . import _hybrid_neural_core as _core
from ._hybrid_neural_core import *  # noqa: F401,F403

# Private gates are intentionally used by tensor trainers and therefore remain
# part of the internal cross-module contract.
_dielectric_loss_gate = _core._dielectric_loss_gate
_reactance_gate = _core._reactance_gate


def _feature_stack(samples, getter):
    arrays = []
    width = None
    for sample in samples:
        value = np.asarray(getter(sample), dtype=float)
        if value.ndim < 1:
            raise ValueError("hybrid feature arrays must have at least one axis")
        if width is None:
            width = int(value.shape[-1])
        elif int(value.shape[-1]) != width:
            raise ValueError("hybrid feature widths must be consistent")
        arrays.append(value.reshape(-1, int(value.shape[-1])))
    if width is None:
        raise ValueError("at least one hybrid sample is required")
    return np.concatenate(arrays, axis=0)


def _feature_stats(values, floor):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("hybrid normalization expects rank-2 flattened features")
    if values.shape[0] == 0:
        # An absent entity family carries no data.  Zero mean / unit scale is
        # the unique neutral affine normalization and avoids NaN statistics.
        return (
            np.zeros(values.shape[1], dtype=float),
            np.ones(values.shape[1], dtype=float),
        )
    return (
        values.mean(axis=0),
        np.maximum(values.std(axis=0), float(floor)),
    )


class HybridNormalizer(_core.HybridNormalizer):
    """Hybrid normalizer with a neutral definition for empty entity families."""

    @staticmethod
    def fit(samples, *, floor: float = 1e-8) -> "HybridNormalizer":
        samples = tuple(samples)
        if not samples:
            raise ValueError("at least one hybrid sample is required")
        if not np.isfinite(floor) or float(floor) <= 0.0:
            raise ValueError("normalization floor must be positive and finite")

        coil_node = _feature_stack(
            samples, lambda sample: sample.encoded.coil.node_features
        )
        coil_pair = _feature_stack(
            samples, lambda sample: sample.encoded.coil.pair_features
        )
        package = _feature_stack(
            samples, lambda sample: sample.encoded.package_features
        )
        coil_package = _feature_stack(
            samples, lambda sample: sample.encoded.coil_package_features
        )
        package_pair = _feature_stack(
            samples, lambda sample: sample.encoded.package_pair_features
        )

        coil_node_mean, coil_node_scale = _feature_stats(coil_node, floor)
        coil_pair_mean, coil_pair_scale = _feature_stats(coil_pair, floor)
        package_mean, package_scale = _feature_stats(package, floor)
        coil_package_mean, coil_package_scale = _feature_stats(coil_package, floor)
        package_pair_mean, package_pair_scale = _feature_stats(package_pair, floor)

        resistance_values = []
        reactance_values = []
        for sample in samples:
            target = np.asarray(sample.target_impedance, dtype=complex)
            resistance_values.append(
                (target.real - np.asarray(sample.baseline_resistance, dtype=float)).ravel()
            )
            reactance_values.append(
                (target.imag - np.asarray(sample.baseline_reactance, dtype=float)).ravel()
            )
        resistance_values = np.concatenate(resistance_values)
        reactance_values = np.concatenate(reactance_values)

        resistance_scale = max(
            float(np.sqrt(np.mean(resistance_values**2))), float(floor)
        )
        reactance_scale = max(
            float(np.sqrt(np.mean(reactance_values**2))), float(floor)
        )
        return HybridNormalizer(
            coil_node_mean,
            coil_node_scale,
            coil_pair_mean,
            coil_pair_scale,
            package_mean,
            package_scale,
            coil_package_mean,
            coil_package_scale,
            package_pair_mean,
            package_pair_scale,
            resistance_scale,
            reactance_scale,
        )

    @staticmethod
    def from_dict(data):
        base = _core.HybridNormalizer.from_dict(data)
        return HybridNormalizer(
            base.coil_node_mean,
            base.coil_node_scale,
            base.coil_pair_mean,
            base.coil_pair_scale,
            base.package_mean,
            base.package_scale,
            base.coil_package_mean,
            base.coil_package_scale,
            base.package_pair_mean,
            base.package_pair_scale,
            base.resistance_scale,
            base.reactance_scale,
        )


class HybridPhysicsFactoredResidualNet(_core.HybridPhysicsFactoredResidualNet):
    """Hybrid graph network whose package set may be empty."""

    def _latent(
        self,
        coil_features,
        coil_pair_features,
        package_features,
        coil_package_features,
        package_pair_features,
    ):
        if int(package_features.shape[0]) != 0:
            return super()._latent(
                coil_features,
                coil_pair_features,
                package_features,
                coil_package_features,
                package_pair_features,
            )

        torch = _core.torch
        math = _core.math
        coil0 = self.coil_encoder(coil_features)
        package = self.package_encoder(package_features)
        n_coils = int(coil0.shape[0])

        coil_pair_messages = []
        for i in range(n_coils):
            messages = []
            for j in range(n_coils):
                if i == j:
                    continue
                messages.append(
                    self.coil_pair_encoder(
                        torch.cat(
                            (coil0[i], coil0[j], coil_pair_features[i, j]),
                            dim=-1,
                        )
                    )
                )
            if messages:
                pair_message = torch.stack(messages, dim=0).sum(dim=0) / math.sqrt(
                    len(messages)
                )
            else:
                pair_message = torch.zeros_like(coil0[i])
            coil_pair_messages.append(pair_message)

        coil = torch.stack(
            [
                self.coil_update(
                    torch.cat(
                        (
                            coil0[index],
                            coil_pair_messages[index],
                            torch.zeros_like(coil0[index]),
                        ),
                        dim=-1,
                    )
                )
                for index in range(n_coils)
            ],
            dim=0,
        )
        return coil, package


# Factory functions and class methods defined in the core resolve these globals
# at call time.  Rebinding them keeps scalar/package training behavior identical
# while ensuring every public construction path receives the empty-safe classes.
_core.HybridNormalizer = HybridNormalizer
_core.HybridPhysicsFactoredResidualNet = HybridPhysicsFactoredResidualNet
