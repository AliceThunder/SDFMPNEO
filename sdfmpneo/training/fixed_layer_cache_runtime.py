"""Dense compiled programs for layerwise fixed-response training.

A response layer is linear in its trainable amplitude block while all earlier
layers and feature directions are frozen.  The expensive symbolic ExpPoly work
therefore belongs to a *compile* step, not to every LM iteration/candidate.

This module compiles one contiguous numeric program per active layer and working
collocation set.  Jacobian assembly and exact candidate evaluation consume the
same arrays.  Accepted amplitude updates do not invalidate the program because
the signature deliberately excludes the active amplitude block.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import numpy as np

from .fixed_physics_runtime import (
    _prepare_operating_contexts,
    _prepare_working_set,
    physics_vector_field,
)
from .parallel_runtime import _ordered_map
from .research_linearization import _balanced_physics_subset


@dataclass
class CompiledLayerProgram:
    layer: int
    ids: np.ndarray
    targets: np.ndarray
    signature: bytes
    points: np.ndarray
    point_lookup: dict[bytes, int]
    base_a: np.ndarray
    base_da: np.ndarray
    ja_rows: np.ndarray
    jda_rows: np.ndarray

    @property
    def n_parameters(self) -> int:
        return int(self.ids.size)

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    def indices_for(self, points) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        out = np.empty(len(values), dtype=int)
        for i, row in enumerate(values):
            key = np.ascontiguousarray(row, dtype=np.float64).tobytes()
            try:
                out[i] = self.point_lookup[key]
            except KeyError as exc:
                raise KeyError("collocation point is not in the compiled layer program") from exc
        return out

    def reconstruct(self, network, points):
        """Return exact ``a`` and ``da/dt`` for the current amplitude vector."""
        index = self.indices_for(points)
        theta = np.asarray(network.parameters, dtype=float)[self.ids]
        a = np.asarray(self.base_a[index], dtype=float).copy()
        da = np.asarray(self.base_da[index], dtype=float).copy()
        if self.targets.size and self.ids.size:
            a[:, self.targets] += np.einsum(
                "ntp,p->nt", self.ja_rows[index], theta, optimize=True
            )
            da[:, self.targets] += np.einsum(
                "ntp,p->nt", self.jda_rows[index], theta, optimize=True
            )
        return a, da

    def state_parameter_jacobian(self, points):
        """Materialize the sparse-in-modes current-layer state Jacobian."""
        index = self.indices_for(points)
        count = len(index)
        n_modes = self.base_a.shape[1]
        ja = np.zeros((count, n_modes, self.n_parameters), dtype=float)
        jda = np.zeros_like(ja)
        if self.targets.size and self.ids.size:
            ja[:, self.targets, :] = self.ja_rows[index]
            jda[:, self.targets, :] = self.jda_rows[index]
        return ja, jda


_PROGRAM_CACHE: dict[tuple[int, bytes, bytes], CompiledLayerProgram] = {}
_ACTIVE_FEATURE_SIGNATURE: bytes | None = None


def clear_layer_program_cache() -> None:
    global _ACTIVE_FEATURE_SIGNATURE
    _PROGRAM_CACHE.clear()
    _ACTIVE_FEATURE_SIGNATURE = None


def _affine_supported(network, layer: int) -> bool:
    layer = int(layer)
    if layer <= 0 or layer >= network.depth:
        return False
    ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
    if ids.size == 0 or ids.size > 4096:
        return False
    return not any(
        not network._layer_is_zero(index)
        for index in range(layer + 1, network.depth)
    )


def layer_feature_signature(network, layer: int, ids=None) -> bytes:
    """Hash every quantity that fixes a layer basis, excluding its amplitudes."""
    layer = int(layer)
    if ids is None:
        ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
    else:
        ids = np.asarray(ids, dtype=int)
    theta = np.asarray(network.parameters, dtype=np.float64).copy()
    theta[ids] = 0.0
    digest = hashlib.blake2b(digest_size=20)
    digest.update(theta.tobytes())
    digest.update(np.asarray(network.lambdas, dtype=np.float64).tobytes())
    digest.update(np.asarray(network.input_center, dtype=np.float64).tobytes())
    digest.update(np.asarray(network.input_scale, dtype=np.float64).tobytes())
    digest.update(np.asarray(network.layer_widths[: layer + 1], dtype=np.int64).tobytes())
    for targets in network.layer_targets[: layer + 1]:
        digest.update(np.asarray(targets, dtype=np.int64).tobytes())
    return digest.digest()


def _point_set_digest(points) -> bytes:
    values = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.digest()


def _activate_feature_signature(signature: bytes) -> None:
    """Keep only programs for the currently active frozen feature bank."""
    global _ACTIVE_FEATURE_SIGNATURE
    if _ACTIVE_FEATURE_SIGNATURE != signature:
        _PROGRAM_CACHE.clear()
        _ACTIVE_FEATURE_SIGNATURE = signature


def _compile_program(network, points, layer, ids, signature, *, monitor=None, work_label=None):
    from . import research_helpers as rh

    values = np.ascontiguousarray(np.asarray(points, dtype=float))
    targets = np.unique(np.asarray(network.layer_targets[layer], dtype=int))
    theta = np.asarray(network.parameters, dtype=float)[ids]
    total = len(values)
    label = work_label or f"physics_layer_{layer + 1}_basis"

    def one(point):
        initial = point[: network.n_modes]
        operating = point[network.n_modes:-1]
        time = float(point[-1])
        a, da, ja, jda, actual_ids = network.evaluate_layer_amplitude_jacobian(
            time, a0=initial, operating=operating, layer=layer
        )
        actual_ids = np.asarray(actual_ids, dtype=int)
        if not np.array_equal(actual_ids, ids):
            raise RuntimeError("layer amplitude block changed during program compilation")
        ja_rows = np.ascontiguousarray(np.asarray(ja, dtype=float)[targets])
        jda_rows = np.ascontiguousarray(np.asarray(jda, dtype=float)[targets])
        base_a = np.asarray(a, dtype=float).copy()
        base_da = np.asarray(da, dtype=float).copy()
        if ids.size and targets.size:
            base_a[targets] -= ja_rows @ theta
            base_da[targets] -= jda_rows @ theta
        return base_a, base_da, ja_rows, jda_rows

    progress = None
    if monitor is not None:
        progress = lambda completed, count: rh._work(monitor, label, completed, count)
    rows = _ordered_map(one, values, monitor=monitor, progress=progress)
    if rows:
        base_a = np.stack([row[0] for row in rows])
        base_da = np.stack([row[1] for row in rows])
        ja_rows = np.stack([row[2] for row in rows])
        jda_rows = np.stack([row[3] for row in rows])
    else:
        n = network.n_modes
        p = len(ids)
        base_a = np.empty((0, n), dtype=float)
        base_da = np.empty((0, n), dtype=float)
        ja_rows = np.empty((0, len(targets), p), dtype=float)
        jda_rows = np.empty_like(ja_rows)
    lookup = {
        np.ascontiguousarray(row, dtype=np.float64).tobytes(): i
        for i, row in enumerate(values)
    }
    return CompiledLayerProgram(
        layer=int(layer),
        ids=np.asarray(ids, dtype=int),
        targets=targets,
        signature=signature,
        points=values.copy(),
        point_lookup=lookup,
        base_a=base_a,
        base_da=base_da,
        ja_rows=ja_rows,
        jda_rows=jda_rows,
    )


def layer_program(network, points, layer, *, monitor=None, work_label=None):
    """Return a dense exact program, compiling it once if necessary."""
    layer = int(layer)
    values = np.asarray(points, dtype=float)
    if not _affine_supported(network, layer) or len(values) == 0:
        return None
    ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
    signature = layer_feature_signature(network, layer, ids)
    _activate_feature_signature(signature)
    key = (layer, signature, _point_set_digest(values))
    program = _PROGRAM_CACHE.get(key)
    if program is None:
        program = _compile_program(
            network,
            values,
            layer,
            ids,
            signature,
            monitor=monitor,
            work_label=work_label,
        )
        _PROGRAM_CACHE[key] = program
    return program


def prewarm_layer_basis(network, points, layer, *, monitor=None, work_label=None) -> bool:
    return layer_program(
        network, points, layer, monitor=monitor, work_label=work_label
    ) is not None


def evaluate_layer_physics(
    network,
    field,
    points,
    layer,
    *,
    monitor=None,
    work_label="physics_layer_jacobian",
):
    """Exact residual plus inexact-GN Jacobian using one compiled layer program."""
    from . import research_helpers as rh

    points = np.asarray(points, dtype=float)
    layer = int(layer)
    _prepare_working_set(field, points)
    n = network.n_modes
    if len(points):
        _prepare_operating_contexts(field, points[:, n:-1])

    program = layer_program(network, points, layer, monitor=monitor)
    ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
    if program is None:
        from .fixed_physics_runtime import evaluate_layer_physics as fallback
        return fallback(
            network,
            field,
            points,
            layer,
            monitor=monitor,
            work_label=work_label,
        )

    a_batch, da_batch = program.reconstruct(network, points)
    ja_batch, jda_batch = program.state_parameter_jacobian(points)

    def one(index):
        point = points[index]
        operating = point[n:-1]
        a = a_batch[index]
        da = da_batch[index]
        ja = ja_batch[index]
        jda = jda_batch[index]
        F = physics_vector_field(field, a, operating)
        JFJa = rh._gn_apply_field_jacobian(field, a, operating, ja)
        return SimpleNamespace(
            residual=np.asarray(da - F, dtype=float),
            parameter_jacobian=np.asarray(jda - JFJa, dtype=float),
            parameter_indices=ids,
        )

    progress = None
    if monitor is not None:
        progress = lambda completed, total: rh._work(monitor, work_label, completed, total)
    records = _ordered_map(one, range(len(points)), monitor=monitor, progress=progress)
    return records, ids


def layer_linearization(
    network,
    field,
    points,
    evaluated,
    layer,
    config,
    monitor=None,
):
    """Compile the active layer once, then select a bounded Jacobian subset."""
    from . import research_helpers as rh

    points = np.asarray(points, dtype=float)
    layer = int(layer)
    # Compile the *whole* training set before selecting the Jacobian subset so
    # candidate acceptance and every later iteration reuse the exact same arrays.
    prewarm_layer_basis(network, points, layer, monitor=monitor)

    budget = min(int(config.jacobian_point_budget), len(points))
    subset = _balanced_physics_subset(
        points, evaluated.physics_norms, budget, config
    )
    program = layer_program(network, points, layer)
    if program is not None:
        # Evaluate a subset through the full-set program; no symbolic rebuild.
        n = network.n_modes
        _prepare_working_set(field, points)
        if len(points):
            _prepare_operating_contexts(field, points[:, n:-1])
        a_batch, da_batch = program.reconstruct(network, subset)
        ja_batch, jda_batch = program.state_parameter_jacobian(subset)
        ids = program.ids

        def one(index):
            point = subset[index]
            operating = point[n:-1]
            a = a_batch[index]
            da = da_batch[index]
            ja = ja_batch[index]
            jda = jda_batch[index]
            F = physics_vector_field(field, a, operating)
            JFJa = rh._gn_apply_field_jacobian(field, a, operating, ja)
            return SimpleNamespace(
                residual=np.asarray(da - F, dtype=float),
                parameter_jacobian=np.asarray(jda - JFJa, dtype=float),
                parameter_indices=ids,
            )

        progress = None
        if monitor is not None:
            progress = lambda completed, total: rh._work(
                monitor, f"physics_layer_{layer + 1}_jacobian", completed, total
            )
        records = _ordered_map(
            one, range(len(subset)), monitor=monitor, progress=progress
        )
    else:
        records, ids = evaluate_layer_physics(
            network,
            field,
            subset,
            layer,
            monitor=monitor,
            work_label=f"physics_layer_{layer + 1}_jacobian",
        )
    result = rh._combined_metrics(records, [])
    result.physics_subset = subset
    result.parameter_indices = np.asarray(ids, dtype=int)
    return result


def install_layer_cache_acceleration() -> None:
    from . import research_multilayer as rm

    if getattr(rm, "_sdfmpneo_layer_cache_acceleration", False):
        return
    rm.evaluate_layer_physics = evaluate_layer_physics
    rm.layer_linearization = layer_linearization
    rm._sdfmpneo_layer_cache_acceleration = True


__all__ = [
    "CompiledLayerProgram",
    "clear_layer_program_cache",
    "evaluate_layer_physics",
    "install_layer_cache_acceleration",
    "layer_feature_signature",
    "layer_linearization",
    "layer_program",
    "prewarm_layer_basis",
]
