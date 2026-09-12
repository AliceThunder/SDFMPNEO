from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .fixed_physics_runtime import (
    _prepare_operating_contexts,
    _prepare_working_set,
    physics_vector_field,
)
from .parallel_runtime import _ordered_map
from .research_linearization import _balanced_physics_subset


def _affine_supported(network, layer: int) -> bool:
    """Return whether the active layer is exactly affine in its trainable block."""
    layer = int(layer)
    if layer <= 0 or layer >= network.depth:
        return False
    ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
    if ids.size == 0 or ids.size > 2048:
        return False
    # A later active layer would consume this layer as an input, so changing the
    # current amplitudes would no longer be affine in the final network output.
    return not any(
        not network._layer_is_zero(index)
        for index in range(layer + 1, network.depth)
    )


def _affine_context(network, layer: int):
    from . import fixed_trial_runtime as ftr

    ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
    signature = ftr._affine_signature(network, int(layer), ids)
    ftr._activate_affine_signature(signature)
    return ftr, ids, signature


def _cache_key(signature: bytes, point) -> tuple[bytes, bytes]:
    values = np.ascontiguousarray(np.asarray(point, dtype=np.float64))
    return signature, values.tobytes()


def prewarm_layer_basis(
    network,
    points,
    layer,
    *,
    monitor=None,
    work_label=None,
) -> bool:
    """Compile the exact current-layer response basis once for all collocation points."""
    from . import research_helpers as rh

    points = np.asarray(points, dtype=float)
    layer = int(layer)
    if not _affine_supported(network, layer) or len(points) == 0:
        return False
    ftr, ids, signature = _affine_context(network, layer)
    missing = [
        row for row in points
        if _cache_key(signature, row) not in ftr._AFFINE_POINT_CACHE
    ]
    if not missing:
        return True

    label = work_label or f"physics_layer_{layer + 1}_basis"

    def one(row):
        ftr._affine_basis_for_point(network, row, layer, ids, signature)
        return None

    progress = None
    if monitor is not None:
        progress = lambda completed, total: rh._work(
            monitor, label, completed, total
        )
    _ordered_map(one, missing, monitor=monitor, progress=progress)
    return True


def _affine_state_jacobian(network, point, layer, ids, signature):
    from . import fixed_trial_runtime as ftr

    targets, base_a, base_da, ja_rows, jda_rows = ftr._affine_basis_for_point(
        network, point, int(layer), ids, signature
    )
    theta = np.asarray(network.parameters, dtype=float)[ids]
    a = np.asarray(base_a, dtype=float).copy()
    da = np.asarray(base_da, dtype=float).copy()
    a[targets] += ja_rows @ theta
    da[targets] += jda_rows @ theta

    ja = np.zeros((network.n_modes, len(ids)), dtype=float)
    jda = np.zeros_like(ja)
    ja[targets] = ja_rows
    jda[targets] = jda_rows
    return a, da, ja, jda


def evaluate_layer_physics(
    network,
    field,
    points,
    layer,
    *,
    monitor=None,
    work_label="physics_layer_jacobian",
):
    """Exact layer residual/Jacobian with a persistent analytic-basis cache."""
    from . import research_helpers as rh

    points = np.asarray(points, dtype=float)
    layer = int(layer)
    _prepare_working_set(field, points)
    n = network.n_modes
    if len(points):
        _prepare_operating_contexts(field, points[:, n:-1])

    affine = _affine_supported(network, layer)
    if affine:
        _, ids, signature = _affine_context(network, layer)
    else:
        ids = np.asarray(network.layer_amplitude_parameter_indices(layer), dtype=int)
        signature = None

    def one(point):
        initial, operating, time = point[:n], point[n:-1], float(point[-1])
        if affine:
            a, da, ja, jda = _affine_state_jacobian(
                network, point, layer, ids, signature
            )
            local_ids = ids
        else:
            a, da, ja, jda, local_ids = network.evaluate_layer_amplitude_jacobian(
                time, a0=initial, operating=operating, layer=layer
            )
            local_ids = np.asarray(local_ids, dtype=int)
        F = physics_vector_field(field, a, operating)
        JFJa = rh._gn_apply_field_jacobian(field, a, operating, ja)
        return SimpleNamespace(
            residual=np.asarray(da - F, dtype=float),
            parameter_jacobian=np.asarray(jda - JFJa, dtype=float),
            parameter_indices=np.asarray(local_ids, dtype=int),
        )

    progress = None
    if monitor is not None:
        progress = lambda completed, total: rh._work(
            monitor, work_label, completed, total
        )
    records = _ordered_map(one, points, monitor=monitor, progress=progress)
    for record in records:
        if not np.array_equal(ids, record.parameter_indices):
            raise RuntimeError(
                "layer amplitude parameter block changed across collocation points"
            )
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
    """Build one LM linearization while compiling the deep basis at most once."""
    from . import research_helpers as rh

    points = np.asarray(points, dtype=float)
    layer = int(layer)
    # Compile all physics collocation points once when a deep layer becomes active.
    # The same cache is consumed by both Jacobian assembly and exact candidate
    # acceptance, so accepted amplitude updates do not trigger symbolic rebuilds.
    prewarm_layer_basis(network, points, layer, monitor=monitor)

    budget = min(int(config.jacobian_point_budget), len(points))
    subset = _balanced_physics_subset(
        points, evaluated.physics_norms, budget, config
    )
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
    result.parameter_indices = ids
    return result


def install_layer_cache_acceleration() -> None:
    from . import research_multilayer as rm

    if getattr(rm, "_sdfmpneo_layer_cache_acceleration", False):
        return
    rm.evaluate_layer_physics = evaluate_layer_physics
    rm.layer_linearization = layer_linearization
    rm._sdfmpneo_layer_cache_acceleration = True


__all__ = [
    "evaluate_layer_physics",
    "install_layer_cache_acceleration",
    "layer_linearization",
    "prewarm_layer_basis",
]
