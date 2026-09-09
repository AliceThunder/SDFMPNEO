from __future__ import annotations

import numpy as np

from sdfmpneo.analytic.long_time import realization_observation_action
from sdfmpneo.analytic.parametric_realization import compile_parametric_nodes


def sparse_weight_value_jacobian_observed(graph, point):
    """Exact sparse-ancestry value/Jacobian using cached output observations.

    The derivative recursion is identical to the late-stage implementation.  It
    differs only at the final semigroup contraction: c^T exp(A t) is cached as a
    row and applied directly to the node value/derivative right-hand sides.
    """
    n_modes = graph.n_modes
    n_weights = len(graph.response_nodes)
    initial = np.asarray(point[:n_modes], dtype=float)
    operating = np.asarray(point[n_modes:-1], dtype=float)
    time = float(point[-1])
    compiled = compile_parametric_nodes(graph, a0=initial, operating=operating)

    sparse_db: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in graph.initial_names + graph.operating_names
    }
    for index, node in enumerate(graph.response_nodes):
        source_b = np.array([complex(node.weight)], dtype=complex)
        source_d = {index: np.ones(1, dtype=complex)}
        for parent in node.parents:
            right_b = compiled.node_realizations[parent].b
            right_d = sparse_db[parent]
            keys = source_d.keys() | right_d.keys()
            next_d: dict[int, np.ndarray] = {}
            for key in keys:
                left = source_d.get(key)
                right = right_d.get(key)
                value = np.zeros(source_b.size * right_b.size, dtype=complex)
                if left is not None:
                    value += np.kron(left, right_b)
                if right is not None:
                    value += np.kron(source_b, right)
                next_d[key] = value
            source_b = np.kron(source_b, right_b)
            source_d = next_d
        sparse_db[node.name] = {
            key: np.concatenate([value, np.zeros(1, dtype=complex)])
            for key, value in source_d.items()
        }

    a = np.zeros(n_modes, dtype=float)
    da = np.zeros(n_modes, dtype=float)
    ja = np.zeros((n_modes, n_weights), dtype=float)
    jda = np.zeros_like(ja)

    for mode, name in enumerate(graph.initial_names):
        realization = compiled.node_realizations[name]
        value, slope = realization_observation_action(
            realization.A, realization.b, realization.c, time
        )
        a[mode] += float(np.real(value))
        da[mode] += float(np.real(slope))

    for node in graph.response_nodes:
        realization = compiled.node_realizations[node.name]
        derivatives = sparse_db[node.name]
        indices = tuple(sorted(derivatives))
        B = np.column_stack(
            [realization.b] + [derivatives[index] for index in indices]
        )
        values, slopes = realization_observation_action(
            realization.A, B, realization.c, time
        )
        values = np.real(values)
        slopes = np.real(slopes)
        target = node.target_mode
        a[target] += float(values[0])
        da[target] += float(slopes[0])
        if indices:
            columns = list(indices)
            ja[target, columns] += values[1:]
            jda[target, columns] += slopes[1:]

    return a, da, ja, jda


def install_observation_training_acceleration() -> None:
    """Install the exact observation-row contraction in the late-stage trainer."""
    from . import late_stage_runtime

    late_stage_runtime._sparse_weight_value_jacobian = sparse_weight_value_jacobian_observed
