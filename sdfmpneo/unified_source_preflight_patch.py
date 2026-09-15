"""Production source preflight for an open two-terminal stranded current.

The theory explicitly forbids turning an open two-terminal spiral into a closed
divergence-free loop.  The correct discrete continuity audit therefore checks
that the deposited source has a genuine terminal divergence, that its total
nodal injection is exactly balanced, and that the first moment of that nodal
divergence reproduces the physical terminal-separation vector.
"""
from __future__ import annotations

import numpy as np

from .unified_gradient_block_maxwell import source_terminal_divergence


_MAX_NET_TERMINAL_BALANCE_ERROR = 1e-12
_MAX_TERMINAL_MOMENT_RELATIVE_ERROR = 1e-12
_MIN_TERMINAL_DIVERGENCE_RELATIVE_NORM = 1e-12


def install(truth_preflight_module):
    if bool(getattr(truth_preflight_module, "_open_terminal_source_preflight_installed", False)):
        return truth_preflight_module
    original = truth_preflight_module._source_and_loss_partition

    def source_and_loss_partition(background, geometry):
        result = dict(original(background, geometry))
        context = background.geometry_context(geometry, assemble_thermal=False)
        source = np.asarray(context.source_shape, float)
        if source.ndim != 2 or source.shape[0] != background.n_edges:
            raise ValueError("production source matrix has invalid edge/port shape")

        rows = []
        for port in range(source.shape[1]):
            q, net_error, moment = source_terminal_divergence(background, source[:, port])
            coil = context.geometry.coils[port]
            points = np.asarray(background._physical_centerline(coil), float)
            endpoint_vector = np.asarray(points[-1] - points[0], float)
            path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            moment_error = float(
                np.linalg.norm(moment - endpoint_vector)
                / max(path_length, np.finfo(float).tiny)
            )
            divergence_relative = float(
                np.linalg.norm(q)
                / max(float(np.linalg.norm(source[:, port])), np.finfo(float).tiny)
            )
            rows.append(
                {
                    "port": int(port),
                    "net_terminal_balance_relative_error": float(net_error),
                    "terminal_divergence_relative_norm": divergence_relative,
                    "terminal_first_moment_relative_error": moment_error,
                    "terminal_divergence_nonzero": bool(
                        divergence_relative >= _MIN_TERMINAL_DIVERGENCE_RELATIVE_NORM
                    ),
                }
            )

        terminal_ok = bool(
            rows
            and all(
                row["net_terminal_balance_relative_error"]
                <= _MAX_NET_TERMINAL_BALANCE_ERROR
                and row["terminal_first_moment_relative_error"]
                <= _MAX_TERMINAL_MOMENT_RELATIVE_ERROR
                and row["terminal_divergence_nonzero"]
                for row in rows
            )
        )
        result["open_two_terminal_charge_balance"] = terminal_ok
        result["maximum_net_terminal_balance_relative_error"] = max(
            (row["net_terminal_balance_relative_error"] for row in rows),
            default=float("inf"),
        )
        result["minimum_terminal_divergence_relative_norm"] = min(
            (row["terminal_divergence_relative_norm"] for row in rows),
            default=0.0,
        )
        result["maximum_terminal_first_moment_relative_error"] = max(
            (row["terminal_first_moment_relative_error"] for row in rows),
            default=float("inf"),
        )
        result["terminal_divergence_audit"] = rows
        result["terminal_path_conservation"] = bool(
            result.get("terminal_path_conservation", False) and terminal_ok
        )
        return result

    truth_preflight_module._source_and_loss_partition = source_and_loss_partition
    truth_preflight_module._open_terminal_source_preflight_installed = True
    return truth_preflight_module


__all__ = ["install"]
