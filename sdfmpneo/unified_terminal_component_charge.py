"""Terminal-component source locking for local dissipative scalar refinement.

A Cartesian terminal refinement inserts whole coordinate planes.  Reassembling the
full balanced terminal charge on that mixed grid can therefore redistribute the
*other* terminal even when only one terminal is meant to be certified.  The local
additive defect must change exactly one source component at a time.

For a selected terminal we keep the opposite terminal's production coarse nodal
load byte-for-byte (embedded on the refined node set) and replace only the
selected feed/return component by its refined physical quadrature load.  Since
both coarse and refined components carry exactly one unit of terminal charge,
the hybrid full-port source remains globally balanced.
"""
from __future__ import annotations

import numpy as np

from .unified_charge_regularized_source import terminal_charge_target


_TERMINAL_NAMES = ("feed", "return")


def _axis_embedding(coarse_axis, fine_axis):
    coarse = np.asarray(coarse_axis, float)
    fine = np.asarray(fine_axis, float)
    scale = max(float(np.max(np.abs(coarse))), float(np.max(np.abs(fine))), 1.0)
    tol = 256.0 * np.finfo(float).eps * scale
    index = np.searchsorted(fine, coarse, side="left")
    if np.any(index >= fine.size):
        raise RuntimeError("coarse terminal-charge node is outside refined axis")
    if np.any(np.abs(fine[index] - coarse) > tol):
        raise RuntimeError("refined terminal patch does not retain all coarse nodes")
    return np.asarray(index, np.int64)


def _embed_coarse_vector(coarse_background, fine_background, vector):
    q = np.asarray(vector, float).reshape(
        len(coarse_background.x), len(coarse_background.y), len(coarse_background.z)
    )
    ix = _axis_embedding(coarse_background.x, fine_background.x)
    iy = _axis_embedding(coarse_background.y, fine_background.y)
    iz = _axis_embedding(coarse_background.z, fine_background.z)
    out = np.zeros(
        (len(fine_background.x), len(fine_background.y), len(fine_background.z)),
        float,
    )
    out[np.ix_(ix, iy, iz)] = q
    return out.reshape(-1)


def hybrid_terminal_charge_from_targets(
    coarse_background,
    fine_background,
    coarse_target,
    fine_target,
    terminal,
):
    """Replace one terminal component while leaving the opposite component coarse."""
    t = int(terminal)
    if t not in (0, 1):
        raise ValueError("terminal component must be feed=0 or return=1")
    q_coarse = np.asarray(coarse_target, float).reshape(-1)
    q_fine = np.asarray(fine_target, float).reshape(-1)

    coarse_selected = np.minimum(q_coarse, 0.0) if t == 0 else np.maximum(q_coarse, 0.0)
    coarse_other = np.maximum(q_coarse, 0.0) if t == 0 else np.minimum(q_coarse, 0.0)
    fine_selected = np.minimum(q_fine, 0.0) if t == 0 else np.maximum(q_fine, 0.0)

    embedded_other = _embed_coarse_vector(coarse_background, fine_background, coarse_other)
    hybrid = embedded_other + fine_selected

    selected_total = float(np.sum(fine_selected))
    other_total = float(np.sum(embedded_other))
    expected_selected = -1.0 if t == 0 else 1.0
    expected_other = 1.0 if t == 0 else -1.0
    if abs(selected_total - expected_selected) > 5e-13:
        raise FloatingPointError(
            f"refined {_TERMINAL_NAMES[t]} charge total is invalid: {selected_total:.16e}"
        )
    if abs(other_total - expected_other) > 5e-13:
        raise FloatingPointError(
            f"coarse locked opposite-terminal charge total is invalid: {other_total:.16e}"
        )
    net = float(np.sum(hybrid))
    if abs(net) > 5e-13:
        raise FloatingPointError(f"hybrid terminal component source is not balanced: {net:.3e}")

    return hybrid, {
        "terminal_component_lock": True,
        "selected_terminal": t,
        "selected_terminal_name": _TERMINAL_NAMES[t],
        "selected_terminal_total": selected_total,
        "locked_opposite_terminal_total": other_total,
        "terminal_component_net_charge": net,
        "locked_opposite_terminal_support_nodes": int(np.count_nonzero(np.abs(embedded_other) > 1e-15)),
        "selected_terminal_support_nodes": int(np.count_nonzero(np.abs(fine_selected) > 1e-15)),
    }


def terminal_component_locked_target(
    coarse_background,
    fine_background,
    coil,
    terminal,
):
    """Build the balanced hybrid q_target for a single-terminal additive defect."""
    coarse_target, coarse_meta = terminal_charge_target(coarse_background, coil)
    fine_target, fine_meta = terminal_charge_target(fine_background, coil)
    hybrid, lock_meta = hybrid_terminal_charge_from_targets(
        coarse_background,
        fine_background,
        coarse_target,
        fine_target,
        int(terminal),
    )
    meta = dict(fine_meta)
    meta.update(lock_meta)
    meta["coarse_terminal_charge_support_nodes"] = int(
        coarse_meta.get("terminal_charge_support_nodes", 0)
    )
    meta["terminal_charge_support_nodes"] = int(np.count_nonzero(np.abs(hybrid) > 1e-15))
    return hybrid, meta


__all__ = [
    "hybrid_terminal_charge_from_targets",
    "terminal_component_locked_target",
]
