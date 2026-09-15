"""Install compatible coarse-to-fine warm transfer for local Maxwell solves."""
from __future__ import annotations

import numpy as np

from .unified_hcurl_transfer import build_hcurl_prolongation


def install(local_solver_module):
    """Replace component interpolation by a commuting H(curl) edge transfer."""

    def pack_warm_state(local, local_geometry, port, fine_step, field):
        value = np.asarray(field, complex).reshape(-1)
        if value.shape != (local.n_edges,) or np.any(~np.isfinite(value)):
            raise ValueError("local Maxwell warm state field is invalid")
        return {
            "key": local_solver_module._canonical_key(local_geometry, port),
            "fine_step": float(fine_step),
            "axes": tuple(np.asarray(axis, float).copy() for axis in (local.x, local.y, local.z)),
            "field": value.copy(),
            "local_geometry": local_geometry,
        }

    def warm_start(parent, local, local_geometry, port, fine_step):
        state = getattr(parent, "_local_self_warm_state", None)
        if not isinstance(state, dict) or state.get("key") != local_solver_module._canonical_key(local_geometry, port):
            return None, False
        previous_step = float(state.get("fine_step", np.inf))
        if float(fine_step) > previous_step * (1.0 + 1e-12):
            return None, False
        axes = state.get("axes")
        coarse_field = np.asarray(state.get("field", ()), complex).reshape(-1)
        if axes is None:
            return None, False
        try:
            P = build_hcurl_prolongation(axes, local)
            if coarse_field.shape != (P.shape[1],):
                return None, False
            guess = np.asarray(P @ coarse_field, complex).reshape(-1)
        except (RuntimeError, ValueError, TypeError, FloatingPointError):
            return None, False
        if guess.shape != (local.n_edges,) or np.any(~np.isfinite(guess)):
            return None, False

        # Keep the coarse information only on this fine background for the
        # duration of the solve.  The two-level preconditioner can reuse P and
        # rebuild the much smaller coarse operator without retaining it between
        # audit stages.
        local._sdfmpneo_coarse_state = {
            **state,
            "prolongation": P,
            "target_fine_step": float(fine_step),
        }
        print(
            "local Maxwell H(curl) transfer: "
            f"coarse_edges={P.shape[1]}, fine_edges={P.shape[0]}, nnz={P.nnz}",
            flush=True,
        )
        return guess, True

    local_solver_module._pack_warm_state = pack_warm_state
    local_solver_module._warm_start = warm_start
    local_solver_module._compatible_hcurl_warm_start_installed = True
    return local_solver_module


__all__ = ["install"]
