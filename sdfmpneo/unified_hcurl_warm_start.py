"""Install compatible coarse-to-fine warm transfer for local Maxwell solves."""
from __future__ import annotations

import time

import numpy as np

from .unified_compensated_field import collapsed_field
from .unified_hcurl_transfer import build_hcurl_prolongation
from .unified_two_level_local_krylov import _two_level_minimum_dofs


def install(local_solver_module):
    """Replace component interpolation by a commuting H(curl) edge transfer."""

    def pack_warm_state(local, local_geometry, port, fine_step, field):
        # Warm state is only an initial guess for a later solve.  A compensated
        # high/low field is therefore intentionally collapsed here; the current
        # solve has already consumed the high/low representation for its truth
        # certificate and physical contractions before any later reuse.
        value = collapsed_field(field)
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
        cfg = dict(getattr(parent, "self_correction_config", {}) or {})
        two_level_required = local.n_edges >= _two_level_minimum_dofs(cfg)
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
        started = time.perf_counter()
        try:
            P = build_hcurl_prolongation(axes, local)
            if coarse_field.shape != (P.shape[1],):
                raise ValueError("coarse warm field dimension does not match H(curl) transfer")
            guess = np.asarray(P @ coarse_field, complex).reshape(-1)
        except (RuntimeError, ValueError, TypeError, FloatingPointError) as exc:
            message = str(exc).replace("\n", " ")
            print(
                "local Maxwell H(curl) transfer FAILED: "
                f"coarse_step={previous_step:g}m, fine_step={float(fine_step):g}m, "
                f"fine_edges={local.n_edges}, error={type(exc).__name__}: {message}",
                flush=True,
            )
            # Once the direct band ends, the two-level hierarchy is the robust
            # iterative path.  Do not silently route a failed transfer back into
            # the one-level ILU/LGMRES gap that production geometry can hit.
            if two_level_required:
                raise RuntimeError("required H(curl) local transfer failed") from exc
            return None, False
        if guess.shape != (local.n_edges,) or np.any(~np.isfinite(guess)):
            if two_level_required:
                raise FloatingPointError("required H(curl) local warm field is invalid")
            return None, False

        local._sdfmpneo_coarse_state = {
            **state,
            "prolongation": P,
            "target_fine_step": float(fine_step),
        }
        print(
            "local Maxwell H(curl) transfer: "
            f"coarse_edges={P.shape[1]}, fine_edges={P.shape[0]}, nnz={P.nnz}, "
            f"time={time.perf_counter()-started:.1f}s",
            flush=True,
        )
        return guess, True

    local_solver_module._pack_warm_state = pack_warm_state
    local_solver_module._warm_start = warm_start
    local_solver_module._compatible_hcurl_warm_start_installed = True
    return local_solver_module


__all__ = ["install"]
