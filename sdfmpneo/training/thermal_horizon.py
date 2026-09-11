"""Finite-horizon diagnostics for an already-computed steady modal envelope."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .automatic_thermal_rank import select_rank_from_modal_response_envelope


def finite_horizon_rank_diagnostic(
    lambdas,
    steady_response_envelope,
    horizons=(1.0, 10.0, 30.0, 100.0),
    *,
    relative_tolerance,
    absolute_tolerance=0.0,
    safety_factor=1.0,
    boundary_fraction=0.25,
):
    """Compare steady-envelope rank with finite-time modal response ranks.

    If ``E_inf[j] = |q_j| / lambda_j`` is the existing steady response envelope,
    then a constant modal source reaches ``E_inf[j] * (1-exp(-lambda_j H))`` by
    time H.  This is a design diagnostic only: production truncation remains the
    conservative steady-envelope criterion.
    """
    rates = np.asarray(lambdas, dtype=float).reshape(-1)
    steady = np.asarray(steady_response_envelope, dtype=float).reshape(-1)
    if rates.shape != steady.shape or rates.size < 1:
        raise ValueError("thermal rates and steady response envelope must have the same nonzero length")
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("thermal decay rates must be finite and positive")
    if np.any(~np.isfinite(steady)) or np.any(steady < 0.0):
        raise ValueError("steady response envelope must be finite and non-negative")

    result = {}
    for value in horizons:
        horizon = float(value)
        if not np.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("diagnostic horizons must be finite and positive")
        finite = steady * (-np.expm1(-rates * horizon))
        rank, details = select_rank_from_modal_response_envelope(
            finite,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            safety_factor=safety_factor,
            boundary_fraction=boundary_fraction,
            unresolved=False,
        )
        result[f"{horizon:g}s"] = {
            "rank": int(rank),
            "horizon_s": horizon,
            "total_response_norm": float(details["total_response_norm"]),
            "resolved_tail_norm": float(details["resolved_tail_norm"]),
            "selection_limit": float(details["selection_limit"]),
        }
    return result


def _full_decay_rates(core, report, full_dimension):
    cache_path = report.get("spectrum_cache_path") if isinstance(report, dict) else None
    if cache_path:
        try:
            with np.load(Path(cache_path), allow_pickle=False) as data:
                rates = np.asarray(data["lambdas"], dtype=float).reshape(-1)
            if (
                rates.size == int(full_dimension)
                and np.all(np.isfinite(rates))
                and np.all(rates > 0.0)
            ):
                return rates, None
        except (OSError, ValueError, TypeError, KeyError):
            pass

    try:
        rates = np.asarray(core.thermal_model.lambdas, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return None, "full_spectrum_decay_rates_unavailable"
    if (
        rates.size != int(full_dimension)
        or np.any(~np.isfinite(rates))
        or np.any(rates <= 0.0)
    ):
        return None, "full_spectrum_decay_rates_unavailable"
    return rates, None


def diagnostic_from_rank_report(core, report, horizons=(1.0, 10.0, 30.0, 100.0)):
    """Build a horizon comparison from the saved full envelope and spectrum cache.

    The retained surrogate model normally stores only the selected thermal
    prefix.  The layered automatic-rank cache, however, already contains the
    complete spectrum.  Reusing it makes this diagnostic cheap and avoids
    invalidating the existing rank cache.
    """
    if not isinstance(report, dict):
        return {"available": False, "reason": "thermal_rank_report_missing"}
    envelope = report.get("maximum_modal_steady_response")
    if envelope is None:
        return {"available": False, "reason": "steady_response_envelope_missing"}
    envelope = np.asarray(envelope, dtype=float).reshape(-1)
    full_dimension = int(report.get("full_dimension", envelope.size))
    if envelope.size != full_dimension:
        return {
            "available": False,
            "reason": "steady_response_envelope_incomplete",
            "full_dimension": full_dimension,
            "envelope_dimension": int(envelope.size),
        }
    rates, reason = _full_decay_rates(core, report, full_dimension)
    if rates is None:
        return {
            "available": False,
            "reason": reason,
            "retained_rank": int(getattr(core.thermal_model, "rank", 0)),
            "full_dimension": full_dimension,
        }
    values = finite_horizon_rank_diagnostic(
        rates,
        envelope,
        horizons,
        relative_tolerance=float(report.get("relative_tolerance", 1e-3)),
        absolute_tolerance=float(report.get("absolute_tolerance", 0.0)),
        safety_factor=float(report.get("source_bound_safety_factor", 1.0)),
        boundary_fraction=float(report.get("boundary_fraction", 0.25)),
    )
    return {
        "available": True,
        "method": "steady_envelope_times_exact_first_order_horizon_factor",
        "production_rank_unchanged": True,
        "steady_selected_rank": int(report.get("selected_rank", full_dimension)),
        "full_dimension": full_dimension,
        "horizons": values,
        "scope": (
            "design diagnostic from the cached steady-response envelope; "
            "not a replacement for the production steady-envelope rank criterion"
        ),
    }


__all__ = ["finite_horizon_rank_diagnostic", "diagnostic_from_rank_report"]
