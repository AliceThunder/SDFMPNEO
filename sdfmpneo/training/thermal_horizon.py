"""Finite-horizon diagnostics for an already-computed steady modal envelope."""
from __future__ import annotations

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
    time H.  This is a diagnostic only: the production truncation rule remains
    the conservative steady-envelope selection until the user explicitly opts
    into a finite-horizon certificate.
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
            "maximum_modal_response": finite.tolist(),
            **details,
        }
    return result


def diagnostic_from_rank_report(model, report, horizons=(1.0, 10.0, 30.0, 100.0)):
    """Build a horizon comparison when the saved report contains the full envelope.

    Older layered caches do not store the discarded full-spectrum lambdas.  The
    diagnostic is therefore exact only when the retained thermal model already
    equals the report's full diagnostic dimension.  In all other cases return a
    clear unavailable reason instead of underestimating the required rank.
    """
    envelope = report.get("maximum_modal_steady_response") if isinstance(report, dict) else None
    if envelope is None:
        return {"available": False, "reason": "steady_response_envelope_missing"}
    rates = np.asarray(model.thermal_model.lambdas, dtype=float).reshape(-1)
    envelope = np.asarray(envelope, dtype=float).reshape(-1)
    full_dimension = int(report.get("full_dimension", envelope.size))
    if envelope.size != full_dimension or rates.size != full_dimension:
        return {
            "available": False,
            "reason": "full_spectrum_decay_rates_not_retained_in_this_checkpoint",
            "retained_rank": int(rates.size),
            "full_dimension": int(full_dimension),
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
        "selection_remains_steady_envelope": True,
        "horizons": values,
    }


__all__ = ["finite_horizon_rank_diagnostic", "diagnostic_from_rank_report"]
