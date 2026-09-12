"""Batched online inference for independent queries sharing geometry and time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .integrators import GeneralizedThermalSpectrum, _etd_coefficients


@dataclass(frozen=True)
class BatchedNeuralROMPrediction:
    time: float
    states: np.ndarray
    derivatives: np.ndarray
    heat_sources: np.ndarray
    steps: int
    step_sizes: tuple[float, ...]


def _validate_batch_states(model, states: np.ndarray, *, allow_extrapolation: bool) -> None:
    if allow_extrapolation or not model.training_domain:
        if np.any(~np.isfinite(states)):
            raise ValueError("batched thermal states must be finite")
        return
    lower = model.training_domain.get("state_lower")
    upper = model.training_domain.get("state_upper")
    if lower is None or upper is None:
        return
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if states.ndim != 2 or states.shape[1] != lo.size or np.any(~np.isfinite(states)):
        raise ValueError("batched thermal states are incompatible with the trained state domain")
    if np.any(states < lo[None, :]) or np.any(states > hi[None, :]):
        raise ValueError("a batched trajectory left the trained thermal-state domain")


def predict_batch_fixed_etd2(
    model,
    time: float,
    *,
    initial_states: np.ndarray,
    geometry: np.ndarray,
    operating: np.ndarray,
    max_step: float,
    allow_extrapolation: bool = False,
) -> BatchedNeuralROMPrediction:
    """Run aligned queries with one shared geometry/time using batched MLP calls.

    Time stepping remains sequential because the ODE is causal.  Parallelism is
    across independent trajectories: each ETD2 stage evaluates the entire batch
    with one neural-network forward.  The generalized thermal eigenspectrum is
    built once and shared by every query in the batch.
    """
    t = float(time)
    hmax = float(max_step)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("batch prediction time must be finite and non-negative")
    if not np.isfinite(hmax) or hmax <= 0.0:
        raise ValueError("batch max_step must be finite and positive")
    states = np.asarray(initial_states, dtype=float)
    g = np.asarray(geometry, dtype=float).reshape(-1)
    u = np.asarray(operating, dtype=float)
    if states.ndim != 2 or states.shape[1] != model.surrogate.state_dimension or len(states) < 1:
        raise ValueError("initial_states must be a nonempty (batch,thermal_rank) matrix")
    if g.shape != (model.surrogate.geometry_dimension,):
        raise ValueError("shared geometry dimension mismatch")
    if u.ndim != 2 or u.shape != (len(states), model.surrogate.pod.current_dimension):
        raise ValueError("operating must have shape (batch,current_dimension)")
    if np.any(~np.isfinite(states)) or np.any(~np.isfinite(g)) or np.any(~np.isfinite(u)):
        raise ValueError("batch prediction inputs must be finite")

    # Reuse the model's full fail-closed domain logic for every independent
    # operating point.  Geometry is shared, so this also checks it once per row.
    for state, operating_row in zip(states, u):
        model._check_domain(
            state,
            g,
            operating_row,
            allow_extrapolation=allow_extrapolation,
        )
    _validate_batch_states(model, states, allow_extrapolation=allow_extrapolation)

    operator = model.thermal_operators.operator(g)
    spectrum = GeneralizedThermalSpectrum(operator)
    # For row-major batches, column formulas c=V^T M a and s=V^T q become
    # A M V and Q V respectively.
    mass_vectors = operator.mass @ spectrum.vectors

    def to_modal_state(batch_states):
        return np.asarray(batch_states, dtype=float) @ mass_vectors

    def from_modal_state(batch_modal):
        return np.asarray(batch_modal, dtype=float) @ spectrum.vectors.T

    def to_modal_source(batch_source):
        return np.asarray(batch_source, dtype=float) @ spectrum.vectors

    def heat(batch_states):
        geometries = np.repeat(g[None, :], len(batch_states), axis=0)
        return model.surrogate.heat_source_batch_numpy(batch_states, geometries, u)

    current = states.copy()
    elapsed = 0.0
    sizes: list[float] = []
    while elapsed < t:
        h = min(hmax, t - elapsed)
        E, b1, b2 = _etd_coefficients(spectrum.lambdas, h)
        c0 = to_modal_state(current)
        q0 = heat(current)
        s0 = to_modal_source(q0)
        c_stage = E[None, :] * c0 + b1[None, :] * s0
        stage = from_modal_state(c_stage)
        _validate_batch_states(model, stage, allow_extrapolation=allow_extrapolation)
        q1 = heat(stage)
        s1 = to_modal_source(q1)
        c1 = c_stage + b2[None, :] * (s1 - s0)
        current = from_modal_state(c1)
        _validate_batch_states(model, current, allow_extrapolation=allow_extrapolation)
        if np.any(~np.isfinite(current)):
            raise FloatingPointError("batched ETD2 produced non-finite thermal states")
        elapsed = min(t, elapsed + h)
        sizes.append(float(h))

    final_heat = heat(current)
    rhs = -current @ operator.stiffness.T + final_heat
    derivatives = scipy.linalg.solve(
        operator.mass,
        rhs.T,
        assume_a="pos",
        check_finite=True,
    ).T
    return BatchedNeuralROMPrediction(
        time=t,
        states=current,
        derivatives=np.asarray(derivatives, dtype=float),
        heat_sources=np.asarray(final_heat, dtype=float),
        steps=len(sizes),
        step_sizes=tuple(sizes),
    )


__all__ = ["BatchedNeuralROMPrediction", "predict_batch_fixed_etd2"]
