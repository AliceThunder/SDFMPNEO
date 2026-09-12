"""State/geometry sampling for high-dimensional neural Joule-tensor learning."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .dataset import latin_hypercube_box


class StateDomainInsufficientError(ValueError):
    """Raised when sampled real-physics trajectories leave the declared state box."""

    def __init__(self, observed_lower: np.ndarray, observed_upper: np.ndarray):
        self.observed_lower = np.asarray(observed_lower, dtype=float)
        self.observed_upper = np.asarray(observed_upper, dtype=float)
        super().__init__(
            "physical reachable-state sampling left the declared neural state box; "
            "expand state_lower/state_upper before generating tensor labels"
        )


@dataclass(frozen=True)
class SnapshotSamplingReport:
    strategy: str
    sample_count: int
    box_sample_count: int
    reachable_sample_count: int
    trajectory_count: int
    observed_state_lower: np.ndarray
    observed_state_upper: np.ndarray


@dataclass(frozen=True)
class SnapshotSamplingResult:
    states: np.ndarray
    geometries: np.ndarray
    report: SnapshotSamplingReport


def box_state_geometry_samples(
    state_lower,
    state_upper,
    geometry_lower,
    geometry_upper,
    n_samples: int,
    *,
    seed: int = 0,
) -> SnapshotSamplingResult:
    state_lo = np.asarray(state_lower, dtype=float).reshape(-1)
    state_hi = np.asarray(state_upper, dtype=float).reshape(-1)
    geometry_lo = np.asarray(geometry_lower, dtype=float).reshape(-1)
    geometry_hi = np.asarray(geometry_upper, dtype=float).reshape(-1)
    lower = np.concatenate([state_lo, geometry_lo])
    upper = np.concatenate([state_hi, geometry_hi])
    sampled = latin_hypercube_box(lower, upper, int(n_samples), seed=int(seed))
    r = state_lo.size
    states = sampled[:, :r]
    geometries = sampled[:, r:]
    return SnapshotSamplingResult(
        states,
        geometries,
        SnapshotSamplingReport(
            strategy="box",
            sample_count=len(states),
            box_sample_count=len(states),
            reachable_sample_count=0,
            trajectory_count=0,
            observed_state_lower=np.min(states, axis=0),
            observed_state_upper=np.max(states, axis=0),
        ),
    )


def _trajectory_times(time_horizon: float, samples_per_trajectory: int, time_min: float) -> np.ndarray:
    horizon = float(time_horizon)
    count = int(samples_per_trajectory)
    minimum = float(time_min)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("time_horizon must be finite and positive")
    if count < 2:
        raise ValueError("samples_per_trajectory must be at least two")
    if not np.isfinite(minimum) or minimum <= 0.0 or minimum > horizon:
        raise ValueError("time_min must satisfy 0 < time_min <= time_horizon")
    if count == 2:
        return np.array([0.0, horizon])
    positive = np.geomspace(minimum, horizon, count - 1)
    return np.concatenate([[0.0], positive])


def hybrid_reachable_state_geometry_samples(
    physical_vector_field,
    *,
    state_lower,
    state_upper,
    geometry_lower,
    geometry_upper,
    operating_lower,
    operating_upper,
    n_samples: int,
    initial_lower=None,
    initial_upper=None,
    box_fraction: float = 0.25,
    trajectory_count: int = 32,
    samples_per_trajectory: int = 16,
    time_horizon: float,
    time_min: float = 1e-6,
    seed: int = 0,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> SnapshotSamplingResult:
    """Mix full-box coverage with real-physics reachable-state locations.

    The physical trajectories are used only to choose state locations.  Tensor
    labels are still local ``G(a,g)`` evaluations and no transient state is used
    as a supervised network target.

    If any sampled physical trajectory exits the declared neural state box the
    function raises ``StateDomainInsufficientError``.  It never clips reachable
    states back into the box because doing so would hide a domain-design error.
    """
    state_lo = np.asarray(state_lower, dtype=float).reshape(-1)
    state_hi = np.asarray(state_upper, dtype=float).reshape(-1)
    geom_lo = np.asarray(geometry_lower, dtype=float).reshape(-1)
    geom_hi = np.asarray(geometry_upper, dtype=float).reshape(-1)
    op_lo = np.asarray(operating_lower, dtype=float).reshape(-1)
    op_hi = np.asarray(operating_upper, dtype=float).reshape(-1)
    init_lo = state_lo if initial_lower is None else np.asarray(initial_lower, dtype=float).reshape(-1)
    init_hi = state_hi if initial_upper is None else np.asarray(initial_upper, dtype=float).reshape(-1)
    n = int(n_samples)
    fraction = float(box_fraction)
    paths = int(trajectory_count)
    per_path = int(samples_per_trajectory)
    if n < 3:
        raise ValueError("n_samples must be at least three")
    if state_lo.shape != state_hi.shape or init_lo.shape != state_lo.shape or init_hi.shape != state_lo.shape:
        raise ValueError("state/initial bound dimensions do not match")
    if geom_lo.shape != geom_hi.shape or op_lo.shape != op_hi.shape:
        raise ValueError("geometry/operating bound dimensions do not match")
    if np.any(state_hi <= state_lo) or np.any(init_hi <= init_lo):
        raise ValueError("state/initial bounds must be strictly ordered")
    if np.any(geom_hi <= geom_lo) or np.any(op_hi <= op_lo):
        raise ValueError("geometry/operating bounds must be strictly ordered")
    if np.any(init_lo < state_lo) or np.any(init_hi > state_hi):
        raise ValueError("initial-state box must be contained in declared state box")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("box_fraction must lie in [0,1]")
    if paths < 1:
        raise ValueError("trajectory_count must be positive")

    n_box = int(round(n * fraction))
    n_box = min(max(n_box, 1), n - 1)
    box = box_state_geometry_samples(
        state_lo,
        state_hi,
        geom_lo,
        geom_hi,
        n_box,
        seed=seed,
    )
    n_reachable = n - n_box
    times = _trajectory_times(time_horizon, per_path, time_min)

    seed_lower = np.concatenate([init_lo, geom_lo, op_lo])
    seed_upper = np.concatenate([init_hi, geom_hi, op_hi])
    seeds = latin_hypercube_box(seed_lower, seed_upper, paths, seed=seed + 1)
    r = state_lo.size
    d = geom_lo.size
    candidates_state: list[np.ndarray] = []
    candidates_geometry: list[np.ndarray] = []
    observed_lower = np.full(r, np.inf)
    observed_upper = np.full(r, -np.inf)

    for row in seeds:
        initial = row[:r]
        geometry = row[r:r + d]
        operating = row[r + d:]

        def rhs(_time, state):
            return np.asarray(physical_vector_field(state, geometry, operating), dtype=float)

        solution = solve_ivp(
            rhs,
            (0.0, float(times[-1])),
            initial,
            method="Radau",
            dense_output=True,
            rtol=float(rtol),
            atol=float(atol),
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        states = np.asarray(solution.sol(times).T, dtype=float)
        observed_lower = np.minimum(observed_lower, np.min(states, axis=0))
        observed_upper = np.maximum(observed_upper, np.max(states, axis=0))
        if np.any(states < state_lo[None, :]) or np.any(states > state_hi[None, :]):
            raise StateDomainInsufficientError(observed_lower, observed_upper)
        candidates_state.extend(states)
        candidates_geometry.extend(np.repeat(geometry[None, :], len(states), axis=0))

    candidate_state = np.asarray(candidates_state, dtype=float)
    candidate_geometry = np.asarray(candidates_geometry, dtype=float)
    if len(candidate_state) < n_reachable:
        raise ValueError("reachable sampler produced fewer candidates than requested samples")
    rng = np.random.default_rng(int(seed) + 2)
    selected = rng.choice(len(candidate_state), size=n_reachable, replace=False)
    states = np.vstack([box.states, candidate_state[selected]])
    geometries = np.vstack([box.geometries, candidate_geometry[selected]])
    order = rng.permutation(n)
    states = states[order]
    geometries = geometries[order]
    observed_lower = np.minimum(observed_lower, np.min(box.states, axis=0))
    observed_upper = np.maximum(observed_upper, np.max(box.states, axis=0))
    return SnapshotSamplingResult(
        states,
        geometries,
        SnapshotSamplingReport(
            strategy="hybrid_reachable",
            sample_count=n,
            box_sample_count=n_box,
            reachable_sample_count=n_reachable,
            trajectory_count=paths,
            observed_state_lower=observed_lower,
            observed_state_upper=observed_upper,
        ),
    )


__all__ = [
    "SnapshotSamplingReport",
    "SnapshotSamplingResult",
    "StateDomainInsufficientError",
    "box_state_geometry_samples",
    "hybrid_reachable_state_geometry_samples",
]
