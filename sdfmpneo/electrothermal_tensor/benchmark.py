"""Wall-clock benchmarks for Gate 7; no theoretical FLOP substitutions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class TimingStats:
    samples: int
    total_seconds: float
    mean_seconds: float
    median_seconds: float
    percentile_95_seconds: float
    throughput_per_second: float


def _stats(durations) -> TimingStats:
    value = np.asarray(tuple(durations), dtype=float)
    if value.size < 1 or np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("benchmark durations must be finite and non-negative")
    total = float(np.sum(value))
    return TimingStats(
        samples=int(value.size),
        total_seconds=total,
        mean_seconds=float(np.mean(value)),
        median_seconds=float(np.median(value)),
        percentile_95_seconds=float(np.percentile(value, 95)),
        throughput_per_second=float(value.size / max(total, np.finfo(float).tiny)),
    )


def benchmark_calls(
    callables: Iterable[Callable[[], object]],
    *,
    warmup: int = 0,
) -> TimingStats:
    calls = list(callables)
    if not calls:
        raise ValueError("at least one benchmark call is required")
    for call in calls[: max(0, int(warmup))]:
        call()
    durations = []
    for call in calls:
        start = perf_counter()
        call()
        durations.append(perf_counter() - start)
    return _stats(durations)


def benchmark_snapshot_generation(tensor_factory, states, geometries) -> TimingStats:
    a = np.asarray(states, dtype=float)
    g = np.asarray(geometries, dtype=float)
    if a.ndim != 2 or g.ndim != 2 or len(a) != len(g):
        raise ValueError("states/geometries must be aligned matrices")
    return benchmark_calls(
        (lambda ai=ai, gi=gi: tensor_factory(ai, gi) for ai, gi in zip(a, g))
    )


def benchmark_vector_field(field, state, geometry, operating, *, repeats: int = 100) -> TimingStats:
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    a = np.asarray(state, dtype=float)
    g = np.asarray(geometry, dtype=float)
    u = np.asarray(operating, dtype=float)
    field.vector_field(a, g, u)
    return benchmark_calls(
        (lambda: field.vector_field(a, g, u) for _ in range(repeats))
    )


def benchmark_trajectory_queries(
    model,
    times,
    *,
    initial_state,
    geometry,
    operating,
    max_step,
    method: str = "etd2",
    predict_options: dict | None = None,
    repeats: int = 3,
) -> dict[float, TimingStats]:
    """Benchmark exactly the integrator/options intended for deployment."""
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    options = {} if predict_options is None else dict(predict_options)
    result = {}
    for time in times:
        t = float(time)

        def call(t=t):
            return model.predict(
                t,
                initial_state=initial_state,
                geometry=geometry,
                operating=operating,
                max_step=max_step,
                method=method,
                **options,
            )

        call()
        result[t] = benchmark_calls([call for _ in range(repeats)])
    return result


def model_file_size(path: str | Path) -> int:
    return int(Path(path).stat().st_size)


__all__ = [
    "TimingStats",
    "benchmark_calls",
    "benchmark_snapshot_generation",
    "benchmark_trajectory_queries",
    "benchmark_vector_field",
    "model_file_size",
]
