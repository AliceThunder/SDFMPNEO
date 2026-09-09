from __future__ import annotations

import os
from threading import Barrier, get_ident

from sdfmpneo.training.parallel_runtime import (
    _ordered_map,
    training_blas_threads,
    training_parallelism,
    training_point_workers,
)


def test_preexisting_threaded_blas_does_not_disable_outer_point_workers(monkeypatch):
    monkeypatch.delenv("SDFMPNEO_POINT_WORKERS", raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setattr(os, "cpu_count", lambda: 12)
    assert training_point_workers() == 12


def test_explicit_point_and_blas_settings_are_respected(monkeypatch):
    monkeypatch.setenv("SDFMPNEO_POINT_WORKERS", "6")
    monkeypatch.setenv("SDFMPNEO_BLAS_THREADS", "2")
    assert training_point_workers() == 6
    assert training_blas_threads() == 2
    assert training_parallelism()["point_workers"] == 6
    assert training_parallelism()["blas_threads_per_point"] == 2


def test_ordered_map_really_uses_multiple_worker_threads(monkeypatch):
    workers = 4
    monkeypatch.setenv("SDFMPNEO_POINT_WORKERS", str(workers))
    monkeypatch.setenv("SDFMPNEO_BLAS_THREADS", "1")
    gate = Barrier(workers)

    def one(value):
        gate.wait(timeout=5.0)
        return value, get_ident()

    out = _ordered_map(one, range(workers))
    assert [value for value, _ in out] == list(range(workers))
    assert len({thread_id for _, thread_id in out}) == workers
