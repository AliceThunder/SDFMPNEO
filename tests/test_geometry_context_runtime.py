from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
from types import SimpleNamespace

import numpy as np

from sdfmpneo.training.geometry_context_runtime import (
    enable_concurrent_geometry_context_cache,
)


class _FakeGeometryModel:
    def __init__(self):
        self.cache_size = 1
        self._cache = OrderedDict()
        self.geometry_names = ("g",)
        self.thermal_model = SimpleNamespace(rank=1)
        self._stats_lock = Lock()
        # A shallow proxy intentionally shares this dictionary, matching the kind
        # of immutable/heavy state the production proxy shares with its parent.
        self.stats = {"active": 0, "maximum": 0, "counts": {}}

    @property
    def build_counts(self):
        return self.stats["counts"]

    def geometry_vector(self, geometry):
        value = np.asarray(geometry, dtype=float)
        if value.shape != (1,):
            raise ValueError("bad geometry")
        return value

    def denormalize(self, z):
        return np.asarray(z, dtype=float)

    def context(self, geometry):
        g = self.geometry_vector(geometry)
        key = tuple(float(value) for value in g)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        with self._stats_lock:
            self.stats["active"] += 1
            self.stats["maximum"] = max(
                self.stats["maximum"], self.stats["active"]
            )
            counts = self.stats["counts"]
            counts[key] = counts.get(key, 0) + 1
        time.sleep(0.03)
        result = object()
        with self._stats_lock:
            self.stats["active"] -= 1
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result


def test_geometry_working_set_is_prewarmed_once_and_kept_resident(monkeypatch):
    monkeypatch.setenv("SDFMPNEO_POINT_WORKERS", "4")
    monkeypatch.setenv("SDFMPNEO_GEOMETRY_CACHE_MAX", "16")
    model = _FakeGeometryModel()
    assert enable_concurrent_geometry_context_cache(model)

    points = np.asarray([
        [0.0, -1.0, 0.2, 0.0],
        [0.1, 0.0, 0.4, 1.0],
        [0.2, 1.0, 0.6, 2.0],
        [0.3, 0.0, 0.8, 3.0],
    ])
    required = model.prepare_training_contexts(points)
    assert required == 3
    assert model.cache_size >= 3
    assert len(model._cache) == 3
    assert model.stats["maximum"] >= 2
    assert model.build_counts == {(-1.0,): 1, (0.0,): 1, (1.0,): 1}

    model.prepare_training_contexts(points)
    assert model.build_counts == {(-1.0,): 1, (0.0,): 1, (1.0,): 1}


def test_duplicate_geometry_builds_are_coalesced(monkeypatch):
    monkeypatch.setenv("SDFMPNEO_POINT_WORKERS", "4")
    model = _FakeGeometryModel()
    enable_concurrent_geometry_context_cache(model)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(model.context, ([0.4], [0.4], [0.4], [0.4])))
    assert all(value is results[0] for value in results)
    assert model.build_counts[(0.4,)] == 1
