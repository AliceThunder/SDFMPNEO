from __future__ import annotations

"""Concurrent exact geometry-context cache for training working sets.

Geometry construction (deformed mesh, thermal M/K, ports and reduced EM helpers)
is independent for distinct geometry keys.  Training used to build these contexts
serially and could also evict them while repeatedly visiting the same collocation
set.  This runtime keeps the working set resident when practical, coalesces
same-key construction, and builds distinct geometries concurrently.
"""

from collections import OrderedDict
from copy import copy
import os
from threading import Event, RLock

import numpy as np

from .parallel_runtime import _ordered_map


def install_concurrent_geometry_context_cache(model_class) -> None:
    if getattr(model_class, "_concurrent_context_cache_installed", False):
        return
    if not all(hasattr(model_class, name) for name in ("context", "geometry_vector")):
        return

    original_init = model_class.__init__
    original_context = model_class.context

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._context_cache_lock = RLock()
        self._context_inflight = {}

    def _ensure_runtime(self):
        if not hasattr(self, "_context_cache_lock"):
            self._context_cache_lock = RLock()
        if not hasattr(self, "_context_inflight"):
            self._context_inflight = {}

    def _uncached_build(self, geometry):
        # The historical context() mutates an LRU while constructing.  Give it a
        # private one-entry cache on a shallow proxy so expensive construction can
        # happen outside the shared lock and distinct geometries can overlap.
        proxy = copy(self)
        proxy._cache = OrderedDict()
        proxy.cache_size = 1
        proxy._context_cache_lock = RLock()
        proxy._context_inflight = {}
        return original_context(proxy, geometry)

    def context(self, geometry):
        _ensure_runtime(self)
        g = self.geometry_vector(geometry)
        key = tuple(float(value) for value in g)
        lock = self._context_cache_lock

        while True:
            with lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached
                event = self._context_inflight.get(key)
                if event is None:
                    event = Event()
                    self._context_inflight[key] = event
                    builder = True
                else:
                    builder = False
            if builder:
                break
            event.wait()

        try:
            result = _uncached_build(self, g)
        except Exception:
            with lock:
                current = self._context_inflight.pop(key, None)
                if current is not None:
                    current.set()
            raise

        with lock:
            cached = self._cache.get(key)
            if cached is None:
                self._cache[key] = result
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
                cached = result
            else:
                self._cache.move_to_end(key)
            current = self._context_inflight.pop(key, None)
            if current is not None:
                current.set()
            return cached

    def prepare_training_contexts(self, *point_sets):
        _ensure_runtime(self)
        arrays = []
        for values in point_sets:
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            if array.size:
                if array.ndim != 2:
                    raise ValueError("training point sets must be matrices")
                arrays.append(array)
        if not arrays:
            return 0

        n_initial = int(self.thermal_model.rank)
        n_geometry = len(self.geometry_names)
        normalized = []
        seen = set()
        for values in arrays:
            for row in values:
                z = tuple(float(v) for v in row[n_initial:n_initial + n_geometry])
                if z not in seen:
                    seen.add(z)
                    normalized.append(np.asarray(z, dtype=float))

        required = len(normalized)
        maximum = max(1, int(os.environ.get("SDFMPNEO_GEOMETRY_CACHE_MAX", "512")))
        target = min(required, maximum)
        with self._context_cache_lock:
            if target > self.cache_size:
                self.cache_size = target
            fits = required <= self.cache_size

        if fits and normalized:
            _ordered_map(
                lambda z: self.context(self.denormalize(z)),
                normalized,
                monitor=None,
            )
        return required

    model_class.__init__ = init
    model_class.context = context
    model_class.prepare_training_contexts = prepare_training_contexts
    model_class._concurrent_context_cache_installed = True


def enable_concurrent_geometry_context_cache(model) -> bool:
    """Enable the runtime for an already-created geometry model instance."""
    if not all(
        hasattr(model, name)
        for name in ("context", "geometry_vector", "geometry_names", "denormalize")
    ):
        return False
    install_concurrent_geometry_context_cache(type(model))
    if not hasattr(model, "_context_cache_lock"):
        model._context_cache_lock = RLock()
    if not hasattr(model, "_context_inflight"):
        model._context_inflight = {}
    return True


__all__ = [
    "enable_concurrent_geometry_context_cache",
    "install_concurrent_geometry_context_cache",
]
