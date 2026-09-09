from __future__ import annotations

from collections import OrderedDict
from copy import copy
import os
from threading import Event, RLock

import numpy as np

from .parallel_runtime import _ordered_map


def install_concurrent_geometry_context_cache(model_class) -> None:
    """Install exact per-key geometry caching without serializing construction.

    The original ``context`` method is executed on a shallow proxy with a private
    cache, so its expensive mesh/thermal/EM construction touches no shared LRU.
    The real model cache is accessed only under a short lock.  One Event per key
    coalesces duplicate builds while different geometries construct concurrently.
    """
    if getattr(model_class, "_concurrent_context_cache_installed", False):
        return

    original_init = model_class.__init__
    original_context = model_class.context

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._context_cache_lock = RLock()
        self._context_inflight = {}

    def _uncached_build(self, geometry):
        # original_context owns cache mutation as part of its historical API.
        # Give it a private one-entry cache so the expensive construction can
        # run outside the shared model lock with exactly the original equations.
        proxy = copy(self)
        proxy._cache = OrderedDict()
        proxy.cache_size = 1
        proxy._context_cache_lock = RLock()
        proxy._context_inflight = {}
        return original_context(proxy, geometry)

    def context(self, geometry):
        g = self.geometry_vector(geometry)
        key = tuple(float(value) for value in g)
        lock = getattr(self, "_context_cache_lock", None)
        if lock is None:
            self._context_cache_lock = RLock()
            self._context_inflight = {}
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
            # No other builder can own this key, but a defensive cache check
            # preserves deterministic identity if a custom subclass inserted it.
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
        arrays = []
        for values in point_sets:
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            if array.size:
                arrays.append(array)
        if not arrays:
            return 0

        n_initial = int(self.thermal_model.rank)
        n_geometry = len(self.geometry_names)
        normalized = []
        seen = set()
        for values in arrays:
            if values.ndim != 2:
                raise ValueError("training point sets must be matrices")
            for row in values:
                z = tuple(float(value) for value in row[n_initial:n_initial+n_geometry])
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

        # Prewarm independent geometry contexts concurrently. Collection order
        # is deterministic; only the expensive construction overlaps in time.
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
