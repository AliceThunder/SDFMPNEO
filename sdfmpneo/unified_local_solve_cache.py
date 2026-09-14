"""Exact LRU reuse for repeated canonical local-self solves.

Only phi=None solves are cached.  The key contains the canonical intrinsic
coil/package geometry, material laws, frequency, local-domain settings, source
semantics and residual tolerance.  Global translation/rotation and the parent
coarse mesh are deliberately absent because _local_background removes them and
they do not enter the canonical local operator.

This is exact memoization, not a surrogate or rounded-parameter approximation.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json

import numpy as np


_CACHE = OrderedDict()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _cache_key(module, parent, geometry, port, fine_step):
    _, local = module._canonical_port_geometry(geometry, port)
    p = int(port)
    cfg = module._config(parent)
    local_cfg = {
        key: cfg.get(key)
        for key in (
            "core_padding",
            "boundary_padding",
            "growth",
            "max_step",
            "linear_relative_residual_tolerance",
        )
    }
    payload = {
        "fine_step": float(fine_step),
        "frequency_hz": float(parent.frequency_hz),
        "ambient_temperature": float(parent.ambient_temperature),
        "coil": local.coils[0].to_mapping(),
        "package": local.packages[0].to_mapping(),
        "coil_material": parent.materials[parent.coil_materials[p]],
        "package_material": parent.materials[parent.package_materials[p]],
        "seawater_material": parent.materials[parent.seawater_material],
        "local_config": local_cfg,
        "source_model": getattr(parent, "source_model", "unknown"),
        "terminal_model": getattr(parent, "terminal_model", "unknown"),
        "transverse_source_model": getattr(parent, "transverse_source_model", "none"),
        "source_centerline_step": float(getattr(parent, "source_centerline_step", 0.0)),
    }
    text = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def install(self_correction_module):
    if bool(getattr(self_correction_module, "_exact_local_solve_cache_installed", False)):
        return self_correction_module
    original = self_correction_module._solve_local

    def cached_solve_local(parent, geometry, port, fine_step, phi=None):
        # Modal contractions depend on the geometry-specific thermal basis and
        # therefore are intentionally left to the physical solver path.
        if phi is not None:
            return original(parent, geometry, port, fine_step, phi=phi)

        key = _cache_key(self_correction_module, parent, geometry, port, fine_step)
        if key in _CACHE:
            _CACHE.move_to_end(key)
            result = copy.deepcopy(_CACHE[key])
            result["linear_solver_cache_hit"] = True
            print(
                f"local Maxwell exact cache hit: port={int(port)+1}, step={float(fine_step):g}m, "
                f"edges={int(result.get('n_edges', 0))}",
                flush=True,
            )
            return result

        result = original(parent, geometry, port, fine_step, phi=None)
        stored = copy.deepcopy(result)
        stored["linear_solver_cache_hit"] = False
        _CACHE[key] = stored
        _CACHE.move_to_end(key)
        limit = int(self_correction_module._config(parent).get("linear_result_cache_size", 64))
        if limit < 1:
            raise ValueError("self_correction.linear_result_cache_size must be positive")
        while len(_CACHE) > limit:
            _CACHE.popitem(last=False)
        return copy.deepcopy(stored)

    self_correction_module._solve_local = cached_solve_local
    self_correction_module._exact_local_solve_cache_installed = True
    return self_correction_module


__all__ = ["install"]
