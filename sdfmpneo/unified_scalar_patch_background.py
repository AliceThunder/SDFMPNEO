"""Lightweight backgrounds for refined longitudinal scalar references.

A scalar patch needs cells, edge incidence/storage and edge-cell mass weights, but
never the Maxwell curl/face Hodge or cell-field reconstruction.  Constructing
those Maxwell-only structures on 0.1--0.5 million-cell terminal patches costs far
more than the scalar solve itself.  This adapter preserves the inherited
OpenBoundaryBackground geometry/source/material methods while skipping only the
unused topology for genuinely refined scalar patches.  Coarse parent-consistency
patches continue to use the historical full background byte-for-byte.
"""
from __future__ import annotations

import time
import numpy as np


_CLASS_CACHE = {}


def _scalar_patch_class(base_cls):
    cached = _CLASS_CACHE.get(base_cls)
    if cached is not None:
        return cached

    class ScalarPatchBackground(base_cls):
        def __init__(
            self,
            x,
            y,
            z,
            *,
            frequency_hz,
            materials,
            coil_materials,
            package_materials,
            seawater_material,
            thermal_basis=None,
            thermal_library=None,
            ambient_temperature=293.15,
            **_ignored,
        ):
            started = time.perf_counter()
            self.x = np.asarray(x, float)
            self.y = np.asarray(y, float)
            self.z = np.asarray(z, float)
            for axis in (self.x, self.y, self.z):
                if axis.ndim != 1 or len(axis) < 3 or np.any(np.diff(axis) <= 0):
                    raise ValueError("scalar patch axes must be strictly increasing")
            self.dx = np.diff(self.x)
            self.dy = np.diff(self.y)
            self.dz = np.diff(self.z)
            self.nx, self.ny, self.nz = len(self.dx), len(self.dy), len(self.dz)
            self.frequency_hz = float(frequency_hz)
            self.omega = 2.0 * np.pi * self.frequency_hz
            self.ambient_temperature = float(ambient_temperature)
            self.materials = {str(k): dict(v) for k, v in materials.items()}
            self.coil_materials = tuple(map(str, coil_materials))
            self.package_materials = tuple(map(str, package_materials))
            self.seawater_material = str(seawater_material)
            if len(self.coil_materials) != len(self.package_materials):
                raise ValueError("coil/package material lists must match")
            required = set(self.coil_materials + self.package_materials + (self.seawater_material,))
            if required - set(self.materials):
                raise ValueError("scalar patch region materials are incomplete")

            self._build_cells()
            cells_seconds = float(time.perf_counter() - started)
            edge_started = time.perf_counter()
            self._build_edges()
            edge_seconds = float(time.perf_counter() - edge_started)
            self.thermal_basis = None
            self.thermal_library = None
            self._sdfmpneo_scalar_only_topology = True
            self._sdfmpneo_scalar_topology_seconds = {
                "cells": cells_seconds,
                "edges": edge_seconds,
            }
            if thermal_basis is not None:
                self.set_thermal_basis(thermal_basis)
            if thermal_library is not None:
                self.set_thermal_library(thermal_library)
            print(
                "longitudinal scalar topology: "
                f"cells={self.n_cells}, edges={self.n_edges}, "
                f"cell_build={cells_seconds:.1f}s, edge_build={edge_seconds:.1f}s, "
                "skipped=curl+face_hodge+reconstruction",
                flush=True,
            )

    ScalarPatchBackground.__name__ = f"ScalarPatch{base_cls.__name__}"
    _CLASS_CACHE[base_cls] = ScalarPatchBackground
    return ScalarPatchBackground


def _background_step(background):
    cfg = getattr(background, "background_config", None)
    if isinstance(cfg, dict) and "fine_step" in cfg:
        return float(cfg["fine_step"])
    return float(min(np.min(background.dx), np.min(background.dy), np.min(background.dz)))


def _make_lightweight(parent, axes, *, fine_step):
    cls = _scalar_patch_class(parent.__class__)
    patch = cls(
        *axes,
        frequency_hz=parent.frequency_hz,
        materials=parent.materials,
        coil_materials=tuple(parent.coil_materials),
        package_materials=tuple(parent.package_materials),
        seawater_material=parent.seawater_material,
        ambient_temperature=parent.ambient_temperature,
    )
    patch.background_config = {
        "fine_step": float(fine_step),
        "self_correction": {"enabled": False},
    }
    patch.self_correction_config = {"enabled": False}
    return patch


def install(consistency_module):
    if bool(getattr(consistency_module, "_lightweight_scalar_patch_installed", False)):
        return consistency_module
    original = consistency_module._make_full_patch_background

    def make_full_patch_background(module, parent, axes, *, fine_step):
        parent_step = _background_step(parent)
        if float(fine_step) >= parent_step * (1.0 - 1e-12):
            return original(module, parent, axes, fine_step=fine_step)
        return _make_lightweight(parent, axes, fine_step=fine_step)

    consistency_module._make_full_patch_background = make_full_patch_background
    consistency_module._lightweight_scalar_patch_installed = True
    return consistency_module


__all__ = ["install"]
