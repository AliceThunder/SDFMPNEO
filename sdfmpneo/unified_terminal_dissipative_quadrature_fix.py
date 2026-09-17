"""Lock terminal dissipative source quadrature across scalar h-refinement.

The terminal dissipative Gate compares two scalar Galerkin spaces.  Both spaces
must integrate the same physical terminal charge measure; otherwise the Gate
mixes discretization error with a simultaneous change of source quadrature.

This adapter keeps production/global source assembly unchanged.  For the
terminal-local dissipative reference only, the reference and validation solves
share the quadrature resolution implied by the validation grid.  The coarse
parent restriction deliberately keeps the production source definition, so the
applied defect still contains the physical coarse-to-resolved source correction.

The adapter also removes the internal ``prepared`` tuple from public audit
payloads.  That tuple contains an OpenBoundaryBackground object and is an
implementation cache, not physics evidence; retaining it made failed preflight
reports non-JSON-serializable.

After the common quadrature rule is installed, the single-terminal component
lock is installed as the outermost source adapter: only the selected feed/return
cloud is reprojected on the refined tensor grid, while the opposite terminal
retains the production coarse nodal load exactly.
"""
from __future__ import annotations

from contextvars import ContextVar

import numpy as np


_MODEL_SUFFIX = "shared_validation_terminal_source_quadrature_v1"
_ACTIVE_RESOLUTION = ContextVar(
    "sdfmpneo_terminal_dissipative_source_quadrature_resolution",
    default=None,
)
_OVERRIDE_ATTR = "_sdfmpneo_source_quadrature_resolution_override"


def _validation_resolution(terminal_defect_module, module, background, prepared, port, terminal):
    _parent_potential, coarse_axes, full_geometry, _coarse_background, _coarse_meta = prepared
    cfg = terminal_defect_module._config(background)
    validation_cells = float(cfg["validation_cells_per_support"])
    axes, _steps, _boxes, _contact, _halo = terminal_defect_module._terminal_axes(
        module,
        background,
        full_geometry,
        int(port),
        int(terminal),
        coarse_axes,
        validation_cells,
    )
    resolution = float(min(np.min(np.diff(np.asarray(axis, float))) for axis in axes))
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("terminal dissipative validation quadrature resolution must be positive")
    return resolution


def install(
    terminal_source_module,
    charge_source_module,
    terminal_defect_module,
    longitudinal_module,
):
    if bool(getattr(terminal_defect_module, "_shared_source_quadrature_fix_installed", False)):
        return terminal_defect_module

    original_quadrature = terminal_source_module._cross_section_quadrature

    def cross_section_quadrature(background, coil):
        override = getattr(background, _OVERRIDE_ATTR, None)
        if override is None:
            return original_quadrature(background, coil)
        resolution = float(override)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("source quadrature resolution override must be positive and finite")
        target = max(
            float(terminal_source_module._QUADRATURE_PANEL_TO_MESH) * resolution,
            np.finfo(float).tiny,
        )
        u, wu, up = terminal_source_module._composite_gauss_1d(
            float(coil.conductor_width), target
        )
        v, wv, vp = terminal_source_module._composite_gauss_1d(
            float(coil.conductor_thickness), target
        )
        return u, wu, v, wv, int(up), int(vp), resolution

    # The terminal source installer resolves this module global dynamically.
    terminal_source_module._cross_section_quadrature = cross_section_quadrature
    # unified_charge_regularized_source imported the helper by value, so update
    # that binding explicitly as well.
    charge_source_module._cross_section_quadrature = cross_section_quadrature

    original_balanced_state = terminal_defect_module._balanced_state

    def balanced_state(*args, **kwargs):
        exact_refined = bool(kwargs.get("exact_refined", False))
        patch = args[2] if len(args) > 2 else kwargs.get("patch")
        resolution = _ACTIVE_RESOLUTION.get()
        if not exact_refined or patch is None or resolution is None:
            return original_balanced_state(*args, **kwargs)

        existed = hasattr(patch, _OVERRIDE_ATTR)
        previous = getattr(patch, _OVERRIDE_ATTR, None)
        setattr(patch, _OVERRIDE_ATTR, float(resolution))
        try:
            result = original_balanced_state(*args, **kwargs)
        finally:
            if existed:
                setattr(patch, _OVERRIDE_ATTR, previous)
            else:
                delattr(patch, _OVERRIDE_ATTR)
        result = dict(result)
        result["source_quadrature_resolution"] = float(resolution)
        result["source_quadrature_locked_to_validation"] = True
        return result

    terminal_defect_module._balanced_state = balanced_state

    original_terminal_reference = terminal_defect_module._terminal_reference

    def terminal_reference(
        module,
        background,
        geometry,
        port,
        terminal,
        *,
        cells_per_support,
        phi=None,
        prepared=None,
    ):
        resolution = None
        token = None
        if prepared is not None:
            resolution = _validation_resolution(
                terminal_defect_module,
                module,
                background,
                prepared,
                int(port),
                int(terminal),
            )
            token = _ACTIVE_RESOLUTION.set(float(resolution))
        try:
            result = original_terminal_reference(
                module,
                background,
                geometry,
                port,
                terminal,
                cells_per_support=cells_per_support,
                phi=phi,
                prepared=prepared,
            )
        finally:
            if token is not None:
                _ACTIVE_RESOLUTION.reset(token)

        result = dict(result)
        # Internal cache state is intentionally not part of the public physics
        # audit and contains a background object that json.dumps cannot encode.
        result.pop("prepared", None)
        if resolution is not None:
            result["source_quadrature_resolution"] = float(resolution)
            result["source_quadrature_semantics"] = (
                "reference_and_validation_share_validation_grid_quadrature"
            )
        return result

    terminal_defect_module._terminal_reference = terminal_reference
    terminal_defect_module._MODEL = f"{terminal_defect_module._MODEL}+{_MODEL_SUFFIX}"
    longitudinal_module._MODEL = f"{longitudinal_module._MODEL}+{_MODEL_SUFFIX}"
    terminal_defect_module._shared_source_quadrature_fix_installed = True

    # Install after the quadrature wrapper so the component-lock context encloses
    # it: selected fine charge uses the common validation quadrature while the
    # opposite terminal remains the coarse production nodal load.
    from .unified_terminal_component_lock import install as install_component_lock

    install_component_lock(terminal_defect_module, longitudinal_module)
    return terminal_defect_module


__all__ = ["install"]
