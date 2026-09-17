"""Install single-terminal source locking for dissipative reference patches.

The terminal dissipative mesh is a tensor-product Cartesian grid.  Refining one
terminal inserts coordinate planes across the whole patch, so blindly rebuilding
q_target redistributes both feed and return nodal clouds.  This adapter makes the
additive defect literal: only the selected terminal component is reprojected on
the refined grid; the opposite component is embedded exactly from the coarse
production patch.
"""
from __future__ import annotations

from contextvars import ContextVar

from .unified_terminal_component_charge import terminal_component_locked_target


_ACTIVE = ContextVar("sdfmpneo_selected_terminal_component_lock", default=None)
_MODEL_SUFFIX = "single_terminal_component_locked_source_v1"


def install(terminal_defect_module, longitudinal_module):
    if bool(getattr(terminal_defect_module, "_terminal_component_lock_installed", False)):
        return terminal_defect_module

    # Import the fast adapter module, not merely its install function: it imported
    # terminal_charge_target by value and therefore needs its binding updated too.
    from . import unified_fast_terminal_dissipative_scalar as fast_terminal

    original_reference = terminal_defect_module._terminal_reference
    original_core_target = terminal_defect_module.terminal_charge_target
    original_fast_target = fast_terminal.terminal_charge_target

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
        token = None
        if prepared is not None:
            # prepared = (parent_potential, coarse_axes, full_geometry,
            #             coarse_background, coarse_meta)
            coarse_background = prepared[3]
            token = _ACTIVE.set((coarse_background, int(terminal)))
        try:
            return original_reference(
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
                _ACTIVE.reset(token)

    def locked_target(background, coil):
        active = _ACTIVE.get()
        if active is None:
            return original_core_target(background, coil)
        coarse_background, terminal = active
        # Coarse parent-restriction calls must remain exactly production-defined.
        if background is coarse_background:
            return original_core_target(background, coil)
        return terminal_component_locked_target(
            coarse_background,
            background,
            coil,
            int(terminal),
        )

    terminal_defect_module._terminal_reference = terminal_reference
    terminal_defect_module.terminal_charge_target = locked_target
    fast_terminal.terminal_charge_target = locked_target

    original_balanced = terminal_defect_module._balanced_state

    def balanced_state(*args, **kwargs):
        result = original_balanced(*args, **kwargs)
        active = _ACTIVE.get()
        if active is None or not bool(kwargs.get("exact_refined", False)):
            return result
        _coarse_background, terminal = active
        out = dict(result)
        out["terminal_component_source_locked"] = True
        out["selected_terminal_component"] = int(terminal)
        out["selected_terminal_name"] = ("feed", "return")[int(terminal)]
        out["opposite_terminal_discretization"] = "production_coarse_nodal_load"
        return out

    terminal_defect_module._balanced_state = balanced_state
    terminal_defect_module._MODEL = f"{terminal_defect_module._MODEL}+{_MODEL_SUFFIX}"
    longitudinal_module._MODEL = f"{longitudinal_module._MODEL}+{_MODEL_SUFFIX}"
    terminal_defect_module._terminal_component_lock_installed = True
    return terminal_defect_module


__all__ = ["install"]
