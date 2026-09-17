"""Reuse scalar reference states already certified during preflight.

The early longitudinal audit and the later correction request the same phi=None
scalar fields.  Their public state dictionaries contain only compact diagnostics,
so caching them on the background avoids repeated factorization without retaining
large matrices or field arrays.  Modal/thermal calls (phi is not None) are never
served from this cache.
"""
from __future__ import annotations

import copy


def _geometry_key(module, geometry):
    helper = getattr(module, "_geometry_key", None)
    return helper(geometry) if callable(helper) else repr(geometry)


def install(module, terminal_defect_module):
    if bool(getattr(module, "_longitudinal_state_cache_installed", False)):
        return module

    original_port_states = module._port_states

    def port_states(
        background,
        geometry,
        port,
        global_potential,
        target_steps,
        *,
        phi=None,
    ):
        if phi is not None:
            return original_port_states(
                background,
                geometry,
                port,
                global_potential,
                target_steps,
                phi=phi,
            )
        cache = getattr(background, "_sdfmpneo_longitudinal_port_state_cache", None)
        if cache is None:
            cache = {}
            background._sdfmpneo_longitudinal_port_state_cache = cache
        base = (_geometry_key(module, geometry), int(port))
        requested = tuple(float(step) for step in target_steps)
        complete = all((base, step) in cache for step in requested) and (base, "coarse") in cache
        if complete:
            out = {"coarse": copy.deepcopy(cache[(base, "coarse")])}
            for step in requested:
                out[step] = copy.deepcopy(cache[(base, step)])
            return out

        states = original_port_states(
            background,
            geometry,
            port,
            global_potential,
            target_steps,
            phi=None,
        )
        cache[(base, "coarse")] = copy.deepcopy(states["coarse"])
        for step in requested:
            cache[(base, step)] = copy.deepcopy(states[step])
        return copy.deepcopy(states)

    module._port_states = port_states

    original_terminal_reference = terminal_defect_module._terminal_reference

    def terminal_reference(
        module_arg,
        background,
        geometry,
        port,
        terminal,
        *,
        cells_per_support,
        phi=None,
        prepared=None,
    ):
        if phi is not None:
            return original_terminal_reference(
                module_arg,
                background,
                geometry,
                port,
                terminal,
                cells_per_support=cells_per_support,
                phi=phi,
                prepared=prepared,
            )
        cache = getattr(background, "_sdfmpneo_terminal_dissipative_state_cache", None)
        if cache is None:
            cache = {}
            background._sdfmpneo_terminal_dissipative_state_cache = cache
        key = (
            _geometry_key(module, geometry),
            int(port),
            int(terminal),
            float(cells_per_support),
        )
        if key not in cache:
            cache[key] = copy.deepcopy(
                original_terminal_reference(
                    module_arg,
                    background,
                    geometry,
                    port,
                    terminal,
                    cells_per_support=cells_per_support,
                    phi=None,
                    prepared=prepared,
                )
            )
        return copy.deepcopy(cache[key])

    terminal_defect_module._terminal_reference = terminal_reference
    module._longitudinal_state_cache_installed = True
    return module


__all__ = ["install"]
