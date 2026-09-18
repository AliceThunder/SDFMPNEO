"""Keep the production terminal dissipative correction fixed during mesh Gates.

The terminal-local dissipative correction is part of the production truth model.
A global 12->9 mm mesh-convergence audit must therefore compare the same subgrid
model on both Maxwell grids.  Recomputing the terminal correction on the 9 mm
validation background changes the model being compared: the converged local
reference itself depends on the parent boundary/material discretization.

This adapter freezes only the terminal dissipative diagonal contribution from the
production/base background while a mesh Gate is active.  Reactive longitudinal
and transverse/cross corrections continue to be evaluated on each Maxwell grid.
Production truth generation outside mesh audits is unchanged.
"""
from __future__ import annotations

from contextvars import ContextVar
import copy
import numpy as np


_PREFLIGHT_STATE = ContextVar("sdfmpneo_mesh_gate_terminal_dissipative_preflight", default=None)
_POSTBASIS_STATE = ContextVar("sdfmpneo_mesh_gate_terminal_dissipative_postbasis", default=None)


def _geometry_key(longitudinal_module, geometry):
    helper = getattr(longitudinal_module, "_geometry_key", None)
    return helper(geometry) if callable(helper) else repr(geometry)


def _terminal_delta(global_audit):
    if not isinstance(global_audit, dict):
        return None
    value = global_audit.get("terminal_dissipative_delta_d_vol")
    if value is None:
        return None
    out = np.asarray(value, float).reshape(-1)
    return out if out.size else None


def _terminal_modal(global_audit, rank, nports):
    rows = global_audit.get("terminal_dissipative_ports") if isinstance(global_audit, dict) else None
    if not isinstance(rows, list):
        return None
    out = np.zeros((int(rank), int(nports)), float)
    found = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        p = int(row.get("port", -1))
        if p < 0 or p >= int(nports):
            continue
        for terminal in row.get("terminals", ()):
            if not isinstance(terminal, dict):
                continue
            value = terminal.get("delta_modal_h")
            if value is None:
                continue
            vec = np.asarray(value, float).reshape(-1)
            if vec.shape != (int(rank),):
                raise ValueError("terminal dissipative modal correction rank mismatch")
            out[:, p] += vec
            found = True
    return out if found else None


def _mark_audit(global_audit, *, original_delta, frozen_delta, source_step):
    out = copy.deepcopy(global_audit)
    out["mesh_gate_terminal_dissipative_original_delta_d_vol"] = (
        np.asarray(original_delta, float).tolist()
    )
    out["terminal_dissipative_delta_d_vol"] = np.asarray(frozen_delta, float).tolist()
    out["terminal_dissipative_delta_z_real"] = np.asarray(frozen_delta, float).tolist()
    out["terminal_dissipative_mesh_gate_frozen_to_production"] = True
    out["terminal_dissipative_mesh_gate_source_fine_step"] = float(source_step)
    return out


def _apply_diagonal_shift(z, d, shift):
    zc = np.asarray(z, complex).copy()
    dc = np.asarray(d, complex).copy()
    value = np.asarray(shift, float).reshape(-1)
    if zc.shape[0] != zc.shape[1] or dc.shape != zc.shape or value.size != zc.shape[0]:
        raise ValueError("terminal dissipative mesh-Gate diagonal shift shape mismatch")
    idx = np.arange(value.size)
    zc[idx, idx] += value
    dc[idx, idx] += value
    return zc, dc


def install(preflight_module, physics_gate_module, longitudinal_module):
    if bool(getattr(preflight_module, "_frozen_terminal_dissipative_mesh_gate_installed", False)):
        return preflight_module

    # Pre-basis: wrap the already-installed global-longitudinal correction.
    original_preflight_correct = preflight_module._correct
    original_preflight_mesh = preflight_module.audit_em_mesh_preflight

    def preflight_correct(background, geometry, z, d, d_out):
        zc, dc, oc, audit = original_preflight_correct(background, geometry, z, d, d_out)
        state = _PREFLIGHT_STATE.get()
        if state is None:
            return zc, dc, oc, audit
        global_audit = audit.get("global_longitudinal_reference") if isinstance(audit, dict) else None
        current = _terminal_delta(global_audit)
        if current is None:
            return zc, dc, oc, audit

        key = _geometry_key(longitudinal_module, geometry)
        step = float(longitudinal_module._background_step(background))
        base_step = float(state["base_step"])
        cache = state["cache"]
        if abs(step - base_step) <= 1e-12 * max(abs(base_step), 1.0):
            cache[key] = np.asarray(current, float).copy()
            marked = _mark_audit(
                global_audit,
                original_delta=current,
                frozen_delta=current,
                source_step=base_step,
            )
            out = dict(audit)
            out["global_longitudinal_reference"] = marked
            return zc, dc, oc, out

        base = cache.get(key)
        if base is None:
            raise RuntimeError("refined mesh Gate requested before production terminal correction")
        shift = np.asarray(base, float) - np.asarray(current, float)
        zc, dc = _apply_diagonal_shift(zc, dc, shift)
        out = dict(audit)
        out["global_longitudinal_reference"] = _mark_audit(
            global_audit,
            original_delta=current,
            frozen_delta=base,
            source_step=base_step,
        )
        return zc, dc, oc, out

    def preflight_mesh(settings, background, geometries, monitor=None):
        state = {
            "base_step": float(longitudinal_module._background_step(background)),
            "cache": {},
        }
        token = _PREFLIGHT_STATE.set(state)
        try:
            return original_preflight_mesh(settings, background, geometries, monitor)
        finally:
            _PREFLIGHT_STATE.reset(token)

    preflight_module._correct = preflight_correct
    preflight_module.audit_em_mesh_preflight = preflight_mesh

    # Post-basis: freeze both D/Z diagonal correction and its modal Joule term.
    original_fields = physics_gate_module._corrected_fields
    original_mesh = physics_gate_module.audit_mesh_convergence

    def corrected_fields(background, geometry, phi=None):
        context, z, d, d_out, modal, audit = original_fields(background, geometry, phi)
        state = _POSTBASIS_STATE.get()
        if state is None:
            return context, z, d, d_out, modal, audit
        global_audit = audit.get("global_longitudinal_reference") if isinstance(audit, dict) else None
        current = _terminal_delta(global_audit)
        if current is None:
            return context, z, d, d_out, modal, audit

        key = _geometry_key(longitudinal_module, geometry)
        step = float(longitudinal_module._background_step(background))
        base_step = float(state["base_step"])
        cache = state["cache"]
        current_modal = None
        if modal is not None:
            current_modal = _terminal_modal(global_audit, np.asarray(modal).shape[0], len(current))

        if abs(step - base_step) <= 1e-12 * max(abs(base_step), 1.0):
            cache[key] = (
                np.asarray(current, float).copy(),
                None if current_modal is None else np.asarray(current_modal, float).copy(),
            )
            out = dict(audit)
            out["global_longitudinal_reference"] = _mark_audit(
                global_audit,
                original_delta=current,
                frozen_delta=current,
                source_step=base_step,
            )
            return context, z, d, d_out, modal, out

        cached = cache.get(key)
        if cached is None:
            raise RuntimeError("refined post-basis mesh Gate requested before production correction")
        base_delta, base_modal = cached
        shift = np.asarray(base_delta, float) - np.asarray(current, float)
        z, d = _apply_diagonal_shift(z, d, shift)
        if modal is not None and base_modal is not None and current_modal is not None:
            hm = np.asarray(modal, complex).copy()
            delta_h = np.asarray(base_modal, float) - np.asarray(current_modal, float)
            for p in range(delta_h.shape[1]):
                hm[:, p, p] += delta_h[:, p]
            modal = hm
        out = dict(audit)
        out["global_longitudinal_reference"] = _mark_audit(
            global_audit,
            original_delta=current,
            frozen_delta=base_delta,
            source_step=base_step,
        )
        return context, z, d, d_out, modal, out

    def postbasis_mesh(settings, background, geometries, monitor=None):
        state = {
            "base_step": float(longitudinal_module._background_step(background)),
            "cache": {},
        }
        token = _POSTBASIS_STATE.set(state)
        try:
            return original_mesh(settings, background, geometries, monitor)
        finally:
            _POSTBASIS_STATE.reset(token)

    physics_gate_module._corrected_fields = corrected_fields
    physics_gate_module.audit_mesh_convergence = postbasis_mesh
    preflight_module._frozen_terminal_dissipative_mesh_gate_installed = True
    physics_gate_module._frozen_terminal_dissipative_mesh_gate_installed = True
    return preflight_module


__all__ = ["_apply_diagonal_shift", "_terminal_delta", "_terminal_modal", "install"]
