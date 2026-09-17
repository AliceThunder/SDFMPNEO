"""Reuse certified terminal scalar fields for dissipative self correction.

The reactive terminal audit already solves the complete balanced full-port scalar
problem on a patch that refines *both* feed and return contacts.  Running two
additional feed/return scalar solves per level duplicates the dominant work.

This adapter disables the older separate terminal-dissipative solves internally
and contracts the two disjoint terminal Joule windows carried by the shared
reactive states.  Reference and validation therefore compare the same scalar
fields used by the reactive Gate.  No Gate is relaxed: every terminal local-loss
defect must still converge within the configured relative tolerance, and the
underlying scalar field must still satisfy its original residual certificate.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy

import numpy as np

from . import unified_terminal_dissipative_defect as _legacy_terminal


_MODEL = "global_balanced_shared_terminal_patch_longitudinal_dissipative_defect_v3"


@contextmanager
def _disable_legacy_terminal_solves(background):
    root = getattr(background, "background_config", None)
    if not isinstance(root, dict):
        yield
        return
    own = root.setdefault("terminal_dissipative_reference", {})
    had = "enabled" in own
    previous = own.get("enabled")
    own["enabled"] = False
    try:
        yield
    finally:
        if had:
            own["enabled"] = previous
        else:
            own.pop("enabled", None)


def _local_values(state):
    values = np.asarray(state.get("terminal_local_d_vol", ()), float).reshape(-1)
    if values.shape != (2,) or np.any(~np.isfinite(values)):
        raise RuntimeError("shared scalar state is missing two terminal local Dvol values")
    return values


def _local_modal(state):
    value = state.get("terminal_local_modal_h")
    if value is None:
        return None
    array = np.asarray(value, float)
    if array.ndim != 2 or array.shape[0] != 2 or np.any(~np.isfinite(array)):
        raise RuntimeError("shared scalar state has invalid terminal modal heat")
    return array


def _relative(reference_delta, validation_delta, reference_state, validation_state, coarse_state, terminal):
    t = int(terminal)
    ref_local = _local_values(reference_state)[t]
    val_local = _local_values(validation_state)[t]
    coarse_local = _local_values(coarse_state)[t]
    scale = max(
        abs(float(reference_delta)),
        abs(float(validation_delta)),
        abs(float(ref_local)),
        abs(float(val_local)),
        abs(float(coarse_local)),
        np.finfo(float).tiny,
    )
    return float(abs(float(reference_delta) - float(validation_delta)) / scale)


def install(module):
    if bool(getattr(module, "_shared_terminal_dissipative_installed", False)):
        return module

    original_correction = module._correction
    original_audit = module.audit_reference_convergence

    def correction(background, geometry, *, phi=None):
        cfg = _legacy_terminal._config(background)
        with _disable_legacy_terminal_solves(background):
            raw = original_correction(background, geometry, phi=phi)
        audit = dict(raw.get("audit", {}))
        audit["terminal_dissipative_reference_model"] = _MODEL
        if not cfg["enabled"]:
            audit["terminal_dissipative_reference_enabled"] = False
            raw["audit"] = audit
            return raw

        ports = list(audit.get("ports", ()))
        n = len(background.coil_materials)
        if len(ports) != n:
            raise RuntimeError("shared terminal dissipative correction is missing reactive port states")
        delta = np.zeros(n, float)
        modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], n), float)
        compact = []
        for p, row in enumerate(ports):
            coarse = row.get("coarse", {})
            fine = row.get("fine", {})
            coarse_local = _local_values(coarse)
            fine_local = _local_values(fine)
            terminal_delta = fine_local - coarse_local
            delta[p] = float(np.sum(terminal_delta))
            terminal_modal = []
            if modal is not None:
                coarse_modal = _local_modal(coarse)
                fine_modal = _local_modal(fine)
                if coarse_modal is None or fine_modal is None:
                    raise RuntimeError("shared terminal dissipative modal correction is missing")
                modal[:, p] = np.sum(fine_modal - coarse_modal, axis=0)
                terminal_modal = (fine_modal - coarse_modal).tolist()
            compact.append({
                "port": int(p),
                "terminal_delta_d_vol": terminal_delta.tolist(),
                "delta_d_vol": float(delta[p]),
                "terminal_delta_modal_h": terminal_modal,
            })

        raw["delta_z"] = np.asarray(raw["delta_z"], complex) + delta.astype(complex)
        raw["delta_d_vol"] = np.asarray(raw["delta_d_vol"], float) + delta
        if modal is not None:
            if raw.get("delta_modal_h") is None:
                raw["delta_modal_h"] = np.zeros_like(modal)
            raw["delta_modal_h"] = np.asarray(raw["delta_modal_h"], float) + modal
        audit.update(
            terminal_dissipative_reference_enabled=True,
            terminal_dissipative_reference_model=_MODEL,
            terminal_dissipative_delta_d_vol=delta.tolist(),
            terminal_dissipative_delta_z_real=delta.tolist(),
            terminal_dissipative_ports=compact,
            terminal_dissipative_semantics=(
                "shared_balanced_full_port_scalar_field_disjoint_terminal_energy_windows"
            ),
            terminal_dissipative_extra_scalar_solves=0,
        )
        raw["audit"] = audit
        return raw

    module._correction = correction

    def audit_reference_convergence(background, geometry):
        cfg = _legacy_terminal._config(background)
        with _disable_legacy_terminal_solves(background):
            report = dict(original_audit(background, geometry))
        if not cfg["enabled"] or not bool(report.get("converged", False)):
            return report

        rows = []
        worst = 0.0
        for row in report.get("samples", []):
            p = int(row.get("port", len(rows)))
            coarse = row.get("coarse", {})
            fine = row.get("fine", {})
            validation = row.get("validation", {})
            coarse_local = _local_values(coarse)
            fine_local = _local_values(fine)
            validation_local = _local_values(validation)
            terminal_rows = []
            ref_total = 0.0
            val_total = 0.0
            for terminal in range(2):
                reference_delta = float(fine_local[terminal] - coarse_local[terminal])
                validation_delta = float(validation_local[terminal] - coarse_local[terminal])
                error = _relative(
                    reference_delta,
                    validation_delta,
                    fine,
                    validation,
                    coarse,
                    terminal,
                )
                worst = max(worst, error)
                ref_total += reference_delta
                val_total += validation_delta
                terminal_rows.append({
                    "terminal": int(terminal),
                    "terminal_name": ("feed", "return")[terminal],
                    "reference_delta_d_vol": reference_delta,
                    "validation_delta_d_vol": validation_delta,
                    "relative_d_vol_defect_error": float(error),
                    "reference_local_d_vol": float(fine_local[terminal]),
                    "validation_local_d_vol": float(validation_local[terminal]),
                    "coarse_local_d_vol": float(coarse_local[terminal]),
                })
                print(
                    "terminal longitudinal dissipative defect Gate: "
                    f"port={p + 1}, terminal={(\"feed\", \"return\")[terminal]}, "
                    f"Dvol={error:.3e}, ref={reference_delta:.6e}, "
                    f"val={validation_delta:.6e} (shared scalar field)",
                    flush=True,
                )
            rows.append({
                "port": p,
                "reference_delta_d_vol": float(ref_total),
                "validation_delta_d_vol": float(val_total),
                "terminals": terminal_rows,
            })

        converged = bool(worst <= float(cfg["relative_tolerance"]))
        report["terminal_dissipative_reference"] = {
            "model": _MODEL,
            "reference_cells_per_support": float(cfg["reference_cells_per_support"]),
            "validation_cells_per_support": float(cfg["validation_cells_per_support"]),
            "relative_tolerance": float(cfg["relative_tolerance"]),
            "maximum_relative_error": float(worst),
            "converged": converged,
            "samples": rows,
            "semantics": "shared_balanced_full_port_scalar_field_disjoint_terminal_energy_windows",
            "extra_scalar_solves": 0,
        }
        report["maximum_terminal_dissipative_relative_error"] = float(worst)
        report["maximum_relative_error"] = max(
            float(report.get("maximum_relative_error", 0.0)), float(worst)
        )
        report["converged"] = bool(report.get("converged", False) and converged)
        report["model"] = f"{report.get('model', module._MODEL)}+{_MODEL}"
        return report

    module.audit_reference_convergence = audit_reference_convergence
    module._MODEL = f"{module._MODEL}+{_MODEL}"
    module._shared_terminal_dissipative_installed = True
    return module


__all__ = ["install"]
