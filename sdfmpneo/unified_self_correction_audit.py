"""Independent convergence audit for the canonical local self correction."""
from __future__ import annotations

import numpy as np

from .unified_self_correction import _config, _solve_local


def _relative(a, b):
    return float(abs(a - b) / max(abs(b), np.finfo(float).tiny))


def audit_local_self_correction(background, geometries, monitor=None):
    cfg = _config(background)
    tolerance = float(cfg.get("relative_tolerance", 1e-1))
    fine = float(cfg["fine_step"])
    validation = float(cfg.get("validation_fine_step", 0.75 * fine))
    if not 0.0 < validation < fine:
        raise ValueError("self_correction.validation_fine_step must be smaller than fine_step")
    geometries = list(geometries)
    rows = []
    for gi, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        ports = []
        for port in range(len(background.coil_materials)):
            print(
                f"local self correction convergence……geometry {gi+1}/{len(geometries)} "
                f"port {port+1}/{len(background.coil_materials)}  {fine:g}m -> {validation:g}m",
                flush=True,
            )
            a = _solve_local(background, geometry, port, fine, phi=None)
            b = _solve_local(background, geometry, port, validation, phi=None)
            row = {
                "port": int(port),
                "relative_z_error": _relative(a["z"], b["z"]),
                "relative_resistive_error": _relative(a["z"].real, b["z"].real),
                "relative_reactive_error": _relative(a["z"].imag, b["z"].imag),
                "relative_d_vol_error": _relative(a["d_vol"], b["d_vol"]),
                "relative_d_out_error": _relative(a["d_out"], b["d_out"]),
                "fine": {k: v for k, v in a.items() if k != "modal_h"},
                "validation": {k: v for k, v in b.items() if k != "modal_h"},
            }
            row["maximum_relative_error"] = max(
                row["relative_z_error"],
                row["relative_resistive_error"],
                row["relative_reactive_error"],
                row["relative_d_vol_error"],
                row["relative_d_out_error"],
            )
            ports.append(row)
        rows.append({"geometry_index": int(gi), "geometry": geometry, "ports": ports})
    worst = max(
        (port["maximum_relative_error"] for row in rows for port in row["ports"]),
        default=float("inf"),
    )
    return {
        "sample_count": len(rows),
        "fine_step": fine,
        "validation_fine_step": validation,
        "relative_tolerance": tolerance,
        "maximum_relative_error": float(worst),
        "converged": bool(rows and worst <= tolerance),
        "samples": rows,
    }


__all__ = ["audit_local_self_correction"]
