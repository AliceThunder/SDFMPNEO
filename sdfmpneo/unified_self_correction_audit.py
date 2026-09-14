"""Independent convergence audit for the canonical local self correction."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .unified_self_correction import _config, _parent_fine_step, _solve_local


class _AuditParentView:
    """Share immutable background data but keep a private warm state per port."""

    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_parent"), name)

    def __setattr__(self, name, value):
        if name.startswith("_local_self_"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_parent"), name, value)


def _relative(a, b):
    return float(abs(a - b) / max(abs(b), np.finfo(float).tiny))


def audit_local_self_correction(background, geometries, monitor=None):
    cfg = _config(background)
    tolerance = float(cfg.get("relative_tolerance", 1e-1))
    joule_tolerance = float(cfg.get("joule_identity_tolerance", 1e-10))
    linear_tolerance = float(cfg.get("linear_relative_residual_tolerance", 1e-9))
    fine = float(cfg["fine_step"])
    validation = float(cfg.get("validation_fine_step", 0.75 * fine))
    if not 0.0 < validation < fine:
        raise ValueError("self_correction.validation_fine_step must be smaller than fine_step")
    if joule_tolerance <= 0.0:
        raise ValueError("self_correction.joule_identity_tolerance must be positive")
    if linear_tolerance <= 0.0:
        raise ValueError("self_correction.linear_relative_residual_tolerance must be positive")
    geometries = list(geometries)
    rows = []
    n_ports = len(background.coil_materials)
    workers = min(
        n_ports,
        max(1, int(cfg.get("parallel_preflight_ports", min(2, n_ports)))),
    )

    for gi, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()

        def solve_port(port):
            parent = _AuditParentView(background)
            seed_step = float(_parent_fine_step(parent))
            if bool(cfg.get("linear_warm_start_from_parent", True)) and seed_step > fine:
                print(
                    f"local self warm start……geometry {gi+1}/{len(geometries)} "
                    f"port {port+1}/{n_ports}  {seed_step:g}m -> {fine:g}m",
                    flush=True,
                )
                _solve_local(parent, geometry, port, seed_step, phi=None)

            print(
                f"local self correction convergence……geometry {gi+1}/{len(geometries)} "
                f"port {port+1}/{n_ports}  {fine:g}m -> {validation:g}m",
                flush=True,
            )
            a = _solve_local(parent, geometry, port, fine, phi=None)
            b = _solve_local(parent, geometry, port, validation, phi=None)
            dissipation_scale = max(
                abs(float(np.real(b["z"]))),
                abs(float(b["d_vol"] + b["d_out"])),
                np.finfo(float).tiny,
            )
            outward_significance = float(abs(a["d_out"] - b["d_out"]) / dissipation_scale)
            linear_residual = max(
                float(a.get("linear_relative_residual", np.inf)),
                float(b.get("linear_relative_residual", np.inf)),
            )
            linear_converged = bool(
                a.get("linear_solver_converged", float(a.get("linear_relative_residual", np.inf)) <= linear_tolerance)
                and b.get("linear_solver_converged", float(b.get("linear_relative_residual", np.inf)) <= linear_tolerance)
                and linear_residual <= linear_tolerance
            )
            row = {
                "port": int(port),
                "warm_start_seed_step": seed_step if seed_step > fine else None,
                "relative_z_error": _relative(a["z"], b["z"]),
                "relative_resistive_error": _relative(a["z"].real, b["z"].real),
                "relative_reactive_error": _relative(a["z"].imag, b["z"].imag),
                "relative_d_vol_error": _relative(a["d_vol"], b["d_vol"]),
                "relative_outward_partition_significance": outward_significance,
                "raw_relative_d_out_error": _relative(a["d_out"], b["d_out"]),
                "maximum_joule_total_power_relative_error": max(
                    float(a["joule_total_power_relative_error"]),
                    float(b["joule_total_power_relative_error"]),
                ),
                "maximum_linear_relative_residual": linear_residual,
                "linear_solver_converged": linear_converged,
                "fine": {k: v for k, v in a.items() if k != "modal_h"},
                "validation": {k: v for k, v in b.items() if k != "modal_h"},
            }
            row["maximum_relative_error"] = max(
                row["relative_z_error"],
                row["relative_resistive_error"],
                row["relative_reactive_error"],
                row["relative_d_vol_error"],
                row["relative_outward_partition_significance"],
            )
            return row

        if workers > 1 and n_ports > 1:
            print(f"local self convergence audit: parallel ports={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="local-self-audit") as pool:
                ports = list(pool.map(solve_port, range(n_ports)))
        else:
            ports = [solve_port(port) for port in range(n_ports)]
        ports.sort(key=lambda row: row["port"])
        rows.append({"geometry_index": int(gi), "geometry": geometry, "ports": ports})

    worst = max(
        (port["maximum_relative_error"] for row in rows for port in row["ports"]),
        default=float("inf"),
    )
    worst_joule = max(
        (port["maximum_joule_total_power_relative_error"] for row in rows for port in row["ports"]),
        default=float("inf"),
    )
    worst_linear = max(
        (port["maximum_linear_relative_residual"] for row in rows for port in row["ports"]),
        default=float("inf"),
    )
    all_linear = bool(
        rows
        and all(port["linear_solver_converged"] for row in rows for port in row["ports"])
    )
    return {
        "sample_count": len(rows),
        "fine_step": fine,
        "validation_fine_step": validation,
        "relative_tolerance": tolerance,
        "joule_identity_tolerance": joule_tolerance,
        "linear_relative_residual_tolerance": linear_tolerance,
        "parallel_ports": int(workers),
        "maximum_relative_error": float(worst),
        "maximum_joule_total_power_relative_error": float(worst_joule),
        "maximum_linear_relative_residual": float(worst_linear),
        "linear_solver_converged": all_linear,
        "loss_partition_semantics": "D_out_is_scaled_by_total_local_self_dissipation",
        "converged": bool(
            rows
            and worst <= tolerance
            and worst_joule <= joule_tolerance
            and worst_linear <= linear_tolerance
            and all_linear
        ),
        "samples": rows,
    }


__all__ = ["audit_local_self_correction"]
