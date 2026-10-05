"""Independent convergence audit for the canonical localized self correction."""
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


def _scaled_difference(a, b, *scales):
    denominator = max(
        *(abs(complex(value)) for value in scales),
        np.finfo(float).tiny,
    )
    return float(abs(complex(a) - complex(b)) / denominator)


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
    validation_workers = min(
        n_ports,
        max(1, int(cfg.get("parallel_validation_ports", 1))),
    )

    for gi, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()

        def solve_fine(port):
            parent = _AuditParentView(background)
            seed_step = float(_parent_fine_step(parent))
            if not seed_step > fine:
                raise ValueError("local self audit requires parent coarse step > fine step")
            print(
                f"local self warm start……geometry {gi+1}/{len(geometries)} "
                f"port {port+1}/{n_ports}  {seed_step:g}m -> {fine:g}m",
                flush=True,
            )
            seed = _solve_local(parent, geometry, port, seed_step, phi=None)

            print(
                f"local self correction fine solve……geometry {gi+1}/{len(geometries)} "
                f"port {port+1}/{n_ports}  step={fine:g}m",
                flush=True,
            )
            a = _solve_local(parent, geometry, port, fine, phi=None)
            return int(port), parent, seed_step, seed, a

        if workers > 1 and n_ports > 1:
            print(f"local self convergence audit: parallel fine ports={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="local-self-fine") as pool:
                fine_states = list(pool.map(solve_fine, range(n_ports)))
        else:
            fine_states = [solve_fine(port) for port in range(n_ports)]
        fine_states.sort(key=lambda item: item[0])

        def solve_validation(item):
            port, parent, seed_step, seed, a = item
            print(
                f"local self correction validation……geometry {gi+1}/{len(geometries)} "
                f"port {port+1}/{n_ports}  {fine:g}m -> {validation:g}m",
                flush=True,
            )
            b = _solve_local(parent, geometry, port, validation, phi=None)

            # The production object is a fine-minus-coarse *localized defect*.
            # Audit exactly that object using one shared coarse seed rather than
            # comparing raw local-box terminal responses, whose pure longitudinal
            # component is intentionally not part of the correction.
            dz_f = complex(a["localized_z"] - seed["localized_z"])
            dz_v = complex(b["localized_z"] - seed["localized_z"])
            dd_f = float(a["localized_d_vol"] - seed["localized_d_vol"])
            dd_v = float(b["localized_d_vol"] - seed["localized_d_vol"])
            do_f = float(a["localized_d_out"] - seed["localized_d_out"])
            do_v = float(b["localized_d_out"] - seed["localized_d_out"])

            dissipation_scale = max(
                abs(float(np.real(b["localized_z"]))),
                abs(float(b["localized_d_vol"] + b["localized_d_out"])),
                abs(dd_v) + abs(do_v),
                np.finfo(float).tiny,
            )
            z_scale = max(abs(dz_v), abs(complex(b["localized_z"])), dissipation_scale)
            reactive_scale = max(
                abs(float(dz_v.imag)),
                abs(float(np.imag(b["localized_z"]))),
                dissipation_scale,
            )
            dvol_scale = max(
                abs(dd_v),
                abs(float(b["localized_d_vol"])),
                dissipation_scale,
            )

            relative_z_error = _scaled_difference(dz_f, dz_v, z_scale)
            relative_resistive_error = _scaled_difference(
                dz_f.real, dz_v.real, dissipation_scale, abs(dz_v.real)
            )
            relative_reactive_error = _scaled_difference(
                dz_f.imag, dz_v.imag, reactive_scale
            )
            relative_d_vol_error = _scaled_difference(dd_f, dd_v, dvol_scale)
            outward_significance = float(abs(do_f - do_v) / dissipation_scale)

            linear_residual = max(
                float(seed.get("linear_relative_residual", np.inf)),
                float(a.get("linear_relative_residual", np.inf)),
                float(b.get("linear_relative_residual", np.inf)),
            )
            linear_converged = bool(
                seed.get("linear_solver_converged", float(seed.get("linear_relative_residual", np.inf)) <= linear_tolerance)
                and a.get("linear_solver_converged", float(a.get("linear_relative_residual", np.inf)) <= linear_tolerance)
                and b.get("linear_solver_converged", float(b.get("linear_relative_residual", np.inf)) <= linear_tolerance)
                and linear_residual <= linear_tolerance
            )
            row = {
                "port": int(port),
                "warm_start_seed_step": seed_step,
                "defect_model": "localized_transverse_fine_minus_coarse_v2",
                "fine_delta_z": dz_f,
                "validation_delta_z": dz_v,
                "fine_delta_d_vol": dd_f,
                "validation_delta_d_vol": dd_v,
                "fine_delta_d_out": do_f,
                "validation_delta_d_out": do_v,
                "relative_z_error": relative_z_error,
                "relative_resistive_error": relative_resistive_error,
                "relative_reactive_error": relative_reactive_error,
                "relative_d_vol_error": relative_d_vol_error,
                "relative_outward_partition_significance": outward_significance,
                "raw_relative_d_out_error": _scaled_difference(
                    a["d_out"], b["d_out"], abs(float(b["d_out"]))
                ),
                "maximum_localized_power_balance_relative_error": max(
                    float(seed["localized_power_balance_relative_error"]),
                    float(a["localized_power_balance_relative_error"]),
                    float(b["localized_power_balance_relative_error"]),
                ),
                "maximum_joule_total_power_relative_error": max(
                    float(seed["joule_total_power_relative_error"]),
                    float(a["joule_total_power_relative_error"]),
                    float(b["joule_total_power_relative_error"]),
                ),
                "maximum_linear_relative_residual": linear_residual,
                "linear_solver_converged": linear_converged,
                "coarse": {
                    k: v
                    for k, v in seed.items()
                    if k not in (
                        "modal_h",
                        "localized_modal_h",
                        "localized_heat_cells",
                        "local_cell_centers",
                    )
                },
                "fine": {
                    k: v
                    for k, v in a.items()
                    if k not in (
                        "modal_h",
                        "localized_modal_h",
                        "localized_heat_cells",
                        "local_cell_centers",
                    )
                },
                "validation": {
                    k: v
                    for k, v in b.items()
                    if k not in (
                        "modal_h",
                        "localized_modal_h",
                        "localized_heat_cells",
                        "local_cell_centers",
                    )
                },
            }
            row["maximum_relative_error"] = max(
                row["relative_z_error"],
                row["relative_resistive_error"],
                row["relative_reactive_error"],
                row["relative_d_vol_error"],
                row["relative_outward_partition_significance"],
            )
            print(
                "local self defect convergence: "
                f"port={port+1}, z={relative_z_error:.3e}, "
                f"R={relative_resistive_error:.3e}, X={relative_reactive_error:.3e}, "
                f"Dvol={relative_d_vol_error:.3e}, Dout={outward_significance:.3e}",
                flush=True,
            )
            return row

        if validation_workers > 1 and n_ports > 1:
            print(
                f"local self convergence audit: parallel validation ports={validation_workers}",
                flush=True,
            )
            with ThreadPoolExecutor(
                max_workers=validation_workers,
                thread_name_prefix="local-self-validation",
            ) as pool:
                ports = list(pool.map(solve_validation, fine_states))
        else:
            if n_ports > 1:
                print("local self convergence audit: validation ports serialized for memory", flush=True)
            ports = [solve_validation(item) for item in fine_states]

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
    worst_localized_balance = max(
        (port["maximum_localized_power_balance_relative_error"] for row in rows for port in row["ports"]),
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
        "parallel_validation_ports": int(validation_workers),
        "self_correction_model": "canonical_local_transverse_fine_minus_coarse_self_defect_v2",
        "maximum_relative_error": float(worst),
        "maximum_localized_power_balance_relative_error": float(worst_localized_balance),
        "maximum_joule_total_power_relative_error": float(worst_joule),
        "maximum_linear_relative_residual": float(worst_linear),
        "linear_solver_converged": all_linear,
        "loss_partition_semantics": "localized_defect_D_out_scaled_by_local_dissipation",
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
