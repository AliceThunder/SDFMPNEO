"""Install the v3 convergence audit for the complete local self mesh defect."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numpy as np


_MODEL = "canonical_local_full_fine_minus_coarse_self_defect_v3"


def install(audit_module, self_correction_module):
    if bool(getattr(audit_module, "_full_self_defect_v3_audit_installed", False)):
        return audit_module

    ParentView = audit_module._AuditParentView
    scaled_difference = audit_module._scaled_difference

    def audit_local_self_correction(background, geometries, monitor=None):
        cfg = self_correction_module._config(background)
        tolerance = float(cfg.get("relative_tolerance", 1e-1))
        joule_tolerance = float(cfg.get("joule_identity_tolerance", 1e-10))
        linear_tolerance = float(cfg.get("linear_relative_residual_tolerance", 1e-9))
        fine = float(cfg["fine_step"])
        validation = float(cfg.get("validation_fine_step", 0.75 * fine))
        if not 0.0 < validation < fine:
            raise ValueError("self_correction.validation_fine_step must be smaller than fine_step")
        if joule_tolerance <= 0.0 or linear_tolerance <= 0.0:
            raise ValueError("local self audit tolerances must be positive")

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
                parent = ParentView(background)
                seed_step = float(self_correction_module._parent_fine_step(parent))
                if not seed_step > fine:
                    raise ValueError("local self audit requires parent coarse step > fine step")
                print(
                    f"local self warm start……geometry {gi+1}/{len(geometries)} "
                    f"port {port+1}/{n_ports}  {seed_step:g}m -> {fine:g}m",
                    flush=True,
                )
                seed = self_correction_module._solve_local(
                    parent, geometry, port, seed_step, phi=None
                )
                print(
                    f"local self correction fine solve……geometry {gi+1}/{len(geometries)} "
                    f"port {port+1}/{n_ports}  step={fine:g}m",
                    flush=True,
                )
                solved = self_correction_module._solve_local(
                    parent, geometry, port, fine, phi=None
                )
                return int(port), parent, seed_step, seed, solved

            if workers > 1 and n_ports > 1:
                print(f"local self convergence audit: parallel fine ports={workers}", flush=True)
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="local-self-fine"
                ) as pool:
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
                b = self_correction_module._solve_local(
                    parent, geometry, port, validation, phi=None
                )

                # v3 audits exactly the production object: the complete local
                # fine-minus-coarse defect.  Absolute local terminal response is
                # never compared or substituted for the global response.
                dz_f = complex(a["z"] - seed["z"])
                dz_v = complex(b["z"] - seed["z"])
                dd_f = float(a["d_vol"] - seed["d_vol"])
                dd_v = float(b["d_vol"] - seed["d_vol"])
                do_f = float(a["d_out"] - seed["d_out"])
                do_v = float(b["d_out"] - seed["d_out"])

                dissipation_scale = max(
                    abs(float(dz_v.real)),
                    abs(dd_v) + abs(do_v),
                    np.finfo(float).tiny,
                )
                z_scale = max(abs(dz_v), dissipation_scale)
                reactive_scale = max(abs(float(dz_v.imag)), dissipation_scale)
                dvol_scale = max(abs(dd_v), dissipation_scale)

                relative_z_error = scaled_difference(dz_f, dz_v, z_scale)
                relative_resistive_error = scaled_difference(
                    dz_f.real, dz_v.real, dissipation_scale, abs(dz_v.real)
                )
                relative_reactive_error = scaled_difference(
                    dz_f.imag, dz_v.imag, reactive_scale
                )
                relative_d_vol_error = scaled_difference(dd_f, dd_v, dvol_scale)
                outward_significance = float(abs(do_f - do_v) / dissipation_scale)

                # Keep the v2 localized defect difference as a diagnosis only.
                localized_dz_f = complex(a["localized_z"] - seed["localized_z"])
                localized_dz_v = complex(b["localized_z"] - seed["localized_z"])
                localized_error = scaled_difference(
                    localized_dz_f,
                    localized_dz_v,
                    max(abs(localized_dz_v), dissipation_scale),
                )

                linear_residual = max(
                    float(seed.get("linear_relative_residual", np.inf)),
                    float(a.get("linear_relative_residual", np.inf)),
                    float(b.get("linear_relative_residual", np.inf)),
                )
                linear_converged = bool(
                    seed.get(
                        "linear_solver_converged",
                        float(seed.get("linear_relative_residual", np.inf)) <= linear_tolerance,
                    )
                    and a.get(
                        "linear_solver_converged",
                        float(a.get("linear_relative_residual", np.inf)) <= linear_tolerance,
                    )
                    and b.get(
                        "linear_solver_converged",
                        float(b.get("linear_relative_residual", np.inf)) <= linear_tolerance,
                    )
                    and linear_residual <= linear_tolerance
                )

                row = {
                    "port": int(port),
                    "warm_start_seed_step": seed_step,
                    "defect_model": "full_local_fine_minus_coarse_v3",
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
                    "diagnostic_localized_defect_relative_error": localized_error,
                    "maximum_localized_power_balance_relative_error": max(
                        float(seed["localized_power_balance_relative_error"]),
                        float(a["localized_power_balance_relative_error"]),
                        float(b["localized_power_balance_relative_error"]),
                    ),
                    "maximum_full_local_power_balance_relative_error": max(
                        float(seed["power_balance_relative_error"]),
                        float(a["power_balance_relative_error"]),
                        float(b["power_balance_relative_error"]),
                    ),
                    "maximum_joule_total_power_relative_error": max(
                        float(seed["joule_total_power_relative_error"]),
                        float(a["joule_total_power_relative_error"]),
                        float(b["joule_total_power_relative_error"]),
                    ),
                    "maximum_linear_relative_residual": linear_residual,
                    "linear_solver_converged": linear_converged,
                    "coarse": {
                        k: v for k, v in seed.items()
                        if k not in ("modal_h", "localized_modal_h")
                    },
                    "fine": {
                        k: v for k, v in a.items()
                        if k not in ("modal_h", "localized_modal_h")
                    },
                    "validation": {
                        k: v for k, v in b.items()
                        if k not in ("modal_h", "localized_modal_h")
                    },
                }
                row["maximum_relative_error"] = max(
                    relative_z_error,
                    relative_resistive_error,
                    relative_reactive_error,
                    relative_d_vol_error,
                    outward_significance,
                )
                print(
                    "local full self defect convergence: "
                    f"port={port+1}, z={relative_z_error:.3e}, "
                    f"R={relative_resistive_error:.3e}, X={relative_reactive_error:.3e}, "
                    f"Dvol={relative_d_vol_error:.3e}, Dout={outward_significance:.3e}, "
                    f"localized={localized_error:.3e}",
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
                    print(
                        "local self convergence audit: validation ports serialized for memory",
                        flush=True,
                    )
                ports = [solve_validation(item) for item in fine_states]

            ports.sort(key=lambda row: row["port"])
            rows.append({"geometry_index": int(gi), "geometry": geometry, "ports": ports})

        all_ports = [port for row in rows for port in row["ports"]]
        worst = max((p["maximum_relative_error"] for p in all_ports), default=float("inf"))
        worst_joule = max(
            (p["maximum_joule_total_power_relative_error"] for p in all_ports),
            default=float("inf"),
        )
        worst_linear = max(
            (p["maximum_linear_relative_residual"] for p in all_ports),
            default=float("inf"),
        )
        worst_localized_balance = max(
            (p["maximum_localized_power_balance_relative_error"] for p in all_ports),
            default=float("inf"),
        )
        worst_full_balance = max(
            (p["maximum_full_local_power_balance_relative_error"] for p in all_ports),
            default=float("inf"),
        )
        all_linear = bool(all_ports and all(p["linear_solver_converged"] for p in all_ports))
        return {
            "sample_count": len(rows),
            "fine_step": fine,
            "validation_fine_step": validation,
            "relative_tolerance": tolerance,
            "joule_identity_tolerance": joule_tolerance,
            "linear_relative_residual_tolerance": linear_tolerance,
            "parallel_ports": int(workers),
            "parallel_validation_ports": int(validation_workers),
            "self_correction_model": _MODEL,
            "maximum_relative_error": float(worst),
            "maximum_localized_power_balance_relative_error": float(worst_localized_balance),
            "maximum_full_local_power_balance_relative_error": float(worst_full_balance),
            "maximum_joule_total_power_relative_error": float(worst_joule),
            "maximum_linear_relative_residual": float(worst_linear),
            "linear_solver_converged": all_linear,
            "loss_partition_semantics": "full_local_defect_with_independent_D_out",
            "converged": bool(
                rows
                and worst <= tolerance
                and worst_joule <= joule_tolerance
                and worst_linear <= linear_tolerance
                and all_linear
            ),
            "rows": rows,
            "samples": rows,
        }

    audit_module.audit_local_self_correction = audit_local_self_correction
    audit_module._full_self_defect_v3_audit_installed = True
    return audit_module


__all__ = ["install"]
