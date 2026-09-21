"""Post-truth Physics Gate for spatial-Joule + online thermal architecture."""
from __future__ import annotations

import numpy as np

from .unified_corrected_truth import solve_spatial_truth_tensors
from .unified_online_thermal import audit_online_thermal_trajectories


_SELF_CORRECTION_MODEL = (
    "canonical_local_transverse_fine_minus_coarse_self_defect_v2"
)


def _relative(a, b):
    x = np.asarray(a)
    y = np.asarray(b)
    return float(
        np.linalg.norm(x - y)
        / max(
            float(np.linalg.norm(x)),
            float(np.linalg.norm(y)),
            np.finfo(float).tiny,
        )
    )


def run_spatial_physics_gate(
    settings,
    background,
    dataset,
    geometries,
    *,
    preflight,
    monitor=None,
):
    """Validate corrected spatial Joule truth and geometry-local thermal ROM."""
    geometries = list(geometries)
    if not geometries:
        raise ValueError("spatial Physics Gate needs held-out geometry")
    if not bool(preflight.get("certified", False)):
        raise RuntimeError(
            "spatial Physics Gate requires certified truth preflight"
        )

    audit = dict(dataset.audit)
    thermal_target = float(
        settings["TRAINING"].get(
            "thermal_basis_energy_tolerance",
            5e-2,
        )
    )
    time_scales = settings["TRAINING"].get(
        "thermal_time_scales",
        [0.1, 1.0, 10.0],
    )
    trajectory_times = settings["TRAINING"].get(
        "thermal_trajectory_times",
        [0.1, 1.0, 10.0, 100.0],
    )
    condition_limit = float(
        settings["TRAINING"].get(
            "thermal_basis_conditioning_limit",
            1e10,
        )
    )

    online_rows = []
    heldout_spatial_rows = []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        z, d, cells, truth_audit = solve_spatial_truth_tensors(
            background,
            geometry,
        )
        total_error = float(
            np.linalg.norm(np.sum(cells, axis=0) - d)
            / max(float(np.linalg.norm(d)), np.finfo(float).tiny)
        )
        minimum_eigenvalue = float(
            np.min(np.linalg.eigvalsh(cells).real)
        )
        heldout_spatial_rows.append(
            {
                "index": int(index),
                "aggregate_d_relative_error": total_error,
                "minimum_cell_eigenvalue": minimum_eigenvalue,
                "minimum_d_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(d)).real
                ),
                "reciprocity_relative_error": float(
                    truth_audit["reciprocity_relative_error"]
                ),
                "poynting_relative_error": float(
                    truth_audit[
                        "open_boundary_power_balance_relative_error"
                    ]
                ),
            }
        )

        thermal = audit_online_thermal_trajectories(
            background,
            geometry,
            cells,
            times=trajectory_times,
            time_scales=time_scales,
            conditioning_limit=condition_limit,
            target_relative_error=thermal_target,
        )
        online_rows.append(thermal)
        print(
            "spatial Physics Gate……"
            f"{index + 1}/{len(geometries)} "
            f"online-rank={thermal['rank']} "
            f"thermal={thermal['maximum_mass_relative_error']:.3e} "
            f"sumD={total_error:.3e}",
            flush=True,
        )

    maximum_online_error = max(
        row["maximum_mass_relative_error"]
        for row in online_rows
    )
    maximum_online_resolvent = max(
        row["resolvent_error"] for row in online_rows
    )
    maximum_online_condition = max(
        row["conditioning"] for row in online_rows
    )
    maximum_heldout_total_error = max(
        row["aggregate_d_relative_error"]
        for row in heldout_spatial_rows
    )
    minimum_heldout_cell_eigenvalue = min(
        row["minimum_cell_eigenvalue"]
        for row in heldout_spatial_rows
    )

    checks = {
        "linear_solve_ok": (
            audit["maximum_linear_relative_residual"] <= 1e-8
        ),
        "reciprocity_ok": (
            audit["maximum_reciprocity_relative_error"] <= 1e-8
        ),
        "volume_passivity_ok": (
            audit["minimum_d_vol_eigenvalue"] >= -1e-9
        ),
        "physical_outward_passivity_ok": (
            audit["minimum_physical_outward_eigenvalue"] >= -1e-9
        ),
        "implied_outward_passivity_ok": (
            audit["minimum_implied_outward_eigenvalue"] >= -1e-9
        ),
        "independent_poynting_balance_ok": (
            audit["maximum_open_boundary_power_balance_relative_error"]
            <= 1e-7
        ),
        "spatial_cell_psd_ok": (
            audit["minimum_cell_joule_tensor_eigenvalue"] >= -1e-10
            and minimum_heldout_cell_eigenvalue >= -1e-10
        ),
        "spatial_sum_to_d_ok": (
            audit["maximum_spatial_joule_total_mismatch"] <= 1e-10
            and maximum_heldout_total_error <= 1e-10
        ),
        "joule_total_power_identity_ok": (
            audit["maximum_joule_total_power_relative_error"] <= 1e-10
        ),
        "local_self_joule_total_identity_ok": (
            audit.get(
                "maximum_local_self_joule_total_power_relative_error",
                np.inf,
            )
            <= 1e-10
        ),
        "local_self_correction_available": (
            audit.get("local_self_correction_available", 0.0) >= 0.5
        ),
        "local_self_correction_power_balance_ok": (
            audit.get(
                "maximum_local_self_correction_power_balance_relative_error",
                np.inf,
            )
            <= 1e-7
        ),
        "material_fraction_closure_ok": bool(
            preflight["material_fraction_closure_ok"]
            and audit["maximum_material_fraction_closure_error"]
            <= 1e-10
        ),
        "finite_support_source_ok": bool(
            preflight["finite_support_source_ok"]
            and audit["source_regularization_available"] >= 0.5
        ),
        "terminal_source_continuity_ok": bool(
            preflight["terminal_source_continuity_ok"]
        ),
        "wire_loss_not_double_counted": bool(
            preflight["wire_loss_not_double_counted"]
        ),
        "outward_power_form_available": bool(
            audit["independent_outward_power_available"] >= 0.5
        ),
        "open_boundary_domain_converged": bool(
            preflight["open_boundary_domain_converged"]
        ),
        "low_frequency_formulation_converged": bool(
            preflight["low_frequency_formulation_converged"]
        ),
        "em_mesh_converged": bool(
            preflight["em_mesh_converged"]
        ),
        "online_thermal_resolvent_ok": (
            maximum_online_resolvent <= thermal_target
        ),
        "online_thermal_trajectory_ok": (
            maximum_online_error <= thermal_target
        ),
        "online_thermal_conditioning_ok": (
            maximum_online_condition <= condition_limit
        ),
    }
    certified = all(bool(v) for v in checks.values())
    return {
        **{k: bool(v) for k, v in checks.items()},
        "certified": bool(certified),
        "self_correction_model": _SELF_CORRECTION_MODEL,
        "tensor_representation": "cellwise_joule_tensor_v1",
        "thermal_representation": "geometry_local_rational_krylov_v1",
        "maximum_online_thermal_relative_error": float(
            maximum_online_error
        ),
        "maximum_online_thermal_resolvent_error": float(
            maximum_online_resolvent
        ),
        "maximum_online_thermal_condition": float(
            maximum_online_condition
        ),
        "online_thermal_samples": online_rows,
        "heldout_spatial_samples": heldout_spatial_rows,
    }


__all__ = ["run_spatial_physics_gate"]
