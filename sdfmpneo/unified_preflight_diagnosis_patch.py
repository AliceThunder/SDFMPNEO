"""Sharper failure diagnosis for corrected spatial-truth preflight."""
from __future__ import annotations

import numpy as np


def install(corrected_preflight_module):
    if bool(getattr(corrected_preflight_module, "_linear_diagnosis_patch_installed", False)):
        return corrected_preflight_module
    original_local = corrected_preflight_module._local_self_failure_diagnosis

    def diagnose_local(local_self):
        if not bool(local_self.get("linear_solver_converged", True)):
            residual = float(local_self.get("maximum_linear_relative_residual", np.inf))
            tolerance = float(local_self.get("linear_relative_residual_tolerance", 1e-9))
            return {
                "code": "local_self_linear_solve_not_converged",
                "maximum_linear_relative_residual": residual,
                "linear_relative_residual_tolerance": tolerance,
                "fine_step": float(local_self.get("fine_step", np.nan)),
                "validation_fine_step": float(local_self.get("validation_fine_step", np.nan)),
                "recommendation": (
                    "The local Maxwell reference solve is not numerically certified. "
                    "Do not interpret this as physical mesh nonconvergence and do not relax the "
                    "mesh Gate. Repair/condition the local solve or source formulation first."
                ),
            }
        return original_local(local_self)

    def diagnose_mesh(mesh):
        if bool(mesh.get("converged", False)) or not mesh.get("samples"):
            return None
        row = max(
            mesh["samples"],
            key=lambda item: float(item.get("maximum_relative_error", 0.0)),
        )
        path_error = float(row.get("relative_source_path_length_error", np.inf))
        if path_error > float(mesh.get("source_path_relative_tolerance", 1e-10)):
            return {
                "code": "mesh_refinement_changed_physical_source_geometry",
                "geometry": row.get("geometry"),
                "relative_source_path_length_error": path_error,
                "recommendation": "Mesh refinement must compare the same physical source geometry.",
            }
        self_z = float(row.get("diagnostic_z_self_relative_error", np.inf))
        mutual_z = float(row.get("relative_mutual_impedance_error", np.inf))
        self_d = float(row.get("diagnostic_d_vol_self_relative_error", np.inf))
        mutual_d = float(row.get("diagnostic_d_vol_mutual_relative_error", np.inf))
        self_dominated = bool(self_z > mutual_z and self_d > mutual_d)
        return {
            "code": (
                "corrected_global_self_remainder_not_converged"
                if self_dominated
                else "general_em_mesh_nonconvergence"
            ),
            "geometry": row.get("geometry"),
            "corrected_self_z_relative_error": self_z,
            "raw_self_z_relative_error": float(
                row.get("diagnostic_raw_z_self_relative_error", np.inf)
            ),
            "mutual_z_relative_error": mutual_z,
            "corrected_self_d_vol_relative_error": self_d,
            "mutual_d_vol_relative_error": mutual_d,
            "maximum_relative_error": float(row.get("maximum_relative_error", np.inf)),
            "recommendation": (
                "The independently certified localized transverse/cross self defect has been "
                "applied, but the remaining global self response is still mesh dependent. "
                "The canonical local box must not be used to replace pure longitudinal terminal "
                "response. Diagnose the remaining global longitudinal/source/material "
                "discretization while keeping the unchanged mesh Gate."
                if self_dominated
                else "Refine the remaining non-self EM truth discretization and rerun the Gate; "
                "do not relax the tolerance."
            ),
        }

    corrected_preflight_module._local_self_failure_diagnosis = diagnose_local
    corrected_preflight_module._mesh_failure_diagnosis = diagnose_mesh
    corrected_preflight_module._linear_diagnosis_patch_installed = True
    return corrected_preflight_module


__all__ = ["install"]
