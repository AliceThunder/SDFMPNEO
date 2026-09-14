"""Sharper failure diagnosis for corrected spatial-truth preflight."""
from __future__ import annotations

import numpy as np


def install(corrected_preflight_module):
    if bool(getattr(corrected_preflight_module, "_linear_diagnosis_patch_installed", False)):
        return corrected_preflight_module
    original = corrected_preflight_module._local_self_failure_diagnosis

    def diagnose(local_self):
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
        return original(local_self)

    corrected_preflight_module._local_self_failure_diagnosis = diagnose
    corrected_preflight_module._linear_diagnosis_patch_installed = True
    return corrected_preflight_module


__all__ = ["install"]
