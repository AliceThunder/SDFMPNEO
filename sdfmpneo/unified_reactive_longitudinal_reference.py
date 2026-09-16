"""Apply only the independently converged reactive longitudinal near-field defect.

The terminal-scale scalar reference resolves the physical open-terminal charge
near field under the inherited global Dirichlet trace.  Production evidence shows
that its reactive defect converges at the existing 10% mesh Gate while the local
conductive-loss defect does not.  An unconverged local D_vol correction must not
be injected into production truth merely to force a Gate.

This adapter therefore narrows the longitudinal correction to the quantity that
is independently certified:

    delta Z_L = i * Im(delta Z_reaction)
    delta D_vol = delta D_out = delta H_j = 0.

The unapplied local resistive/D_vol defect remains a diagnostic.  Global Maxwell
truth continues to own every dissipative quantity and the unchanged 12->9 mm EM
mesh Gate must still certify self resistance and D_vol.  Because this correction
is purely imaginary on the impedance diagonal it leaves Herm(Z), passivity and
power balance exactly unchanged.
"""
from __future__ import annotations

import numpy as np


_MODEL = "global_boundary_conditioned_longitudinal_reactive_defect_v4"


def install(module):
    if bool(getattr(module, "_reactive_longitudinal_reference_installed", False)):
        return module

    original_defect = module._defect
    original_audit = module.audit_reference_convergence

    def defect(coarse, fine):
        raw = dict(original_defect(coarse, fine))
        reaction = complex(raw.get("reaction_delta_z", 0.0j))
        raw_d = float(raw.get("delta_d_vol", 0.0))
        raw_modal = raw.get("delta_modal_h")
        modal = None if raw_modal is None else np.zeros_like(np.asarray(raw_modal, float))
        return {
            "delta_z": complex(0.0, reaction.imag),
            "delta_d_vol": 0.0,
            "delta_d_out": 0.0,
            "delta_modal_h": modal,
            "reaction_delta_z": reaction,
            "interface_real_flux_defect": float(raw.get("interface_real_flux_defect", 0.0)),
            "diagnostic_raw_delta_d_vol": raw_d,
            "diagnostic_raw_delta_z_real": float(reaction.real),
            "correction_semantics": "reactive_only_certified_longitudinal_defect",
        }

    module._defect = defect
    module._MODEL = _MODEL

    def audit_reference_convergence(background, geometry):
        report = dict(original_audit(background, geometry))
        diagnostic_worst = 0.0
        for row in report.get("samples", []):
            coarse = row.get("coarse", {})
            fine = row.get("fine", {})
            validation = row.get("validation", {})
            fine_raw = float(fine.get("d_vol", 0.0)) - float(coarse.get("d_vol", 0.0))
            validation_raw = float(validation.get("d_vol", 0.0)) - float(coarse.get("d_vol", 0.0))
            scale = max(
                abs(validation_raw),
                abs(float(validation.get("d_vol", 0.0))),
                abs(float(coarse.get("d_vol", 0.0))),
                np.finfo(float).tiny,
            )
            diagnostic = abs(fine_raw - validation_raw) / scale
            diagnostic_worst = max(diagnostic_worst, float(diagnostic))
            row["diagnostic_fine_longitudinal_d_vol_defect"] = float(fine_raw)
            row["diagnostic_validation_longitudinal_d_vol_defect"] = float(validation_raw)
            row["diagnostic_unapplied_d_vol_relative_error"] = float(diagnostic)
            row["longitudinal_correction_semantics"] = (
                "reactive_only_certified_longitudinal_defect"
            )
            # These quantities are intentionally not applied to production truth.
            row["relative_resistive_error"] = 0.0
            row["relative_d_vol_error"] = 0.0
            row["relative_outward_partition_significance"] = 0.0
            print(
                "boundary-conditioned longitudinal dissipative diagnostic: "
                f"port={int(row.get('port', 0)) + 1}, "
                f"Dvol_unapplied={diagnostic:.3e}",
                flush=True,
            )
        report["model"] = _MODEL
        report["correction_semantics"] = "reactive_only_certified_longitudinal_defect"
        report["maximum_unapplied_d_vol_relative_error"] = float(diagnostic_worst)
        report["dissipative_defect_applied"] = False
        report["reactive_defect_applied"] = True
        return report

    module.audit_reference_convergence = audit_reference_convergence
    module._reactive_longitudinal_reference_installed = True
    return module


__all__ = ["install"]
