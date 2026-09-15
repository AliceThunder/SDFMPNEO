"""Install the v3 full local fine-minus-coarse self defect.

The canonical local problem is never used as an absolute terminal model.  It is
used only for a mesh defect.  Once the impressed source has genuinely finite
mesh-resolved support, the singular pure-longitudinal near-field defect is just
as local as the transverse/cross defect and must also be removed from the coarse
global truth.

Production therefore uses

    global corrected = global coarse + local fine - local coarse

for the complete diagonal Z/D/D_out/modal response.  The global coarse solve
still supplies all nonlocal return-path, dielectric and open-boundary physics;
those smooth contributions cancel from the local fine-minus-coarse difference.
The independent v3 audit and the final global 12mm->9mm Gate certify that this
local defect is transferable before any training labels are accepted.
"""
from __future__ import annotations

import numpy as np


def install(self_correction_module):
    if bool(getattr(self_correction_module, "_full_self_defect_v3_installed", False)):
        return self_correction_module

    Result = self_correction_module.SelfCorrectionResult

    def apply_local_self_correction(
        background,
        geometry,
        z,
        d_vol,
        d_out,
        *,
        phi=None,
        modal_h=None,
    ):
        zc = np.asarray(z, complex).copy()
        dc = np.asarray(d_vol, complex).copy()
        oc = np.asarray(d_out, complex).copy()
        hc = None if modal_h is None else np.asarray(modal_h, complex).copy()
        cfg = self_correction_module._config(background)
        n = zc.shape[0]
        if zc.shape != (n, n) or dc.shape != (n, n) or oc.shape != (n, n):
            raise ValueError("self correction requires shape-compatible square port tensors")
        if not bool(cfg.get("enabled", True)):
            return Result(zc, dc, oc, hc, {"enabled": False, "ports": []})
        if phi is not None and hc is None:
            raise ValueError("modal_h is required when phi is supplied to self correction")
        if hc is not None and (hc.ndim != 3 or hc.shape[1:] != (n, n)):
            raise ValueError("modal_h has incompatible self-correction shape")

        coarse_step = self_correction_module._parent_fine_step(background)
        fine_step = float(cfg["fine_step"])
        if not 0.0 < fine_step < coarse_step:
            raise ValueError(
                f"self_correction.fine_step={fine_step:g} must be smaller than "
                f"parent fine_step={coarse_step:g}"
            )

        ports = []
        maximum_total_identity_error = 0.0
        maximum_modal_identity_error = 0.0
        maximum_localized_balance_error = 0.0
        maximum_full_balance_error = 0.0
        for p in range(n):
            coarse = self_correction_module._solve_local(
                background, geometry, p, coarse_step, phi=phi
            )
            fine = self_correction_module._solve_local(
                background, geometry, p, fine_step, phi=phi
            )
            maximum_total_identity_error = max(
                maximum_total_identity_error,
                float(coarse["joule_total_power_relative_error"]),
                float(fine["joule_total_power_relative_error"]),
            )
            maximum_modal_identity_error = max(
                maximum_modal_identity_error,
                float(coarse["joule_modal_contraction_relative_error"]),
                float(fine["joule_modal_contraction_relative_error"]),
            )
            maximum_localized_balance_error = max(
                maximum_localized_balance_error,
                float(coarse["localized_power_balance_relative_error"]),
                float(fine["localized_power_balance_relative_error"]),
            )
            maximum_full_balance_error = max(
                maximum_full_balance_error,
                float(coarse["power_balance_relative_error"]),
                float(fine["power_balance_relative_error"]),
            )

            # v3: correct the complete local mesh defect.  This includes the
            # terminal longitudinal near-field but never substitutes the local
            # absolute response for the global one.
            dz = complex(fine["z"] - coarse["z"])
            dd = float(fine["d_vol"] - coarse["d_vol"])
            do = float(fine["d_out"] - coarse["d_out"])
            zc[p, p] += dz
            dc[p, p] += dd
            oc[p, p] += do

            modal_delta = None
            if phi is not None:
                modal_delta = np.asarray(fine["modal_h"] - coarse["modal_h"], float)
                hc[:, p, p] += modal_delta

            ports.append({
                "port": int(p),
                "coarse_step": float(coarse_step),
                "fine_step": float(fine_step),
                "delta_z_real": float(dz.real),
                "delta_z_imag": float(dz.imag),
                "delta_d_vol": float(dd),
                "delta_d_out": float(do),
                "localized_delta_z_real": float(
                    np.real(fine["localized_z"] - coarse["localized_z"])
                ),
                "localized_delta_z_imag": float(
                    np.imag(fine["localized_z"] - coarse["localized_z"])
                ),
                "coarse": {
                    k: v for k, v in coarse.items()
                    if k not in ("modal_h", "localized_modal_h")
                },
                "fine": {
                    k: v for k, v in fine.items()
                    if k not in ("modal_h", "localized_modal_h")
                },
                "maximum_modal_delta": (
                    None if modal_delta is None else float(np.max(np.abs(modal_delta)))
                ),
            })

        zc = 0.5 * (zc + zc.T)
        dc = 0.5 * (dc + dc.conj().T)
        oc = 0.5 * (oc + oc.conj().T)
        if hc is not None:
            hc = 0.5 * (hc + np.swapaxes(hc.conj(), 1, 2))

        herm_z = 0.5 * (zc + zc.conj().T)
        corrected_balance = float(
            np.linalg.norm(herm_z - dc - oc)
            / max(
                float(np.linalg.norm(herm_z)),
                float(np.linalg.norm(dc + oc)),
                np.finfo(float).tiny,
            )
        )
        return Result(
            zc,
            dc,
            oc,
            hc,
            {
                "enabled": True,
                "model": "canonical_local_full_fine_minus_coarse_self_defect_v3",
                "coarse_step": float(coarse_step),
                "fine_step": float(fine_step),
                "corrected_power_balance_relative_error": corrected_balance,
                # Retain the old diagnostic for compatibility; v3 acceptance is
                # governed by the complete defect audit below.
                "maximum_localized_power_balance_relative_error": float(
                    maximum_localized_balance_error
                ),
                "maximum_full_local_power_balance_relative_error": float(
                    maximum_full_balance_error
                ),
                "maximum_joule_total_power_relative_error": float(
                    maximum_total_identity_error
                ),
                "maximum_joule_modal_contraction_relative_error": float(
                    maximum_modal_identity_error
                ),
                "ports": ports,
            },
        )

    self_correction_module.apply_local_self_correction = apply_local_self_correction
    self_correction_module._full_self_defect_v3_installed = True
    return self_correction_module


__all__ = ["install"]
