"""Parallel port scheduling for canonical local self corrections.

Each diagonal self defect is an independent one-port canonical Maxwell problem.
Running those port pairs concurrently changes only scheduling.  A lightweight
parent view gives every worker its own coarse->fine warm-start state while all
immutable production-background data remain shared.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


class _ParentView:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_parent"), name)

    def __setattr__(self, name, value):
        if name.startswith("_local_self_"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_parent"), name, value)


def install(self_correction_module):
    if bool(getattr(self_correction_module, "_parallel_self_correction_installed", False)):
        return self_correction_module

    Result = self_correction_module.SelfCorrectionResult

    def apply_local_self_correction(background, geometry, z, d_vol, d_out, *, phi=None, modal_h=None):
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
                f"self_correction.fine_step={fine_step:g} must be smaller than parent fine_step={coarse_step:g}"
            )

        def solve_pair(p):
            # Independent warm-state namespace per port; all physical parent
            # arrays/operators remain read-only through __getattr__.
            parent = _ParentView(background)
            coarse = self_correction_module._solve_local(
                parent, geometry, p, coarse_step, phi=phi
            )
            fine = self_correction_module._solve_local(
                parent, geometry, p, fine_step, phi=phi
            )
            return int(p), coarse, fine

        workers = min(n, max(1, int(cfg.get("parallel_local_ports", min(2, n)))))
        if workers > 1 and n > 1:
            print(f"local self correction: parallel ports={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="local-self") as pool:
                solved = list(pool.map(solve_pair, range(n)))
        else:
            solved = [solve_pair(p) for p in range(n)]
        solved.sort(key=lambda item: item[0])

        ports = []
        maximum_total_identity_error = 0.0
        maximum_modal_identity_error = 0.0
        for p, coarse, fine in solved:
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
            dz = fine["z"] - coarse["z"]
            dd = float(fine["d_vol"] - coarse["d_vol"])
            do = float(fine["d_out"] - coarse["d_out"])
            zc[p, p] += dz
            dc[p, p] += dd
            oc[p, p] += do
            modal_delta = None
            if phi is not None:
                modal_delta = np.asarray(fine["modal_h"] - coarse["modal_h"], float)
                hc[:, p, p] += modal_delta
            ports.append(
                {
                    "port": int(p),
                    "coarse_step": coarse_step,
                    "fine_step": fine_step,
                    "delta_z_real": float(dz.real),
                    "delta_z_imag": float(dz.imag),
                    "delta_d_vol": dd,
                    "delta_d_out": do,
                    "coarse": {k: v for k, v in coarse.items() if k != "modal_h"},
                    "fine": {k: v for k, v in fine.items() if k != "modal_h"},
                    "maximum_modal_delta": (
                        None if modal_delta is None else float(np.max(np.abs(modal_delta)))
                    ),
                }
            )

        zc = 0.5 * (zc + zc.T)
        dc = 0.5 * (dc + dc.conj().T)
        oc = 0.5 * (oc + oc.conj().T)
        if hc is not None:
            hc = 0.5 * (hc + np.swapaxes(hc.conj(), 1, 2))
        corrected_balance = float(
            np.linalg.norm(0.5 * (zc + zc.conj().T) - dc - oc)
            / max(
                float(np.linalg.norm(0.5 * (zc + zc.conj().T))),
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
                "model": "canonical_local_fine_minus_coarse_self_defect_v1",
                "coarse_step": coarse_step,
                "fine_step": fine_step,
                "parallel_ports": int(workers),
                "corrected_power_balance_relative_error": corrected_balance,
                "maximum_joule_total_power_relative_error": float(maximum_total_identity_error),
                "maximum_joule_modal_contraction_relative_error": float(maximum_modal_identity_error),
                "ports": ports,
            },
        )

    self_correction_module.apply_local_self_correction = apply_local_self_correction
    self_correction_module._parallel_self_correction_installed = True
    return self_correction_module


__all__ = ["install"]
