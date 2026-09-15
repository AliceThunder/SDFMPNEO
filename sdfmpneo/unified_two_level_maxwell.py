"""Two-level compatible preconditioner for the finest local Maxwell validation.

The 3-mm local Maxwell problem is already certified cheaply while the 2.25-mm
validation problem is too large for a scale-robust one-level ILU.  This module
uses the certified coarse grid as an auxiliary H(curl) space instead of trying
to factor the 254k-edge fine operator.

The transfer is the commuting Nedelec prolongation from
``unified_hcurl_transfer``.  The coarse physical operator is rediscretized on
the previous grid and receives the same compatible gradient+transverse ILU that
is known to work at about 118k edges.  A small scalar alignment maps that
rediscretized inverse to the Galerkin coarse residual ``P.T A_f P``.  Fine-grid
high-frequency error is damped only by fixed weighted Jacobi sweeps.  The
result is a fixed linear V-cycle suitable as an LGMRES preconditioner; the
physical fine operator and the final true-residual certificate are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.sparse.linalg as spla

from .unified_gradient_block_maxwell import build_gradient_block, compose_block_preconditioner
from .unified_transverse_ilu import build_transverse_ilu


def _finite_complex(value):
    value = complex(value)
    return bool(np.isfinite(value.real) and np.isfinite(value.imag))


def _rebuild_coarse_background(fine_background, coarse_state):
    axes = coarse_state.get("axes")
    geometry = coarse_state.get("local_geometry")
    if axes is None or geometry is None:
        raise ValueError("two-level Maxwell coarse state is incomplete")
    cls = type(fine_background)
    coarse = cls(
        *[np.asarray(axis, float) for axis in axes],
        frequency_hz=fine_background.frequency_hz,
        materials=fine_background.materials,
        coil_materials=fine_background.coil_materials,
        package_materials=fine_background.package_materials,
        seawater_material=fine_background.seawater_material,
        ambient_temperature=fine_background.ambient_temperature,
    )
    context = coarse.geometry_context(geometry, assemble_thermal=False)
    A = coarse.em_operator(context, None)
    return coarse, context, A


def _inverse_diagonal(A):
    diagonal = np.asarray(A.diagonal(), complex).reshape(-1)
    magnitude = np.abs(diagonal)
    positive = magnitude[np.isfinite(magnitude) & (magnitude > np.finfo(float).tiny)]
    if positive.size == 0:
        raise ValueError("fine Maxwell diagonal has no usable entries")
    reference = float(np.median(positive))
    floor = max(reference * 1e-12, np.finfo(float).tiny)
    return np.conj(diagonal) / (magnitude * magnitude + floor * floor)


@dataclass
class TwoLevelMaxwell:
    fine_A: object
    prolongation: object
    coarse_A: object
    coarse_M: object
    fine_gradient: object
    coarse_scale: complex
    inverse_diagonal: np.ndarray
    smoother_weight: float
    build_seconds: float
    consistency_error: float
    coarse_edges: int

    def _smooth(self, rhs, sweeps):
        r = np.asarray(rhs, complex).reshape(-1).copy()
        z = np.zeros_like(r)
        for _ in range(int(sweeps)):
            delta = float(self.smoother_weight) * self.inverse_diagonal * r
            z += delta
            r -= self.fine_A @ delta
        return z

    def _coarse_solve(self, rhs, corrections):
        P = self.prolongation
        b = np.asarray(rhs, complex).reshape(-1)
        y = self.coarse_scale * np.asarray(self.coarse_M @ b, complex).reshape(-1)
        # Fixed Richardson corrections against the actual fine-grid Galerkin
        # coarse operator.  Because the number of steps is fixed and x0=0, the
        # resulting preconditioner remains a linear map.
        for _ in range(max(0, int(corrections) - 1)):
            galerkin_residual = b - P.T @ (self.fine_A @ (P @ y))
            y += self.coarse_scale * np.asarray(
                self.coarse_M @ np.asarray(galerkin_residual, complex).reshape(-1),
                complex,
            ).reshape(-1)
        return y

    def operator(self, *, coarse_corrections=1, smoother_sweeps=1):
        A = self.fine_A
        P = self.prolongation
        G = self.fine_gradient
        corrections = max(1, int(coarse_corrections))
        sweeps = max(0, int(smoother_sweeps))

        def apply(vector):
            v = np.asarray(vector, complex).reshape(-1)
            # Exact longitudinal solve first.
            zg = G.solve(v)
            rt = v - A @ zg

            # Symmetric-in-spirit V-cycle on the transverse residual: cheap
            # fine smoothing, coarse H(curl) correction, cheap fine smoothing.
            zpre = self._smooth(rt, sweeps) if sweeps else np.zeros_like(rt)
            r1 = rt - A @ zpre
            coarse_rhs = np.asarray(P.T @ r1, complex).reshape(-1)
            yc = self._coarse_solve(coarse_rhs, corrections)
            zc = np.asarray(P @ yc, complex).reshape(-1)
            r2 = r1 - A @ zc
            zpost = self._smooth(r2, sweeps) if sweeps else np.zeros_like(rt)

            z = zg + zpre + zc + zpost
            # Jacobi/coarse corrections can reintroduce a longitudinal
            # residual.  Remove it exactly with the same scalar-gradient block.
            remainder = v - A @ z
            z += G.solve(remainder)
            return np.asarray(z, complex).reshape(-1)

        return spla.LinearOperator(A.shape, matvec=apply, dtype=A.dtype)


def build_two_level_maxwell(
    fine_A,
    fine_background,
    fine_gradient,
    coarse_state,
    cfg,
):
    started = time.perf_counter()
    P = coarse_state.get("prolongation")
    if P is None:
        raise ValueError("two-level Maxwell requires the compatible H(curl) prolongation")
    if P.shape[0] != fine_A.shape[0]:
        raise ValueError("two-level Maxwell prolongation has wrong fine dimension")

    coarse, coarse_context, coarse_A = _rebuild_coarse_background(fine_background, coarse_state)
    if P.shape[1] != coarse_A.shape[0]:
        raise ValueError("two-level Maxwell prolongation has wrong coarse dimension")

    coarse_gradient = build_gradient_block(coarse, coarse_context, check_topology=True)
    coarse_edge_M, coarse_stats = build_transverse_ilu(
        coarse_A,
        coarse,
        coarse_context,
        coarse_gradient,
        drop_tol=float(cfg.get("linear_two_level_coarse_ilu_drop_tolerance", cfg.get("linear_ilu_drop_tolerance", 5e-3))),
        fill_factor=float(cfg.get("linear_two_level_coarse_ilu_fill_factor", cfg.get("linear_ilu_fill_factor", 4.0))),
        stabilization_factor=float(cfg.get("linear_two_level_coarse_stabilization_factor", cfg.get("linear_transverse_stabilization_factor", 3e-2))),
    )
    coarse_M = compose_block_preconditioner(
        coarse_A,
        coarse_edge_M,
        coarse_gradient,
        post_correct=True,
    )

    coarse_field = np.asarray(coarse_state.get("field", ()), complex).reshape(-1)
    if coarse_field.shape != (coarse_A.shape[0],) or np.any(~np.isfinite(coarse_field)):
        raise ValueError("two-level Maxwell coarse field is invalid")
    prolonged = np.asarray(P @ coarse_field, complex).reshape(-1)
    physical_action = np.asarray(coarse_A @ coarse_field, complex).reshape(-1)
    galerkin_action = np.asarray(P.T @ (fine_A @ prolonged), complex).reshape(-1)
    denominator = complex(np.vdot(physical_action, physical_action))
    if abs(denominator) <= np.finfo(float).tiny:
        alignment = 1.0 + 0.0j
        consistency = float("inf")
    else:
        alignment = complex(np.vdot(physical_action, galerkin_action) / denominator)
        if not _finite_complex(alignment) or not 1e-8 <= abs(alignment) <= 1e8:
            alignment = 1.0 + 0.0j
        consistency = float(
            np.linalg.norm(galerkin_action - alignment * physical_action)
            / max(float(np.linalg.norm(galerkin_action)), np.finfo(float).tiny)
        )
    coarse_scale = 1.0 / alignment

    smoother_weight = float(cfg.get("linear_two_level_jacobi_weight", 0.5))
    if not np.isfinite(smoother_weight) or not 0.0 < smoother_weight <= 1.0:
        raise ValueError("linear_two_level_jacobi_weight must lie in (0, 1]")
    inverse_diagonal = _inverse_diagonal(fine_A)
    elapsed = float(time.perf_counter() - started)
    print(
        "Maxwell two-level H(curl) block: "
        f"coarse_edges={coarse_A.shape[0]}, fine_edges={fine_A.shape[0]}, "
        f"P_nnz={P.nnz}, scale={coarse_scale.real:.3e}{coarse_scale.imag:+.3e}j, "
        f"consistency={consistency:.3e}, coarse_factor={coarse_stats.get('seconds', 0.0):.1f}s, "
        f"build={elapsed:.1f}s",
        flush=True,
    )
    return TwoLevelMaxwell(
        fine_A=fine_A,
        prolongation=P,
        coarse_A=coarse_A,
        coarse_M=coarse_M,
        fine_gradient=fine_gradient,
        coarse_scale=complex(coarse_scale),
        inverse_diagonal=np.asarray(inverse_diagonal, complex),
        smoother_weight=smoother_weight,
        build_seconds=elapsed,
        consistency_error=consistency,
        coarse_edges=int(coarse_A.shape[0]),
    )


__all__ = ["TwoLevelMaxwell", "build_two_level_maxwell"]
