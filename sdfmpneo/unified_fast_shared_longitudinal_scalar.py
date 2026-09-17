"""Fast shared scalar state for terminal longitudinal certification.

The terminal reactive and dissipative certificates use the same balanced
full-port scalar field.  Re-solving separate feed/return patches multiplies the
largest 3-D factorizations without adding physics.  This adapter makes one
refined scalar solve carry both the impedance reaction and two disjoint terminal
Joule-window contractions.

Large refined patches use a two-level LGMRES solve instead of a 3-D SuperLU
factorization.  The coarse space is the exact parent-grid nodal subset retained
by terminal refinement; its operator is the true Galerkin ``P.T A P``.  Damped
fine Jacobi removes high-frequency error.  Every accepted field is checked
against the original fine matrix and must reach the unchanged 1e-9 residual
certificate.  Direct SuperLU remains a correctness fallback.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import unified_certified_local_solve as _local_solve
from . import unified_longitudinal_patch_consistency as _consistency
from . import unified_terminal_dissipative_defect as _terminal_defect
from . import unified_terminal_longitudinal_refinement as _terminal_refinement
from .unified_charge_regularized_source import terminal_charge_target


_MODEL_SUFFIX = "shared_terminal_field_two_level_scalar_v1"
_OVERRIDE_ATTR = "_sdfmpneo_source_quadrature_resolution_override"
_DIRECT_THRESHOLD = 120000
_RESIDUAL_TOLERANCE = 1e-9


def _parent_subset_axis(parent_axis, patch_axis):
    parent = np.asarray(parent_axis, float)
    patch = np.asarray(patch_axis, float)
    scale = max(float(np.max(np.abs(parent))), float(np.max(np.abs(patch))), 1.0)
    tol = 256.0 * np.finfo(float).eps * scale
    mask = (parent >= patch[0] - tol) & (parent <= patch[-1] + tol)
    coarse = parent[mask].copy()
    if coarse.size < 3:
        raise RuntimeError("shared scalar patch has fewer than two parent coarse cells")
    if abs(float(coarse[0] - patch[0])) > tol or abs(float(coarse[-1] - patch[-1])) > tol:
        raise RuntimeError("shared scalar patch boundary is not aligned to parent nodes")
    coarse[0] = patch[0]
    coarse[-1] = patch[-1]
    return coarse


def _coarse_axes(parent, patch):
    return tuple(
        _parent_subset_axis(parent_axis, patch_axis)
        for parent_axis, patch_axis in zip(
            (parent.x, parent.y, parent.z), (patch.x, patch.y, patch.z)
        )
    )


def _interpolation_1d(coarse_axis, fine_axis):
    coarse = np.asarray(coarse_axis, float)
    fine = np.asarray(fine_axis, float)
    nf = max(0, fine.size - 2)
    nc = max(0, coarse.size - 2)
    rows, cols, data = [], [], []
    scale = max(float(np.max(np.abs(coarse))), float(np.max(np.abs(fine))), 1.0)
    tol = 256.0 * np.finfo(float).eps * scale
    for row, value in enumerate(fine[1:-1]):
        index = int(np.searchsorted(coarse, value, side="left"))
        if index < coarse.size and abs(float(coarse[index] - value)) <= tol:
            if 0 < index < coarse.size - 1:
                rows.append(row); cols.append(index - 1); data.append(1.0)
            continue
        right = min(max(index, 1), coarse.size - 1)
        left = right - 1
        width = float(coarse[right] - coarse[left])
        if width <= 0.0:
            raise FloatingPointError("coarse interpolation axis is not monotone")
        wr = float((value - coarse[left]) / width)
        wl = 1.0 - wr
        if 0 < left < coarse.size - 1 and abs(wl) > np.finfo(float).tiny:
            rows.append(row); cols.append(left - 1); data.append(wl)
        if 0 < right < coarse.size - 1 and abs(wr) > np.finfo(float).tiny:
            rows.append(row); cols.append(right - 1); data.append(wr)
    return sp.csr_matrix((data, (rows, cols)), shape=(nf, nc))


def _nodal_prolongation(coarse_axes, fine_axes):
    pieces = [_interpolation_1d(c, f) for c, f in zip(coarse_axes, fine_axes)]
    P = sp.kron(sp.kron(pieces[0], pieces[1], format="csr"), pieces[2], format="csr")
    expected_fine = int(np.prod([len(axis) - 2 for axis in fine_axes], dtype=np.int64))
    expected_coarse = int(np.prod([len(axis) - 2 for axis in coarse_axes], dtype=np.int64))
    if P.shape != (expected_fine, expected_coarse):
        raise AssertionError("scalar two-level prolongation shape mismatch")
    return P


def _direct_solve(A, rhs):
    try:
        factor = spla.splu(
            A.tocsc(),
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        factor = spla.splu(A.tocsc())
    return np.asarray(factor.solve(rhs), complex).reshape(-1)


def _inverse_diagonal(A):
    diagonal = np.asarray(A.diagonal(), complex).reshape(-1)
    magnitude = np.abs(diagonal)
    good = magnitude[np.isfinite(magnitude) & (magnitude > np.finfo(float).tiny)]
    if good.size == 0:
        raise RuntimeError("scalar patch diagonal has no usable entries")
    floor = max(float(np.median(good)) * 1e-12, np.finfo(float).tiny)
    return np.conj(diagonal) / (magnitude * magnitude + floor * floor)


def _two_level_operator(A, P, coarse_factor, inverse_diagonal, *, sweeps, weight=0.65):
    sweeps = max(1, int(sweeps))

    def smooth(residual):
        r = np.asarray(residual, complex).reshape(-1).copy()
        z = np.zeros_like(r)
        for _ in range(sweeps):
            delta = float(weight) * inverse_diagonal * r
            z += delta
            r -= A @ delta
        return z, r

    def apply(vector):
        rhs = np.asarray(vector, complex).reshape(-1)
        zpre, r1 = smooth(rhs)
        coarse_rhs = np.asarray(P.T @ r1, complex).reshape(-1)
        coarse = np.asarray(coarse_factor.solve(coarse_rhs), complex).reshape(-1)
        zc = np.asarray(P @ coarse, complex).reshape(-1)
        r2 = r1 - A @ zc
        zpost, _r3 = smooth(r2)
        return np.asarray(zpre + zc + zpost, complex).reshape(-1)

    return spla.LinearOperator(A.shape, matvec=apply, dtype=A.dtype)


def _solve_refined(A, rhs, *, parent, patch, x0):
    n = int(A.shape[0])
    rhs = np.asarray(rhs, complex).reshape(-1)
    if n < _DIRECT_THRESHOLD:
        started = time.perf_counter()
        field = _direct_solve(A, rhs)
        residual = float(_local_solve._relative_residual(A, field, rhs))
        print(
            f"longitudinal scalar direct: dofs={n}, residual={residual:.3e}, "
            f"time={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return field, residual, "direct"

    coarse_axes = _coarse_axes(parent, patch)
    fine_axes = tuple(np.asarray(axis, float) for axis in (patch.x, patch.y, patch.z))
    started = time.perf_counter()
    try:
        P = _nodal_prolongation(coarse_axes, fine_axes)
        Ac = (P.T @ (A @ P)).tocsc()
        Ac.sum_duplicates(); Ac.eliminate_zeros()
        if Ac.shape[0] == 0:
            raise RuntimeError("scalar two-level coarse space is empty")
        try:
            coarse_factor = spla.splu(
                Ac,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.01,
                options={"Equil": True},
            )
        except (RuntimeError, ValueError):
            coarse_factor = spla.splu(Ac)
        inverse_diagonal = _inverse_diagonal(A)
        build_seconds = float(time.perf_counter() - started)
        best = None if x0 is None else np.asarray(x0, complex).reshape(-1).copy()
        best_residual = (
            float("inf") if best is None else float(_local_solve._relative_residual(A, best, rhs))
        )
        history = []
        for sweeps, maxiter, inner_m in ((1, 24, 24), (2, 40, 30)):
            M = _two_level_operator(
                A, P, coarse_factor, inverse_diagonal, sweeps=sweeps
            )
            t0 = time.perf_counter()
            candidate, info = _local_solve._lgmres(
                A,
                rhs,
                x0=best,
                M=M,
                rtol=max(0.2 * _RESIDUAL_TOLERANCE, 1e-12),
                maxiter=maxiter,
                inner_m=inner_m,
            )
            candidate = np.asarray(candidate, complex).reshape(-1)
            residual = float(_local_solve._relative_residual(A, candidate, rhs))
            elapsed = float(time.perf_counter() - t0)
            history.append((sweeps, int(info), residual, elapsed))
            print(
                "longitudinal scalar two-level: "
                f"dofs={n}, coarse={Ac.shape[0]}, P_nnz={P.nnz}, sweeps={sweeps}, "
                f"residual={residual:.3e}, info={int(info)}, build={build_seconds:.1f}s, "
                f"solve={elapsed:.1f}s",
                flush=True,
            )
            if np.isfinite(residual) and residual < best_residual:
                best = candidate
                best_residual = residual
            if best_residual <= _RESIDUAL_TOLERANCE:
                return best, best_residual, "two-level-lgmres"
        print(
            "longitudinal scalar two-level did not reach residual Gate; "
            "falling back to sparse direct solve",
            flush=True,
        )
    except (RuntimeError, ValueError, MemoryError, FloatingPointError) as exc:
        print(
            f"longitudinal scalar two-level unavailable ({exc}); falling back to direct",
            flush=True,
        )

    t0 = time.perf_counter()
    field = _direct_solve(A, rhs)
    residual = float(_local_solve._relative_residual(A, field, rhs))
    print(
        f"longitudinal scalar fallback direct: dofs={n}, residual={residual:.3e}, "
        f"time={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return field, residual, "fallback-direct"


def _terminal_windows(module, parent, patch, geometry, port):
    coarse_axes = _coarse_axes(parent, patch)
    boxes, _contact, coil = _terminal_refinement._contact_boxes(
        module, parent, geometry, int(port)
    )
    cfg = _terminal_defect._config(parent)
    halo = float(cfg["core_padding_factor"]) * max(
        float(coil.conductor_width), float(coil.conductor_thickness)
    )
    windows = tuple(
        _terminal_defect._snap_window(coarse_axes, box, halo) for box in boxes
    )
    if not _terminal_defect._windows_disjoint(windows[0], windows[1]):
        raise RuntimeError("shared terminal energy windows overlap")
    return coarse_axes, windows


def _validation_resolution(module, parent, geometry, port, coarse_axes):
    cfg = module._config(parent)
    axes, _steps, _boxes, _contact = _terminal_refinement._refined_axes_for_request(
        module,
        parent,
        geometry,
        int(port),
        coarse_axes,
        float(cfg["validation_fine_step"]),
        float(cfg["fine_step"]),
    )
    value = float(min(np.min(np.diff(np.asarray(axis, float))) for axis in axes))
    if not np.isfinite(value) or value <= 0.0:
        raise FloatingPointError("shared scalar validation quadrature resolution is invalid")
    return value


def _local_energy(module, parent, patch, q_cells, windows, *, phi=None):
    local_d = []
    local_modal = [] if phi is not None else None
    local_phi = None if phi is None else module._interpolate_cell_basis(parent, patch, phi)
    for window in windows:
        mask = _terminal_defect._cell_mask(patch, window)
        local_q = np.where(mask, np.asarray(q_cells, float), 0.0)
        local_d.append(float(2.0 * np.sum(local_q)))
        if local_modal is not None:
            local_modal.append(np.asarray(2.0 * (local_phi.T @ local_q), float).tolist())
    return local_d, local_modal


def install(module):
    if bool(getattr(module, "_fast_shared_longitudinal_scalar_installed", False)):
        return module

    original = _consistency._full_patch_state

    def full_patch_state(
        module_arg,
        parent,
        patch,
        geometry,
        global_potential,
        *,
        source_port,
        phi=None,
        fine_step,
        certify_parent=False,
    ):
        p = int(source_port)
        refined = bool(
            not certify_parent
            and float(fine_step) < module._background_step(parent) * (1.0 - 1e-12)
        )
        if not refined and not certify_parent:
            return original(
                module_arg, parent, patch, geometry, global_potential,
                source_port=p, phi=phi, fine_step=fine_step,
                certify_parent=certify_parent,
            )

        coarse_axes, windows = _terminal_windows(module, parent, patch, geometry, p)
        override = None
        if refined:
            override = _validation_resolution(module, parent, geometry, p, coarse_axes)
            patch._sdfmpneo_scalar_charge_target_only = True
            setattr(patch, _OVERRIDE_ATTR, float(override))
        try:
            context = patch.geometry_context(geometry, assemble_thermal=False)
            G = module.gradient_operator(patch, gauge_fixed=False)
            diagonal, sigma, _hs = module._volume_edge_diagonal(patch, context)
            scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
            scalar.sum_duplicates(); scalar.eliminate_zeros()
            boundary = module._boundary_node_mask(patch)
            interior = ~boundary
            restricted = _consistency._restricted_parent_potential(
                module, parent, global_potential, patch
            )

            if certify_parent:
                B = np.asarray(patch.rhs_matrix(context), complex)
                source = np.asarray(context.source_shape, float)
                rhs = B[:, p]
                scalar_rhs = np.asarray(G.T @ rhs, complex).reshape(-1)
                action = np.asarray(scalar[interior] @ restricted, complex).reshape(-1)
                residual_vec = np.asarray(scalar_rhs[interior] - action, complex).reshape(-1)
                scale = max(
                    float(np.linalg.norm(scalar_rhs[interior])),
                    float(np.linalg.norm(action)),
                    np.finfo(float).tiny,
                )
                residual = float(np.linalg.norm(residual_vec) / scale)
                phi_nodes = np.asarray(restricted, complex).copy()
                field = np.asarray(G @ phi_nodes, complex).reshape(-1)
                z_reaction = complex(-source[:, p] @ field)
                solver_label = "parent-restriction"
                charge_meta = {}
            else:
                q_target, charge_meta = terminal_charge_target(
                    patch, context.geometry.coils[p]
                )
                scalar_rhs = (-1j * float(patch.omega)) * np.asarray(q_target, complex)
                phi_nodes = np.zeros(G.shape[1], complex)
                phi_nodes[boundary] = module._boundary_values(
                    parent, global_potential, patch, boundary
                )
                Sii = scalar[interior][:, interior].tocsr()
                Sib = scalar[interior][:, boundary].tocsr()
                rhs_i = np.asarray(
                    scalar_rhs[interior] - Sib @ phi_nodes[boundary], complex
                ).reshape(-1)
                x0 = np.asarray(restricted[interior], complex).reshape(-1)
                solved, residual, solver_label = _solve_refined(
                    Sii, rhs_i, parent=parent, patch=patch, x0=x0
                )
                phi_nodes[interior] = solved
                field = np.asarray(G @ phi_nodes, complex).reshape(-1)
                z_reaction = complex(-np.asarray(q_target, float) @ phi_nodes)

            abs2 = np.abs(field) ** 2
            sigma = np.asarray(sigma, float)
            edge_loss = np.asarray(patch.edge_cell_hodge @ sigma, float).reshape(-1)
            d_vol = float(np.dot(edge_loss, abs2))
            q_cells = np.asarray(
                0.5 * sigma * np.asarray(patch.edge_cell_hodge.T @ abs2).reshape(-1),
                float,
            )
            modal = None
            if phi is not None:
                local_phi = module._interpolate_cell_basis(parent, patch, phi)
                modal = np.asarray(2.0 * (local_phi.T @ q_cells), float)
            terminal_d, terminal_modal = _local_energy(
                module, parent, patch, q_cells, windows, phi=phi
            )
            return {
                "z_reaction": z_reaction,
                "d_vol": d_vol,
                "modal_h": modal,
                "scalar_relative_residual": float(residual),
                "n_cells": int(patch.n_cells),
                "scalar_dofs": int(np.count_nonzero(interior)),
                "fine_step": float(fine_step),
                "parent_restriction_relative_residual": (
                    float(residual) if certify_parent else None
                ),
                "full_geometry_material_assembly": True,
                "source_port": p,
                "scalar_charge_rhs_direct": bool(refined),
                "scalar_charge_support_nodes": int(
                    charge_meta.get("terminal_charge_support_nodes", 0)
                ),
                "scalar_charge_contact_length": float(
                    charge_meta.get("terminal_charge_contact_length", 0.0)
                ),
                "source_quadrature_resolution": (
                    None if override is None else float(override)
                ),
                "source_quadrature_locked_to_validation": bool(refined),
                "terminal_local_d_vol": [float(v) for v in terminal_d],
                "terminal_local_modal_h": terminal_modal,
                "terminal_energy_windows": [
                    {"lo": np.asarray(lo, float).tolist(), "hi": np.asarray(hi, float).tolist()}
                    for lo, hi in windows
                ],
                "scalar_solver": solver_label,
            }
        finally:
            if refined and hasattr(patch, _OVERRIDE_ATTR):
                delattr(patch, _OVERRIDE_ATTR)

    _consistency._full_patch_state = full_patch_state
    module._MODEL = f"{module._MODEL}+{_MODEL_SUFFIX}"
    module._fast_shared_longitudinal_scalar_installed = True
    return module


__all__ = [
    "_interpolation_1d",
    "_nodal_prolongation",
    "install",
]
