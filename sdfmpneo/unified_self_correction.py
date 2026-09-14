"""Local multiscale correction for unresolved Maxwell self response.

The production background is intentionally coarse enough to make many-geometry
truth generation practical.  Mutual/far-field coupling is already converged on
that grid, while the diagonal source self response is not when the conductor
cross section is much smaller than the global cell size.

This module performs a deterministic defect correction in each port's rigid
local frame:

    global corrected self = global coarse self
                          + local fine self - local coarse self.

The local problem keeps the physical finite-cross-section stranded source and
its own package, but removes global translation/rotation.  This is legitimate
for the local defect because the surrounding seawater/material laws are
isotropic; global pose, the other port and long-range boundary interaction stay
in the global solve.  The correction is applied consistently to Z, D_vol,
D_out and the diagonal entries of every modal H_j.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from .unified_geometry import UnifiedUWPTGeometry
from .unified_open_boundary import OpenBoundaryBackground


@dataclass(frozen=True)
class SelfCorrectionResult:
    z: np.ndarray
    d_vol: np.ndarray
    d_out: np.ndarray
    modal_h: np.ndarray | None
    audit: dict


def _config(background):
    cfg = copy.deepcopy(dict(getattr(background, "self_correction_config", {}) or {}))
    cfg.setdefault("enabled", True)
    cfg.setdefault("fine_step", 0.003)
    cfg.setdefault("core_padding", 0.006)
    cfg.setdefault("boundary_padding", 0.04)
    cfg.setdefault("growth", 1.5)
    cfg.setdefault("max_step", 0.02)
    return cfg


def _parent_fine_step(background):
    cfg = getattr(background, "background_config", None)
    if isinstance(cfg, dict) and "fine_step" in cfg:
        return float(cfg["fine_step"])
    return float(min(np.min(background.dx), np.min(background.dy), np.min(background.dz)))


def _canonical_port_geometry(geometry, port):
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    p = int(port)
    coil = g.coils[p]
    package = g.packages[p]
    # Production packages are rigidly attached to their coil.  Do not silently
    # apply a canonical correction if a future geometry introduces relative pose.
    if (
        np.linalg.norm(np.asarray(package.pose.translation) - np.asarray(coil.pose.translation)) > 1e-10
        or np.linalg.norm(np.asarray(package.pose.rotation) - np.asarray(coil.pose.rotation)) > 1e-10
    ):
        raise ValueError("local self correction requires each package to be rigidly attached to its coil")
    cm = coil.to_mapping()
    cm["name"] = coil.name
    cm["translation"] = [0.0, 0.0, 0.0]
    cm["angles"] = [0.0, 0.0, 0.0]
    pm = package.to_mapping()
    pm["translation"] = [0.0, 0.0, 0.0]
    pm["angles"] = [0.0, 0.0, 0.0]
    local = UnifiedUWPTGeometry.from_mapping({"coils": [cm], "packages": [pm]})
    return g, local


def _local_background(parent, geometry, port, fine_step):
    g, local_geometry = _canonical_port_geometry(geometry, port)
    p = int(port)
    package = local_geometry.packages[0]
    cfg = _config(parent)
    core_padding = float(cfg["core_padding"])
    boundary_padding = float(cfg["boundary_padding"])
    if core_padding <= 0.0 or boundary_padding <= core_padding:
        raise ValueError("self_correction requires boundary_padding > core_padding > 0")
    half = np.asarray(package.half_extent, float)
    bounds_half = half + boundary_padding
    core_half = half + core_padding
    local_cfg = {
        "bounds": [[-float(v), float(v)] for v in bounds_half],
        "core_center": [0.0, 0.0, 0.0],
        "core_half_extent": core_half.tolist(),
        "fine_step": float(fine_step),
        "growth": float(cfg["growth"]),
        "max_step": max(float(fine_step), float(cfg["max_step"])),
        # Prevent recursive use if a caller later routes local truth through a
        # higher-level tensor helper rather than the direct solve below.
        "self_correction": {"enabled": False},
    }
    background = OpenBoundaryBackground.from_config(
        local_cfg,
        frequency_hz=parent.frequency_hz,
        materials=parent.materials,
        coil_materials=(parent.coil_materials[p],),
        package_materials=(parent.package_materials[p],),
        seawater_material=parent.seawater_material,
        ambient_temperature=parent.ambient_temperature,
    )
    return g, local_geometry, background


def _phi_on_local_grid(parent, global_geometry, port, local_background, phi):
    phi = np.asarray(phi, float)
    if phi.ndim != 2 or phi.shape[0] != parent.n_cells:
        raise ValueError("self-correction modal basis has incompatible shape")
    coil = global_geometry.coils[int(port)]
    global_points = coil.pose.apply(local_background.cell_centers)
    values = phi.reshape(parent.nx, parent.ny, parent.nz, phi.shape[1])
    interpolation = RegularGridInterpolator(
        parent.cell_axes,
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    local_phi = np.asarray(interpolation(global_points), float)
    if local_phi.shape != (local_background.n_cells, phi.shape[1]) or np.any(~np.isfinite(local_phi)):
        raise ValueError(
            "local self-correction support left the parent thermal grid; enlarge BACKGROUND bounds"
        )
    return local_phi


def _solve_local(parent, geometry, port, fine_step, phi=None):
    global_geometry, local_geometry, local = _local_background(parent, geometry, port, fine_step)
    context = local.geometry_context(local_geometry, assemble_thermal=False)
    A = local.em_operator(context, None)
    B = np.asarray(local.rhs_matrix(context), complex)
    if B.shape[1] != 1:
        raise AssertionError("canonical self-correction problem must have exactly one port")
    rhs = B[:, 0]
    try:
        lu = spla.splu(A.tocsc())
        field = np.asarray(lu.solve(rhs), complex).reshape(-1)
    except RuntimeError:
        field = np.asarray(spla.spsolve(A, rhs), complex).reshape(-1)
    if np.any(~np.isfinite(field)):
        raise FloatingPointError("local self-correction Maxwell solve produced non-finite fields")
    residual = float(
        np.linalg.norm(rhs - A @ field) / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    )
    source = np.asarray(context.source_shape[:, 0], float)
    z = complex(-source @ field)
    sigma = np.asarray(local.cell_properties(context, None, em=True)[0], float)
    edge_loss = np.asarray(local.edge_cell_hodge @ sigma).reshape(-1)
    d = float(np.real(field.conj() @ (edge_loss * field)))
    outward_weights = np.asarray(local.outward_loss_weights(), float).reshape(-1)
    d_out = float(np.real(field.conj() @ (outward_weights * field)))
    q_cells = np.asarray(
        0.5 * sigma * np.asarray(local.edge_cell_hodge.T @ (np.abs(field) ** 2)).reshape(-1),
        float,
    )
    modal = None
    if phi is not None:
        local_phi = _phi_on_local_grid(parent, global_geometry, port, local, phi)
        modal = np.asarray(2.0 * (local_phi.T @ q_cells), float)
    scale = max(abs(z.real), abs(d) + abs(d_out), np.finfo(float).tiny)
    balance = float(abs(z.real - d - d_out) / scale)
    return {
        "z": z,
        "d_vol": d,
        "d_out": d_out,
        "modal_h": modal,
        "linear_relative_residual": residual,
        "power_balance_relative_error": balance,
        "n_cells": int(local.n_cells),
        "n_edges": int(local.n_edges),
        "fine_step": float(fine_step),
    }


def apply_local_self_correction(background, geometry, z, d_vol, d_out, *, phi=None, modal_h=None):
    """Apply diagonal local fine-minus-coarse self defects to port tensors."""
    zc = np.asarray(z, complex).copy()
    dc = np.asarray(d_vol, complex).copy()
    oc = np.asarray(d_out, complex).copy()
    hc = None if modal_h is None else np.asarray(modal_h, complex).copy()
    cfg = _config(background)
    n = zc.shape[0]
    if zc.shape != (n, n) or dc.shape != (n, n) or oc.shape != (n, n):
        raise ValueError("self correction requires shape-compatible square port tensors")
    if not bool(cfg.get("enabled", True)):
        return SelfCorrectionResult(zc, dc, oc, hc, {"enabled": False, "ports": []})
    if phi is not None and hc is None:
        raise ValueError("modal_h is required when phi is supplied to self correction")
    if hc is not None and (hc.ndim != 3 or hc.shape[1:] != (n, n)):
        raise ValueError("modal_h has incompatible self-correction shape")

    coarse_step = _parent_fine_step(background)
    fine_step = float(cfg["fine_step"])
    if not 0.0 < fine_step < coarse_step:
        raise ValueError(
            f"self_correction.fine_step={fine_step:g} must be smaller than parent fine_step={coarse_step:g}"
        )
    ports = []
    for p in range(n):
        coarse = _solve_local(background, geometry, p, coarse_step, phi=phi)
        fine = _solve_local(background, geometry, p, fine_step, phi=phi)
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
        ports.append({
            "port": int(p),
            "coarse_step": coarse_step,
            "fine_step": fine_step,
            "delta_z_real": float(dz.real),
            "delta_z_imag": float(dz.imag),
            "delta_d_vol": dd,
            "delta_d_out": do,
            "coarse": {k: v for k, v in coarse.items() if k != "modal_h"},
            "fine": {k: v for k, v in fine.items() if k != "modal_h"},
            "maximum_modal_delta": None if modal_delta is None else float(np.max(np.abs(modal_delta))),
        })

    # Exact symmetry/Hermiticity is restored after diagonal replacement.
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
    return SelfCorrectionResult(
        zc,
        dc,
        oc,
        hc,
        {
            "enabled": True,
            "model": "canonical_local_fine_minus_coarse_self_defect_v1",
            "coarse_step": coarse_step,
            "fine_step": fine_step,
            "corrected_power_balance_relative_error": corrected_balance,
            "ports": ports,
        },
    )


__all__ = ["SelfCorrectionResult", "apply_local_self_correction"]
