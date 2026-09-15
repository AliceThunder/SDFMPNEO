"""Local multiscale correction for unresolved Maxwell self response.

The production background is intentionally coarse enough to make many-geometry
truth generation practical. Mutual/far-field coupling is already converged on
that grid, while the diagonal source self response is not when the conductor
cross section is much smaller than the global cell size.

This module performs a deterministic defect correction in each port's rigid
local frame:

    global corrected self = global coarse self
                          + local fine defect - local coarse defect.

The full global Maxwell solve always retains the theory-defined open two-terminal
source and its longitudinal terminal/charge response.  A canonical local box
cannot reproduce the external return path, global dielectric environment, or
full-domain terminal capacitance, so the *local defect extractor* removes only
the pure scalar-gradient self energy and refines the remaining localizable
transverse/cross response.  This is deliberately not a projection of the
physical global source.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from .unified_compensated_field import field_abs2, field_linear_dot
from .unified_geometry import UnifiedUWPTGeometry
from .unified_gradient_block_maxwell import build_gradient_block
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


def _relative_identity_error(value, reference):
    return float(
        np.linalg.norm(np.asarray(value) - np.asarray(reference))
        / max(
            float(np.linalg.norm(np.asarray(value))),
            float(np.linalg.norm(np.asarray(reference))),
            np.finfo(float).tiny,
        )
    )


def _localized_self_response(
    local,
    context,
    A,
    rhs,
    field,
    source,
    sigma,
    edge_loss,
    outward_weights,
    *,
    local_phi=None,
):
    """Extract the localizable self response without changing the physical solve.

    The full field contains a scalar-gradient terminal response whose length
    scale is global: it depends on the return path, dielectric environment and
    open-domain boundary.  Replacing that term with a canonical local-box value
    is not a legitimate local fine-minus-coarse correction.  We therefore keep
    the full field solve intact, compute its exact compatible gradient component,
    and subtract only the *pure longitudinal self energy* from local defect
    quantities.  Transverse/longitudinal cross terms remain in the refinable
    quantity, so only the nonlocal pure-gradient term is frozen at global scale.
    """
    gradient = build_gradient_block(local, context, check_topology=True)
    longitudinal = np.asarray(gradient.solve(rhs), complex).reshape(-1)
    if longitudinal.shape != (local.n_edges,) or np.any(~np.isfinite(longitudinal)):
        raise FloatingPointError("local Maxwell longitudinal field is invalid")

    full_abs2 = field_abs2(field)
    longitudinal_abs2 = np.abs(longitudinal) ** 2
    full_z = -field_linear_dot(source, field)
    grad_action = np.asarray(A @ longitudinal, complex).reshape(-1)
    grad_energy = complex(np.vdot(longitudinal, grad_action))
    # For e^{+i wt}, E^H A E = i w Z* on a physical one-port solution, hence
    # Z_energy = Im(E^H A E)/w + i Re(E^H A E)/w.  Applying this to the exact
    # gradient component identifies the pure longitudinal terminal self energy.
    longitudinal_z = complex(
        float(np.imag(grad_energy) / local.omega),
        float(np.real(grad_energy) / local.omega),
    )
    refinable_z = complex(full_z - longitudinal_z)

    full_d = float(np.dot(np.asarray(edge_loss, float), full_abs2))
    longitudinal_d = float(np.dot(np.asarray(edge_loss, float), longitudinal_abs2))
    refinable_d = float(full_d - longitudinal_d)
    full_out = float(np.dot(np.asarray(outward_weights, float), full_abs2))
    longitudinal_out = float(np.dot(np.asarray(outward_weights, float), longitudinal_abs2))
    refinable_out = float(full_out - longitudinal_out)

    q_full = np.asarray(
        0.5
        * np.asarray(sigma, float)
        * np.asarray(local.edge_cell_hodge.T @ full_abs2).reshape(-1),
        float,
    )
    q_longitudinal = np.asarray(
        0.5
        * np.asarray(sigma, float)
        * np.asarray(local.edge_cell_hodge.T @ longitudinal_abs2).reshape(-1),
        float,
    )
    q_refinable = np.asarray(q_full - q_longitudinal, float)

    modal_refinable = None
    if local_phi is not None:
        modal_refinable = np.asarray(2.0 * (local_phi.T @ q_refinable), float)

    scale = max(
        abs(float(np.real(refinable_z))),
        abs(refinable_d) + abs(refinable_out),
        np.finfo(float).tiny,
    )
    balance = float(abs(refinable_z.real - refinable_d - refinable_out) / scale)
    return {
        "localized_z": refinable_z,
        "localized_d_vol": refinable_d,
        "localized_d_out": refinable_out,
        "localized_modal_h": modal_refinable,
        "localized_power_balance_relative_error": balance,
        "longitudinal_z": longitudinal_z,
        "longitudinal_d_vol": longitudinal_d,
        "longitudinal_d_out": longitudinal_out,
        "longitudinal_field_relative_norm": float(
            np.linalg.norm(longitudinal)
            / max(np.linalg.norm(np.asarray(field, complex)), np.finfo(float).tiny)
        ),
    }


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

    direct_d = float(2.0 * np.sum(q_cells))
    joule_total_error = _relative_identity_error(d, direct_d)

    local_phi = None
    modal = None
    modal_error = 0.0
    if phi is not None:
        local_phi = _phi_on_local_grid(parent, global_geometry, port, local, phi)
        direct_modal = np.asarray(2.0 * (local_phi.T @ q_cells), float)
        modal = direct_modal.copy()
        modal_error = _relative_identity_error(modal, direct_modal)

    scale = max(abs(z.real), abs(d) + abs(d_out), np.finfo(float).tiny)
    balance = float(abs(z.real - d - d_out) / scale)
    localized = _localized_self_response(
        local,
        context,
        A,
        rhs,
        field,
        source,
        sigma,
        edge_loss,
        outward_weights,
        local_phi=local_phi,
    )
    return {
        "z": z,
        "d_vol": d,
        "d_out": d_out,
        "modal_h": modal,
        **localized,
        "linear_relative_residual": residual,
        "power_balance_relative_error": balance,
        "joule_total_power_relative_error": joule_total_error,
        "joule_modal_contraction_relative_error": modal_error,
        "n_cells": int(local.n_cells),
        "n_edges": int(local.n_edges),
        "fine_step": float(fine_step),
    }


def apply_local_self_correction(background, geometry, z, d_vol, d_out, *, phi=None, modal_h=None):
    """Apply diagonal localized fine-minus-coarse self defects to port tensors."""
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
    maximum_total_identity_error = 0.0
    maximum_modal_identity_error = 0.0
    maximum_localized_balance_error = 0.0
    for p in range(n):
        coarse = _solve_local(background, geometry, p, coarse_step, phi=phi)
        fine = _solve_local(background, geometry, p, fine_step, phi=phi)
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
        dz = fine["localized_z"] - coarse["localized_z"]
        dd = float(fine["localized_d_vol"] - coarse["localized_d_vol"])
        do = float(fine["localized_d_out"] - coarse["localized_d_out"])
        zc[p, p] += dz
        dc[p, p] += dd
        oc[p, p] += do
        modal_delta = None
        if phi is not None:
            modal_delta = np.asarray(
                fine["localized_modal_h"] - coarse["localized_modal_h"], float
            )
            hc[:, p, p] += modal_delta
        ports.append({
            "port": int(p),
            "coarse_step": coarse_step,
            "fine_step": fine_step,
            "delta_z_real": float(dz.real),
            "delta_z_imag": float(dz.imag),
            "delta_d_vol": dd,
            "delta_d_out": do,
            "coarse": {k: v for k, v in coarse.items() if k not in ("modal_h", "localized_modal_h")},
            "fine": {k: v for k, v in fine.items() if k not in ("modal_h", "localized_modal_h")},
            "maximum_modal_delta": None if modal_delta is None else float(np.max(np.abs(modal_delta))),
        })

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
            "model": "canonical_local_transverse_fine_minus_coarse_self_defect_v2",
            "coarse_step": coarse_step,
            "fine_step": fine_step,
            "corrected_power_balance_relative_error": corrected_balance,
            "maximum_localized_power_balance_relative_error": float(maximum_localized_balance_error),
            "maximum_joule_total_power_relative_error": float(maximum_total_identity_error),
            "maximum_joule_modal_contraction_relative_error": float(maximum_modal_identity_error),
            "ports": ports,
        },
    )


__all__ = [
    "SelfCorrectionResult",
    "_localized_self_response",
    "apply_local_self_correction",
]
