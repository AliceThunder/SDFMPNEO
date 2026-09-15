"""Boundary-conditioned longitudinal near-field defect for production truth.

The canonical local Maxwell correction intentionally excludes pure longitudinal
terminal self energy because an isolated local box cannot reproduce the global
return path, dielectric environment or open boundary.  A uniformly refined
*global* scalar reference was also found to converge too slowly: 9 mm -> 6.75 mm
changed the longitudinal self resistance by O(40%), while the global outward
term was already converged.  That identifies a source-near-field discretization
error rather than a far-boundary error.

This module keeps the nonlocal part global and refines only that near field:

1. solve the compatible global scalar problem on the current production grid;
2. extract a Cartesian patch whose coarse axes are an exact subset of the
   production x/y/z grid;
3. interpolate the global scalar potential onto the patch boundary and impose
   it as Dirichlet data (the artificial patch boundary has no Silver-Muller
   term);
4. solve the same physical terminal source/material problem on the coarse patch
   and a locally refined patch; and
5. add only the fine-minus-coarse self defect to the global diagonal.

Thus the global scalar solution still owns return-path/open-boundary physics,
while the local patch supplies only unresolved source-near-field resolution.
For power consistency, Re(delta Z) is set exactly equal to delta D_vol; the
Dirichlet patch source-reaction real part contains artificial interface flux and
is retained only as a diagnostic.  Im(delta Z) comes from the same-boundary
fine-minus-coarse terminal reaction.  D_out is not locally corrected because it
is a physical outer-boundary quantity, not a patch-interface quantity.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from .unified_background import stretched_axis
from .unified_geometry import UnifiedUWPTGeometry
from .unified_gradient_block_maxwell import build_gradient_block, gradient_operator, node_coordinates


_MODEL = "global_boundary_conditioned_longitudinal_nearfield_defect_v1"


def _hermitian(value):
    a = np.asarray(value, complex)
    return 0.5 * (a + a.conj().T)


def _relative(value, reference, *scales):
    a = np.asarray(value)
    b = np.asarray(reference)
    denominator = max(
        float(np.linalg.norm(a)),
        float(np.linalg.norm(b)),
        *(float(abs(complex(scale))) for scale in scales),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(a - b) / denominator)


def _geometry_key(geometry):
    if hasattr(geometry, "canonical_json"):
        return geometry.canonical_json()
    try:
        return UnifiedUWPTGeometry.from_mapping(geometry).canonical_json()
    except Exception:
        return repr(geometry)


def _background_step(background):
    cfg = getattr(background, "background_config", None)
    if isinstance(cfg, dict) and "fine_step" in cfg:
        return float(cfg["fine_step"])
    return float(min(np.min(background.dx), np.min(background.dy), np.min(background.dz)))


def _config(background):
    root = copy.deepcopy(dict(getattr(background, "background_config", {}) or {}))
    self_cfg = copy.deepcopy(
        dict(getattr(background, "self_correction_config", {}) or root.get("self_correction", {}) or {})
    )
    own = copy.deepcopy(dict(root.get("global_longitudinal_correction", {}) or {}))
    mesh = copy.deepcopy(dict(root.get("mesh_check", {}) or {}))
    fine = float(own.get("fine_step", self_cfg.get("fine_step", 0.003)))
    validation = float(
        own.get(
            "validation_fine_step",
            self_cfg.get("validation_fine_step", 0.75 * fine),
        )
    )
    if not (0.0 < validation < fine):
        raise ValueError("longitudinal defect validation_fine_step must be smaller than fine_step")
    return {
        "enabled": bool(own.get("enabled", True)),
        "fine_step": fine,
        "validation_fine_step": validation,
        "relative_tolerance": float(
            own.get("relative_tolerance", mesh.get("relative_tolerance", 1e-1))
        ),
        "core_padding": float(own.get("core_padding", self_cfg.get("core_padding", 0.006))),
        "boundary_padding": float(
            own.get("boundary_padding", self_cfg.get("boundary_padding", 0.04))
        ),
        "growth": float(own.get("growth", self_cfg.get("growth", 1.5))),
        "max_step": float(own.get("max_step", self_cfg.get("max_step", 0.02))),
    }


def _remember_context(background, geometry, context):
    background._sdfmpneo_global_longitudinal_last_context = (
        _geometry_key(geometry), context
    )


def _cached_context(background, geometry):
    row = getattr(background, "_sdfmpneo_global_longitudinal_last_context", None)
    if not isinstance(row, tuple) or len(row) != 2:
        return None
    key, context = row
    return context if key == _geometry_key(geometry) else None


def _single_port_geometry(geometry, port):
    g = geometry if isinstance(geometry, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(geometry)
    p = int(port)
    coil = g.coils[p]
    package = g.packages[p]
    cm = coil.to_mapping()
    cm["name"] = coil.name
    pm = package.to_mapping()
    return g, UnifiedUWPTGeometry.from_mapping({"coils": [cm], "packages": [pm]})


def _select_axis(axis, lo, hi):
    values = np.asarray(axis, float)
    left = max(0, int(np.searchsorted(values, float(lo), side="right")) - 1)
    right = min(len(values) - 1, int(np.searchsorted(values, float(hi), side="left")))
    while right - left < 2 and (left > 0 or right < len(values) - 1):
        if left > 0:
            left -= 1
        if right < len(values) - 1 and right - left < 2:
            right += 1
    if right - left < 2:
        raise ValueError("longitudinal patch needs at least two cells per axis")
    return values[left:right + 1].copy()


def _patch_axes(parent, package, cfg):
    rotation = np.asarray(package.pose.rotation, float)
    half = np.asarray(package.half_extent, float)
    center = np.asarray(package.pose.translation, float)
    aabb_half = np.abs(rotation) @ half
    padding = float(cfg["boundary_padding"])
    axes = tuple(
        _select_axis(axis, center[k] - aabb_half[k] - padding, center[k] + aabb_half[k] + padding)
        for k, axis in enumerate((parent.x, parent.y, parent.z))
    )
    return axes, center, aabb_half


def _refined_axes(coarse_axes, center, aabb_half, cfg, fine_step):
    out = []
    core_half = np.asarray(aabb_half, float) + float(cfg["core_padding"])
    for k, axis in enumerate(coarse_axes):
        lo = float(axis[0])
        hi = float(axis[-1])
        available = max(min(float(center[k] - lo), float(hi - center[k])), np.finfo(float).tiny)
        h = min(float(core_half[k]), 0.999 * available)
        out.append(
            stretched_axis(
                (lo, hi),
                h,
                float(fine_step),
                float(cfg["growth"]),
                max(float(fine_step), float(cfg["max_step"])),
                center=float(center[k]),
            )
        )
    return tuple(out)


def _make_patch_background(parent, port, axes, *, fine_step):
    p = int(port)
    patch = parent.__class__(
        *axes,
        frequency_hz=parent.frequency_hz,
        materials=parent.materials,
        coil_materials=(parent.coil_materials[p],),
        package_materials=(parent.package_materials[p],),
        seawater_material=parent.seawater_material,
        ambient_temperature=parent.ambient_temperature,
    )
    patch.background_config = {
        "fine_step": float(fine_step),
        "self_correction": {"enabled": False},
    }
    patch.self_correction_config = {"enabled": False}
    return patch


def _global_scalar_potentials(background, geometry):
    context = _cached_context(background, geometry)
    if context is None:
        context = background.geometry_context(geometry, assemble_thermal=False)
        _remember_context(background, geometry, context)
    block = build_gradient_block(background, context, check_topology=True)
    B = np.asarray(background.rhs_matrix(context), complex)
    source = np.asarray(context.source_shape, float)
    G = block.gradient
    n_nodes = (background.nx + 1) * (background.ny + 1) * (background.nz + 1)
    potentials = np.zeros((n_nodes, B.shape[1]), complex)
    residuals = []
    z = np.zeros(B.shape[1], complex)
    d = np.zeros(B.shape[1], float)
    sigma = np.asarray(background.cell_properties(context, None, em=True)[0], float)
    edge_loss = np.asarray(background.edge_cell_hodge @ sigma, float).reshape(-1)
    for p in range(B.shape[1]):
        scalar_rhs = np.asarray(G.T @ B[:, p], complex).reshape(-1)
        phi = np.asarray(block.factor.solve(scalar_rhs), complex).reshape(-1)
        residuals.append(
            float(
                np.linalg.norm(scalar_rhs - block.scalar_matrix @ phi)
                / max(float(np.linalg.norm(scalar_rhs)), np.finfo(float).tiny)
            )
        )
        potentials[1:, p] = phi
        field = np.asarray(G @ phi, complex).reshape(-1)
        z[p] = complex(-source[:, p] @ field)
        d[p] = float(np.dot(edge_loss, np.abs(field) ** 2))
    return context, potentials, {
        "fine_step": _background_step(background),
        "scalar_dofs": int(block.scalar_dofs),
        "maximum_projection_relative_residual": max(residuals, default=0.0),
        "z_real": np.real(z).tolist(),
        "z_imag": np.imag(z).tolist(),
        "d_vol": d.tolist(),
    }


def _boundary_values(parent, global_potential, patch, boundary_mask):
    values = np.asarray(global_potential, complex).reshape(
        parent.nx + 1, parent.ny + 1, parent.nz + 1
    )
    interpolation = RegularGridInterpolator(
        (parent.x, parent.y, parent.z),
        values,
        method="linear",
        bounds_error=True,
    )
    coords = node_coordinates(patch)
    return np.asarray(interpolation(coords[boundary_mask]), complex).reshape(-1)


def _interpolate_cell_basis(parent, patch, phi):
    values = np.asarray(phi, float)
    if values.ndim != 2 or values.shape[0] != parent.n_cells:
        raise ValueError("longitudinal modal basis has incompatible parent shape")
    grid = values.reshape(parent.nx, parent.ny, parent.nz, values.shape[1])
    interpolation = RegularGridInterpolator(
        parent.cell_axes,
        grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    out = np.asarray(interpolation(patch.cell_centers), float)
    if out.shape != (patch.n_cells, values.shape[1]) or np.any(~np.isfinite(out)):
        raise ValueError("longitudinal patch left the parent thermal-basis domain")
    return out


def _volume_edge_diagonal(background, context):
    sigma, eps, _mu_inv, _k, _cap, _temperature = background.cell_properties(
        context, None, em=True
    )
    sigma = np.asarray(sigma, float)
    eps = np.asarray(eps, float)
    hs = np.asarray(background.edge_cell_hodge @ sigma).reshape(-1)
    he = np.asarray(background.edge_cell_hodge @ eps).reshape(-1)
    diagonal = 1j * background.omega * hs.astype(complex) - (background.omega ** 2) * he
    if diagonal.shape != (background.n_edges,) or np.any(~np.isfinite(diagonal)):
        raise FloatingPointError("longitudinal patch edge mass diagonal is invalid")
    return np.asarray(diagonal, complex), sigma, hs


def _boundary_node_mask(background):
    mask = np.zeros((background.nx + 1, background.ny + 1, background.nz + 1), bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask.reshape(-1)


def _patch_state(parent, patch, local_geometry, global_potential, *, phi=None, fine_step):
    context = patch.geometry_context(local_geometry, assemble_thermal=False)
    B = np.asarray(patch.rhs_matrix(context), complex)
    if B.shape[1] != 1:
        raise AssertionError("longitudinal self patch must contain exactly one port")
    rhs = B[:, 0]
    source = np.asarray(context.source_shape[:, 0], float)
    G = gradient_operator(patch, gauge_fixed=False)
    diagonal, sigma, _hs = _volume_edge_diagonal(patch, context)
    scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
    scalar.sum_duplicates()
    scalar.eliminate_zeros()
    scalar_rhs = np.asarray(G.T @ rhs, complex).reshape(-1)

    boundary = _boundary_node_mask(patch)
    interior = ~boundary
    phi_nodes = np.zeros(G.shape[1], complex)
    phi_nodes[boundary] = _boundary_values(parent, global_potential, patch, boundary)
    Sii = scalar[interior][:, interior].tocsc()
    Sib = scalar[interior][:, boundary].tocsr()
    rhs_i = scalar_rhs[interior] - Sib @ phi_nodes[boundary]
    try:
        factor = spla.splu(
            Sii,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        factor = spla.splu(Sii)
    phi_nodes[interior] = np.asarray(factor.solve(rhs_i), complex).reshape(-1)
    residual = float(
        np.linalg.norm(rhs_i - Sii @ phi_nodes[interior])
        / max(float(np.linalg.norm(rhs_i)), np.finfo(float).tiny)
    )
    field = np.asarray(G @ phi_nodes, complex).reshape(-1)
    abs2 = np.abs(field) ** 2
    z_reaction = complex(-source @ field)
    edge_loss = np.asarray(patch.edge_cell_hodge @ sigma, float).reshape(-1)
    d_vol = float(np.dot(edge_loss, abs2))
    q_cells = np.asarray(
        0.5 * sigma * np.asarray(patch.edge_cell_hodge.T @ abs2).reshape(-1),
        float,
    )
    modal = None
    if phi is not None:
        local_phi = _interpolate_cell_basis(parent, patch, phi)
        modal = np.asarray(2.0 * (local_phi.T @ q_cells), float)
    return {
        "z_reaction": z_reaction,
        "d_vol": d_vol,
        "modal_h": modal,
        "scalar_relative_residual": residual,
        "n_cells": int(patch.n_cells),
        "scalar_dofs": int(np.count_nonzero(interior)),
        "fine_step": float(fine_step),
    }


def _port_states(background, geometry, port, global_potential, target_steps, *, phi=None):
    cfg = _config(background)
    global_geometry, local_geometry = _single_port_geometry(geometry, port)
    package = global_geometry.packages[int(port)]
    coarse_axes, center, aabb_half = _patch_axes(background, package, cfg)
    coarse = _make_patch_background(
        background, port, coarse_axes, fine_step=_background_step(background)
    )
    states = {
        "coarse": _patch_state(
            background,
            coarse,
            local_geometry,
            global_potential,
            phi=phi,
            fine_step=_background_step(background),
        )
    }
    for step in target_steps:
        axes = _refined_axes(coarse_axes, center, aabb_half, cfg, float(step))
        patch = _make_patch_background(background, port, axes, fine_step=float(step))
        states[float(step)] = _patch_state(
            background,
            patch,
            local_geometry,
            global_potential,
            phi=phi,
            fine_step=float(step),
        )
    return states


def _defect(coarse, fine):
    dd = float(fine["d_vol"] - coarse["d_vol"])
    reaction_delta = complex(fine["z_reaction"] - coarse["z_reaction"])
    dz = complex(dd, reaction_delta.imag)
    modal = None
    if fine["modal_h"] is not None:
        modal = np.asarray(fine["modal_h"] - coarse["modal_h"], float)
    return {
        "delta_z": dz,
        "delta_d_vol": dd,
        "delta_d_out": 0.0,
        "delta_modal_h": modal,
        "reaction_delta_z": reaction_delta,
        "interface_real_flux_defect": float(reaction_delta.real - dd),
    }


def _correction(background, geometry, *, phi=None):
    cfg = _config(background)
    n = len(background.coil_materials)
    zero_modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], n), float)
    if not cfg["enabled"] or _background_step(background) <= cfg["fine_step"]:
        return {
            "delta_z": np.zeros(n, complex),
            "delta_d_vol": np.zeros(n, float),
            "delta_d_out": np.zeros(n, float),
            "delta_modal_h": zero_modal,
            "audit": {"enabled": False, "model": _MODEL},
        }

    _context, potentials, global_audit = _global_scalar_potentials(background, geometry)
    dz = np.zeros(n, complex)
    dd = np.zeros(n, float)
    dm = zero_modal
    rows = []
    max_patch_residual = 0.0
    fine = float(cfg["fine_step"])
    for p in range(n):
        states = _port_states(
            background, geometry, p, potentials[:, p], [fine], phi=phi
        )
        row = _defect(states["coarse"], states[fine])
        dz[p] = row["delta_z"]
        dd[p] = row["delta_d_vol"]
        if dm is not None:
            dm[:, p] = row["delta_modal_h"]
        max_patch_residual = max(
            max_patch_residual,
            float(states["coarse"]["scalar_relative_residual"]),
            float(states[fine]["scalar_relative_residual"]),
        )
        rows.append({
            "port": int(p),
            "coarse": states["coarse"],
            "fine": states[fine],
            "delta_z_real": float(row["delta_z"].real),
            "delta_z_imag": float(row["delta_z"].imag),
            "delta_d_vol": float(row["delta_d_vol"]),
            "reaction_delta_z_real": float(row["reaction_delta_z"].real),
            "interface_real_flux_defect": float(row["interface_real_flux_defect"]),
        })
    return {
        "delta_z": dz,
        "delta_d_vol": dd,
        "delta_d_out": np.zeros(n, float),
        "delta_modal_h": dm,
        "audit": {
            "enabled": True,
            "model": _MODEL,
            "current_fine_step": float(_background_step(background)),
            "local_reference_fine_step": fine,
            "delta_z_real": np.real(dz).tolist(),
            "delta_z_imag": np.imag(dz).tolist(),
            "delta_d_vol": dd.tolist(),
            "delta_d_out": np.zeros(n, float).tolist(),
            "correction_power_balance_relative_error": 0.0,
            "maximum_patch_scalar_relative_residual": float(max_patch_residual),
            "global_scalar": global_audit,
            "ports": rows,
        },
    }


def apply_global_longitudinal_reference(
    background,
    geometry,
    z,
    d_vol,
    d_out=None,
    *,
    phi=None,
    modal_h=None,
):
    """Apply the boundary-conditioned longitudinal near-field defect.

    The historical function name is retained as a compatibility surface for the
    installed truth/Gate adapters; ``audit['model']`` records the new semantics.
    """
    zc = np.asarray(z, complex).copy()
    dc = np.asarray(d_vol, complex).copy()
    oc = None if d_out is None else np.asarray(d_out, complex).copy()
    hc = None if modal_h is None else np.asarray(modal_h, complex).copy()
    correction = _correction(background, geometry, phi=phi)
    for p, value in enumerate(correction["delta_z"]):
        zc[p, p] += value
        dc[p, p] += float(correction["delta_d_vol"][p])
        if oc is not None:
            oc[p, p] += 0.0
        if hc is not None:
            hc[:, p, p] += np.asarray(correction["delta_modal_h"][:, p], float)
    return zc, dc, oc, hc, correction["audit"]


def audit_reference_convergence(background, geometry):
    """Certify the local longitudinal defect with a finer patch, same boundary trace."""
    cfg = _config(background)
    if not cfg["enabled"]:
        return {
            "model": _MODEL,
            "converged": True,
            "maximum_relative_error": 0.0,
            "relative_tolerance": float(cfg["relative_tolerance"]),
            "samples": [],
        }
    _context, potentials, global_audit = _global_scalar_potentials(background, geometry)
    fine = float(cfg["fine_step"])
    validation = float(cfg["validation_fine_step"])
    rows = []
    worst = 0.0
    for p in range(len(background.coil_materials)):
        states = _port_states(
            background,
            geometry,
            p,
            potentials[:, p],
            [fine, validation],
            phi=None,
        )
        df = _defect(states["coarse"], states[fine])
        dv = _defect(states["coarse"], states[validation])
        dissipation_scale = max(
            abs(float(dv["delta_d_vol"])),
            abs(float(states[validation]["d_vol"])),
            abs(float(states["coarse"]["d_vol"])),
            np.finfo(float).tiny,
        )
        z_scale = max(abs(complex(dv["delta_z"])), dissipation_scale)
        reactive_scale = max(
            abs(float(dv["delta_z"].imag)),
            abs(float(states[validation]["z_reaction"].imag)),
            dissipation_scale,
        )
        zerr = _relative(df["delta_z"], dv["delta_z"], z_scale)
        rerr = abs(float(df["delta_d_vol"] - dv["delta_d_vol"])) / dissipation_scale
        xerr = abs(float(df["delta_z"].imag - dv["delta_z"].imag)) / reactive_scale
        derr = abs(float(df["delta_d_vol"] - dv["delta_d_vol"])) / dissipation_scale
        max_residual = max(
            float(states["coarse"]["scalar_relative_residual"]),
            float(states[fine]["scalar_relative_residual"]),
            float(states[validation]["scalar_relative_residual"]),
        )
        row_worst = max(zerr, rerr, xerr, derr)
        worst = max(worst, row_worst)
        row = {
            "port": int(p),
            "fine_step": fine,
            "validation_fine_step": validation,
            "fine_delta_z": df["delta_z"],
            "validation_delta_z": dv["delta_z"],
            "fine_delta_d_vol": float(df["delta_d_vol"]),
            "validation_delta_d_vol": float(dv["delta_d_vol"]),
            "relative_z_error": float(zerr),
            "relative_resistive_error": float(rerr),
            "relative_reactive_error": float(xerr),
            "relative_d_vol_error": float(derr),
            "relative_outward_partition_significance": 0.0,
            "maximum_patch_scalar_relative_residual": float(max_residual),
            "maximum_relative_error": float(row_worst),
            "coarse": states["coarse"],
            "fine": states[fine],
            "validation": states[validation],
        }
        rows.append(row)
        print(
            "boundary-conditioned longitudinal defect Gate: "
            f"port={p + 1}, z={zerr:.3e}, R={rerr:.3e}, X={xerr:.3e}, "
            f"Dvol={derr:.3e}, max={row_worst:.3e}",
            flush=True,
        )
    return {
        "model": _MODEL,
        "fine_step": fine,
        "validation_fine_step": validation,
        "relative_tolerance": float(cfg["relative_tolerance"]),
        "maximum_relative_error": float(worst),
        "converged": bool(worst <= float(cfg["relative_tolerance"])),
        "global_scalar": global_audit,
        "samples": rows,
    }


def _resolve_settings(settings, background):
    bg_cfg = settings["BACKGROUND"]
    self_cfg = dict(bg_cfg.get("self_correction", {}) or {})
    mesh_cfg = dict(bg_cfg.get("mesh_check", {}) or {})
    own = bg_cfg.setdefault("global_longitudinal_correction", {})
    fine = float(own.get("fine_step", self_cfg.get("fine_step", 0.003)))
    own.setdefault("enabled", True)
    own.setdefault("fine_step", fine)
    own.setdefault(
        "validation_fine_step",
        float(self_cfg.get("validation_fine_step", 0.75 * fine)),
    )
    own.setdefault("relative_tolerance", float(mesh_cfg.get("relative_tolerance", 1e-1)))
    own.setdefault("core_padding", float(self_cfg.get("core_padding", 0.006)))
    own.setdefault("boundary_padding", float(self_cfg.get("boundary_padding", 0.04)))
    own.setdefault("growth", float(self_cfg.get("growth", 1.5)))
    own.setdefault("max_step", float(self_cfg.get("max_step", 0.02)))
    if isinstance(getattr(background, "background_config", None), dict):
        background.background_config["global_longitudinal_correction"] = copy.deepcopy(own)


def install(corrected_preflight_module, corrected_truth_module):
    if bool(getattr(corrected_preflight_module, "_global_longitudinal_reference_installed", False)):
        return corrected_preflight_module

    original_pf_solve = corrected_preflight_module._solve_fields

    def preflight_solve(background, geometry, *args, **kwargs):
        result = original_pf_solve(background, geometry, *args, **kwargs)
        _remember_context(background, geometry, result[0])
        return result

    corrected_preflight_module._solve_fields = preflight_solve

    original_truth_port = corrected_truth_module._port_truth_from_context

    def truth_port(background, context):
        geometry = getattr(context, "geometry", None)
        if geometry is not None:
            _remember_context(background, geometry, context)
        return original_truth_port(background, context)

    corrected_truth_module._port_truth_from_context = truth_port

    original_correct = corrected_preflight_module._correct

    def correct(background, geometry, z, d, d_out):
        z1, d1, o1, _h, global_audit = apply_global_longitudinal_reference(
            background, geometry, z, d, d_out
        )
        z2, d2, o2, local_audit = original_correct(background, geometry, z1, d1, o1)
        audit = dict(local_audit)
        audit["global_longitudinal_reference"] = global_audit
        return z2, d2, o2, audit

    corrected_preflight_module._correct = correct

    original_mesh = corrected_preflight_module.audit_em_mesh_preflight

    def audit_mesh(settings, background, geometries, monitor=None):
        geometries = list(geometries)
        scalar_rows = []
        for index, geometry in enumerate(geometries):
            if monitor is not None:
                monitor.checkpoint()
            row = audit_reference_convergence(background, geometry)
            row["index"] = int(index)
            scalar_rows.append(row)
        scalar_worst = max((r["maximum_relative_error"] for r in scalar_rows), default=0.0)
        scalar_ok = all(bool(r["converged"]) for r in scalar_rows)
        scalar_report = {
            "model": _MODEL,
            "sample_count": len(scalar_rows),
            "maximum_relative_error": float(scalar_worst),
            "converged": bool(scalar_ok),
            "samples": scalar_rows,
        }
        if not scalar_ok:
            cfg = dict(settings["BACKGROUND"].get("mesh_check", {}))
            print(
                "pre-basis corrected EM mesh Gate……skipped "
                "(boundary-conditioned longitudinal defect failed)",
                flush=True,
            )
            return {
                "sample_count": len(geometries),
                "refinement_factor": float(cfg.get("refinement_factor", 0.75)),
                "base_fine_step": float(settings["BACKGROUND"]["fine_step"]),
                "refined_fine_step": float(cfg.get("refinement_factor", 0.75))
                * float(settings["BACKGROUND"]["fine_step"]),
                "relative_tolerance": float(cfg.get("relative_tolerance", 1e-1)),
                "source_path_relative_tolerance": float(
                    cfg.get("source_path_relative_tolerance", 1e-10)
                ),
                "maximum_relative_error": float(scalar_worst),
                "maximum_source_path_length_relative_error": 0.0,
                "source_geometry_invariant": True,
                "self_correction_model": corrected_preflight_module._SELF_CORRECTION_MODEL,
                "global_longitudinal_reference_convergence": scalar_report,
                "converged": False,
                "skipped_full_em_mesh_gate": True,
                "skip_reason": "boundary-conditioned longitudinal defect not converged",
                "samples": [],
            }
        report = original_mesh(settings, background, geometries, monitor)
        report["global_longitudinal_reference_convergence"] = scalar_report
        report["converged"] = bool(report.get("converged", False) and scalar_ok)
        report["maximum_relative_error"] = max(
            float(report.get("maximum_relative_error", 0.0)), float(scalar_worst)
        )
        return report

    corrected_preflight_module.audit_em_mesh_preflight = audit_mesh

    original_run = corrected_preflight_module.run_truth_preflight

    def run_truth_preflight(settings, background, geometries, monitor=None):
        _resolve_settings(settings, background)
        return original_run(settings, background, geometries, monitor)

    corrected_preflight_module.run_truth_preflight = run_truth_preflight

    original_diagnose = corrected_preflight_module._mesh_failure_diagnosis

    def diagnose_mesh(mesh):
        scalar = mesh.get("global_longitudinal_reference_convergence")
        if isinstance(scalar, dict) and not bool(scalar.get("converged", False)):
            return {
                "code": "boundary_conditioned_longitudinal_defect_not_converged",
                "maximum_relative_error": float(scalar.get("maximum_relative_error", np.inf)),
                "recommendation": (
                    "The global return-path potential is retained, but its source-near-field "
                    "longitudinal defect is not yet converged under the inherited Dirichlet "
                    "trace. Refine only the local scalar patch/reference or its physical source "
                    "representation; do not refine the global Maxwell solve and do not relax "
                    "the mesh Gate."
                ),
            }
        return original_diagnose(mesh)

    corrected_preflight_module._mesh_failure_diagnosis = diagnose_mesh

    original_solve_port = corrected_truth_module.solve_port_truth_tensors

    def solve_port_truth_tensors(background, geometry):
        z, d, d_out, audit = original_solve_port(background, geometry)
        zc, dc, oc, _h, global_audit = apply_global_longitudinal_reference(
            background, geometry, z, d, d_out
        )
        out = dict(audit)
        out["global_longitudinal_reference"] = global_audit
        implied = _hermitian(zc) - dc
        out["minimum_d_vol_eigenvalue"] = float(np.min(np.linalg.eigvalsh(_hermitian(dc))).real)
        out["minimum_physical_outward_eigenvalue"] = float(
            np.min(np.linalg.eigvalsh(_hermitian(oc))).real
        )
        out["minimum_implied_outward_eigenvalue"] = float(
            np.min(np.linalg.eigvalsh(_hermitian(implied))).real
        )
        scale = max(
            float(np.linalg.norm(_hermitian(zc))),
            float(np.linalg.norm(dc + oc)),
            np.finfo(float).tiny,
        )
        out["open_boundary_power_balance_relative_error"] = float(
            np.linalg.norm(_hermitian(zc) - dc - oc) / scale
        )
        return zc, dc, oc, out

    corrected_truth_module.solve_port_truth_tensors = solve_port_truth_tensors

    original_solve_truth = corrected_truth_module.solve_truth_tensors

    def solve_truth_tensors(background, geometry):
        z, d, modal, phi_min, phi_max, audit = original_solve_truth(background, geometry)
        context = background.geometry_context(geometry, assemble_thermal=True)
        phi = np.asarray(context.thermal_basis, float)
        physical_out = _hermitian(z) - d
        zc, dc, oc, hc, global_audit = apply_global_longitudinal_reference(
            background,
            geometry,
            z,
            d,
            physical_out,
            phi=phi,
            modal_h=modal,
        )
        out = dict(audit)
        out["global_longitudinal_reference"] = global_audit
        implied = _hermitian(zc) - dc
        out["minimum_d_vol_eigenvalue"] = float(np.min(np.linalg.eigvalsh(_hermitian(dc))).real)
        out["minimum_physical_outward_eigenvalue"] = float(
            np.min(np.linalg.eigvalsh(_hermitian(oc))).real
        )
        out["minimum_implied_outward_eigenvalue"] = float(
            np.min(np.linalg.eigvalsh(_hermitian(implied))).real
        )
        scale = max(
            float(np.linalg.norm(_hermitian(zc))),
            float(np.linalg.norm(dc + oc)),
            np.finfo(float).tiny,
        )
        out["open_boundary_power_balance_relative_error"] = float(
            np.linalg.norm(_hermitian(zc) - dc - oc) / scale
        )
        out["maximum_relative_loewner_violation"] = corrected_truth_module._loewner_violation(
            hc, dc, np.asarray(phi_min, float), np.asarray(phi_max, float)
        )
        return zc, dc, hc, phi_min, phi_max, out

    corrected_truth_module.solve_truth_tensors = solve_truth_tensors
    corrected_preflight_module._global_longitudinal_reference_installed = True
    corrected_truth_module._global_longitudinal_reference_installed = True
    return corrected_preflight_module


__all__ = [
    "apply_global_longitudinal_reference",
    "audit_reference_convergence",
    "install",
]
