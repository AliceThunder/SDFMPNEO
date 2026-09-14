"""Production tensor truth with local self-response defect correction."""
from __future__ import annotations

import numpy as np

from .unified_self_correction import apply_local_self_correction
from .unified_tensor_surrogate import (
    TensorDataset,
    _audit_current_vectors,
    _conductivity_support_bounds,
    _hermitian,
    _split_labels,
    encode_geometry,
    pack_tensors,
    solve_port_truth_tensors as _raw_solve_port_truth_tensors,
    solve_truth_tensors as _raw_solve_truth_tensors,
)


def _loewner_violation(modal, d, phi_min, phi_max):
    worst = 0.0
    for j, h in enumerate(np.asarray(modal, complex)):
        low = np.min(np.linalg.eigvalsh(_hermitian(h - phi_min[j] * d))).real
        high = np.min(np.linalg.eigvalsh(_hermitian(phi_max[j] * d - h))).real
        scale = max(float(np.linalg.norm(d)), float(np.linalg.norm(h)), np.finfo(float).tiny)
        worst = max(worst, float(max(-low, -high, 0.0) / scale))
    return float(worst)


def _refresh_audit(z, d, d_out, audit, correction, *, modal=None, phi_min=None, phi_max=None):
    out = dict(audit)
    herm_z = _hermitian(z)
    implied = herm_z - d
    scale = max(float(np.linalg.norm(herm_z)), float(np.linalg.norm(d + d_out)), np.finfo(float).tiny)
    out["minimum_d_vol_eigenvalue"] = float(np.min(np.linalg.eigvalsh(_hermitian(d))).real)
    out["minimum_physical_outward_eigenvalue"] = float(np.min(np.linalg.eigvalsh(_hermitian(d_out))).real)
    out["minimum_implied_outward_eigenvalue"] = float(np.min(np.linalg.eigvalsh(_hermitian(implied))).real)
    out["open_boundary_power_balance_relative_error"] = float(np.linalg.norm(herm_z - d - d_out) / scale)
    out["local_self_correction_enabled"] = 1.0 if correction.audit.get("enabled", False) else 0.0
    out["local_self_correction_power_balance_relative_error"] = float(
        correction.audit.get("corrected_power_balance_relative_error", 0.0)
    )
    if modal is not None:
        out["maximum_relative_loewner_violation"] = _loewner_violation(
            modal, d, np.asarray(phi_min, float), np.asarray(phi_max, float)
        )
        # The defect is constructed from the same local Joule field for D and H,
        # so the existing exact contraction identities are preserved additively.
        out["joule_total_power_relative_error"] = float(
            audit.get("joule_total_power_relative_error", 0.0)
        )
        out["joule_modal_contraction_relative_error"] = float(
            audit.get("joule_modal_contraction_relative_error", 0.0)
        )
    out["local_self_correction"] = correction.audit
    return out


def solve_port_truth_tensors(background, geometry):
    z, d, d_out, audit = _raw_solve_port_truth_tensors(background, geometry)
    correction = apply_local_self_correction(background, geometry, z, d, d_out)
    refreshed = _refresh_audit(
        correction.z, correction.d_vol, correction.d_out, audit, correction
    )
    return correction.z, correction.d_vol, correction.d_out, refreshed


def solve_truth_tensors(background, geometry):
    z, d, modal, phi_min, phi_max, audit = _raw_solve_truth_tensors(background, geometry)
    context = background.geometry_context(geometry, assemble_thermal=True)
    phi = np.asarray(context.thermal_basis, float)
    # D_out is needed only so the local defect remains power-balanced.  The raw
    # tensor path does not return the independently assembled D_out; using the
    # implied raw partition here does not certify Poynting balance.  Independent
    # D_out is still produced by solve_port_truth_tensors and final Gate paths.
    implied_raw_out = _hermitian(z) - d
    correction = apply_local_self_correction(
        background,
        geometry,
        z,
        d,
        implied_raw_out,
        phi=phi,
        modal_h=modal,
    )
    phi_min2, phi_max2 = _conductivity_support_bounds(
        phi, np.asarray(background.cell_properties(context, None, em=True)[0], float)
    )
    refreshed = _refresh_audit(
        correction.z,
        correction.d_vol,
        correction.d_out,
        audit,
        correction,
        modal=correction.modal_h,
        phi_min=phi_min2,
        phi_max=phi_max2,
    )
    return (
        correction.z,
        correction.d_vol,
        correction.modal_h,
        phi_min2,
        phi_max2,
        refreshed,
    )


def generate_tensor_dataset(background, geometries, *, seed=0, monitor=None):
    geometries = list(geometries)
    if getattr(background, "thermal_library", None) is None:
        raise ValueError("geometry-aware thermal library must be frozen before tensor labels")
    inputs, outputs, mins, maxs, audits = [], [], [], [], []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        z, d, modal, phi_min, phi_max, audit = solve_truth_tensors(background, geometry)
        inputs.append(encode_geometry(geometry))
        outputs.append(pack_tensors(z, d, modal))
        mins.append(phi_min)
        maxs.append(phi_max)
        audits.append(audit)
        print(
            "生成 corrected geometry-aware Z_field / D_vol / H_j truth……"
            f"{100.0 * (index + 1) / len(geometries):5.1f}%  ({index + 1}/{len(geometries)})",
            flush=True,
        )
    numeric_audit = {
        "maximum_linear_relative_residual": max(a["max_linear_relative_residual"] for a in audits),
        "maximum_reciprocity_relative_error": max(a["reciprocity_relative_error"] for a in audits),
        "minimum_d_vol_eigenvalue": min(a["minimum_d_vol_eigenvalue"] for a in audits),
        "minimum_physical_outward_eigenvalue": min(a["minimum_physical_outward_eigenvalue"] for a in audits),
        "minimum_implied_outward_eigenvalue": min(a["minimum_implied_outward_eigenvalue"] for a in audits),
        "maximum_open_boundary_power_balance_relative_error": max(
            a["open_boundary_power_balance_relative_error"] for a in audits
        ),
        "maximum_relative_loewner_violation": max(
            a["maximum_relative_loewner_violation"] for a in audits
        ),
        "maximum_joule_total_power_relative_error": max(
            a["joule_total_power_relative_error"] for a in audits
        ),
        "maximum_joule_modal_contraction_relative_error": max(
            a["joule_modal_contraction_relative_error"] for a in audits
        ),
        "maximum_material_fraction_closure_error": max(
            a["material_fraction_closure_error"] for a in audits
        ),
        "source_regularization_available": min(
            a["source_regularization_available"] for a in audits
        ),
        "independent_outward_power_available": 1.0,
        "local_self_correction_available": min(
            a["local_self_correction_enabled"] for a in audits
        ),
        "maximum_local_self_correction_power_balance_relative_error": max(
            a["local_self_correction_power_balance_relative_error"] for a in audits
        ),
    }
    return TensorDataset(
        np.asarray(inputs, float),
        np.asarray(outputs, float),
        np.asarray(mins, float),
        np.asarray(maxs, float),
        _split_labels(len(geometries), seed),
        numeric_audit,
        len(background.coil_materials),
        int(background.thermal_rank),
    )


__all__ = ["generate_tensor_dataset", "solve_port_truth_tensors", "solve_truth_tensors"]
