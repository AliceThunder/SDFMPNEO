"""Production tensor truth with local self-response defect correction."""
from __future__ import annotations

import numpy as np

from .unified_self_correction import apply_local_self_correction
from .unified_tensor_surrogate import (
    TensorDataset,
    SpatialTensorDataset,
    _audit_current_vectors,
    _cell_volume_heat,
    _conductivity_support_bounds,
    _hermitian,
    _port_truth_from_context,
    cell_joule_tensors_from_port_fields,
    normalize_cell_joule_tensors,
    _require_static_field_materials,
    _split_labels,
    encode_geometry,
    pack_tensors,
    pack_spatial_tensors,
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
    out["local_self_joule_total_power_relative_error"] = float(
        correction.audit.get("maximum_joule_total_power_relative_error", np.inf)
    )
    out["local_self_joule_modal_contraction_relative_error"] = float(
        correction.audit.get("maximum_joule_modal_contraction_relative_error", 0.0)
    )
    if modal is not None:
        out["maximum_relative_loewner_violation"] = _loewner_violation(
            modal, d, np.asarray(phi_min, float), np.asarray(phi_max, float)
        )
    out["local_self_correction"] = correction.audit
    return out


def solve_port_truth_tensors(background, geometry):
    _require_static_field_materials(background)
    context = background.geometry_context(geometry, assemble_thermal=False)
    z, d, d_out, _X, _sigma, audit = _port_truth_from_context(background, context)
    correction = apply_local_self_correction(background, geometry, z, d, d_out)
    refreshed = _refresh_audit(
        correction.z, correction.d_vol, correction.d_out, audit, correction
    )
    return correction.z, correction.d_vol, correction.d_out, refreshed


def solve_truth_tensors(background, geometry):
    """Generate corrected Z/D/H using the current geometry-specific Phi(g)."""
    _require_static_field_materials(background)
    context = background.geometry_context(geometry, assemble_thermal=True)
    phi = np.asarray(context.thermal_basis, float)
    z_raw, d_raw, d_out_raw, X, sigma, audit = _port_truth_from_context(background, context)

    modal_raw = []
    for j in range(phi.shape[1]):
        weighted_edge = np.asarray(
            background.edge_cell_hodge @ (sigma * phi[:, j])
        ).reshape(-1)
        modal_raw.append(_hermitian(X.conj().T @ (weighted_edge[:, None] * X)))
    modal_raw = np.asarray(modal_raw, complex)

    correction = apply_local_self_correction(
        background,
        geometry,
        z_raw,
        d_raw,
        d_out_raw,
        phi=phi,
        modal_h=modal_raw,
    )
    z = correction.z
    d = correction.d_vol
    d_out = correction.d_out
    modal = correction.modal_h
    phi_min, phi_max = _conductivity_support_bounds(phi, sigma)

    # Verify the corrected multiscale Joule identities over a complete Hermitian
    # current span. The local defect contributes only |c_p|^2 diagonal terms.
    delta_d = np.real(np.diag(d - d_raw))
    delta_h = np.real(
        np.stack([np.diag(modal[j] - modal_raw[j]) for j in range(modal.shape[0])], axis=0)
    )
    power_consistency = 0.0
    modal_consistency = 0.0
    for current in _audit_current_vectors(X.shape[1]):
        c = np.asarray(current, complex)
        field = X @ c
        q_cells = _cell_volume_heat(background, sigma, field)
        weights = np.abs(c) ** 2
        direct_power = float(np.sum(q_cells) + 0.5 * np.dot(weights, delta_d))
        tensor_power = float(0.5 * np.real(c.conj() @ d @ c))
        pscale = max(abs(direct_power), abs(tensor_power), np.finfo(float).tiny)
        power_consistency = max(power_consistency, abs(direct_power - tensor_power) / pscale)

        direct_modal = phi.T @ q_cells + 0.5 * (delta_h @ weights)
        tensor_modal = 0.5 * np.real(
            np.einsum("p,rpq,q->r", c.conj(), modal, c, optimize=True)
        )
        mscale = max(
            float(np.linalg.norm(direct_modal)),
            float(np.linalg.norm(tensor_modal)),
            np.finfo(float).tiny,
        )
        modal_consistency = max(
            modal_consistency,
            float(np.linalg.norm(direct_modal - tensor_modal) / mscale),
        )

    refreshed = _refresh_audit(
        z,
        d,
        d_out,
        audit,
        correction,
        modal=modal,
        phi_min=phi_min,
        phi_max=phi_max,
    )
    refreshed["joule_total_power_relative_error"] = float(power_consistency)
    refreshed["joule_modal_contraction_relative_error"] = float(modal_consistency)
    return z, d, modal, phi_min, phi_max, refreshed


def solve_spatial_truth_tensors(background, geometry):
    """Corrected Z/D plus cellwise Hermitian Joule tensor field.

    This truth representation is independent of any thermal basis.  The global
    Maxwell solve supplies the coarse spatial tensor field.  The existing
    localized fine-minus-coarse correction changes only diagonal self terms; its
    D defect is deposited on the corresponding physical line-heat support before
    one PSD/total-D normalization.
    """
    _require_static_field_materials(background)
    context = background.geometry_context(
        geometry,
        assemble_thermal=False,
    )
    z_raw, d_raw, d_out_raw, X, sigma, audit = _port_truth_from_context(
        background,
        context,
    )
    cells_raw = cell_joule_tensors_from_port_fields(
        background,
        sigma,
        X,
    )

    correction = apply_local_self_correction(
        background,
        geometry,
        z_raw,
        d_raw,
        d_out_raw,
        spatial=True,
    )
    z = correction.z
    d = correction.d_vol
    d_out = correction.d_out

    spatial_delta = correction.spatial_d_vol
    if (
        spatial_delta is None
        or spatial_delta.shape
        != (background.n_cells, X.shape[1], X.shape[1])
        or np.any(~np.isfinite(spatial_delta))
    ):
        raise RuntimeError(
            "local self correction did not provide a valid spatial D defect"
        )
    cells = (
        np.asarray(cells_raw, complex)
        + np.asarray(spatial_delta, complex)
    )
    d, cells = normalize_cell_joule_tensors(cells, d)

    minimum_cell_eigenvalue = float(
        np.min(np.linalg.eigvalsh(cells).real)
    )
    aggregate_error = float(
        np.linalg.norm(np.sum(cells, axis=0) - d)
        / max(float(np.linalg.norm(d)), np.finfo(float).tiny)
    )
    power_error = 0.0
    minimum_cell_power = float("inf")
    for current in _audit_current_vectors(X.shape[1]):
        current = np.asarray(current, complex)
        q = 0.5 * np.real(
            np.einsum(
                "p,kpq,q->k",
                current.conj(),
                cells,
                current,
                optimize=True,
            )
        )
        direct = float(np.sum(q))
        tensor = float(
            0.5 * np.real(current.conj() @ d @ current)
        )
        scale = max(
            abs(direct),
            abs(tensor),
            np.finfo(float).tiny,
        )
        power_error = max(
            power_error,
            abs(direct - tensor) / scale,
        )
        minimum_cell_power = min(
            minimum_cell_power,
            float(np.min(q)),
        )

    refreshed = _refresh_audit(
        z,
        d,
        d_out,
        audit,
        correction,
    )
    refreshed.update(
        minimum_cell_joule_tensor_eigenvalue=minimum_cell_eigenvalue,
        maximum_spatial_joule_total_mismatch=aggregate_error,
        maximum_spatial_joule_power_relative_error=float(power_error),
        minimum_spatial_joule_cell_power=float(minimum_cell_power),
        spatial_joule_representation="cellwise_hermitian_psd_v1",
    )
    return z, d, cells, refreshed


def generate_spatial_tensor_dataset(
    background,
    geometries,
    *,
    seed=0,
    monitor=None,
):
    geometries = list(geometries)
    inputs, outputs, audits = [], [], []
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        z, d, cells, audit = solve_spatial_truth_tensors(
            background,
            geometry,
        )
        inputs.append(encode_geometry(geometry))
        outputs.append(pack_spatial_tensors(z, d, cells))
        audits.append(audit)
        print(
            "生成 corrected geometry→Z/D/cell-Joule truth……"
            f"{100.0 * (index + 1) / len(geometries):5.1f}%  "
            f"({index + 1}/{len(geometries)})",
            flush=True,
        )

    numeric_audit = {
        "maximum_linear_relative_residual": max(
            a["max_linear_relative_residual"] for a in audits
        ),
        "maximum_reciprocity_relative_error": max(
            a["reciprocity_relative_error"] for a in audits
        ),
        "minimum_d_vol_eigenvalue": min(
            a["minimum_d_vol_eigenvalue"] for a in audits
        ),
        "minimum_physical_outward_eigenvalue": min(
            a["minimum_physical_outward_eigenvalue"] for a in audits
        ),
        "minimum_implied_outward_eigenvalue": min(
            a["minimum_implied_outward_eigenvalue"] for a in audits
        ),
        "maximum_open_boundary_power_balance_relative_error": max(
            a["open_boundary_power_balance_relative_error"] for a in audits
        ),
        "maximum_joule_total_power_relative_error": max(
            a["maximum_spatial_joule_power_relative_error"] for a in audits
        ),
        "minimum_cell_joule_tensor_eigenvalue": min(
            a["minimum_cell_joule_tensor_eigenvalue"] for a in audits
        ),
        "maximum_spatial_joule_total_mismatch": max(
            a["maximum_spatial_joule_total_mismatch"] for a in audits
        ),
        "minimum_spatial_joule_cell_power": min(
            a["minimum_spatial_joule_cell_power"] for a in audits
        ),
        "maximum_local_self_joule_total_power_relative_error": max(
            a["local_self_joule_total_power_relative_error"] for a in audits
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
            a["local_self_correction_power_balance_relative_error"]
            for a in audits
        ),
    }
    return SpatialTensorDataset(
        np.asarray(inputs, float),
        np.asarray(outputs, float),
        _split_labels(len(geometries), seed),
        numeric_audit,
        len(background.coil_materials),
        int(background.n_cells),
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
        "maximum_local_self_joule_total_power_relative_error": max(
            a["local_self_joule_total_power_relative_error"] for a in audits
        ),
        "maximum_local_self_joule_modal_contraction_relative_error": max(
            a["local_self_joule_modal_contraction_relative_error"] for a in audits
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


__all__ = ["generate_tensor_dataset", "generate_spatial_tensor_dataset", "solve_port_truth_tensors", "solve_truth_tensors", "solve_spatial_truth_tensors"]
