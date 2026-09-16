"""Use geometry-resolved edge-dual complex mass in scalar loss references.

This adapter deliberately does *not* replace the production full-Maxwell
operator.  The full field keeps the already-certified coarse global operator and
local transverse/cross correction.  Only the independently certified
longitudinal dissipative reference is upgraded:

* the current/background scalar state remains the legacy scalar component of the
  same operator used by full Maxwell;
* refined whole-domain reference/validation scalar states integrate the
  package/seawater conductivity and permittivity on Cartesian edge-dual wedges.

Production therefore replaces the unresolved longitudinal self-loss part by a
common geometry-resolved reference while leaving mutual/transverse physics
untouched.  The embedded stranded-coil dielectric volume keeps the existing
conservative coil-fraction semantics; wire conductivity remains excluded from
Maxwell exactly as in the production constitutive model.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .unified_charge_regularized_source import terminal_charge_target
from .unified_gradient_block_maxwell import _edge_mass_diagonal, gradient_operator
from .unified_resolved_conductive_hodge import _MODEL as _SIGMA_HODGE_MODEL
from .unified_resolved_conductive_hodge import _build_conductivity_hodge
from .unified_resolved_admittance_hodge import _MODEL as _EPS_HODGE_MODEL
from .unified_resolved_admittance_hodge import _build_permittivity_weights


_MODEL = "global_longitudinal_dissipative_exact_edge_dual_complex_mass_reference_v3"


def _relative_difference(left, right):
    a = np.asarray(left, float).reshape(-1)
    b = np.asarray(right, float).reshape(-1)
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    )


def _exact_scalar_state(module, implementation_module, parent, background, geometry, *, phi=None):
    context = background.geometry_context(geometry, assemble_thermal=False)
    conductivity_hodge, edge_loss = _build_conductivity_hodge(background, context)
    exact_eps, legacy_eps, eps_meta = _build_permittivity_weights(background, context)
    G = gradient_operator(background, gauge_fixed=True)

    # Start from the exact compatible production scalar block and replace only
    # its material edge-mass discretization.  Open-boundary mass is untouched.
    diagonal = np.asarray(_edge_mass_diagonal(background, context), complex).reshape(-1)
    sigma, *_ = background.cell_properties(context, None, em=True)
    legacy_edge_loss = np.asarray(
        background.edge_cell_hodge @ np.asarray(sigma, float), float
    ).reshape(-1)
    diagonal = (
        diagonal
        + 1j * float(background.omega) * (edge_loss - legacy_edge_loss)
        - (float(background.omega) ** 2) * (exact_eps - legacy_eps)
    )
    scalar = (G.T @ sp.diags(diagonal, format="csr") @ G).tocsc()
    scalar.sum_duplicates()
    scalar.eliminate_zeros()
    try:
        factor = spla.splu(
            scalar,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        factor = spla.splu(scalar)

    n = len(background.coil_materials)
    d = np.zeros(n, float)
    z = np.zeros(n, complex)
    modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], n), float)
    local_phi = (
        None
        if phi is None
        else implementation_module._interpolate_basis(parent, background, phi)
    )
    residuals = []
    supports = []
    sigma_field_difference = []
    epsilon_field_difference = []

    for p, coil in enumerate(context.geometry.coils):
        q_full, meta = terminal_charge_target(background, coil)
        q = np.asarray(q_full[1:], complex)
        scalar_rhs = (-1j * float(background.omega)) * q
        potential = np.asarray(factor.solve(scalar_rhs), complex).reshape(-1)
        residuals.append(
            float(
                np.linalg.norm(scalar_rhs - scalar @ potential)
                / max(float(np.linalg.norm(scalar_rhs)), np.finfo(float).tiny)
            )
        )
        field = np.asarray(G @ potential, complex).reshape(-1)
        abs2 = np.abs(field) ** 2
        d[p] = float(np.dot(edge_loss, abs2))
        z[p] = complex(-q @ potential)
        supports.append(int(meta.get("terminal_charge_support_nodes", 0)))

        legacy_d = float(np.dot(legacy_edge_loss, abs2))
        sigma_field_difference.append(
            abs(float(d[p]) - legacy_d)
            / max(abs(float(d[p])), abs(legacy_d), np.finfo(float).tiny)
        )
        exact_e = float(np.dot(exact_eps, abs2))
        legacy_e = float(np.dot(legacy_eps, abs2))
        epsilon_field_difference.append(
            abs(exact_e - legacy_e)
            / max(abs(exact_e), abs(legacy_e), np.finfo(float).tiny)
        )

        if modal is not None:
            q_cells = np.asarray(0.5 * (conductivity_hodge.T @ abs2), float).reshape(-1)
            modal[:, p] = np.asarray(2.0 * (local_phi.T @ q_cells), float)

    return {
        "fine_step": float(module._background_step(background)),
        "n_cells": int(background.n_cells),
        "scalar_dofs": int(G.shape[1]),
        "d_vol": d,
        "z_reaction": z,
        "modal_h": modal,
        "maximum_scalar_relative_residual": max(residuals, default=0.0),
        "terminal_charge_support_nodes": supports,
        "conductive_hodge_model": _SIGMA_HODGE_MODEL,
        "conductive_hodge_legacy_relative_difference": _relative_difference(
            edge_loss, legacy_edge_loss
        ),
        "conductive_hodge_field_weighted_relative_difference": [
            float(v) for v in sigma_field_difference
        ],
        "dielectric_hodge_model": str(eps_meta["dielectric_hodge_model"]),
        "dielectric_hodge_legacy_relative_difference": float(
            eps_meta["dielectric_hodge_legacy_relative_difference"]
        ),
        "dielectric_hodge_field_weighted_relative_difference": [
            float(v) for v in epsilon_field_difference
        ],
        "complex_mass_models": {
            "sigma": _SIGMA_HODGE_MODEL,
            "epsilon": _EPS_HODGE_MODEL,
        },
    }


def install(module, implementation_module):
    """Install the exact reference on the aggregate and implementation modules.

    ``module`` is the already-enhanced longitudinal aggregate used by production
    preflight/truth. ``implementation_module`` is the
    ``unified_global_dissipative_reference`` module whose file-level helper
    functions are resolved by the installed correction/audit closures at call
    time. Patching those helpers is required so the production correction and
    convergence Gate both consume the exact reference rather than only exposing
    exact values in diagnostics.
    """
    if bool(getattr(module, "_resolved_dissipative_reference_installed", False)):
        return module

    required = ("_scalar_state", "_reference_state", "_config", "_interpolate_basis")
    missing = [name for name in required if not hasattr(implementation_module, name)]
    if missing:
        raise AttributeError(
            "global dissipative implementation is missing required hooks: "
            + ", ".join(missing)
        )
    if not hasattr(module, "audit_reference_convergence"):
        raise AttributeError(
            "resolved dissipative reference must be installed after the base "
            "global dissipative reference"
        )

    original_scalar_state = implementation_module._scalar_state
    original_reference_state = implementation_module._reference_state
    original_audit = module.audit_reference_convergence

    def scalar_state(module_arg, parent, background, geometry, *, phi=None):
        # The current scalar state is intentionally the same legacy longitudinal
        # component as the full-Maxwell operator. Refined reference backgrounds
        # use the geometry-resolved dual complex mass so the correction replaces,
        # rather than double-counts, unresolved coarse longitudinal loss.
        if background is parent:
            return original_scalar_state(
                module_arg, parent, background, geometry, phi=phi
            )
        return _exact_scalar_state(
            module_arg,
            implementation_module,
            parent,
            background,
            geometry,
            phi=phi,
        )

    implementation_module._scalar_state = scalar_state

    def reference_state(module_arg, background, geometry, *, step, max_step, phi=None):
        state = original_reference_state(
            module_arg,
            background,
            geometry,
            step=step,
            max_step=max_step,
            phi=phi,
        )
        if "conductive_hodge_model" in state:
            print(
                "global longitudinal dissipative Hodge: "
                f"step={float(step):.6g}m, "
                f"sigma={state['conductive_hodge_model']}, "
                f"sigma_legacy={float(state.get('conductive_hodge_legacy_relative_difference', 0.0)):.3e}, "
                f"epsilon={state.get('dielectric_hodge_model', 'legacy')}, "
                f"epsilon_legacy={float(state.get('dielectric_hodge_legacy_relative_difference', 0.0)):.3e}, "
                f"sigma_field={state.get('conductive_hodge_field_weighted_relative_difference', [])}, "
                f"epsilon_field={state.get('dielectric_hodge_field_weighted_relative_difference', [])}",
                flush=True,
            )
        return state

    implementation_module._reference_state = reference_state

    def audit_reference_convergence(background, geometry):
        report = dict(original_audit(background, geometry))
        cfg = implementation_module._config(background)
        if cfg.get("enabled", False):
            reference = implementation_module._reference_state(
                module,
                background,
                geometry,
                step=cfg["reference_step"],
                max_step=cfg["reference_max_step"],
                phi=None,
            )
            validation = implementation_module._reference_state(
                module,
                background,
                geometry,
                step=cfg["validation_step"],
                max_step=cfg["validation_max_step"],
                phi=None,
            )
            diagnostic = report.get("global_dissipative_reference")
            if isinstance(diagnostic, dict):
                diagnostic.update(
                    conductive_hodge_model=_SIGMA_HODGE_MODEL,
                    dielectric_hodge_model=_EPS_HODGE_MODEL,
                    reference_conductive_hodge_legacy_relative_difference=float(
                        reference.get("conductive_hodge_legacy_relative_difference", 0.0)
                    ),
                    validation_conductive_hodge_legacy_relative_difference=float(
                        validation.get("conductive_hodge_legacy_relative_difference", 0.0)
                    ),
                    reference_dielectric_hodge_legacy_relative_difference=float(
                        reference.get("dielectric_hodge_legacy_relative_difference", 0.0)
                    ),
                    validation_dielectric_hodge_legacy_relative_difference=float(
                        validation.get("dielectric_hodge_legacy_relative_difference", 0.0)
                    ),
                    reference_conductive_hodge_field_weighted_relative_difference=list(
                        reference.get("conductive_hodge_field_weighted_relative_difference", [])
                    ),
                    validation_conductive_hodge_field_weighted_relative_difference=list(
                        validation.get("conductive_hodge_field_weighted_relative_difference", [])
                    ),
                    reference_dielectric_hodge_field_weighted_relative_difference=list(
                        reference.get("dielectric_hodge_field_weighted_relative_difference", [])
                    ),
                    validation_dielectric_hodge_field_weighted_relative_difference=list(
                        validation.get("dielectric_hodge_field_weighted_relative_difference", [])
                    ),
                    reference_semantics="exact_edge_dual_complex_material_mass",
                )
        report["model"] = (
            "global_boundary_conditioned_longitudinal_reactive_defect_v4+" + _MODEL
        )
        return report

    module.audit_reference_convergence = audit_reference_convergence
    module._MODEL = _MODEL
    module._resolved_dissipative_reference_installed = True
    return module


__all__ = ["install"]
