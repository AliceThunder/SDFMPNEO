from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .electrothermal_domain import ElectroThermalDomainBounds
from .geometry_analytic_residual import GeometryPhysicalLipschitzProof


@dataclass(frozen=True)
class GeometryDerivativeProof:
    """Theorem-derived geometry derivative bounds for one branch.

    ``entry_bounds[i,k]`` bounds ``|dF_i/dG_k|`` on the complete induced
    thermal/operating domain of the branch.  The object deliberately carries no
    finite-difference or sampled estimate path: an uncertified provider remains
    fail-closed in the cross-geometry analytic residual certificate.
    """

    entry_bounds: np.ndarray
    certified: bool
    provenance: str

    def __post_init__(self) -> None:
        values = np.asarray(self.entry_bounds, dtype=float)
        if values.ndim != 2 or np.any(values < 0.0) or np.any(~np.isfinite(values)):
            raise ValueError("geometry derivative bounds must be a finite non-negative matrix")
        if self.certified and not str(self.provenance).strip():
            raise ValueError("certified geometry derivative proof requires provenance")
        object.__setattr__(self, "entry_bounds", values)


def compose_geometry_physical_lipschitz_proof(
    electrothermal: ElectroThermalDomainBounds,
    geometry: GeometryDerivativeProof,
    *,
    provenance: str,
) -> GeometryPhysicalLipschitzProof:
    """Compose the exact proof object required by the cross-geometry certifier.

    State and operating derivatives are taken directly from the continuous
    electrothermal-domain theorem.  Geometry derivatives remain a separate
    theorem obligation because they depend on the declared geometry chart and
    canonical thermal transfer.  This composition prevents callers from manually
    substituting a Joule-source Jacobian for the full vector-field Jacobian.
    """

    G = np.asarray(geometry.entry_bounds, dtype=float)
    U = np.asarray(electrothermal.explicit_operating_jacobian_entry_bounds, dtype=float)
    if G.shape[0] != U.shape[0]:
        raise ValueError("geometry/electrothermal proof output dimensions do not match")
    text = str(provenance).strip()
    if geometry.certified and not text:
        raise ValueError("certified composed proof requires provenance")
    return GeometryPhysicalLipschitzProof(
        geometry_jacobian_entry_bounds=G,
        state_jacobian_norm_bound=float(
            electrothermal.vector_field_state_jacobian_norm_bound
        ),
        operating_jacobian_entry_bounds=U,
        contraction_margin_lower_bound=float(
            electrothermal.contraction_margin_lower_bound
        ),
        certified=bool(geometry.certified),
        provenance=text,
    )


def make_geometry_physical_proof_factory(
    electrothermal_factory: Callable[[object], ElectroThermalDomainBounds],
    geometry_factory: Callable[[object], GeometryDerivativeProof],
    *,
    provenance: str,
):
    """Return a branch-local proof factory for ``[a0,G,U,t]`` certification.

    Both providers receive the complete branch box and therefore remain
    responsible for covering its induced thermal state image.  The returned
    callable plugs directly into ``certify_geometry_analytic_residual_domain``.
    """

    if not callable(electrothermal_factory) or not callable(geometry_factory):
        raise TypeError("proof providers must be callable")

    def factory(branch_box):
        return compose_geometry_physical_lipschitz_proof(
            electrothermal_factory(branch_box),
            geometry_factory(branch_box),
            provenance=provenance,
        )

    return factory
