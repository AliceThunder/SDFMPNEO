from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ledger import CertifiedErrorTerm


@dataclass(frozen=True)
class MeshApproximationProof:
    """Proof data for a conforming H(curl)-P1 approximation theorem.

    The object deliberately contains theorem constants rather than estimating
    them from solution differences.  A certified instance means that, on the
    declared geometry/material chart,

        ||u-u_h||_H <= C_I h^s |u|_{1+s,graph}.

    ``regularity_bound`` is a proved upper bound for the exact solution graph
    seminorm and ``interpolation_constant`` is the proved interpolation/reliability
    constant for the conforming Nedelec/P1 pair.  No value is inferred from a
    finite refinement sequence.
    """

    regularity_order: float
    regularity_bound: float
    interpolation_constant: float
    certified: bool
    provenance: str

    def __post_init__(self) -> None:
        s = float(self.regularity_order)
        r = float(self.regularity_bound)
        c = float(self.interpolation_constant)
        if not 0.0 < s <= 1.0:
            raise ValueError("regularity_order must satisfy 0 < s <= 1")
        if r < 0.0 or not np.isfinite(r):
            raise ValueError("regularity_bound must be finite and non-negative")
        if c <= 0.0 or not np.isfinite(c):
            raise ValueError("interpolation_constant must be finite and positive")
        if self.certified and not str(self.provenance).strip():
            raise ValueError("a certified mesh proof requires provenance")


@dataclass(frozen=True)
class MeshErrorCertificate:
    h_max: float
    regularity_order: float
    state_energy_error_bound: float
    certified: bool
    provenance: str

    def as_ledger_term(self) -> CertifiedErrorTerm:
        return CertifiedErrorTerm(
            "mesh",
            self.state_energy_error_bound,
            self.certified,
            self.provenance,
        )


def tetrahedral_h_max(mesh) -> float:
    """Maximum tetrahedron diameter computed from the actual physical mesh."""

    xyz = np.asarray(mesh.vertices, dtype=float)
    tets = np.asarray(mesh.tetrahedra, dtype=int)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("mesh must expose tetrahedra with shape (n,4)")
    h = 0.0
    for tet in tets:
        points = xyz[tet]
        for i in range(4):
            for j in range(i + 1, 4):
                h = max(h, float(np.linalg.norm(points[i] - points[j])))
    if h <= 0.0:
        raise ValueError("mesh has zero diameter")
    return h


def certify_mesh_error(mesh, proof: MeshApproximationProof) -> MeshErrorCertificate:
    """Evaluate a proved conforming finite-element approximation majorant.

    If the regularity/interpolation theorem has not been certified, the same
    numerical expression is returned only as an *uncertified* diagnostic.  This
    makes it impossible for the unified ledger to silently promote a refinement
    study into a mathematical error certificate.
    """

    h = tetrahedral_h_max(mesh)
    bound = (
        float(proof.interpolation_constant)
        * h ** float(proof.regularity_order)
        * float(proof.regularity_bound)
    )
    return MeshErrorCertificate(
        h_max=h,
        regularity_order=float(proof.regularity_order),
        state_energy_error_bound=float(np.nextafter(bound, np.inf)),
        certified=bool(proof.certified),
        provenance=str(proof.provenance),
    )


@dataclass(frozen=True)
class HomogeneousConductiveExterior:
    """Certified spherical source-free conductive exterior shell.

    The exterior of ``inner_radius`` must be homogeneous, isotropic and free of
    impressed sources.  For the magnetoquasistatic diffusion wavenumber

        k=(1+i)/delta,  delta=sqrt(2/(omega*mu*sigma)),

    every outgoing spherical mode decays at least as

        (R1/R2) exp(-(R2-R1)/delta).

    The slowest algebraic factor 1/r is used, so higher vector-spherical modes
    are covered by the same contraction bound.
    """

    omega: float
    permeability: float
    conductivity: float
    source_support_radius: float
    inner_radius: float
    outer_radius: float
    certified_geometry: bool
    provenance: str

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.omega,
                self.permeability,
                self.conductivity,
                self.source_support_radius,
                self.inner_radius,
                self.outer_radius,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("conductive exterior parameters must be finite and positive")
        if not self.source_support_radius < self.inner_radius < self.outer_radius:
            raise ValueError("require source_support_radius < inner_radius < outer_radius")
        if self.certified_geometry and not str(self.provenance).strip():
            raise ValueError("certified exterior geometry requires provenance")

    @property
    def skin_depth(self) -> float:
        return float(np.sqrt(2.0 / (self.omega * self.permeability * self.conductivity)))

    @property
    def contraction_factor(self) -> float:
        d = self.outer_radius - self.inner_radius
        q = (self.inner_radius / self.outer_radius) * np.exp(-d / self.skin_depth)
        return float(q)


@dataclass(frozen=True)
class OuterDomainErrorCertificate:
    nested_domain_difference: float
    skin_depth: float
    contraction_factor: float
    state_energy_error_bound: float
    certified: bool
    provenance: str

    def as_ledger_term(self) -> CertifiedErrorTerm:
        return CertifiedErrorTerm(
            "outer",
            self.state_energy_error_bound,
            self.certified,
            self.provenance,
        )


def certify_conductive_outer_domain(
    nested_domain_difference: float,
    exterior: HomogeneousConductiveExterior,
) -> OuterDomainErrorCertificate:
    """Bound the infinite-domain tail from two nested spherical-domain solves.

    Let ``d_R`` be the energy-norm difference restricted to the common inner
    domain.  The certified exterior contraction ``q<1`` gives the Cauchy tail

        ||u_infty-u_R|| <= d_R (1+q+q^2+...) = d_R/(1-q).

    The result is certified only when the homogeneous spherical exterior chart
    itself is certified.  No observed convergence ratio is used.
    """

    difference = float(nested_domain_difference)
    if difference < 0.0 or not np.isfinite(difference):
        raise ValueError("nested_domain_difference must be finite and non-negative")
    q = exterior.contraction_factor
    if not 0.0 <= q < 1.0:
        raise ValueError("conductive exterior contraction factor must satisfy 0 <= q < 1")
    bound = difference / (1.0 - q)
    return OuterDomainErrorCertificate(
        nested_domain_difference=difference,
        skin_depth=exterior.skin_depth,
        contraction_factor=q,
        state_energy_error_bound=float(np.nextafter(bound, np.inf)),
        certified=bool(exterior.certified_geometry),
        provenance=str(exterior.provenance),
    )


@dataclass(frozen=True)
class SpatialOutputErrorCertificate:
    state_energy_error_bound: float
    impedance_abs_error_bound: np.ndarray | None
    resistance_abs_error_bound: np.ndarray | None
    inductance_abs_error_bound: np.ndarray | None
    heat_source_component_error_bounds: np.ndarray | None
    heat_source_vector_error_bound: float | None


def propagate_spatial_state_error(
    *,
    state_energy_error_bound: float,
    omega: float,
    port_rhs_dual_norms: np.ndarray | None = None,
    approximate_state_energy_norm: float | None = None,
    thermal_test_supremum_bounds: np.ndarray | None = None,
) -> SpatialOutputErrorCertificate:
    """Propagate a spatial field-energy error to port and Joule outputs.

    For a work-conjugate unit-current port ``b_i``,

        |Delta Z_ij| <= omega ||b_i||_{H*} ||e_j||_H.

    For projected Joule heat source ``q_j`` the physical-energy inequality used
    by the algebraic certificate remains valid with any certified field error:

        |Delta q_j| <= .5*omega*||phi_j||_inf
                        (2||x_h||_H eps + eps^2).
    """

    eps = float(state_energy_error_bound)
    w = float(omega)
    if eps < 0.0 or not np.isfinite(eps) or w <= 0.0 or not np.isfinite(w):
        raise ValueError("state error must be non-negative and omega positive")

    z = r = l = None
    if port_rhs_dual_norms is not None:
        rhs = np.asarray(port_rhs_dual_norms, dtype=float)
        if rhs.ndim != 1 or np.any(rhs < 0.0) or np.any(~np.isfinite(rhs)):
            raise ValueError("port_rhs_dual_norms must be a finite non-negative vector")
        # One state-error bound is applied to every driven port column.
        z = w * np.outer(rhs, np.full(rhs.size, eps))
        r = z.copy()
        l = z / w

    components = None
    vector = None
    if thermal_test_supremum_bounds is not None:
        if approximate_state_energy_norm is None:
            raise ValueError("approximate_state_energy_norm is required for heat-source propagation")
        xnorm = float(approximate_state_energy_norm)
        sup = np.asarray(thermal_test_supremum_bounds, dtype=float)
        if xnorm < 0.0 or not np.isfinite(xnorm):
            raise ValueError("approximate_state_energy_norm must be finite and non-negative")
        if sup.ndim != 1 or np.any(sup < 0.0) or np.any(~np.isfinite(sup)):
            raise ValueError("thermal_test_supremum_bounds must be finite and non-negative")
        common = 2.0 * xnorm * eps + eps * eps
        components = 0.5 * w * sup * common
        vector = float(np.linalg.norm(components))

    return SpatialOutputErrorCertificate(
        state_energy_error_bound=eps,
        impedance_abs_error_bound=z,
        resistance_abs_error_bound=r,
        inductance_abs_error_bound=l,
        heat_source_component_error_bounds=components,
        heat_source_vector_error_bound=vector,
    )
