from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg

from sdfmpneo.spatial.barycentric_polynomial import (
    Polynomial,
    assemble_polynomial_weighted_nedelec_mass,
    polynomial_constant,
    polynomial_multiply,
    polynomial_p1,
    polynomial_scale,
)
from sdfmpneo.spatial.tetra3d import TetrahedralComplex3D

from .constitutive import (
    AffineConductivity,
    ConductivityRegion,
    ConstantConductivity,
    ReciprocalLinearResistivity,
)
from .reciprocal_series import ReciprocalSeriesCertificate, certified_reciprocal_polynomials


@dataclass(frozen=True)
class TetrahedralConstitutiveCertificate:
    requested_relative_error: float
    maximum_contraction_factor: float
    maximum_inverse_order: int
    maximum_inverse_square_order: int
    maximum_inverse_relative_bound: float
    maximum_inverse_square_relative_bound: float

    @property
    def certified(self) -> bool:
        return (
            self.maximum_inverse_relative_bound <= self.requested_relative_error
            and self.maximum_inverse_square_relative_bound <= self.requested_relative_error
        )


class NonlinearTetrahedralApsiProblem:
    """Certified nonlinear temperature-dependent A-psi problem on tetrahedra.

    Temperature is P1 inside each tetrahedron. Constant and affine conductivity
    laws therefore integrate exactly as barycentric polynomials. For a copper
    law induced by linear resistivity,

        sigma(T)=sigma_ref/[1+alpha(T-T_ref)],

    the reciprocal denominator is expanded around the exact element-wise range
    center. The geometric-series order is the smallest order satisfying the
    declared relative constitutive error budget; every retained polynomial term
    is integrated exactly against the Nedelec basis.
    """

    def __init__(
        self,
        mesh: TetrahedralComplex3D,
        *,
        omega: float,
        reluctivity_tetra: np.ndarray,
        source_current: np.ndarray,
        temperature_reference_local: np.ndarray,
        thermal_modes_local: np.ndarray,
        conductivity_regions: Sequence[ConductivityRegion],
        constitutive_relative_error_budget: float,
        thermal_test_local: np.ndarray | None = None,
    ) -> None:
        self.mesh = mesh
        self.omega = float(omega)
        if self.omega <= 0.0:
            raise ValueError("omega must be positive")

        nu = np.asarray(reluctivity_tetra, dtype=float)
        if nu.shape != (mesh.n_tetrahedra,) or np.any(nu <= 0.0):
            raise ValueError("reluctivity_tetra must be positive with shape (n_tetrahedra,)")
        self.reluctivity_tetra = nu

        source = np.asarray(source_current, dtype=complex)
        if source.shape != (mesh.n_edges,):
            raise ValueError("source_current shape mismatch")
        self.source_current = source

        T0 = np.asarray(temperature_reference_local, dtype=float)
        modes = np.asarray(thermal_modes_local, dtype=float)
        if T0.shape != (mesh.n_tetrahedra, 4):
            raise ValueError("temperature_reference_local must have shape (n_tetrahedra,4)")
        if modes.ndim != 3 or modes.shape[1:] != (mesh.n_tetrahedra, 4):
            raise ValueError("thermal_modes_local must have shape (n_thermal,n_tetrahedra,4)")
        if modes.shape[0] == 0:
            raise ValueError("at least one thermal mode is required")
        tests = modes if thermal_test_local is None else np.asarray(thermal_test_local, dtype=float)
        if tests.shape != modes.shape:
            raise ValueError("thermal_test_local must match thermal_modes_local")
        self.temperature_reference_local = T0
        self.thermal_modes_local = modes
        self.thermal_test_local = tests

        budget = float(constitutive_relative_error_budget)
        if not 0.0 < budget < 1.0:
            raise ValueError("constitutive_relative_error_budget must satisfy 0 < error < 1")
        self.constitutive_relative_error_budget = budget

        regions = tuple(conductivity_regions)
        occupied = np.zeros(mesh.n_tetrahedra, dtype=bool)
        support = np.zeros(mesh.n_tetrahedra, dtype=bool)
        for region in regions:
            mask = np.asarray(region.mask, dtype=bool)
            if mask.shape != (mesh.n_tetrahedra,):
                raise ValueError(f"region {region.name!r} mask shape mismatch")
            if np.any(occupied & mask):
                raise ValueError("tetrahedral conductivity regions must be disjoint")
            occupied |= mask
            if not np.any(mask):
                continue
            values = region.law.evaluate(T0[mask])
            derivatives = region.law.derivative(T0[mask])
            dynamic = np.any(derivatives != 0.0, axis=1)
            positive = np.all(values > 0.0, axis=1)
            nonzero = np.any(values > 0.0, axis=1)
            if np.any(dynamic & ~positive):
                raise ValueError(
                    f"region {region.name!r} would change conductivity support from a non-positive reference state"
                )
            local_indices = np.flatnonzero(mask)
            support[local_indices[nonzero | dynamic]] = True
        self.conductivity_regions = regions

        magnetic, _ = mesh.assemble_nedelec_edge_matrices(
            nu,
            np.zeros(mesh.n_tetrahedra),
        )
        self.magnetic_stiffness = magnetic
        self.a_basis = mesh.gauge_basis()
        self.grad_c = mesh.conductive_gradient(support)
        self.b = self.source_coordinate(source)
        self.H_metric = self._build_reference_riesz_metric()

    @property
    def n_thermal(self) -> int:
        return self.thermal_modes_local.shape[0]

    @property
    def n_A(self) -> int:
        return self.a_basis.shape[1]

    @property
    def n_scalar(self) -> int:
        return self.grad_c.shape[1]

    @property
    def n_em(self) -> int:
        return self.n_A + self.n_scalar

    def temperature_local(self, a: np.ndarray) -> np.ndarray:
        state = np.asarray(a, dtype=float)
        if state.shape != (self.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        return self.temperature_reference_local + np.tensordot(
            state,
            self.thermal_modes_local,
            axes=(0, 0),
        )

    def _empty_polynomials(self) -> list[Polynomial]:
        return [{} for _ in range(self.mesh.n_tetrahedra)]

    def _base_and_derivative_polynomials(
        self,
        a: np.ndarray,
        derivative_mode: int | None,
    ) -> tuple[list[Polynomial], list[ReciprocalSeriesCertificate]]:
        T = self.temperature_local(a)
        out = self._empty_polynomials()
        certificates: list[ReciprocalSeriesCertificate] = []

        if derivative_mode is not None and not 0 <= derivative_mode < self.n_thermal:
            raise ValueError("derivative_mode out of range")

        for region in self.conductivity_regions:
            mask = np.asarray(region.mask, dtype=bool)
            for q in np.flatnonzero(mask):
                law = region.law
                Tq = T[q]
                if isinstance(law, ConstantConductivity):
                    if derivative_mode is None:
                        out[q] = polynomial_constant(law.sigma)
                    continue

                if isinstance(law, AffineConductivity):
                    if derivative_mode is None:
                        sigma_nodal = law.evaluate(Tq)
                        out[q] = polynomial_p1(sigma_nodal)
                    else:
                        out[q] = polynomial_scale(
                            polynomial_p1(self.thermal_modes_local[derivative_mode, q]),
                            law.beta,
                        )
                    continue

                if isinstance(law, ReciprocalLinearResistivity):
                    denominator = 1.0 + law.alpha * (Tq - law.temperature_ref)
                    inverse, inverse_square, certificate = certified_reciprocal_polynomials(
                        denominator,
                        requested_relative_error=self.constitutive_relative_error_budget,
                    )
                    certificates.append(certificate)
                    if derivative_mode is None:
                        out[q] = polynomial_scale(inverse, law.sigma_ref)
                    else:
                        mode = polynomial_p1(self.thermal_modes_local[derivative_mode, q])
                        out[q] = polynomial_scale(
                            polynomial_multiply(inverse_square, mode),
                            -law.sigma_ref * law.alpha,
                        )
                    continue

                raise TypeError(
                    f"unsupported tetrahedral certified conductivity law: {type(law).__name__}"
                )

        return out, certificates

    def _weighted_polynomials(
        self,
        a: np.ndarray,
        *,
        derivative_mode: int | None = None,
        test_mode: int | None = None,
    ) -> tuple[list[Polynomial], list[ReciprocalSeriesCertificate]]:
        polynomials, certificates = self._base_and_derivative_polynomials(a, derivative_mode)
        if test_mode is None:
            return polynomials, certificates
        if not 0 <= test_mode < self.n_thermal:
            raise ValueError("test_mode out of range")
        weighted = []
        for q, poly in enumerate(polynomials):
            test = polynomial_p1(self.thermal_test_local[test_mode, q])
            weighted.append(polynomial_multiply(poly, test))
        return weighted, certificates

    def constitutive_certificate(self, a: np.ndarray) -> TetrahedralConstitutiveCertificate:
        _, certificates = self._base_and_derivative_polynomials(a, None)
        if not certificates:
            return TetrahedralConstitutiveCertificate(
                requested_relative_error=self.constitutive_relative_error_budget,
                maximum_contraction_factor=0.0,
                maximum_inverse_order=0,
                maximum_inverse_square_order=0,
                maximum_inverse_relative_bound=0.0,
                maximum_inverse_square_relative_bound=0.0,
            )
        return TetrahedralConstitutiveCertificate(
            requested_relative_error=self.constitutive_relative_error_budget,
            maximum_contraction_factor=max(c.contraction_factor for c in certificates),
            maximum_inverse_order=max(c.inverse_order for c in certificates),
            maximum_inverse_square_order=max(c.inverse_square_order for c in certificates),
            maximum_inverse_relative_bound=max(c.inverse_relative_bound for c in certificates),
            maximum_inverse_square_relative_bound=max(c.inverse_square_relative_bound for c in certificates),
        )

    def conductivity_matrix(self, a: np.ndarray):
        polynomials, _ = self._weighted_polynomials(a)
        return assemble_polynomial_weighted_nedelec_mass(self.mesh, polynomials)

    def conductivity_derivative_matrix(self, a: np.ndarray, mode: int):
        polynomials, _ = self._weighted_polynomials(a, derivative_mode=mode)
        return assemble_polynomial_weighted_nedelec_mass(self.mesh, polynomials)

    def electric_extraction(self) -> np.ndarray:
        R = self.a_basis.toarray().astype(complex)
        G = self.grad_c.toarray().astype(complex)
        return -1j * self.omega * np.hstack([R, G])

    def source_coordinate(self, source_current: np.ndarray) -> np.ndarray:
        source = np.asarray(source_current, dtype=complex)
        if source.shape != (self.mesh.n_edges,):
            raise ValueError("source_current shape mismatch")
        top = self.a_basis.toarray().astype(complex).conj().T @ source
        return np.concatenate([top, np.zeros(self.n_scalar, dtype=complex)])

    def _assemble_system(self, conductivity, *, include_magnetic: bool) -> np.ndarray:
        R = self.a_basis.toarray().astype(complex)
        G = self.grad_c.toarray().astype(complex)
        S = conductivity.toarray().astype(complex)
        if include_magnetic:
            K = self.magnetic_stiffness.toarray().astype(complex)
            K_A = R.conj().T @ K @ R
        else:
            K_A = np.zeros((self.n_A, self.n_A), dtype=complex)
        jw = 1j * self.omega
        return np.block(
            [
                [K_A + jw * (R.conj().T @ S @ R), jw * (R.conj().T @ S @ G)],
                [jw * (G.conj().T @ S @ R), jw * (G.conj().T @ S @ G)],
            ]
        )

    def _build_reference_riesz_metric(self) -> np.ndarray:
        zero = np.zeros(self.n_thermal)
        S = self.conductivity_matrix(zero).toarray().astype(complex)
        R = self.a_basis.toarray().astype(complex)
        K = self.magnetic_stiffness.toarray().astype(complex)
        K_A = R.conj().T @ K @ R
        L = self.electric_extraction()
        H_mag = np.zeros((self.n_em, self.n_em), dtype=complex)
        H_mag[: self.n_A, : self.n_A] = 0.5 * K_A
        H = H_mag + (0.5 / self.omega) * (L.conj().T @ S @ L)
        H = 0.5 * (H + H.conj().T)
        scipy.linalg.cholesky(H, lower=True, check_finite=True)
        return H

    def operator(self, a: np.ndarray) -> np.ndarray:
        return self._assemble_system(self.conductivity_matrix(a), include_magnetic=True)

    def operator_derivatives(self, a: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                self._assemble_system(
                    self.conductivity_derivative_matrix(a, k),
                    include_magnetic=False,
                )
                for k in range(self.n_thermal)
            ],
            axis=0,
        )

    def loss_operator(self, output_mode: int, a: np.ndarray) -> np.ndarray:
        polynomials, _ = self._weighted_polynomials(a, test_mode=output_mode)
        W = assemble_polynomial_weighted_nedelec_mass(self.mesh, polynomials).toarray()
        L = self.electric_extraction()
        return 0.5 * (L.conj().T @ W @ L)

    def loss_operator_derivative(
        self,
        output_mode: int,
        state_mode: int,
        a: np.ndarray,
    ) -> np.ndarray:
        polynomials, _ = self._weighted_polynomials(
            a,
            derivative_mode=state_mode,
            test_mode=output_mode,
        )
        W = assemble_polynomial_weighted_nedelec_mass(self.mesh, polynomials).toarray()
        L = self.electric_extraction()
        return 0.5 * (L.conj().T @ W @ L)

    def solve_full(self, a: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve(self.operator(a), self.b, assume_a="gen")
