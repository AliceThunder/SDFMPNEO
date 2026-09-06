from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .constitutive import CompositeCellConductivity
from .grid3d import RectilinearComplex3D


@dataclass(frozen=True)
class NonlinearSpatialAphiProblem:
    """Exact nonlinear temperature-dependent compatible A-phi problem.

    Temperature is reconstructed on cells as

        T(a) = T_ref + sum_k a_k Phi_k.

    The scalar-potential coordinate is psi = phi/(j*omega), so

        E = -j*omega (R_A alpha + G psi).

    For reciprocal real material Hodge matrices this makes the discrete field
    operator complex symmetric and preserves reciprocity structurally.
    """

    grid: RectilinearComplex3D
    omega: float
    curl: np.ndarray
    grad_c: np.ndarray
    a_basis: np.ndarray
    reluctivity_hodge: np.ndarray
    source_current: np.ndarray
    temperature_reference: np.ndarray
    thermal_modes: np.ndarray
    thermal_test: np.ndarray
    conductivity_model: CompositeCellConductivity
    b: np.ndarray
    H_metric: np.ndarray

    @classmethod
    def build(
        cls,
        grid: RectilinearComplex3D,
        *,
        omega: float,
        reluctivity_cell: np.ndarray,
        source_current: np.ndarray,
        temperature_reference: np.ndarray,
        thermal_modes: np.ndarray,
        conductivity_model: CompositeCellConductivity,
        thermal_test_cell: np.ndarray | None = None,
    ) -> "NonlinearSpatialAphiProblem":
        if omega <= 0:
            raise ValueError("omega must be positive")
        nu = np.asarray(reluctivity_cell, dtype=float)
        T0 = np.asarray(temperature_reference, dtype=float)
        Phi = np.asarray(thermal_modes, dtype=float)
        if nu.shape != grid.shape_cells or T0.shape != grid.shape_cells:
            raise ValueError("cell field shape mismatch")
        if Phi.ndim != 4 or Phi.shape[1:] != grid.shape_cells:
            raise ValueError("thermal_modes must have shape (n_thermal,*shape_cells)")
        Psi = Phi if thermal_test_cell is None else np.asarray(thermal_test_cell, dtype=float)
        if Psi.shape != Phi.shape:
            raise ValueError("thermal_test_cell must match thermal_modes shape")
        if np.asarray(source_current).shape != (grid.n_edges,):
            raise ValueError("source_current shape mismatch")

        sigma_ref = conductivity_model.evaluate(T0)
        support = sigma_ref > 0
        grad_c = grid.conductive_gradient(support)
        a_basis = grid.gauge_basis()
        C = grid.curl.toarray().astype(complex)
        R = np.asarray(a_basis, dtype=complex)
        G = np.asarray(grad_c, dtype=complex)
        Nu = grid.face_hodge(nu).toarray().astype(complex)
        S0 = grid.edge_hodge(sigma_ref).toarray().astype(complex)

        K_A = R.conj().T @ (C.conj().T @ Nu @ C) @ R
        L_E = -1j * omega * np.hstack([R, G])
        n_total = R.shape[1] + G.shape[1]
        H_mag = np.zeros((n_total, n_total), dtype=complex)
        H_mag[: R.shape[1], : R.shape[1]] = 0.5 * K_A
        H_metric = H_mag + (0.5 / omega) * (L_E.conj().T @ S0 @ L_E)
        H_metric = 0.5 * (H_metric + H_metric.conj().T)
        scipy.linalg.cholesky(H_metric, lower=True, check_finite=True)

        b = np.concatenate(
            [R.conj().T @ np.asarray(source_current, dtype=complex), np.zeros(G.shape[1], dtype=complex)]
        )

        return cls(
            grid=grid,
            omega=omega,
            curl=C,
            grad_c=G,
            a_basis=R,
            reluctivity_hodge=Nu,
            source_current=np.asarray(source_current, dtype=complex),
            temperature_reference=T0,
            thermal_modes=Phi,
            thermal_test=Psi,
            conductivity_model=conductivity_model,
            b=b,
            H_metric=H_metric,
        )

    @property
    def n_thermal(self) -> int:
        return self.thermal_modes.shape[0]

    @property
    def n_em(self) -> int:
        return self.b.size

    @property
    def n_A(self) -> int:
        return self.a_basis.shape[1]

    @property
    def n_scalar(self) -> int:
        return self.grad_c.shape[1]

    def source_coordinate(self, source_current: np.ndarray) -> np.ndarray:
        source = np.asarray(source_current, dtype=complex)
        if source.shape != (self.grid.n_edges,):
            raise ValueError("source_current shape mismatch")
        return np.concatenate(
            [self.a_basis.conj().T @ source, np.zeros(self.n_scalar, dtype=complex)]
        )

    def temperature(self, a: np.ndarray) -> np.ndarray:
        state = np.asarray(a, dtype=float)
        if state.shape != (self.n_thermal,):
            raise ValueError("thermal state dimension mismatch")
        return self.temperature_reference + np.tensordot(state, self.thermal_modes, axes=(0, 0))

    def conductivity_and_derivatives(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = self.temperature(a)
        sigma = self.conductivity_model.evaluate(T)
        dsigma_dT = self.conductivity_model.derivative(T)
        derivatives = dsigma_dT[None, ...] * self.thermal_modes
        return sigma, derivatives

    def _assemble_from_sigma_hodge(self, sigma_hodge: np.ndarray, *, include_curl: bool) -> np.ndarray:
        R = self.a_basis
        G = self.grad_c
        S = np.asarray(sigma_hodge, dtype=complex)

        if include_curl:
            K_A = R.conj().T @ (self.curl.conj().T @ self.reluctivity_hodge @ self.curl) @ R
        else:
            K_A = np.zeros((self.n_A, self.n_A), dtype=complex)

        jw = 1j * self.omega
        top_left = K_A + jw * (R.conj().T @ S @ R)
        top_right = jw * (R.conj().T @ S @ G)
        bottom_left = jw * (G.conj().T @ S @ R)
        bottom_right = jw * (G.conj().T @ S @ G)
        return np.block([[top_left, top_right], [bottom_left, bottom_right]])

    def electric_extraction(self) -> np.ndarray:
        return -1j * self.omega * np.hstack([self.a_basis, self.grad_c])

    def operator(self, a: np.ndarray) -> np.ndarray:
        sigma, _ = self.conductivity_and_derivatives(a)
        S = self.grid.edge_hodge(sigma).toarray()
        return self._assemble_from_sigma_hodge(S, include_curl=True)

    def operator_derivatives(self, a: np.ndarray) -> np.ndarray:
        _, derivatives = self.conductivity_and_derivatives(a)
        return np.stack(
            [
                self._assemble_from_sigma_hodge(
                    self.grid.edge_hodge(derivatives[k]).toarray(),
                    include_curl=False,
                )
                for k in range(self.n_thermal)
            ],
            axis=0,
        )

    def loss_operator(self, output_mode: int, a: np.ndarray) -> np.ndarray:
        sigma, _ = self.conductivity_and_derivatives(a)
        weighted = sigma * self.thermal_test[output_mode]
        W = self.grid.edge_hodge(weighted).toarray()
        L = self.electric_extraction()
        return 0.5 * (L.conj().T @ W @ L)

    def loss_operator_derivative(
        self,
        output_mode: int,
        state_mode: int,
        a: np.ndarray,
    ) -> np.ndarray:
        _, derivatives = self.conductivity_and_derivatives(a)
        weighted = derivatives[state_mode] * self.thermal_test[output_mode]
        W = self.grid.edge_hodge(weighted).toarray()
        L = self.electric_extraction()
        return 0.5 * (L.conj().T @ W @ L)

    def solve_full(self, a: np.ndarray) -> np.ndarray:
        return scipy.linalg.solve(self.operator(a), self.b, assume_a="gen")
