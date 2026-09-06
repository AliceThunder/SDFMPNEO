from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .reduced import ParametricEMProblem


@dataclass(frozen=True)
class CompatibleAphiDiscretization:
    """Compatible magnetoquasistatic A-phi discretization.

    The scalar-potential coordinate is scaled as psi = phi/(j*omega). The
    electric field is

        E = -j*omega (R_A alpha + G psi),

    and reciprocal real material Hodge matrices produce a complex-symmetric
    discrete operator.
    """

    curl: np.ndarray
    grad_c: np.ndarray
    a_basis: np.ndarray
    reluctivity_hodge: np.ndarray
    conductivity0: np.ndarray
    conductivity_state: np.ndarray
    source_current: np.ndarray
    omega: float
    riesz_metric: np.ndarray
    thermal_loss_hodge0: np.ndarray
    thermal_loss_hodge_state: np.ndarray | None = None

    def __post_init__(self) -> None:
        C = np.asarray(self.curl)
        G = np.asarray(self.grad_c)
        R = np.asarray(self.a_basis)
        n_face, n_edge = C.shape
        if G.ndim != 2 or G.shape[0] != n_edge:
            raise ValueError("grad_c must have shape (n_edge,n_scalar)")
        if R.ndim != 2 or R.shape[0] != n_edge or R.shape[1] == 0:
            raise ValueError("a_basis must have shape (n_edge,n_A) with n_A>0")
        if self.reluctivity_hodge.shape != (n_face, n_face):
            raise ValueError("reluctivity_hodge shape mismatch")
        if self.conductivity0.shape != (n_edge, n_edge):
            raise ValueError("conductivity0 shape mismatch")
        if self.conductivity_state.ndim != 3 or self.conductivity_state.shape[1:] != (n_edge, n_edge):
            raise ValueError("conductivity_state must have shape (n_thermal,n_edge,n_edge)")
        n_thermal = self.conductivity_state.shape[0]
        if self.source_current.shape != (n_edge,):
            raise ValueError("source_current shape mismatch")
        n_total = R.shape[1] + G.shape[1]
        if self.riesz_metric.shape != (n_total, n_total):
            raise ValueError("riesz_metric shape mismatch")
        if self.thermal_loss_hodge0.shape != (n_thermal, n_edge, n_edge):
            raise ValueError("thermal_loss_hodge0 shape mismatch")
        if self.thermal_loss_hodge_state is not None:
            expected = (n_thermal, n_thermal, n_edge, n_edge)
            if self.thermal_loss_hodge_state.shape != expected:
                raise ValueError(f"thermal_loss_hodge_state must have shape {expected}")
        if self.omega <= 0:
            raise ValueError("omega must be positive")
        if not np.allclose(C @ G, 0.0):
            raise ValueError("Compatible topology requires curl @ grad_c = 0")

    @property
    def n_edge(self) -> int:
        return self.curl.shape[1]

    @property
    def n_A(self) -> int:
        return self.a_basis.shape[1]

    @property
    def n_phi(self) -> int:
        return self.grad_c.shape[1]

    @property
    def n_thermal(self) -> int:
        return self.conductivity_state.shape[0]

    def _assemble_from_sigma(self, sigma_hodge: np.ndarray, include_curl: bool) -> np.ndarray:
        C = np.asarray(self.curl, dtype=complex)
        G = np.asarray(self.grad_c, dtype=complex)
        R = np.asarray(self.a_basis, dtype=complex)
        S = np.asarray(sigma_hodge, dtype=complex)

        if include_curl:
            Nu = np.asarray(self.reluctivity_hodge, dtype=complex)
            K_A = R.conj().T @ (C.conj().T @ Nu @ C) @ R
        else:
            K_A = np.zeros((self.n_A, self.n_A), dtype=complex)

        jw = 1j * self.omega
        top_left = K_A + jw * (R.conj().T @ S @ R)
        top_right = jw * (R.conj().T @ S @ G)
        bottom_left = jw * (G.conj().T @ S @ R)
        bottom_right = jw * (G.conj().T @ S @ G)
        return np.block([[top_left, top_right], [bottom_left, bottom_right]])

    def electric_extraction(self) -> np.ndarray:
        R = np.asarray(self.a_basis, dtype=complex)
        G = np.asarray(self.grad_c, dtype=complex)
        return -1j * self.omega * np.hstack([R, G])

    def physical_riesz_metric(self) -> np.ndarray:
        C = np.asarray(self.curl, dtype=complex)
        R = np.asarray(self.a_basis, dtype=complex)
        Nu = np.asarray(self.reluctivity_hodge, dtype=complex)
        S0 = np.asarray(self.conductivity0, dtype=complex)
        K_A = R.conj().T @ (C.conj().T @ Nu @ C) @ R
        L = self.electric_extraction()

        n_total = self.n_A + self.n_phi
        H_mag = np.zeros((n_total, n_total), dtype=complex)
        H_mag[: self.n_A, : self.n_A] = 0.5 * K_A
        H = H_mag + (0.5 / self.omega) * (L.conj().T @ S0 @ L)
        H = 0.5 * (H + H.conj().T)
        scipy.linalg.cholesky(H, lower=True, check_finite=True)
        return H

    def source_coordinate(self, source_current: np.ndarray) -> np.ndarray:
        source = np.asarray(source_current, dtype=complex)
        if source.shape != (self.n_edge,):
            raise ValueError("source_current shape mismatch")
        return np.concatenate(
            [
                np.asarray(self.a_basis, dtype=complex).conj().T @ source,
                np.zeros(self.n_phi, dtype=complex),
            ]
        )

    def to_parametric_problem(self) -> ParametricEMProblem:
        A0 = self._assemble_from_sigma(self.conductivity0, include_curl=True)
        A_state = np.stack(
            [self._assemble_from_sigma(Sk, include_curl=False) for Sk in self.conductivity_state],
            axis=0,
        )
        b = self.source_coordinate(self.source_current)

        L = self.electric_extraction()
        H_loss = np.stack(
            [0.5 * (L.conj().T @ W @ L) for W in self.thermal_loss_hodge0],
            axis=0,
        )

        H_loss_state = None
        if self.thermal_loss_hodge_state is not None:
            n_total = self.n_A + self.n_phi
            H_loss_state = np.empty(
                (self.n_thermal, self.n_thermal, n_total, n_total),
                dtype=complex,
            )
            for j in range(self.n_thermal):
                for k in range(self.n_thermal):
                    W = self.thermal_loss_hodge_state[j, k]
                    H_loss_state[j, k] = 0.5 * (L.conj().T @ W @ L)

        return ParametricEMProblem(
            A0=A0,
            A_state=A_state,
            b=b,
            H_metric=self.physical_riesz_metric(),
            H_loss=H_loss,
            H_loss_state=H_loss_state,
        )
