from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from sdfmpneo.spatial.nedelec_weighted import assemble_weighted_nedelec_mass
from sdfmpneo.spatial.tetra3d import TetrahedralComplex3D

from .reduced import ParametricEMProblem


@dataclass(frozen=True)
class TetrahedralApsiDiscretization:
    """Reciprocal magnetoquasistatic A-psi system on a Nedelec tetrahedral mesh.

    The magnetic part is supplied directly as the first-order Nedelec edge
    curl-curl matrix. This is deliberately distinct from the orthogonal-complex
    face-Hodge representation; only the common reduced electromagnetic interface
    is shared above the spatial discretization layer.
    """

    mesh: TetrahedralComplex3D
    magnetic_stiffness: sp.csr_matrix
    grad_c: sp.csr_matrix
    a_basis: sp.csr_matrix
    conductivity0: sp.csr_matrix
    conductivity_state: tuple[sp.csr_matrix, ...]
    source_current: np.ndarray
    omega: float
    thermal_loss0: tuple[sp.csr_matrix, ...]
    thermal_loss_state: tuple[tuple[sp.csr_matrix, ...], ...]

    def __post_init__(self) -> None:
        n_edge = self.mesh.n_edges
        if self.magnetic_stiffness.shape != (n_edge, n_edge):
            raise ValueError("magnetic_stiffness shape mismatch")
        if self.conductivity0.shape != (n_edge, n_edge):
            raise ValueError("conductivity0 shape mismatch")
        if self.a_basis.shape[0] != n_edge or self.a_basis.shape[1] == 0:
            raise ValueError("a_basis shape mismatch")
        if self.grad_c.shape[0] != n_edge:
            raise ValueError("grad_c shape mismatch")
        if np.asarray(self.source_current).shape != (n_edge,):
            raise ValueError("source_current shape mismatch")
        if self.omega <= 0:
            raise ValueError("omega must be positive")
        n_thermal = len(self.conductivity_state)
        if len(self.thermal_loss0) != n_thermal:
            raise ValueError("thermal loss output count must equal thermal state dimension")
        if len(self.thermal_loss_state) != n_thermal:
            raise ValueError("thermal loss derivative output count mismatch")
        for matrix in self.conductivity_state:
            if matrix.shape != (n_edge, n_edge):
                raise ValueError("conductivity_state matrix shape mismatch")
        for row in self.thermal_loss_state:
            if len(row) != n_thermal:
                raise ValueError("thermal_loss_state must be square in thermal indices")
            for matrix in row:
                if matrix.shape != (n_edge, n_edge):
                    raise ValueError("thermal_loss_state matrix shape mismatch")
        product = self.mesh.curl @ self.grad_c
        if product.nnz != 0:
            raise ValueError("compatible topology requires curl @ grad_c = 0 exactly")

    @property
    def n_A(self) -> int:
        return self.a_basis.shape[1]

    @property
    def n_scalar(self) -> int:
        return self.grad_c.shape[1]

    @property
    def n_thermal(self) -> int:
        return len(self.conductivity_state)

    @property
    def n_em(self) -> int:
        return self.n_A + self.n_scalar

    def _dense(self, matrix: sp.spmatrix) -> np.ndarray:
        return matrix.toarray().astype(complex)

    def electric_extraction(self) -> np.ndarray:
        R = self.a_basis.toarray().astype(complex)
        G = self.grad_c.toarray().astype(complex)
        return -1j * self.omega * np.hstack([R, G])

    def _assemble_from_conductivity(
        self,
        conductivity: sp.spmatrix,
        *,
        include_magnetic: bool,
    ) -> np.ndarray:
        R = self.a_basis.toarray().astype(complex)
        G = self.grad_c.toarray().astype(complex)
        S = self._dense(conductivity)
        if include_magnetic:
            K = self._dense(self.magnetic_stiffness)
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

    def source_coordinate(self, source_current: np.ndarray) -> np.ndarray:
        source = np.asarray(source_current, dtype=complex)
        if source.shape != (self.mesh.n_edges,):
            raise ValueError("source_current shape mismatch")
        top = self.a_basis.toarray().astype(complex).conj().T @ source
        return np.concatenate([top, np.zeros(self.n_scalar, dtype=complex)])

    def physical_riesz_metric(self) -> np.ndarray:
        R = self.a_basis.toarray().astype(complex)
        K = self._dense(self.magnetic_stiffness)
        K_A = R.conj().T @ K @ R
        L = self.electric_extraction()
        S0 = self._dense(self.conductivity0)
        H_mag = np.zeros((self.n_em, self.n_em), dtype=complex)
        H_mag[: self.n_A, : self.n_A] = 0.5 * K_A
        H = H_mag + (0.5 / self.omega) * (L.conj().T @ S0 @ L)
        H = 0.5 * (H + H.conj().T)
        scipy.linalg.cholesky(H, lower=True, check_finite=True)
        return H

    def to_parametric_problem(self) -> ParametricEMProblem:
        A0 = self._assemble_from_conductivity(self.conductivity0, include_magnetic=True)
        A_state = np.stack(
            [
                self._assemble_from_conductivity(matrix, include_magnetic=False)
                for matrix in self.conductivity_state
            ],
            axis=0,
        )
        L = self.electric_extraction()
        H_loss = np.stack(
            [0.5 * (L.conj().T @ self._dense(W) @ L) for W in self.thermal_loss0],
            axis=0,
        )
        H_loss_state = np.empty(
            (self.n_thermal, self.n_thermal, self.n_em, self.n_em),
            dtype=complex,
        )
        for j in range(self.n_thermal):
            for k in range(self.n_thermal):
                W = self.thermal_loss_state[j][k]
                H_loss_state[j, k] = 0.5 * (L.conj().T @ self._dense(W) @ L)

        return ParametricEMProblem(
            A0=A0,
            A_state=A_state,
            b=self.source_coordinate(self.source_current),
            H_metric=self.physical_riesz_metric(),
            H_loss=H_loss,
            H_loss_state=H_loss_state,
        )


def tetra_face_loop_source(
    mesh: TetrahedralComplex3D,
    face_index: int,
    amplitude: complex = 1.0,
) -> np.ndarray:
    """Closed impressed-current cochain equal to the oriented boundary of a face."""

    if not 0 <= int(face_index) < mesh.n_faces:
        raise ValueError("face_index out of range")
    return (
        complex(amplitude)
        * mesh.curl.getrow(int(face_index)).toarray().ravel().astype(complex)
    )


def build_tetrahedral_apsi_from_thermal_modes(
    mesh: TetrahedralComplex3D,
    *,
    omega: float,
    reluctivity_tetra: np.ndarray,
    conductivity_reference_tetra: np.ndarray,
    conductivity_temperature_slope_tetra: np.ndarray,
    thermal_mode_local_values: np.ndarray,
    source_current: np.ndarray,
    thermal_test_local_values: np.ndarray | None = None,
) -> TetrahedralApsiDiscretization:
    """Build the affine-in-temperature tetrahedral electromagnetic core exactly.

    `thermal_mode_local_values[k,q,i]` is the value of thermal mode k at local
    vertex i of tetrahedron q. If

        sigma(T) = sigma_ref + (d sigma / dT) * sum_k a_k phi_k,

    the conductivity state matrices are assembled with exact P1-weighted
    Nedelec integrals. Projected Joule heating uses exact P1 weights, and its
    explicit temperature derivative uses exact P1 x P1 weights. No cell-average
    thermal approximation is introduced.
    """

    nu = np.asarray(reluctivity_tetra, dtype=float)
    sigma0 = np.asarray(conductivity_reference_tetra, dtype=float)
    slope = np.asarray(conductivity_temperature_slope_tetra, dtype=float)
    modes = np.asarray(thermal_mode_local_values, dtype=float)
    tests = modes if thermal_test_local_values is None else np.asarray(
        thermal_test_local_values, dtype=float
    )

    if omega <= 0:
        raise ValueError("omega must be positive")
    if nu.shape != (mesh.n_tetrahedra,):
        raise ValueError("reluctivity_tetra shape mismatch")
    if sigma0.shape != (mesh.n_tetrahedra,) or slope.shape != (mesh.n_tetrahedra,):
        raise ValueError("conductivity arrays must have shape (n_tetrahedra,)")
    if modes.ndim != 3 or modes.shape[1:] != (mesh.n_tetrahedra, 4):
        raise ValueError("thermal_mode_local_values must have shape (n_thermal,n_tetrahedra,4)")
    if tests.shape != modes.shape:
        raise ValueError("thermal_test_local_values must match thermal modes")
    if np.any(nu <= 0) or np.any(sigma0 < 0):
        raise ValueError("reluctivity must be positive and conductivity non-negative")
    if np.any((slope != 0.0) & (sigma0 <= 0.0)):
        raise ValueError("temperature dependence may not create new conductivity support")

    magnetic, _ = mesh.assemble_nedelec_edge_matrices(
        nu,
        np.zeros(mesh.n_tetrahedra),
    )
    conductivity0 = assemble_weighted_nedelec_mass(
        mesh,
        scale_tetra=sigma0,
    )
    conductivity_state = tuple(
        assemble_weighted_nedelec_mass(
            mesh,
            scale_tetra=slope,
            p1_factors=modes[k : k + 1],
        )
        for k in range(modes.shape[0])
    )

    thermal_loss0 = tuple(
        assemble_weighted_nedelec_mass(
            mesh,
            scale_tetra=sigma0,
            p1_factors=tests[j : j + 1],
        )
        for j in range(modes.shape[0])
    )
    thermal_loss_state = tuple(
        tuple(
            assemble_weighted_nedelec_mass(
                mesh,
                scale_tetra=slope,
                p1_factors=np.stack([modes[k], tests[j]], axis=0),
            )
            for k in range(modes.shape[0])
        )
        for j in range(modes.shape[0])
    )

    return TetrahedralApsiDiscretization(
        mesh=mesh,
        magnetic_stiffness=magnetic,
        grad_c=mesh.conductive_gradient(sigma0 > 0.0),
        a_basis=mesh.gauge_basis(),
        conductivity0=conductivity0,
        conductivity_state=conductivity_state,
        source_current=np.asarray(source_current, dtype=complex),
        omega=float(omega),
        thermal_loss0=thermal_loss0,
        thermal_loss_state=thermal_loss_state,
    )
