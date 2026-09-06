from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg

from .energy_solver import PhysicalEnergySparseApsiSolver
from .grid3d import RectilinearComplex3D
from .reduced import RieszFactor
from .sparse_solver import (
    CertifiedSparseApsiSolver,
    SparseEnergyLinearSolveCertificate,
    SparseLinearSolveCertificate,
)


@dataclass(frozen=True)
class MultiPortImpedanceResult:
    names: tuple[str, ...]
    flux_linkage: np.ndarray
    impedance: np.ndarray
    resistance: np.ndarray
    inductance: np.ndarray
    reciprocity_defect: float
    minimum_resistance_eigenvalue: float
    maximum_residual_dual_norm: float

    @property
    def self_inductance(self) -> np.ndarray:
        return np.diag(self.inductance).copy()

    @property
    def mutual_inductance(self) -> np.ndarray:
        return self.inductance - np.diag(np.diag(self.inductance))

    def average_input_power(self, currents: np.ndarray) -> float:
        I = np.asarray(currents, dtype=complex)
        if I.shape != (len(self.names),):
            raise ValueError("currents must have shape (n_ports,)")
        return float(0.5 * np.real(np.vdot(I, self.impedance @ I)))


@dataclass(frozen=True)
class CertifiedSparseMultiPortResult:
    """Euclidean sparse multiport result retained for compatibility."""

    names: tuple[str, ...]
    flux_linkage: np.ndarray
    impedance: np.ndarray
    resistance: np.ndarray
    inductance: np.ndarray
    reciprocity_defect: float
    minimum_resistance_eigenvalue: float
    solve_certificates: tuple[SparseLinearSolveCertificate, ...]
    impedance_element_error_bounds: np.ndarray
    resistance_element_error_bounds: np.ndarray
    inductance_element_error_bounds: np.ndarray
    requested_impedance_element_error: float

    @property
    def all_linear_solves_certified(self) -> bool:
        return all(c.certified for c in self.solve_certificates)

    @property
    def maximum_impedance_error_bound(self) -> float:
        return float(np.max(self.impedance_element_error_bounds))

    @property
    def self_inductance(self) -> np.ndarray:
        return np.diag(self.inductance).copy()

    @property
    def mutual_inductance(self) -> np.ndarray:
        return self.inductance - np.diag(np.diag(self.inductance))


@dataclass(frozen=True)
class CertifiedEnergySparseMultiPortResult:
    """Contrast-independent physical-energy sparse multiport certificate."""

    names: tuple[str, ...]
    flux_linkage: np.ndarray
    impedance: np.ndarray
    resistance: np.ndarray
    inductance: np.ndarray
    reciprocity_defect: float
    minimum_resistance_eigenvalue: float
    source_dual_energy_norms: np.ndarray
    solve_certificates: tuple[SparseEnergyLinearSolveCertificate, ...]
    impedance_element_error_bounds: np.ndarray
    resistance_element_error_bounds: np.ndarray
    inductance_element_error_bounds: np.ndarray
    requested_impedance_element_error: float

    @property
    def all_linear_solves_certified(self) -> bool:
        return all(c.certified for c in self.solve_certificates)

    @property
    def maximum_impedance_error_bound(self) -> float:
        return float(np.max(self.impedance_element_error_bounds))

    @property
    def self_inductance(self) -> np.ndarray:
        return np.diag(self.inductance).copy()

    @property
    def mutual_inductance(self) -> np.ndarray:
        return self.inductance - np.diag(np.diag(self.inductance))


@dataclass(frozen=True)
class ImpressedCurrentPortSet:
    """Work-conjugate closed-loop impressed-current ports."""

    names: tuple[str, ...]
    edge_currents: np.ndarray
    coordinate_rhs: np.ndarray
    omega: float

    @classmethod
    def build(
        cls,
        grid: RectilinearComplex3D,
        *,
        a_basis: np.ndarray,
        n_scalar: int,
        omega: float,
        edge_currents: np.ndarray,
        names: Sequence[str] | None = None,
    ) -> "ImpressedCurrentPortSet":
        currents = np.asarray(edge_currents, dtype=float)
        if currents.ndim == 1:
            currents = currents[:, None]
        if currents.ndim != 2 or currents.shape[0] != grid.n_edges:
            raise ValueError("edge_currents must have shape (n_edges,n_ports)")
        if omega <= 0:
            raise ValueError("omega must be positive")

        n_ports = currents.shape[1]
        if names is None:
            port_names = tuple(f"port_{i}" for i in range(n_ports))
        else:
            port_names = tuple(names)
            if len(port_names) != n_ports or len(set(port_names)) != n_ports:
                raise ValueError("port names must be unique and match n_ports")

        G = grid.grad.tocsr()
        Gt = G.T.tocsr()
        Gnorm_fro = float(np.sqrt(np.sum(np.abs(G.data) ** 2)))
        for p in range(n_ports):
            current = currents[:, p]
            defect = np.asarray(Gt @ current).ravel()
            scale = (
                np.finfo(float).eps
                * max(grid.n_nodes, grid.n_edges)
                * max(1.0, Gnorm_fro * float(np.linalg.norm(current)))
            )
            if float(np.linalg.norm(defect)) > scale:
                raise ValueError(
                    f"port {port_names[p]!r} is not a closed divergence-free impressed-current cochain"
                )

        R = np.asarray(a_basis, dtype=complex)
        if R.ndim != 2 or R.shape[0] != grid.n_edges:
            raise ValueError("a_basis shape mismatch")
        if n_scalar < 0:
            raise ValueError("n_scalar must be non-negative")

        top = R.conj().T @ currents.astype(complex)
        bottom = np.zeros((n_scalar, n_ports), dtype=complex)
        rhs = np.vstack([top, bottom])
        return cls(port_names, currents, rhs, float(omega))

    @property
    def n_ports(self) -> int:
        return self.edge_currents.shape[1]

    def rhs_for_currents(self, currents: np.ndarray) -> np.ndarray:
        I = np.asarray(currents, dtype=complex)
        if I.shape != (self.n_ports,):
            raise ValueError("currents must have shape (n_ports,)")
        return self.coordinate_rhs @ I

    def solve_coordinate_states(
        self,
        problem,
        thermal_state: np.ndarray,
        *,
        reduced_basis: np.ndarray | None = None,
    ) -> np.ndarray:
        a = np.asarray(thermal_state, dtype=float)
        A = np.asarray(problem.operator(a), dtype=complex)
        B = self.coordinate_rhs
        if A.shape[0] != B.shape[0]:
            raise ValueError("port coordinates do not match electromagnetic problem")

        if reduced_basis is None:
            return scipy.linalg.solve(A, B, assume_a="gen")

        V = np.asarray(reduced_basis, dtype=complex)
        if V.ndim != 2 or V.shape[0] != A.shape[0]:
            raise ValueError("reduced_basis shape mismatch")
        Ar = V.conj().T @ A @ V
        Br = V.conj().T @ B
        return V @ scipy.linalg.solve(Ar, Br, assume_a="gen")

    @staticmethod
    def _output_matrices(B: np.ndarray, X: np.ndarray, omega: float):
        flux = B.T @ X
        Z = 1j * omega * flux
        Rmat = np.real(Z)
        Lmat = np.imag(Z) / omega
        norm_z = float(np.linalg.norm(Z))
        reciprocity_absolute = float(np.linalg.norm(Z - Z.T))
        reciprocity = reciprocity_absolute if norm_z == 0.0 else reciprocity_absolute / norm_z
        resistance_symmetric = 0.5 * (Rmat + Rmat.T)
        minimum_resistance = float(np.min(np.linalg.eigvalsh(resistance_symmetric)))
        return flux, Z, Rmat, Lmat, reciprocity, minimum_resistance

    def evaluate(
        self,
        problem,
        thermal_state: np.ndarray,
        *,
        reduced_basis: np.ndarray | None = None,
    ) -> MultiPortImpedanceResult:
        a = np.asarray(thermal_state, dtype=float)
        A = np.asarray(problem.operator(a), dtype=complex)
        B = self.coordinate_rhs
        X = self.solve_coordinate_states(problem, a, reduced_basis=reduced_basis)
        flux, Z, Rmat, Lmat, reciprocity, minimum_resistance = self._output_matrices(
            B, X, self.omega
        )

        residuals = B - A @ X
        riesz = RieszFactor.build(problem.H_metric)
        maximum_residual = max(
            (riesz.dual_norm(residuals[:, p]) for p in range(self.n_ports)),
            default=0.0,
        )

        return MultiPortImpedanceResult(
            names=self.names,
            flux_linkage=flux,
            impedance=Z,
            resistance=Rmat,
            inductance=Lmat,
            reciprocity_defect=reciprocity,
            minimum_resistance_eigenvalue=minimum_resistance,
            maximum_residual_dual_norm=float(maximum_residual),
        )

    def evaluate_sparse_certified(
        self,
        problem,
        thermal_state: np.ndarray,
        *,
        stability_lower_bound: float,
        requested_impedance_element_error: float,
    ) -> CertifiedSparseMultiPortResult:
        """Legacy Euclidean sparse certificate requiring an external beta bound."""

        if not hasattr(problem, "operator_sparse") or not hasattr(problem, "n_A"):
            raise TypeError("problem must provide sparse A-psi operator and n_A")
        requested = float(requested_impedance_element_error)
        if requested <= 0.0:
            raise ValueError("requested_impedance_element_error must be positive")
        a = np.asarray(thermal_state, dtype=float)
        A = problem.operator_sparse(a)
        B = np.asarray(self.coordinate_rhs, dtype=complex)
        if A.shape[0] != B.shape[0]:
            raise ValueError("port coordinates do not match electromagnetic problem")

        source_norms = np.linalg.norm(B, axis=0)
        maximum_source_norm = float(np.max(source_norms))
        if maximum_source_norm <= 0.0:
            raise ValueError("all port coordinate sources are zero")
        requested_state_error = requested / (self.omega * maximum_source_norm)

        solver = CertifiedSparseApsiSolver(
            A,
            n_A=int(problem.n_A),
            stability_lower_bound=stability_lower_bound,
        )
        states = []
        certificates = []
        for p in range(self.n_ports):
            x, certificate = solver.solve(
                B[:, p],
                requested_state_error=requested_state_error,
            )
            states.append(x)
            certificates.append(certificate)
        X = np.column_stack(states)
        flux, Z, Rmat, Lmat, reciprocity, minimum_resistance = self._output_matrices(
            B, X, self.omega
        )

        state_bounds = np.array([c.state_error_bound for c in certificates], dtype=float)
        Z_bounds = self.omega * source_norms[:, None] * state_bounds[None, :]
        return CertifiedSparseMultiPortResult(
            names=self.names,
            flux_linkage=flux,
            impedance=Z,
            resistance=Rmat,
            inductance=Lmat,
            reciprocity_defect=reciprocity,
            minimum_resistance_eigenvalue=minimum_resistance,
            solve_certificates=tuple(certificates),
            impedance_element_error_bounds=Z_bounds,
            resistance_element_error_bounds=Z_bounds.copy(),
            inductance_element_error_bounds=Z_bounds / self.omega,
            requested_impedance_element_error=requested,
        )

    def evaluate_sparse_physical_certified(
        self,
        problem,
        thermal_state: np.ndarray,
        *,
        requested_impedance_element_error: float,
    ) -> CertifiedEnergySparseMultiPortResult:
        """Recommended contrast-independent sparse multiport certificate.

        For the physical A-psi energy metric H=K+D,

            ||Delta x||_H <= sqrt(2) ||r||_(H^-1),

        and the reciprocal port pairing gives

            |Delta Z_ij|
            <= omega ||b_i||_(H^-1) ||Delta x_j||_H.

        The requested impedance accuracy therefore determines the linear solve
        accuracy without any external singular-value estimate.
        """

        if not hasattr(problem, "operator_sparse"):
            raise TypeError("problem must provide sparse A-psi operator")
        requested = float(requested_impedance_element_error)
        if requested <= 0.0:
            raise ValueError("requested_impedance_element_error must be positive")
        a = np.asarray(thermal_state, dtype=float)
        A = problem.operator_sparse(a)
        B = np.asarray(self.coordinate_rhs, dtype=complex)
        if A.shape[0] != B.shape[0]:
            raise ValueError("port coordinates do not match electromagnetic problem")

        solver = PhysicalEnergySparseApsiSolver(A)
        source_dual_norms = np.array(
            [solver.energy.dual_norm(B[:, p]) for p in range(self.n_ports)],
            dtype=float,
        )
        maximum_source_dual_norm = float(np.max(source_dual_norms))
        if maximum_source_dual_norm <= 0.0:
            raise ValueError("all port coordinate sources are zero")
        requested_energy_state_error = requested / (
            self.omega * maximum_source_dual_norm
        )

        states = []
        certificates = []
        for p in range(self.n_ports):
            x, certificate = solver.solve(
                B[:, p],
                requested_energy_state_error=requested_energy_state_error,
            )
            states.append(x)
            certificates.append(certificate)
        X = np.column_stack(states)
        flux, Z, Rmat, Lmat, reciprocity, minimum_resistance = self._output_matrices(
            B, X, self.omega
        )

        energy_state_bounds = np.array(
            [c.energy_state_error_bound for c in certificates], dtype=float
        )
        Z_bounds = self.omega * source_dual_norms[:, None] * energy_state_bounds[None, :]
        return CertifiedEnergySparseMultiPortResult(
            names=self.names,
            flux_linkage=flux,
            impedance=Z,
            resistance=Rmat,
            inductance=Lmat,
            reciprocity_defect=reciprocity,
            minimum_resistance_eigenvalue=minimum_resistance,
            source_dual_energy_norms=source_dual_norms,
            solve_certificates=tuple(certificates),
            impedance_element_error_bounds=Z_bounds,
            resistance_element_error_bounds=Z_bounds.copy(),
            inductance_element_error_bounds=Z_bounds / self.omega,
            requested_impedance_element_error=requested,
        )
