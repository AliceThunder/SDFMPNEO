from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg

from .grid3d import RectilinearComplex3D
from .reduced import RieszFactor


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
        """Time-average real port power, 0.5 Re(I^H Z I)."""

        I = np.asarray(currents, dtype=complex)
        if I.shape != (len(self.names),):
            raise ValueError("currents must have shape (n_ports,)")
        return float(0.5 * np.real(np.vdot(I, self.impedance @ I)))


@dataclass(frozen=True)
class ImpressedCurrentPortSet:
    """Work-conjugate closed-loop impressed-current ports.

    Each edge-current column is the spatial source cochain produced by one ampere
    of the corresponding port current. The caller is responsible for that
    physical ampere normalization; this class does not invent a geometric
    current scale.

    The source patterns must be divergence-free closed-current cochains. This
    makes the flux linkage J_p^T A gauge invariant. For real reciprocal material
    operators and the scaled scalar potential psi=phi/(j*omega), the resulting
    transfer matrix is structurally reciprocal.

    This port model is exact for the impressed-current/stranded-source
    formulation. A solid-conductor terminal-current port, in which the source
    conductor current distribution itself is solved subject to terminal current
    constraints, is a distinct formulation and is not silently conflated with
    this one.
    """

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

        # Divergence-free validation uses only sparse incidence data. The scale
        # is a floating-point backward-error bound, not a model tolerance.
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
        """Return one coordinate-state column for every unit port excitation."""

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

    def evaluate(
        self,
        problem,
        thermal_state: np.ndarray,
        *,
        reduced_basis: np.ndarray | None = None,
    ) -> MultiPortImpedanceResult:
        """Evaluate the field-reaction multiport impedance matrix.

        With one-ampere source cochains, the flux-linkage matrix is

            Psi = B^T A(a)^(-1) B,

        and

            Z = j*omega*Psi,
            R = Re(Z),
            L = Im(Z)/omega.

        Transpose, not conjugate transpose, is used in the reciprocal bilinear
        pairing. Source cochains are real by construction.
        """

        a = np.asarray(thermal_state, dtype=float)
        A = np.asarray(problem.operator(a), dtype=complex)
        B = self.coordinate_rhs
        X = self.solve_coordinate_states(problem, a, reduced_basis=reduced_basis)

        flux = B.T @ X
        Z = 1j * self.omega * flux
        Rmat = np.real(Z)
        Lmat = np.imag(Z) / self.omega

        norm_z = float(np.linalg.norm(Z))
        reciprocity_absolute = float(np.linalg.norm(Z - Z.T))
        reciprocity = reciprocity_absolute if norm_z == 0.0 else reciprocity_absolute / norm_z

        resistance_symmetric = 0.5 * (Rmat + Rmat.T)
        minimum_resistance = float(np.min(np.linalg.eigvalsh(resistance_symmetric)))

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
