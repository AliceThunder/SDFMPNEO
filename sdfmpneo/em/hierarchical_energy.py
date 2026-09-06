from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .adaptive_block import AdaptiveAggregateEnergyPreconditioner
from .certified_riesz import CertifiedEnergyPreconditioner
from .pair_block import _deterministic_energy_matching, _hermitian_positive_diagonal


def _pair_split_transform(
    matrix: sp.csr_matrix,
) -> tuple[sp.csr_matrix, sp.csr_matrix, tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Return deterministic fine/coarse pair transforms for one hierarchy level.

    The matching is the strongest normalized-energy matching already used by the
    certified pair preconditioner; no strength threshold is introduced.  For a
    real magnetic energy block and a matched pair (i,j), the low-energy sign is
    selected directly from the sign of B_ij:

        coarse = (e_i + s e_j)/sqrt(2),
        fine   = (e_i - s e_j)/sqrt(2),
        s = -sign(B_ij).

    For Laplace/curl-like negative off-diagonals this gives the expected smooth
    same-sign coarse coordinate.  Each 2x2 transform is nonsingular; unmatched
    coordinates pass directly to the next coarse level.
    """

    B = sp.csr_matrix(matrix, dtype=complex)
    n = B.shape[0]
    if B.shape != (n, n) or n == 0:
        raise ValueError("hierarchy level matrix must be non-empty and square")
    diagonal = _hermitian_positive_diagonal(B)
    blocks = _deterministic_energy_matching(B, diagonal)
    pairs = tuple(tuple(map(int, block)) for block in blocks if len(block) == 2)
    singles = tuple(int(block[0]) for block in blocks if len(block) == 1)

    if not pairs:
        return (
            sp.csr_matrix((n, 0), dtype=complex),
            sp.eye(n, format="csr", dtype=complex),
            tuple(),
            singles,
        )

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    fine_rows: list[int] = []
    fine_cols: list[int] = []
    fine_data: list[complex] = []
    coarse_rows: list[int] = []
    coarse_cols: list[int] = []
    coarse_data: list[complex] = []

    for column, (i, j) in enumerate(pairs):
        coupling = complex(B[i, j])
        imag_scale = max(abs(coupling.real), 1.0)
        if abs(coupling.imag) > 64.0 * np.finfo(float).eps * imag_scale:
            raise ValueError("magnetic hierarchy requires a real symmetric magnetic energy block")
        sign = 1.0 if coupling.real <= 0.0 else -1.0

        fine_rows.extend((i, j))
        fine_cols.extend((column, column))
        fine_data.extend((inv_sqrt2, -sign * inv_sqrt2))

        coarse_rows.extend((i, j))
        coarse_cols.extend((column, column))
        coarse_data.extend((inv_sqrt2, sign * inv_sqrt2))

    coarse_offset = len(pairs)
    for k, i in enumerate(singles):
        coarse_rows.append(i)
        coarse_cols.append(coarse_offset + k)
        coarse_data.append(1.0)

    F = sp.coo_matrix(
        (fine_data, (fine_rows, fine_cols)),
        shape=(n, len(pairs)),
        dtype=complex,
    ).tocsr()
    C = sp.coo_matrix(
        (coarse_data, (coarse_rows, coarse_cols)),
        shape=(n, len(pairs) + len(singles)),
        dtype=complex,
    ).tocsr()
    return F, C, pairs, singles


@dataclass(frozen=True)
class HierarchicalEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """Certificate-driven multilevel energy-coordinate preconditioner.

    A sequence of deterministic pair splittings produces an invertible hierarchy
    transform T whose columns are all fine difference coordinates followed by the
    final unresolved coarse coordinates.  With

        B_h = T^H B T,

    a certified local aggregate action constructs Q_h and proves

        B_h >= m_h Q_h.

    Therefore, with P = T^{-H} Q_h T^{-1},

        B >= m_h P,
        P^{-1} = T Q_h^{-1} T^H.

    The scientific interface contains no smoother count, strength threshold,
    damping, drop tolerance, fixed coarse dimension, or configured block size.
    Hierarchy depth and coarse dimension follow solely from deterministic energy
    matching; the final local block sizes follow solely from the spectral
    block-Gershgorin certificate.
    """

    dimension: int
    lower_spectral_equivalence_bound: float
    hierarchy_depth: int
    fine_dimensions_by_level: tuple[int, ...]
    coarse_dimensions_by_level: tuple[int, ...]
    final_coarse_dimension: int
    maximum_transformed_block_size: int
    transformed_final_block_count: int
    aggregation_steps: int
    _transform: sp.csr_matrix
    _transformed_matrix: sp.csr_matrix
    _local_action: AdaptiveAggregateEnergyPreconditioner

    @classmethod
    def build(cls, B: sp.spmatrix) -> "HierarchicalEnergyPreconditioner":
        matrix = sp.csr_matrix(B, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n) or n == 0:
            raise ValueError("hierarchical energy matrix must be non-empty and square")
        matrix = (0.5 * (matrix + matrix.conj().T)).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        _hermitian_positive_diagonal(matrix)

        current_basis = sp.eye(n, format="csr", dtype=complex)
        current_matrix = matrix
        completed_fine: list[sp.csr_matrix] = []
        fine_dims: list[int] = []
        coarse_dims: list[int] = []

        while current_matrix.shape[0] > 1:
            F, C, pairs, _singles = _pair_split_transform(current_matrix)
            if not pairs:
                break
            fine_basis = (current_basis @ F).tocsr()
            completed_fine.append(fine_basis)
            fine_dims.append(int(F.shape[1]))

            current_basis = (current_basis @ C).tocsr()
            current_matrix = (C.conj().T @ current_matrix @ C).tocsr()
            current_matrix = (0.5 * (current_matrix + current_matrix.conj().T)).tocsr()
            current_matrix.sum_duplicates()
            current_matrix.eliminate_zeros()
            coarse_dims.append(int(current_matrix.shape[0]))

        columns = completed_fine + [current_basis]
        T = sp.hstack(columns, format="csr", dtype=complex)
        if T.shape != (n, n):
            raise RuntimeError("hierarchical coordinate transform is not square")

        transformed = (T.conj().T @ matrix @ T).tocsr()
        transformed = (0.5 * (transformed + transformed.conj().T)).tocsr()
        transformed.sum_duplicates()
        transformed.eliminate_zeros()
        local = AdaptiveAggregateEnergyPreconditioner.build(transformed)
        lower = float(local.lower_spectral_equivalence_bound)
        if lower <= 0.0:
            raise ValueError("hierarchical local action has no positive certified lower bound")

        return cls(
            dimension=n,
            lower_spectral_equivalence_bound=lower,
            hierarchy_depth=len(fine_dims),
            fine_dimensions_by_level=tuple(fine_dims),
            coarse_dimensions_by_level=tuple(coarse_dims),
            final_coarse_dimension=int(current_basis.shape[1]),
            maximum_transformed_block_size=int(local.maximum_block_size),
            transformed_final_block_count=int(local.final_block_count),
            aggregation_steps=int(local.aggregation_steps),
            _transform=T,
            _transformed_matrix=transformed,
            _local_action=local,
        )

    @property
    def inverse_inf_upper_bound(self) -> float:
        """A conservative induced-infinity upper bound for the represented P^{-1}."""

        T = self._transform
        row_sum = np.asarray(np.abs(T).sum(axis=1)).ravel()
        col_sum = np.asarray(np.abs(T).sum(axis=0)).ravel()
        norm_inf_T = float(np.max(row_sum))
        norm_inf_TH = float(np.max(col_sum))
        local = float(self._local_action.inverse_inf_upper_bound)
        return float(np.nextafter(norm_inf_T * local * norm_inf_TH, np.inf))

    def transformed_matrix(self) -> sp.csr_matrix:
        return self._transformed_matrix.copy()

    def transformed_preconditioner_matrix(self) -> sp.csr_matrix:
        return self._local_action.preconditioner_matrix(self._transformed_matrix)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        if vector.shape != (self.dimension,):
            raise ValueError("hierarchical preconditioner rhs dimension mismatch")
        transformed_rhs = np.asarray(self._transform.conj().T @ vector, dtype=complex)
        transformed_solution = self._local_action.solve(transformed_rhs)
        return np.asarray(self._transform @ transformed_solution, dtype=complex)
