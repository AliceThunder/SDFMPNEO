from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp

from .certified_riesz import CertifiedEnergyPreconditioner


def _gamma(operation_count: int) -> float:
    eps = np.finfo(float).eps
    count = max(1, int(operation_count))
    if count * eps >= 1.0:
        raise FloatingPointError("operation count is too large for the gamma_k bound")
    return count * eps / (1.0 - count * eps)


def _hermitian_positive_diagonal(matrix: sp.csr_matrix) -> np.ndarray:
    n = matrix.shape[0]
    asym = matrix - matrix.conj().T
    asym_norm = float(np.sqrt(np.sum(np.abs(asym.data) ** 2))) if asym.nnz else 0.0
    matrix_norm = float(np.sqrt(np.sum(np.abs(matrix.data) ** 2))) if matrix.nnz else 0.0
    backward = _gamma(max(1, 8 * max(1, matrix.nnz))) * matrix_norm
    if asym_norm > backward:
        raise ValueError("block matrix must be Hermitian to floating-point backward-error scale")

    diagonal_complex = np.asarray(matrix.diagonal(), dtype=complex)
    scale = float(np.max(np.abs(diagonal_complex))) if n else 0.0
    imag_bound = _gamma(max(1, 4 * n)) * scale
    if n and float(np.max(np.abs(diagonal_complex.imag))) > imag_bound:
        raise ValueError("block matrix diagonal is not real to backward-error scale")
    diagonal = diagonal_complex.real.astype(float)
    if np.any(diagonal <= 0.0):
        raise ValueError("block matrix must have a strictly positive diagonal")
    return diagonal


def _coupling_upper(value: complex, di: float, dj: float) -> float:
    raw = float(abs(value) / math.sqrt(di * dj))
    g = _gamma(7)
    return float(np.nextafter(raw / (1.0 - g), np.inf))


def _pair_local_lambda_lower(
    matrix: sp.csr_matrix,
    diagonal: np.ndarray,
    block: tuple[int, ...],
) -> float:
    if len(block) == 1:
        return float(np.nextafter(diagonal[block[0]], 0.0))
    if len(block) != 2:
        raise ValueError("coupled-pair blocks must have size one or two")

    i, j = block
    coupling = _coupling_upper(matrix[i, j], diagonal[i], diagonal[j])
    local_normalized_lower = float(np.nextafter(1.0 - coupling, 0.0))
    if local_normalized_lower <= 0.0:
        raise ValueError("paired principal block is not certifiably positive")
    diagonal_lower = float(np.nextafter(min(diagonal[i], diagonal[j]), 0.0))
    return float(np.nextafter(local_normalized_lower * diagonal_lower, 0.0))


def _cross_frobenius_upper(
    matrix: sp.csr_matrix,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> float:
    sub = matrix[list(left), :][:, list(right)].tocsr()
    terms = [float(abs(value) ** 2) for value in sub.data]
    if not terms:
        return 0.0
    sum_hat = math.fsum(terms)
    g = _gamma(6 * len(terms) + 2)
    sum_upper = float(np.nextafter(sum_hat / (1.0 - g), np.inf))
    return float(np.nextafter(math.sqrt(sum_upper), np.inf))


def _deterministic_energy_matching(
    matrix: sp.csr_matrix,
    diagonal: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    """Pair the strongest normalized-energy couplings without a threshold.

    Every off-diagonal edge is ranked by |B_ij|/sqrt(B_ii B_jj).  The currently
    strongest edge whose endpoints are both unpaired is selected.  Ties are
    resolved lexicographically.  This is an algorithmic definition, not a tuned
    strength-of-connection threshold.
    """

    n = matrix.shape[0]
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        start, stop = matrix.indptr[i], matrix.indptr[i + 1]
        for j, value in zip(matrix.indices[start:stop], matrix.data[start:stop]):
            j = int(j)
            if j <= i or value == 0:
                continue
            score = float(abs(value) / math.sqrt(diagonal[i] * diagonal[j]))
            edges.append((-score, i, j))
    edges.sort()

    used = np.zeros(n, dtype=bool)
    blocks: list[tuple[int, ...]] = []
    for _negative_score, i, j in edges:
        if used[i] or used[j]:
            continue
        used[i] = True
        used[j] = True
        blocks.append((i, j))
    for i in range(n):
        if not used[i]:
            blocks.append((i,))
    blocks.sort(key=lambda block: block[0])
    return tuple(blocks)


@dataclass(frozen=True)
class CoupledPairEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """Factorization-free 1x1/2x2 physical-energy block preconditioner.

    Let the deterministic energy matching partition the SPD block B into
    principal blocks B_pp of size one or two and define

        P = blockdiag(B_pp).

    The local blocks are inverted analytically.  No global or sparse
    factorization is used.  For the block-normalized operator

        B_hat = P^{-1/2} B P^{-1/2},

    block Gershgorin gives

        lambda_min(B_hat)
        >= min_p [1 - sum_(q!=p) ||B_hat_pq||_2].

    The implementation upper-bounds each cross norm by

        ||B_pq||_F / sqrt(lambda_min(B_pp) lambda_min(B_qq)),

    with outward floating-point inflation.  A non-positive lower bound is
    rejected rather than repaired by damping or a strength threshold.
    """

    dimension: int
    blocks: tuple[tuple[int, ...], ...]
    local_lambda_lower_bounds: tuple[float, ...]
    normalized_row_radius_upper_bounds: tuple[float, ...]
    lower_spectral_equivalence_bound: float
    maximum_block_size: int
    paired_block_count: int
    _diagonal: np.ndarray
    _pair_data: tuple[tuple[int, int, complex, float, float, float], ...]

    @classmethod
    def build(cls, B: sp.spmatrix) -> "CoupledPairEnergyPreconditioner":
        matrix = sp.csr_matrix(B, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n) or n == 0:
            raise ValueError("block matrix must be non-empty and square")
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        diagonal = _hermitian_positive_diagonal(matrix)
        blocks = _deterministic_energy_matching(matrix, diagonal)
        local_lowers = tuple(
            _pair_local_lambda_lower(matrix, diagonal, block) for block in blocks
        )

        interaction_terms: list[list[float]] = [[] for _ in blocks]
        for p in range(len(blocks)):
            for q in range(p + 1, len(blocks)):
                cross = _cross_frobenius_upper(matrix, blocks[p], blocks[q])
                if cross == 0.0:
                    continue
                denom = math.sqrt(local_lowers[p] * local_lowers[q])
                normalized = float(np.nextafter(cross / denom, np.inf))
                interaction_terms[p].append(normalized)
                interaction_terms[q].append(normalized)

        radii: list[float] = []
        row_lowers: list[float] = []
        for terms in interaction_terms:
            radius_hat = math.fsum(terms)
            g = _gamma(max(1, len(terms) + 1))
            radius_upper = float(np.nextafter(radius_hat / (1.0 - g), np.inf))
            radii.append(radius_upper)
            row_lowers.append(float(np.nextafter(1.0 - radius_upper, -np.inf)))

        lower = float(min(row_lowers)) if row_lowers else 1.0
        if lower <= 0.0:
            raise ValueError(
                "coupled-pair block Gershgorin certificate has no positive spectral-equivalence lower bound"
            )

        pair_data: list[tuple[int, int, complex, float, float, float]] = []
        for block in blocks:
            if len(block) != 2:
                continue
            i, j = block
            a = float(diagonal[i])
            d = float(diagonal[j])
            c = complex(matrix[i, j])
            det = float(np.real(a * d - c * np.conjugate(c)))
            if det <= 0.0:
                raise ValueError("paired principal block determinant is not positive")
            pair_data.append((i, j, c, a, d, det))

        return cls(
            dimension=n,
            blocks=blocks,
            local_lambda_lower_bounds=tuple(float(v) for v in local_lowers),
            normalized_row_radius_upper_bounds=tuple(float(v) for v in radii),
            lower_spectral_equivalence_bound=float(np.nextafter(lower, 0.0)),
            maximum_block_size=max(len(block) for block in blocks),
            paired_block_count=sum(len(block) == 2 for block in blocks),
            _diagonal=diagonal.copy(),
            _pair_data=tuple(pair_data),
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        if vector.shape != (self.dimension,):
            raise ValueError("block preconditioner rhs dimension mismatch")

        out = np.empty_like(vector)
        paired = np.zeros(self.dimension, dtype=bool)
        for i, j, c, a, d, det in self._pair_data:
            bi = vector[i]
            bj = vector[j]
            out[i] = (d * bi - c * bj) / det
            out[j] = (-np.conjugate(c) * bi + a * bj) / det
            paired[i] = True
            paired[j] = True
        singles = ~paired
        out[singles] = vector[singles] / self._diagonal[singles]
        return out
