from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from .certified_riesz import CertifiedEnergyPreconditioner
from .pair_block import (
    _cross_frobenius_upper,
    _deterministic_energy_matching,
    _gamma,
    _hermitian_positive_diagonal,
    _pair_local_lambda_lower,
)


def _dense_inverse_inf_certificate(
    block_matrix: np.ndarray,
    factor,
) -> tuple[float, float]:
    """Return certified upper ||B^-1||_inf and lower lambda_min(B).

    The parent matrix is a physically SPD energy block, hence every principal
    aggregate is SPD.  The dense solve is used only on the local aggregate.
    If X is the computed inverse and R=I-BX, then

        B^-1 = X (I-R)^-1,
        ||B^-1||_inf <= ||X||_inf/(1-||R||_inf)

    whenever the explicitly enclosed residual satisfies ||R||_inf<1.
    """

    A = np.asarray(block_matrix, dtype=complex)
    k = A.shape[0]
    identity = np.eye(k, dtype=complex)
    X = scipy.linalg.cho_solve(factor, identity, check_finite=False)
    product = A @ X
    residual_hat = identity - product

    dot_gamma = _gamma(max(1, 8 * k + 2))
    product_scale = np.abs(A) @ np.abs(X)
    residual_component_upper = np.abs(residual_hat) + dot_gamma * product_scale

    row_sum_gamma = _gamma(max(1, k + 1))
    residual_rows = np.sum(residual_component_upper, axis=1)
    residual_inf = float(
        np.nextafter(
            float(np.max(residual_rows)) / (1.0 - row_sum_gamma),
            np.inf,
        )
    )
    if residual_inf >= 1.0:
        raise ValueError("local aggregate inverse residual cannot certify nonsingularity")

    inverse_rows = np.sum(np.abs(X), axis=1)
    inverse_inf_hat = float(np.max(inverse_rows))
    inverse_inf = float(
        np.nextafter(
            inverse_inf_hat / (1.0 - row_sum_gamma) / (1.0 - residual_inf),
            np.inf,
        )
    )
    if inverse_inf <= 0.0 or not np.isfinite(inverse_inf):
        raise ValueError("local aggregate inverse norm certificate is invalid")
    lambda_lower = float(np.nextafter(1.0 / inverse_inf, 0.0))
    if lambda_lower <= 0.0:
        raise ValueError("local aggregate energy lower bound is non-positive")
    return inverse_inf, lambda_lower


@dataclass(frozen=True)
class _AggregateFactor:
    indices: tuple[int, ...]
    factor: tuple[np.ndarray, bool]
    lambda_lower_bound: float
    inverse_inf_upper_bound: float


@dataclass(frozen=True)
class AdaptiveAggregateEnergyPreconditioner(CertifiedEnergyPreconditioner):
    """Certificate-driven local aggregate energy preconditioner.

    Start from the parameter-free strongest-energy coupled-pair partition.  For a
    current principal-block partition P=blockdiag(B_pp), block Gershgorin yields

        B >= m_BG P,
        m_BG = min_p [1-sum_(q!=p)||P_p^-1/2 B_pq P_q^-1/2||_2].

    Cross norms are rigorously upper-bounded by sparse Frobenius norms divided by
    certified local eigenvalue lower bounds.  If m_BG is non-positive, the two
    aggregates with the largest normalized cross interaction are merged and the
    certificate is recomputed.  The process stops at the first positive
    certificate.  Therefore there is no configured block size, strength
    threshold, damping, or drop tolerance.

    Local aggregate solves use dense Cholesky factors only inside the final
    certificate-selected blocks.  No global sparse factorization is used by this
    preconditioner.  ``maximum_block_size`` is exposed so production runs can
    report whether the certificate remained genuinely local; a large aggregate
    is never hidden behind a scalability claim.
    """

    dimension: int
    blocks: tuple[tuple[int, ...], ...]
    lower_spectral_equivalence_bound: float
    normalized_row_radius_upper_bounds: tuple[float, ...]
    aggregation_steps: int
    initial_block_count: int
    final_block_count: int
    maximum_block_size: int
    _factors: tuple[_AggregateFactor, ...]

    @staticmethod
    def _factor_block(
        matrix: sp.csr_matrix,
        diagonal: np.ndarray,
        block: tuple[int, ...],
    ) -> _AggregateFactor:
        if len(block) <= 2:
            local_lower = _pair_local_lambda_lower(matrix, diagonal, block)
        else:
            local_lower = None

        dense = matrix[list(block), :][:, list(block)].toarray().astype(complex)
        dense = 0.5 * (dense + dense.conj().T)
        try:
            factor = scipy.linalg.cho_factor(
                dense,
                lower=True,
                check_finite=False,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError("local aggregate is not numerically positive definite") from exc

        inverse_inf, inverse_lambda_lower = _dense_inverse_inf_certificate(dense, factor)
        if local_lower is None:
            lambda_lower = inverse_lambda_lower
        else:
            lambda_lower = max(float(local_lower), float(inverse_lambda_lower))
        return _AggregateFactor(
            indices=tuple(block),
            factor=factor,
            lambda_lower_bound=float(lambda_lower),
            inverse_inf_upper_bound=float(inverse_inf),
        )

    @classmethod
    def _partition_certificate(
        cls,
        matrix: sp.csr_matrix,
        diagonal: np.ndarray,
        blocks: tuple[tuple[int, ...], ...],
    ) -> tuple[
        tuple[_AggregateFactor, ...],
        tuple[float, ...],
        float,
        tuple[int, int] | None,
    ]:
        factors = tuple(cls._factor_block(matrix, diagonal, block) for block in blocks)
        interactions: list[list[float]] = [[] for _ in blocks]
        largest_value = -1.0
        largest_pair: tuple[int, int] | None = None

        for p in range(len(blocks)):
            for q in range(p + 1, len(blocks)):
                cross = _cross_frobenius_upper(matrix, blocks[p], blocks[q])
                if cross == 0.0:
                    continue
                denom = math.sqrt(
                    factors[p].lambda_lower_bound * factors[q].lambda_lower_bound
                )
                normalized = float(np.nextafter(cross / denom, np.inf))
                interactions[p].append(normalized)
                interactions[q].append(normalized)
                if (
                    normalized > largest_value
                    or (
                        normalized == largest_value
                        and largest_pair is not None
                        and (blocks[p][0], blocks[q][0])
                        < (blocks[largest_pair[0]][0], blocks[largest_pair[1]][0])
                    )
                    or largest_pair is None
                ):
                    largest_value = normalized
                    largest_pair = (p, q)

        radii: list[float] = []
        row_lowers: list[float] = []
        for terms in interactions:
            radius_hat = math.fsum(terms)
            g = _gamma(max(1, len(terms) + 1))
            radius_upper = float(np.nextafter(radius_hat / (1.0 - g), np.inf))
            radii.append(radius_upper)
            row_lowers.append(float(np.nextafter(1.0 - radius_upper, -np.inf)))

        lower = float(min(row_lowers)) if row_lowers else 1.0
        return factors, tuple(radii), lower, largest_pair

    @classmethod
    def build(cls, B: sp.spmatrix) -> "AdaptiveAggregateEnergyPreconditioner":
        matrix = sp.csr_matrix(B, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n) or n == 0:
            raise ValueError("block matrix must be non-empty and square")
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        diagonal = _hermitian_positive_diagonal(matrix)

        blocks = _deterministic_energy_matching(matrix, diagonal)
        initial_count = len(blocks)
        steps = 0

        while True:
            factors, radii, lower, merge_pair = cls._partition_certificate(
                matrix,
                diagonal,
                blocks,
            )
            if lower > 0.0:
                return cls(
                    dimension=n,
                    blocks=blocks,
                    lower_spectral_equivalence_bound=float(np.nextafter(lower, 0.0)),
                    normalized_row_radius_upper_bounds=radii,
                    aggregation_steps=steps,
                    initial_block_count=initial_count,
                    final_block_count=len(blocks),
                    maximum_block_size=max(len(block) for block in blocks),
                    _factors=factors,
                )
            if len(blocks) == 1:
                raise RuntimeError("single aggregate must have block-Gershgorin lower bound one")
            if merge_pair is None:
                raise ValueError("failed block certificate has no interacting aggregates to merge")

            p, q = merge_pair
            merged = tuple(sorted(blocks[p] + blocks[q]))
            new_blocks = [block for k, block in enumerate(blocks) if k not in (p, q)]
            new_blocks.append(merged)
            new_blocks.sort(key=lambda block: block[0])
            blocks = tuple(new_blocks)
            steps += 1

    @property
    def inverse_inf_upper_bound(self) -> float:
        """Certified upper bound for the inverse infinity norm of the represented block matrix Q.

        Q is block diagonal with the final aggregate principal blocks.  Hence
        ``||Q^-1||_inf`` is the maximum inverse infinity norm over the blocks.
        Each local value is already residual-certified during construction.
        """

        return float(max(local.inverse_inf_upper_bound for local in self._factors))

    def preconditioner_matrix(self, parent: sp.spmatrix) -> sp.csr_matrix:
        """Materialize the exact block-diagonal matrix Q represented by this action.

        This method copies only the certificate-selected principal blocks from
        the sparse parent matrix.  It does not create a dense global matrix and
        is used for nested spectral/trace certificates.
        """

        matrix = sp.csr_matrix(parent, dtype=complex)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("parent matrix dimension mismatch")
        rows: list[int] = []
        cols: list[int] = []
        data: list[complex] = []
        for block in self.blocks:
            idx = np.asarray(block, dtype=int)
            local = matrix[idx, :][:, idx].tocoo()
            for i, j, value in zip(local.row, local.col, local.data):
                rows.append(int(idx[int(i)]))
                cols.append(int(idx[int(j)]))
                data.append(complex(value))
        out = sp.coo_matrix(
            (data, (rows, cols)),
            shape=matrix.shape,
            dtype=complex,
        ).tocsr()
        out.sum_duplicates()
        out.eliminate_zeros()
        return out

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        if vector.shape != (self.dimension,):
            raise ValueError("block preconditioner rhs dimension mismatch")
        out = np.empty_like(vector)
        for local in self._factors:
            idx = np.asarray(local.indices, dtype=int)
            out[idx] = scipy.linalg.cho_solve(
                local.factor,
                vector[idx],
                check_finite=False,
            )
        return out
