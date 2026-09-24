from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .certified import certify_mixed_ports
from .convergence import mixed_impedance_convergence
from .em import MQSConfig


@dataclass(frozen=True)
class CertifiedReleaseAudit:
    samples: int
    certified_samples: int
    fast_domain_valid_samples: int
    maximum_final_residual: float
    maximum_relative_observable_correction: float
    maximum_discretization_change: float
    maximum_certified_truth_relative_error: float
    operator_backend: str
    passed: bool

    def to_dict(self):
        return {
            "samples": self.samples,
            "certified_samples": self.certified_samples,
            "fast_domain_valid_samples": self.fast_domain_valid_samples,
            "maximum_final_residual": self.maximum_final_residual,
            "maximum_relative_observable_correction": (
                self.maximum_relative_observable_correction
            ),
            "maximum_discretization_change": (
                self.maximum_discretization_change
            ),
            "maximum_certified_truth_relative_error": (
                self.maximum_certified_truth_relative_error
            ),
            "operator_backend": self.operator_backend,
            "passed": self.passed,
        }


def _relative_error(
    predicted,
    target,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=complex,
    )
    target = np.asarray(
        target,
        dtype=complex,
    )
    return float(
        np.linalg.norm(
            predicted - target
        )
        / max(
            np.linalg.norm(target),
            1e-30,
        )
    )


def audit_certified_release(
    artifact,
    samples,
    *,
    coarse_config: MQSConfig,
    fine_config: MQSConfig,
    convergence_tolerance: float = 0.02,
    algebraic_tolerance: float = 1e-7,
    correction_rtol: float = 1e-9,
    correction_restart: int = 40,
    correction_maxiter: int = 160,
    fast_domain_correction_limit: float = 0.20,
    truth_consistency_tolerance: float = 0.02,
    operator_backend: str = "matrix_free",
    matrix_free_chunk_size: int = 256,
) -> CertifiedReleaseAudit:
    samples = tuple(samples)
    if not samples:
        raise ValueError(
            "certified release audit requires at least one sample"
        )
    if (
        convergence_tolerance <= 0.0
        or algebraic_tolerance <= 0.0
        or correction_rtol <= 0.0
        or correction_restart < 1
        or correction_maxiter < 1
        or fast_domain_correction_limit < 0.0
        or truth_consistency_tolerance < 0.0
        or matrix_free_chunk_size < 1
    ):
        raise ValueError(
            "invalid certified release tolerances"
        )
    if operator_backend not in (
        "dense",
        "matrix_free",
    ):
        raise ValueError(
            "operator_backend must be 'dense' or 'matrix_free'"
        )

    certified_count = 0
    fast_domain_count = 0
    max_residual = 0.0
    max_correction = 0.0
    max_discretization = 0.0
    max_truth_error = 0.0

    for sample in samples:
        convergence = mixed_impedance_convergence(
            sample.scene,
            sample.frequency_hz,
            (
                coarse_config,
                fine_config,
            ),
            tolerance=convergence_tolerance,
        )
        discretization_change = float(
            convergence.steps[-1].relative_change
        )
        max_discretization = max(
            max_discretization,
            discretization_change,
        )

        certified = certify_mixed_ports(
            sample.scene,
            sample.frequency_hz,
            artifact,
            config=fine_config,
            convergence_report=convergence,
            algebraic_tolerance=algebraic_tolerance,
            correction_rtol=correction_rtol,
            correction_restart=correction_restart,
            correction_maxiter=correction_maxiter,
            allow_reference_fallback=False,
            operator_backend=operator_backend,
            matrix_free_chunk_size=matrix_free_chunk_size,
            fast_domain_correction_limit=(
                fast_domain_correction_limit
            ),
        )
        if certified.certified:
            certified_count += 1
        if certified.fast_domain_valid:
            fast_domain_count += 1

        max_residual = max(
            max_residual,
            float(certified.final_residual),
        )
        max_correction = max(
            max_correction,
            float(
                certified.relative_observable_correction
            ),
        )
        max_truth_error = max(
            max_truth_error,
            _relative_error(
                certified.impedance,
                sample.target_impedance,
            ),
        )

    passed = bool(
        certified_count == len(samples)
        and fast_domain_count == len(samples)
        and max_residual <= algebraic_tolerance
        and max_discretization <= convergence_tolerance
        and max_truth_error <= truth_consistency_tolerance
    )
    return CertifiedReleaseAudit(
        samples=len(samples),
        certified_samples=certified_count,
        fast_domain_valid_samples=fast_domain_count,
        maximum_final_residual=max_residual,
        maximum_relative_observable_correction=max_correction,
        maximum_discretization_change=max_discretization,
        maximum_certified_truth_relative_error=(
            max_truth_error
        ),
        operator_backend=operator_backend,
        passed=passed,
    )
