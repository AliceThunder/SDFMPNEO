from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SurrogateAudit:
    samples: int
    mean_relative_error: float
    p95_relative_error: float
    maximum_relative_error: float
    baseline_mean_relative_error: float
    baseline_maximum_relative_error: float
    baseline_improvement_ratio: float
    maximum_reciprocity_defect: float
    minimum_dissipation_eigenvalue: float
    passed: bool

    def to_dict(self):
        return {
            "samples": self.samples,
            "mean_relative_error": self.mean_relative_error,
            "p95_relative_error": self.p95_relative_error,
            "maximum_relative_error": self.maximum_relative_error,
            "baseline_mean_relative_error": self.baseline_mean_relative_error,
            "baseline_maximum_relative_error": self.baseline_maximum_relative_error,
            "baseline_improvement_ratio": self.baseline_improvement_ratio,
            "maximum_reciprocity_defect": self.maximum_reciprocity_defect,
            "minimum_dissipation_eigenvalue": self.minimum_dissipation_eigenvalue,
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


def audit_surrogate(
    artifact,
    samples,
    *,
    mean_relative_error_limit: float = 0.02,
    maximum_relative_error_limit: float = 0.05,
    reciprocity_tolerance: float = 1e-8,
    passivity_tolerance: float = 1e-10,
) -> SurrogateAudit:
    samples = tuple(samples)
    if not samples:
        raise ValueError(
            "audit requires at least one sample"
        )
    if (
        mean_relative_error_limit <= 0.0
        or maximum_relative_error_limit <= 0.0
        or reciprocity_tolerance < 0.0
        or passivity_tolerance < 0.0
    ):
        raise ValueError(
            "audit tolerances are invalid"
        )

    errors = []
    baseline_errors = []
    reciprocity = []
    minimum_eigenvalue = np.inf

    for sample in samples:
        target = np.asarray(
            sample.target_impedance,
            dtype=complex,
        )
        predicted = np.asarray(
            artifact.predict(
                sample.scene,
                sample.frequency_hz,
            ),
            dtype=complex,
        )
        if predicted.shape != target.shape:
            raise ValueError(
                "artifact returned the wrong port-matrix shape"
            )
        if not np.all(
            np.isfinite(predicted)
        ):
            raise ValueError(
                "artifact returned non-finite impedance"
            )

        baseline = (
            sample.baseline_resistance
            + 1j
            * sample.baseline_reactance
        )
        errors.append(
            _relative_error(
                predicted,
                target,
            )
        )
        baseline_errors.append(
            _relative_error(
                baseline,
                target,
            )
        )
        scale = max(
            float(
                np.linalg.norm(
                    predicted
                )
            ),
            1e-30,
        )
        reciprocity.append(
            float(
                np.linalg.norm(
                    predicted
                    - predicted.T
                )
                / scale
            )
        )
        dissipation = 0.5 * (
            predicted
            + predicted.conj().T
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue,
            float(
                np.min(
                    np.linalg.eigvalsh(
                        dissipation
                    )
                )
            ),
        )

    errors = np.asarray(
        errors,
        dtype=float,
    )
    baseline_errors = np.asarray(
        baseline_errors,
        dtype=float,
    )
    mean_error = float(
        np.mean(errors)
    )
    max_error = float(
        np.max(errors)
    )
    baseline_mean = float(
        np.mean(
            baseline_errors
        )
    )
    improvement = float(
        baseline_mean
        / max(
            mean_error,
            1e-30,
        )
    )
    max_reciprocity = float(
        np.max(
            reciprocity
        )
    )
    passed = bool(
        mean_error
        <= mean_relative_error_limit
        and max_error
        <= maximum_relative_error_limit
        and max_reciprocity
        <= reciprocity_tolerance
        and minimum_eigenvalue
        >= -passivity_tolerance
    )

    return SurrogateAudit(
        samples=len(samples),
        mean_relative_error=mean_error,
        p95_relative_error=float(
            np.quantile(
                errors,
                0.95,
            )
        ),
        maximum_relative_error=max_error,
        baseline_mean_relative_error=baseline_mean,
        baseline_maximum_relative_error=float(
            np.max(
                baseline_errors
            )
        ),
        baseline_improvement_ratio=improvement,
        maximum_reciprocity_defect=max_reciprocity,
        minimum_dissipation_eigenvalue=float(
            minimum_eigenvalue
        ),
        passed=passed,
    )
