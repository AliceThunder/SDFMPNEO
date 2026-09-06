from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


ReferenceKind = Literal["maxwell", "comsol", "experiment", "other"]


@dataclass(frozen=True)
class IndependentValidationCase:
    """One external validation datum, explicitly excluded from training."""

    name: str
    predicted: np.ndarray
    reference: np.ndarray
    reference_kind: ReferenceKind
    certificate_abs_bound: np.ndarray | None = None
    used_for_training: bool = False

    def __post_init__(self) -> None:
        if self.used_for_training:
            raise ValueError("validation evidence may not be used as a training label")
        p = np.asarray(self.predicted, dtype=float)
        r = np.asarray(self.reference, dtype=float)
        if p.shape != r.shape or np.any(~np.isfinite(p)) or np.any(~np.isfinite(r)):
            raise ValueError("predicted/reference validation arrays must be finite with equal shape")
        if self.reference_kind not in {"maxwell", "comsol", "experiment", "other"}:
            raise ValueError("unsupported validation reference_kind")
        bound = None
        if self.certificate_abs_bound is not None:
            bound = np.asarray(self.certificate_abs_bound, dtype=float)
            if bound.shape not in {(), p.shape}:
                raise ValueError("certificate_abs_bound must be scalar or match output shape")
            if np.any(bound < 0.0) or np.any(~np.isfinite(bound)):
                raise ValueError("certificate_abs_bound must be finite and non-negative")
        object.__setattr__(self, "predicted", p)
        object.__setattr__(self, "reference", r)
        object.__setattr__(self, "certificate_abs_bound", bound)


@dataclass(frozen=True)
class ValidationCaseResult:
    name: str
    reference_kind: str
    absolute_error: np.ndarray
    relative_error: np.ndarray
    maximum_absolute_error: float
    maximum_relative_error: float
    within_declared_certificate: bool | None


@dataclass(frozen=True)
class IndependentValidationReport:
    cases: tuple[ValidationCaseResult, ...]

    @property
    def all_certified_cases_contain_reference(self) -> bool:
        checked = [c.within_declared_certificate for c in self.cases if c.within_declared_certificate is not None]
        return bool(checked) and all(checked)


def evaluate_independent_validation(
    cases: Sequence[IndependentValidationCase],
) -> IndependentValidationReport:
    """Compare predictions with external evidence without feeding it to training."""

    if not cases:
        raise ValueError("at least one independent validation case is required")
    results = []
    for case in cases:
        error = np.abs(case.predicted - case.reference)
        scale = np.maximum(np.abs(case.reference), np.finfo(float).tiny)
        relative = error / scale
        inside = None
        if case.certificate_abs_bound is not None:
            inside = bool(np.all(error <= case.certificate_abs_bound))
        results.append(
            ValidationCaseResult(
                name=case.name,
                reference_kind=case.reference_kind,
                absolute_error=error,
                relative_error=relative,
                maximum_absolute_error=float(np.max(error)),
                maximum_relative_error=float(np.max(relative)),
                within_declared_certificate=inside,
            )
        )
    return IndependentValidationReport(tuple(results))
