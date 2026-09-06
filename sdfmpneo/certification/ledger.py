from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CertifiedErrorTerm:
    name: str
    bound: float
    certified: bool
    provenance: str

    def __post_init__(self) -> None:
        value = float(self.bound)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError("error bound must be finite and non-negative")
        object.__setattr__(self, "bound", value)


@dataclass(frozen=True)
class UnifiedErrorCertificate:
    terms: tuple[CertifiedErrorTerm, ...]
    output_lipschitz: float = 1.0

    def __post_init__(self) -> None:
        if self.output_lipschitz <= 0.0 or not np.isfinite(self.output_lipschitz):
            raise ValueError("output_lipschitz must be finite and positive")
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise ValueError("error-term names must be unique")

    @property
    def fully_certified(self) -> bool:
        required = {
            "constitutive", "algebraic", "em_rom", "mesh", "outer",
            "thermal_rom", "analytic_residual",
        }
        available = {term.name for term in self.terms if term.certified}
        return required.issubset(available)

    @property
    def state_error_bound(self) -> float | None:
        if not self.fully_certified:
            return None
        return float(sum(term.bound for term in self.terms))

    @property
    def output_error_bound(self) -> float | None:
        state = self.state_error_bound
        return None if state is None else float(self.output_lipschitz * state)

    @property
    def missing_terms(self) -> tuple[str, ...]:
        required = (
            "constitutive", "algebraic", "em_rom", "mesh", "outer",
            "thermal_rom", "analytic_residual",
        )
        certified = {term.name for term in self.terms if term.certified}
        return tuple(name for name in required if name not in certified)


def certified_difference_term(
    name: str,
    difference_norm: float,
    *,
    reliability_constant: float | None,
    provenance: str,
) -> CertifiedErrorTerm:
    """Turn a nested-discretization difference into an error term only with proof.

    If a theorem provides ``||e|| <= C_rel ||u_h-u_H||``, pass that proven
    ``C_rel``.  A missing constant is not replaced by an empirical value; the
    returned term is explicitly uncertified.
    """

    diff = float(difference_norm)
    if diff < 0.0 or not np.isfinite(diff):
        raise ValueError("difference_norm must be finite and non-negative")
    if reliability_constant is None:
        return CertifiedErrorTerm(name, diff, False, provenance + "; missing reliability proof")
    C = float(reliability_constant)
    if C <= 0.0 or not np.isfinite(C):
        raise ValueError("reliability_constant must be finite and positive")
    return CertifiedErrorTerm(name, C * diff, True, provenance)


def build_unified_error_certificate(
    terms: Mapping[str, tuple[float, bool, str]],
    *,
    output_lipschitz: float = 1.0,
) -> UnifiedErrorCertificate:
    return UnifiedErrorCertificate(
        tuple(
            CertifiedErrorTerm(name, value, certified, provenance)
            for name, (value, certified, provenance) in terms.items()
        ),
        output_lipschitz=float(output_lipschitz),
    )
