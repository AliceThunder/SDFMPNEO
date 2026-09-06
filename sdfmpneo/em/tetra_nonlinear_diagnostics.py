from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdfmpneo.spatial.barycentric_polynomial import (
    Polynomial,
    assemble_polynomial_weighted_nedelec_mass,
    polynomial_constant,
    polynomial_p1,
    polynomial_scale,
)

from .constitutive import AffineConductivity, ConstantConductivity, ReciprocalLinearResistivity
from .reciprocal_series import certified_reciprocal_polynomials
from .tetra_nonlinear import NonlinearTetrahedralApsiProblem


@dataclass(frozen=True)
class NonlinearTetrahedralRegionLossEvaluator:
    problem: NonlinearTetrahedralApsiProblem

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(region.name for region in self.problem.conductivity_regions)

    def _region_polynomials(self, name: str, a: np.ndarray) -> list[Polynomial]:
        matches = [region for region in self.problem.conductivity_regions if region.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        region = matches[0]
        T = self.problem.temperature_local(a)
        out: list[Polynomial] = [{} for _ in range(self.problem.mesh.n_tetrahedra)]
        for q in np.flatnonzero(np.asarray(region.mask, dtype=bool)):
            law = region.law
            if isinstance(law, ConstantConductivity):
                out[q] = polynomial_constant(law.sigma)
            elif isinstance(law, AffineConductivity):
                out[q] = polynomial_p1(law.evaluate(T[q]))
            elif isinstance(law, ReciprocalLinearResistivity):
                denominator = 1.0 + law.alpha * (T[q] - law.temperature_ref)
                inverse, _, _ = certified_reciprocal_polynomials(
                    denominator,
                    requested_relative_error=self.problem.constitutive_relative_error_budget,
                )
                out[q] = polynomial_scale(inverse, law.sigma_ref)
            else:
                raise TypeError(
                    f"unsupported tetrahedral certified conductivity law: {type(law).__name__}"
                )
        return out

    def operator(self, name: str, a: np.ndarray) -> np.ndarray:
        W = assemble_polynomial_weighted_nedelec_mass(
            self.problem.mesh,
            self._region_polynomials(name, a),
        ).toarray()
        L = self.problem.electric_extraction()
        return 0.5 * (L.conj().T @ W @ L)

    def evaluate_state(self, coordinate_state: np.ndarray, a: np.ndarray) -> dict[str, float]:
        x = np.asarray(coordinate_state, dtype=complex)
        return {
            name: float(np.real(np.vdot(x, self.operator(name, a) @ x)))
            for name in self.names
        }

    def evaluate_reduced_model(self, reduced_model, a: np.ndarray) -> dict[str, float]:
        return self.evaluate_state(reduced_model.state(a), a)
