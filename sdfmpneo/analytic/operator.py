from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parametric_realization import evaluate_parametric_stable


@dataclass(frozen=True)
class AnalyticOperatorPrediction:
    time: float
    initial: np.ndarray
    operating: np.ndarray
    state: np.ndarray
    derivative: np.ndarray
    residual: np.ndarray
    residual_norm: float


@dataclass(frozen=True)
class CanonicalAnalyticIR:
    lambdas: tuple[float, ...]
    initial_names: tuple[str, ...]
    operating_names: tuple[str, ...]
    # Per mode: (time_power, decay_signature, parameter_signature, real, imag)
    mode_terms: tuple[tuple[tuple[int, tuple[int, ...], tuple[int, ...], float, float], ...], ...]
    compression_error_bound: float = 0.0


class CertifiedAnalyticEvolutionOperator:
    """Arbitrary-time analytic neural evolution over one certified spectral chart.

    No time marching occurs at inference.  The stable backend realizes the same
    analytic DAG as matrix exponentials and therefore needs no near-resonance
    threshold.  ``canonical_ir`` performs only exact key merging already implied
    by the analytic algebra, so its compression error is identically zero.
    """

    def __init__(self, graph, vector_field) -> None:
        self.graph = graph
        self.vector_field = vector_field
        if graph.n_modes != vector_field.n_modes:
            raise ValueError("graph/vector-field state dimensions do not match")
        if len(graph.operating_names) != vector_field.n_operating:
            raise ValueError("graph/vector-field operating dimensions do not match")
        if not np.array_equal(np.asarray(graph.lambdas), np.asarray(vector_field.thermal_model.lambdas)):
            raise ValueError("graph and physical vector field must use the same thermal spectrum")

    def evaluate(
        self,
        t: float,
        *,
        a0: np.ndarray,
        operating: np.ndarray,
        stable: bool = True,
    ) -> AnalyticOperatorPrediction:
        time = float(t)
        if time < 0.0:
            raise ValueError("time must be non-negative")
        initial = np.asarray(a0, dtype=float)
        u = np.asarray(operating, dtype=float)
        if stable:
            a, da = evaluate_parametric_stable(
                self.graph, time, a0=initial, operating=u
            )
        else:
            a, da = self.graph.evaluate(time, a0=initial, operating=u)
        residual = self.vector_field.residual(a, da, u)
        return AnalyticOperatorPrediction(
            time=time,
            initial=initial.copy(),
            operating=u.copy(),
            state=np.asarray(a, dtype=float),
            derivative=np.asarray(da, dtype=float),
            residual=np.asarray(residual, dtype=float),
            residual_norm=float(np.linalg.norm(residual)),
        )

    def canonical_ir(self) -> CanonicalAnalyticIR:
        compiled = self.graph.compile()
        modes = []
        for series in compiled.mode_series:
            terms = []
            for (power, decay, params), coefficient in sorted(series.terms.items()):
                c = complex(coefficient)
                terms.append(
                    (
                        int(power),
                        tuple(int(v) for v in decay),
                        tuple(int(v) for v in params),
                        float(c.real),
                        float(c.imag),
                    )
                )
            modes.append(tuple(terms))
        return CanonicalAnalyticIR(
            lambdas=tuple(float(v) for v in compiled.lambdas),
            initial_names=tuple(compiled.initial_names),
            operating_names=tuple(compiled.operating_names),
            mode_terms=tuple(modes),
            compression_error_bound=0.0,
        )
