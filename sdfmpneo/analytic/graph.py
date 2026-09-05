from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .algebra import AnalyticSeries, DeferredResponseSeries


@dataclass(frozen=True)
class BaseNode:
    name: str
    mode: int
    amplitude: complex

    def series(self, n_modes: int) -> AnalyticSeries:
        return AnalyticSeries.decay(n_modes, self.mode, self.amplitude)


@dataclass(frozen=True)
class ProductResponseNode:
    name: str
    target_mode: int
    parents: Tuple[str, ...]
    weight: complex


class AnalyticEvolutionGraph:
    """Minimal analytic neural DAG with exact time derivatives."""

    def __init__(self, lambdas: Sequence[float], a0: Sequence[float]):
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.a0 = np.asarray(a0, dtype=float)
        if self.lambdas.ndim != 1 or self.a0.shape != self.lambdas.shape:
            raise ValueError("lambdas and a0 must be one-dimensional and have equal length")
        if np.any(self.lambdas <= 0):
            raise ValueError("Thermal decay rates must be positive")
        self.n_modes = len(self.lambdas)
        self.base_nodes: List[BaseNode] = [BaseNode(f"base_{i}", i, self.a0[i]) for i in range(self.n_modes)]
        self.response_nodes: List[ProductResponseNode] = []

    def add_product_response(self, name: str, target_mode: int, parents: Sequence[str], weight: complex) -> None:
        known = {n.name for n in self.base_nodes} | {n.name for n in self.response_nodes}
        if name in known:
            raise ValueError(f"Duplicate node name: {name}")
        if not parents:
            raise ValueError("A product response requires at least one parent")
        missing = [p for p in parents if p not in known]
        if missing:
            raise ValueError(f"Unknown parent nodes: {missing}")
        if not 0 <= target_mode < self.n_modes:
            raise ValueError("target_mode out of range")
        self.response_nodes.append(ProductResponseNode(name, target_mode, tuple(parents), complex(weight)))

    def _build_symbolic(self):
        series: Dict[str, AnalyticSeries] = {n.name: n.series(self.n_modes) for n in self.base_nodes}
        responses: Dict[str, DeferredResponseSeries] = {}
        for node in self.response_nodes:
            src = AnalyticSeries.constant(self.n_modes, node.weight)
            for parent in node.parents:
                if parent in responses:
                    raise ValueError(
                        "MVP permits product-response sources from explicit base analytic series only; "
                        "higher-order chained response compilation is the next implementation stage."
                    )
                src = src * series[parent]
            responses[node.name] = DeferredResponseSeries.from_source(src, node.target_mode)
        return series, responses

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        series, responses = self._build_symbolic()
        a = np.zeros(self.n_modes, dtype=float)
        da = np.zeros(self.n_modes, dtype=float)
        for node in self.base_nodes:
            s = series[node.name]
            a[node.mode] += np.real(s.evaluate(t, self.lambdas))
            da[node.mode] += np.real(s.derivative(self.lambdas).evaluate(t, self.lambdas))
        for node in self.response_nodes:
            r = responses[node.name]
            a[node.target_mode] += np.real(r.evaluate(t, self.lambdas))
            da[node.target_mode] += np.real(r.derivative_value(t, self.lambdas))
        return a, da
