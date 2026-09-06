from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .algebra import AnalyticSeries, solve_response_series


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


@dataclass(frozen=True)
class CompiledAnalyticGraph:
    lambdas: np.ndarray
    node_series: Mapping[str, AnalyticSeries]
    node_sources: Mapping[str, AnalyticSeries]
    mode_series: Tuple[AnalyticSeries, ...]

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        a = np.array([np.real(s.evaluate(t, self.lambdas)) for s in self.mode_series], dtype=float)
        da = np.array(
            [np.real(s.derivative(self.lambdas).evaluate(t, self.lambdas)) for s in self.mode_series],
            dtype=float,
        )
        return a, da

    def series_for_node(self, name: str) -> AnalyticSeries:
        return self.node_series[name]

    def source_for_node(self, name: str) -> AnalyticSeries:
        return self.node_sources[name]

    def term_counts(self) -> Dict[str, int]:
        return {name: series.term_count() for name, series in self.node_series.items()}


class AnalyticEvolutionGraph:
    """Analytic neural DAG with arbitrary chained response neurons and exact time derivatives."""

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
        self._compiled: CompiledAnalyticGraph | None = None

    def _invalidate(self) -> None:
        self._compiled = None

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
        self._invalidate()

    def compile(self) -> CompiledAnalyticGraph:
        if self._compiled is not None:
            return self._compiled

        node_series: Dict[str, AnalyticSeries] = {n.name: n.series(self.n_modes) for n in self.base_nodes}
        node_sources: Dict[str, AnalyticSeries] = {}
        mode_series = [AnalyticSeries.zero(self.n_modes) for _ in range(self.n_modes)]

        for node in self.base_nodes:
            mode_series[node.mode] = mode_series[node.mode] + node_series[node.name]

        for node in self.response_nodes:
            source = AnalyticSeries.constant(self.n_modes, node.weight)
            for parent in node.parents:
                source = source * node_series[parent]

            response = solve_response_series(source, node.target_mode, self.lambdas)
            node_sources[node.name] = source
            node_series[node.name] = response
            mode_series[node.target_mode] = mode_series[node.target_mode] + response

        self._compiled = CompiledAnalyticGraph(
            lambdas=self.lambdas.copy(),
            node_series=dict(node_series),
            node_sources=dict(node_sources),
            mode_series=tuple(mode_series),
        )
        return self._compiled

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.compile().evaluate(t)

    def node_series(self, name: str) -> AnalyticSeries:
        return self.compile().series_for_node(name)

    def node_source(self, name: str) -> AnalyticSeries:
        return self.compile().source_for_node(name)

    def term_counts(self) -> Dict[str, int]:
        return self.compile().term_counts()
