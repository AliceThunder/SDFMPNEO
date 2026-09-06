from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .algebra import AnalyticSeries, solve_response_series
from .compiler import CompiledAnalyticKernel
from .realization import AnalyticRealization, CompiledRealizationGraph


@dataclass(frozen=True)
class BaseNode:
    name: str
    mode: int
    amplitude: complex

    def series(self, n_modes: int) -> AnalyticSeries:
        return AnalyticSeries.decay(n_modes, self.mode, self.amplitude)

    def realization(self, decay_rate: float) -> AnalyticRealization:
        return AnalyticRealization.decay(decay_rate, self.amplitude)


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
    kernel: CompiledAnalyticKernel

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.kernel.evaluate(t)

    def series_for_node(self, name: str) -> AnalyticSeries:
        return self.node_series[name]

    def source_for_node(self, name: str) -> AnalyticSeries:
        return self.node_sources[name]

    def term_counts(self) -> Dict[str, int]:
        return {name: series.term_count() for name, series in self.node_series.items()}


class AnalyticEvolutionGraph:
    """Analytic neural DAG with fast and resonance-stable exact backends."""

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
        self._realization_compiled: CompiledRealizationGraph | None = None

    def _invalidate(self) -> None:
        self._compiled = None
        self._realization_compiled = None

    def clone(self) -> "AnalyticEvolutionGraph":
        out = AnalyticEvolutionGraph(self.lambdas.copy(), self.a0.copy())
        for node in self.response_nodes:
            out.add_product_response(node.name, node.target_mode, node.parents, node.weight)
        return out

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
        """Compile to the fast finite polynomial-exponential kernel."""

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

        mode_tuple = tuple(mode_series)
        self._compiled = CompiledAnalyticGraph(
            lambdas=self.lambdas.copy(),
            node_series=dict(node_series),
            node_sources=dict(node_sources),
            mode_series=mode_tuple,
            kernel=CompiledAnalyticKernel.build(mode_tuple, self.lambdas),
        )
        return self._compiled

    def compile_realization(self) -> CompiledRealizationGraph:
        """Compile the same DAG to exact state-space analytic realizations.

        The representation is closed under multiplication and response without
        dividing by differences of decay rates, so exact and near resonances do
        not require a closeness threshold.
        """

        if self._realization_compiled is not None:
            return self._realization_compiled

        node_realizations: Dict[str, AnalyticRealization] = {
            n.name: n.realization(self.lambdas[n.mode]) for n in self.base_nodes
        }
        source_realizations: Dict[str, AnalyticRealization] = {}
        mode_realizations = [AnalyticRealization.zero() for _ in range(self.n_modes)]

        for node in self.base_nodes:
            mode_realizations[node.mode] = mode_realizations[node.mode].add(node_realizations[node.name])

        for node in self.response_nodes:
            source = node_realizations[node.parents[0]]
            for parent in node.parents[1:]:
                source = source.product(node_realizations[parent])
            source = source.scaled(node.weight)
            response = source.response(self.lambdas[node.target_mode])
            source_realizations[node.name] = source
            node_realizations[node.name] = response
            mode_realizations[node.target_mode] = mode_realizations[node.target_mode].add(response)

        self._realization_compiled = CompiledRealizationGraph(
            lambdas=self.lambdas.copy(),
            node_realizations=dict(node_realizations),
            source_realizations=dict(source_realizations),
            mode_realizations=tuple(mode_realizations),
        )
        return self._realization_compiled

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.compile().evaluate(t)

    def evaluate_stable(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.compile_realization().evaluate(t)

    def backend_consistency_defect(self, t: float) -> float:
        fast_a, fast_da = self.evaluate(t)
        stable_a, stable_da = self.evaluate_stable(t)
        numerator = np.linalg.norm(np.concatenate([fast_a - stable_a, fast_da - stable_da]))
        denominator = max(
            1.0,
            float(np.linalg.norm(np.concatenate([stable_a, stable_da]))),
        )
        return float(numerator / denominator)

    def node_series(self, name: str) -> AnalyticSeries:
        return self.compile().series_for_node(name)

    def node_source(self, name: str) -> AnalyticSeries:
        return self.compile().source_for_node(name)

    def term_counts(self) -> Dict[str, int]:
        return self.compile().term_counts()
