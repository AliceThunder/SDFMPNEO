from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

DecaySignature = Tuple[int, ...]
ParameterSignature = Tuple[int, ...]
ParametricTermKey = Tuple[int, DecaySignature, ParameterSignature]


def _add_signature(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if len(a) != len(b):
        raise ValueError("signature dimensions must match")
    return tuple(x + y for x, y in zip(a, b))


def _unit_signature(size: int, index: int) -> tuple[int, ...]:
    if not 0 <= index < size:
        raise ValueError("signature index out of range")
    out = [0] * size
    out[index] = 1
    return tuple(out)


@dataclass(frozen=True)
class ParametricAnalyticSeries:
    """Sparse analytic series in static parameters and continuous time.

    Each term is

        c * p^gamma * t^m * exp(-(n dot lambda)t),

    where `p` contains the initial thermal coordinates followed by declared
    operating parameters. Static parameters are zero-dynamics neural nodes, so
    multiplication and the first-order response operator remain inside one
    analytic network algebra.
    """

    n_modes: int
    n_parameters: int
    terms: Dict[ParametricTermKey, complex]

    def __post_init__(self) -> None:
        clean: Dict[ParametricTermKey, complex] = {}
        for (power, decay, params), coeff in self.terms.items():
            if power < 0:
                raise ValueError("time power must be non-negative")
            if len(decay) != self.n_modes:
                raise ValueError("decay signature dimension mismatch")
            if len(params) != self.n_parameters:
                raise ValueError("parameter signature dimension mismatch")
            if any(x < 0 for x in decay) or any(x < 0 for x in params):
                raise ValueError("analytic signatures must be non-negative")
            key = (
                int(power),
                tuple(int(x) for x in decay),
                tuple(int(x) for x in params),
            )
            value = clean.get(key, 0.0) + complex(coeff)
            if value != 0:
                clean[key] = value
            elif key in clean:
                del clean[key]
        object.__setattr__(self, "terms", clean)

    @staticmethod
    def zero(n_modes: int, n_parameters: int) -> "ParametricAnalyticSeries":
        return ParametricAnalyticSeries(n_modes, n_parameters, {})

    @staticmethod
    def constant(n_modes: int, n_parameters: int, value: complex) -> "ParametricAnalyticSeries":
        return ParametricAnalyticSeries(
            n_modes,
            n_parameters,
            {(0, (0,) * n_modes, (0,) * n_parameters): complex(value)},
        )

    @staticmethod
    def static_parameter(
        n_modes: int,
        n_parameters: int,
        parameter_index: int,
    ) -> "ParametricAnalyticSeries":
        return ParametricAnalyticSeries(
            n_modes,
            n_parameters,
            {
                (
                    0,
                    (0,) * n_modes,
                    _unit_signature(n_parameters, parameter_index),
                ): 1.0
            },
        )

    @staticmethod
    def initial_decay(
        n_modes: int,
        n_parameters: int,
        mode: int,
        initial_parameter_index: int,
    ) -> "ParametricAnalyticSeries":
        return ParametricAnalyticSeries(
            n_modes,
            n_parameters,
            {
                (
                    0,
                    _unit_signature(n_modes, mode),
                    _unit_signature(n_parameters, initial_parameter_index),
                ): 1.0
            },
        )

    def scaled(self, scalar: complex) -> "ParametricAnalyticSeries":
        return ParametricAnalyticSeries(
            self.n_modes,
            self.n_parameters,
            {key: scalar * value for key, value in self.terms.items()},
        )

    def __rmul__(self, scalar: complex) -> "ParametricAnalyticSeries":
        if not np.isscalar(scalar):
            return NotImplemented
        return self.scaled(scalar)

    def __add__(self, other: "ParametricAnalyticSeries") -> "ParametricAnalyticSeries":
        if not isinstance(other, ParametricAnalyticSeries):
            return NotImplemented
        if (self.n_modes, self.n_parameters) != (other.n_modes, other.n_parameters):
            raise ValueError("series dimensions must match")
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = terms.get(key, 0.0) + value
        return ParametricAnalyticSeries(self.n_modes, self.n_parameters, terms)

    def __mul__(self, other):
        if np.isscalar(other):
            return self.scaled(other)
        if not isinstance(other, ParametricAnalyticSeries):
            return NotImplemented
        if (self.n_modes, self.n_parameters) != (other.n_modes, other.n_parameters):
            raise ValueError("series dimensions must match")
        terms: Dict[ParametricTermKey, complex] = {}
        for (m1, d1, p1), c1 in self.terms.items():
            for (m2, d2, p2), c2 in other.terms.items():
                key = (m1 + m2, _add_signature(d1, d2), _add_signature(p1, p2))
                terms[key] = terms.get(key, 0.0) + c1 * c2
        return ParametricAnalyticSeries(self.n_modes, self.n_parameters, terms)

    def derivative(self, lambdas: np.ndarray) -> "ParametricAnalyticSeries":
        lambdas = np.asarray(lambdas, dtype=float)
        if lambdas.shape != (self.n_modes,):
            raise ValueError("lambdas shape mismatch")
        terms: Dict[ParametricTermKey, complex] = {}
        for (m, decay, params), coeff in self.terms.items():
            rho = float(np.dot(decay, lambdas))
            if m > 0:
                key = (m - 1, decay, params)
                terms[key] = terms.get(key, 0.0) + m * coeff
            key = (m, decay, params)
            terms[key] = terms.get(key, 0.0) - rho * coeff
        return ParametricAnalyticSeries(self.n_modes, self.n_parameters, terms)

    def evaluate(self, t: float, lambdas: np.ndarray, parameters: np.ndarray) -> complex:
        if t < 0:
            raise ValueError("time must be non-negative")
        lambdas = np.asarray(lambdas, dtype=float)
        p = np.asarray(parameters, dtype=float)
        if lambdas.shape != (self.n_modes,):
            raise ValueError("lambdas shape mismatch")
        if p.shape != (self.n_parameters,):
            raise ValueError("parameter vector shape mismatch")

        total = 0.0 + 0.0j
        for (m, decay, params), coeff in self.terms.items():
            rho = float(np.dot(decay, lambdas))
            monomial = 1.0
            for value, exponent in zip(p, params):
                if exponent:
                    monomial *= value ** exponent
            total += coeff * monomial * (t**m) * np.exp(-rho * t)
        return total

    def parameter_derivative(self, parameter_index: int) -> "ParametricAnalyticSeries":
        if not 0 <= parameter_index < self.n_parameters:
            raise ValueError("parameter_index out of range")
        terms: Dict[ParametricTermKey, complex] = {}
        for (m, decay, params), coeff in self.terms.items():
            exponent = params[parameter_index]
            if exponent == 0:
                continue
            new_params = list(params)
            new_params[parameter_index] -= 1
            key = (m, decay, tuple(new_params))
            terms[key] = terms.get(key, 0.0) + exponent * coeff
        return ParametricAnalyticSeries(self.n_modes, self.n_parameters, terms)


def solve_parametric_response_series(
    source: ParametricAnalyticSeries,
    target_mode: int,
    lambdas: np.ndarray,
) -> ParametricAnalyticSeries:
    """Exact canonical response compilation, preserving static parameter monomials."""

    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.shape != (source.n_modes,):
        raise ValueError("lambdas shape mismatch")
    if not 0 <= target_mode < source.n_modes:
        raise ValueError("target_mode out of range")

    lam = float(lambdas[target_mode])
    target_decay = _unit_signature(source.n_modes, target_mode)
    out: Dict[ParametricTermKey, complex] = {}

    for (m, decay, params), coeff0 in source.terms.items():
        rho = float(np.dot(decay, lambdas))
        delta = lam - rho
        if delta == 0.0:
            key = (m + 1, target_decay, params)
            out[key] = out.get(key, 0.0) + coeff0 / (m + 1)
            continue

        m_fact = factorial(m)
        for k in range(m + 1):
            power = m - k
            coeff = coeff0 * ((-1) ** k) * m_fact / factorial(power) / (delta ** (k + 1))
            key = (power, decay, params)
            out[key] = out.get(key, 0.0) + coeff

        target_coeff = coeff0 * ((-1) ** (m + 1)) * m_fact / (delta ** (m + 1))
        key = (0, target_decay, params)
        out[key] = out.get(key, 0.0) + target_coeff

    return ParametricAnalyticSeries(source.n_modes, source.n_parameters, out)


@dataclass(frozen=True)
class ParametricResponseNode:
    name: str
    target_mode: int
    parents: Tuple[str, ...]
    weight: complex


@dataclass(frozen=True)
class CompiledParametricAnalyticGraph:
    lambdas: np.ndarray
    initial_names: tuple[str, ...]
    operating_names: tuple[str, ...]
    node_series: Mapping[str, ParametricAnalyticSeries]
    mode_series: tuple[ParametricAnalyticSeries, ...]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.initial_names + self.operating_names

    def parameter_vector(self, a0: np.ndarray, operating: np.ndarray) -> np.ndarray:
        initial = np.asarray(a0, dtype=float)
        u = np.asarray(operating, dtype=float)
        if initial.shape != (len(self.initial_names),):
            raise ValueError("initial coordinate dimension mismatch")
        if u.shape != (len(self.operating_names),):
            raise ValueError("operating parameter dimension mismatch")
        return np.concatenate([initial, u])

    def evaluate(
        self,
        t: float,
        *,
        a0: np.ndarray,
        operating: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.parameter_vector(a0, operating)
        a = np.array(
            [np.real(series.evaluate(t, self.lambdas, p)) for series in self.mode_series],
            dtype=float,
        )
        da = np.array(
            [
                np.real(series.derivative(self.lambdas).evaluate(t, self.lambdas, p))
                for series in self.mode_series
            ],
            dtype=float,
        )
        return a, da

    def operating_jacobian(
        self,
        t: float,
        *,
        a0: np.ndarray,
        operating: np.ndarray,
    ) -> np.ndarray:
        p = self.parameter_vector(a0, operating)
        offset = len(self.initial_names)
        J = np.zeros((len(self.mode_series), len(self.operating_names)), dtype=float)
        for i, series in enumerate(self.mode_series):
            for j in range(len(self.operating_names)):
                derivative = series.parameter_derivative(offset + j)
                J[i, j] = np.real(derivative.evaluate(t, self.lambdas, p))
        return J


class ParametricAnalyticEvolutionGraph:
    """One analytic neural graph for arbitrary a0 and declared static U.

    Initial coordinates and operating parameters are explicit zero-dynamics
    neural nodes. Response sources are products of previous dynamic nodes and/or
    static parameter nodes. Consequently the complete forward map is analytic in
    `(a0, U, t)` without a separate conditioning network that predicts weights.

    The thermal spectrum is fixed in this class. Geometry changes that alter the
    spatial operators/eigenvalues remain a separately certified parameterization
    problem rather than being hidden inside an unverified encoder.
    """

    def __init__(self, lambdas: Sequence[float], operating_names: Sequence[str]):
        self.lambdas = np.asarray(lambdas, dtype=float)
        if self.lambdas.ndim != 1 or np.any(self.lambdas <= 0):
            raise ValueError("lambdas must be positive and one-dimensional")
        self.n_modes = self.lambdas.size
        self.initial_names = tuple(f"a0_{i}" for i in range(self.n_modes))
        self.operating_names = tuple(str(name) for name in operating_names)
        if len(set(self.operating_names)) != len(self.operating_names):
            raise ValueError("operating parameter names must be unique")
        if set(self.initial_names) & set(self.operating_names):
            raise ValueError("operating names conflict with initial-state nodes")
        self.response_nodes: list[ParametricResponseNode] = []
        self._compiled: CompiledParametricAnalyticGraph | None = None

    @property
    def n_parameters(self) -> int:
        return self.n_modes + len(self.operating_names)

    def known_names(self) -> tuple[str, ...]:
        response = tuple(node.name for node in self.response_nodes)
        return self.initial_names + self.operating_names + response

    def add_product_response(
        self,
        name: str,
        target_mode: int,
        parents: Sequence[str],
        weight: complex,
    ) -> None:
        if name in self.known_names():
            raise ValueError(f"duplicate node name: {name}")
        if not 0 <= target_mode < self.n_modes:
            raise ValueError("target_mode out of range")
        # Empty product is the constant source 1 (needed for fixed excitation
        # and prescribed thermal forcing, including the zero-parameter case).
        known = set(self.known_names())
        missing = [parent for parent in parents if parent not in known]
        if missing:
            raise ValueError(f"unknown parent nodes: {missing}")
        self.response_nodes.append(
            ParametricResponseNode(name, target_mode, tuple(parents), complex(weight))
        )
        self._compiled = None

    def compile(self) -> CompiledParametricAnalyticGraph:
        if self._compiled is not None:
            return self._compiled

        n_parameters = self.n_parameters
        nodes: Dict[str, ParametricAnalyticSeries] = {}
        modes = [ParametricAnalyticSeries.zero(self.n_modes, n_parameters) for _ in range(self.n_modes)]

        for i, name in enumerate(self.initial_names):
            series = ParametricAnalyticSeries.initial_decay(
                self.n_modes, n_parameters, i, i
            )
            nodes[name] = series
            modes[i] = modes[i] + series

        for j, name in enumerate(self.operating_names):
            nodes[name] = ParametricAnalyticSeries.static_parameter(
                self.n_modes,
                n_parameters,
                self.n_modes + j,
            )

        for node in self.response_nodes:
            source = ParametricAnalyticSeries.constant(
                self.n_modes, n_parameters, node.weight
            )
            for parent in node.parents:
                source = source * nodes[parent]
            response = solve_parametric_response_series(
                source, node.target_mode, self.lambdas
            )
            nodes[node.name] = response
            modes[node.target_mode] = modes[node.target_mode] + response

        self._compiled = CompiledParametricAnalyticGraph(
            lambdas=self.lambdas.copy(),
            initial_names=self.initial_names,
            operating_names=self.operating_names,
            node_series=dict(nodes),
            mode_series=tuple(modes),
        )
        return self._compiled

    def evaluate(
        self,
        t: float,
        *,
        a0: np.ndarray,
        operating: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.compile().evaluate(t, a0=a0, operating=operating)
