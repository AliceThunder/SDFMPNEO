"""Unified geometry-independent residual-corrected neural electrothermal model."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .electrothermal_tensor.integrators import (
    GeneralizedThermalSpectrum,
    integrate_etd2,
    integrate_etd2_adaptive,
    integrate_imex_euler,
    integrate_reference,
)
from .electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .electrothermal_tensor.vector_field import ReducedThermalOperator
from .unified_background import FixedMultiscaleBackground
from .unified_geometry import UnifiedUWPTGeometry
from .unified_maxwell import NeuralMaxwellAccelerator

ARCHITECTURE = "unified-residual-corrected-neural-electrothermal-solver"
FORMAT_VERSION = 2


@dataclass(frozen=True)
class UnifiedPrediction:
    time: float
    state: np.ndarray
    derivative: np.ndarray
    heat_source: np.ndarray
    maximum_temperature: float
    impedance: np.ndarray
    maxwell_initial_residual: tuple
    maxwell_final_residual: tuple
    maxwell_correction_iterations: tuple
    steps: int
    rejected_steps: int


@dataclass(frozen=True)
class UnifiedSteadyState:
    state: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    maximum_temperature: float
    impedance: np.ndarray
    maxwell_final_residual: tuple


class _ThermalFamily:
    geometry_dimension = 0

    def __init__(self, context):
        if not context.has_thermal_operators:
            raise ValueError("thermal integration requires a thermal background context")
        self._operator = ReducedThermalOperator(
            context.thermal_mass_reduced,
            context.thermal_stiffness_reduced,
        )

    def operator(self, _):
        return self._operator


class _Field:
    def __init__(self, model, context, operating):
        self.model = model
        self.context = context
        self.operating = np.asarray(operating, float)
        self.thermal_operators = _ThermalFamily(context)

    def heat_source(self, state, _geometry, _operating):
        return self.model.heat_source(state, self.context, self.operating)[0]

    def vector_field(self, state, _geometry, _operating):
        op = self.thermal_operators.operator(None)
        a = np.asarray(state, float)
        return np.linalg.solve(op.mass, -op.stiffness @ a + self.heat_source(a, None, None))


class UnifiedNeuralElectroThermalModel:
    def __init__(
        self,
        background,
        accelerator,
        *,
        default_geometry,
        current_offset=None,
        current_matrix=None,
        context_cache_size=16,
    ):
        if background.thermal_basis is None or background.thermal_rank < 1:
            raise ValueError("unified model requires an automatically constructed thermal basis")
        self.background = background
        self.accelerator = accelerator
        self.default_geometry = dict(default_geometry)
        n_ports = len(background.coil_materials)
        self.current_offset = (
            np.zeros(n_ports, complex)
            if current_offset is None
            else np.asarray(current_offset, complex).reshape(-1)
        )
        self.current_matrix = (
            np.eye(n_ports, dtype=complex)
            if current_matrix is None
            else np.asarray(current_matrix, complex)
        )
        if (
            self.current_offset.shape != (n_ports,)
            or self.current_matrix.ndim != 2
            or self.current_matrix.shape[0] != n_ports
        ):
            raise ValueError("current affine map does not match port count")
        self.context_cache_size = max(1, int(context_cache_size))
        self._contexts = OrderedDict()
        self._em_cache = OrderedDict()

    @property
    def thermal_rank(self):
        return self.background.thermal_rank

    @property
    def current_dimension(self):
        return self.current_matrix.shape[1]

    def geometry_context(self, geometry=None):
        mapping = self.default_geometry if geometry is None else geometry
        g = mapping if isinstance(mapping, UnifiedUWPTGeometry) else UnifiedUWPTGeometry.from_mapping(mapping)
        key = g.canonical_json()
        if key in self._contexts:
            self._contexts.move_to_end(key)
            return self._contexts[key]
        context = self.background.geometry_context(g, assemble_thermal=True)
        self._contexts[key] = context
        if len(self._contexts) > self.context_cache_size:
            self._contexts.popitem(last=False)
        return context

    def _currents(self, operating):
        u = np.asarray(operating, complex).reshape(-1)
        if u.shape != (self.current_dimension,) or np.any(~np.isfinite(u)):
            raise ValueError("invalid operating current vector")
        return self.current_offset + self.current_matrix @ u

    def _em_solution(self, state, context):
        a = np.asarray(state, float).reshape(-1)
        if a.shape != (self.thermal_rank,) or np.any(~np.isfinite(a)):
            raise ValueError("invalid thermal state")
        key = (context.geometry.canonical_json(), a.tobytes())
        if key in self._em_cache:
            self._em_cache.move_to_end(key)
            return self._em_cache[key]
        A = self.background.em_operator(context, a)
        B = self.background.rhs_matrix(context)
        result = self.accelerator.solve(A, B)
        self._em_cache[key] = result
        if len(self._em_cache) > 8:
            self._em_cache.popitem(last=False)
        return result

    def joule_matrices(self, state, context):
        X, report = self._em_solution(state, context)
        ex, ey, ez, weight = self.background.material_joule_cells(context, state, X)
        Phi = self.background.thermal_basis
        n_ports = X.shape[1]
        G = np.zeros((self.thermal_rank, n_ports, n_ports), float)
        for j in range(self.thermal_rank):
            weighted = weight * Phi[:, j]
            G[j] = np.real(
                ex.conj().T @ (weighted[:, None] * ex)
                + ey.conj().T @ (weighted[:, None] * ey)
                + ez.conj().T @ (weighted[:, None] * ez)
            )
            G[j] = 0.5 * (G[j] + G[j].T)
        resistances = self.background.wire_resistances(context, state)
        for port, (resistance, line_weights) in enumerate(zip(resistances, context.line_heat_weights)):
            projection = Phi.T @ line_weights
            G[:, port, port] += 0.5 * resistance * projection
        return G, X, report, resistances

    def heat_source(self, state, context, operating):
        currents = self._currents(operating)
        G, X, report, resistances = self.joule_matrices(state, context)
        q = np.array([np.real(np.vdot(currents, matrix @ currents)) for matrix in G], float)
        return q, (G, X, report, resistances)

    def temperature_field(self, state):
        a = np.asarray(state, float).reshape(-1)
        if a.shape != (self.thermal_rank,):
            raise ValueError("thermal state dimension mismatch")
        return self.background.ambient_temperature + self.background.thermal_basis @ a

    def maximum_temperature(self, state):
        return float(max(self.background.ambient_temperature, np.max(self.temperature_field(state))))

    def impedance(self, state, context, X=None, resistances=None):
        if X is None or resistances is None:
            _, X, _, resistances = self.joule_matrices(state, context)
        return context.source_shape.T @ X + np.diag(resistances)

    def evaluate(self, state, geometry, operating):
        context = self.geometry_context(geometry)
        q, (G, X, report, resistances) = self.heat_source(state, context, operating)
        operator = ReducedThermalOperator(
            context.thermal_mass_reduced,
            context.thermal_stiffness_reduced,
        )
        a = np.asarray(state, float).reshape(-1)
        derivative = np.linalg.solve(operator.mass, -operator.stiffness @ a + q)
        return {
            "heat_source": q,
            "derivative": derivative,
            "temperature": self.temperature_field(a),
            "maximum_temperature": self.maximum_temperature(a),
            "impedance": self.impedance(a, context, X, resistances),
            "maxwell": report,
            "joule_matrices": G,
        }

    def predict(
        self,
        time,
        *,
        initial_state,
        geometry,
        operating,
        max_step=100.0,
        method="etd2_adaptive",
        rtol=1e-5,
        atol=1e-8,
        initial_step=None,
    ):
        t = float(time)
        a0 = np.asarray(initial_state, float).reshape(-1)
        if a0.shape != (self.thermal_rank,) or np.any(~np.isfinite(a0)) or not np.isfinite(t) or t < 0:
            raise ValueError("invalid prediction state/time")
        context = self.geometry_context(geometry)
        field = _Field(self, context, operating)
        empty = np.empty(0)
        spectrum = (
            GeneralizedThermalSpectrum(field.thermal_operators.operator(empty))
            if t > 0 and method in {"etd2", "etd2_adaptive", "adaptive_etd2"}
            else None
        )
        if method == "etd2":
            result = integrate_etd2(
                field, t, initial_state=a0, geometry=empty, operating=np.empty(0),
                max_step=max_step, spectrum=spectrum,
            )
        elif method in {"etd2_adaptive", "adaptive_etd2"}:
            result = integrate_etd2_adaptive(
                field, t, initial_state=a0, geometry=empty, operating=np.empty(0),
                max_step=max_step, rtol=rtol, atol=atol, initial_step=initial_step,
                spectrum=spectrum,
            )
        elif method == "imex":
            result = integrate_imex_euler(
                field, t, initial_state=a0, geometry=empty, operating=np.empty(0), max_step=max_step,
            )
        elif method == "reference":
            result = integrate_reference(
                field, t, initial_state=a0, geometry=empty, operating=np.empty(0), rtol=rtol, atol=atol,
            )
        else:
            raise ValueError("method must be etd2, etd2_adaptive, imex, or reference")
        evaluated = self.evaluate(result.state, context.geometry, operating)
        report = evaluated["maxwell"]
        return UnifiedPrediction(
            t,
            result.state,
            evaluated["derivative"],
            evaluated["heat_source"],
            evaluated["maximum_temperature"],
            evaluated["impedance"],
            report.initial_relative_residual,
            report.final_relative_residual,
            report.correction_iterations,
            result.steps,
            result.rejected_steps,
        )

    def steady_state(self, *, initial_guess, geometry, operating, tolerance=1e-10, max_iterations=40):
        state = np.asarray(initial_guess, float).reshape(-1).copy()
        if state.shape != (self.thermal_rank,) or np.any(~np.isfinite(state)):
            raise ValueError("invalid steady-state initial guess")
        context = self.geometry_context(geometry)
        operator = ReducedThermalOperator(
            context.thermal_mass_reduced,
            context.thermal_stiffness_reduced,
        )

        def residual(a):
            return np.linalg.solve(
                operator.mass,
                -operator.stiffness @ a + self.heat_source(a, context, operating)[0],
            )

        last_iteration = 0
        for iteration in range(int(max_iterations) + 1):
            last_iteration = iteration
            value = residual(state)
            norm = float(np.linalg.norm(value))
            if norm <= tolerance or iteration == max_iterations:
                break
            jacobian = np.empty((self.thermal_rank, self.thermal_rank))
            epsilon = np.sqrt(np.finfo(float).eps) * (1 + np.abs(state))
            for k in range(self.thermal_rank):
                trial = state.copy()
                trial[k] += epsilon[k]
                jacobian[:, k] = (residual(trial) - value) / epsilon[k]
            try:
                step = np.linalg.solve(jacobian, -value)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jacobian, -value, rcond=None)[0]
            factor = 1.0
            accepted = False
            for _ in range(12):
                trial = state + factor * step
                if np.linalg.norm(residual(trial)) < norm:
                    state = trial
                    accepted = True
                    break
                factor *= 0.5
            if not accepted:
                break
        final_residual = residual(state)
        evaluated = self.evaluate(state, context.geometry, operating)
        return UnifiedSteadyState(
            state,
            float(np.linalg.norm(final_residual)),
            last_iteration,
            float(np.linalg.norm(final_residual)) <= tolerance,
            evaluated["maximum_temperature"],
            evaluated["impedance"],
            evaluated["maxwell"].final_relative_residual,
        )

    def save(self, path, *, metadata=None):
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        network = self.accelerator.network
        first_parameter = next(network.parameters())
        meta = {
            "format_version": FORMAT_VERSION,
            "architecture": ARCHITECTURE,
            "network_config": network.config.to_dict(),
            "network_dtype": "float64" if first_parameter.dtype == torch.float64 else "float32",
            "frequency_hz": self.background.frequency_hz,
            "ambient_temperature": self.background.ambient_temperature,
            "materials": self.background.materials,
            "coil_materials": self.background.coil_materials,
            "package_materials": self.background.package_materials,
            "seawater_material": self.background.seawater_material,
            "thermal_rank": self.thermal_rank,
            "default_geometry": self.default_geometry,
            "residual_tolerance": self.accelerator.residual_tolerance,
            "maxwell_max_iterations": self.accelerator.max_iterations,
            "metadata": dict(metadata or {}),
        }
        arrays = {
            "metadata_json": np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
            "background_x": self.background.x,
            "background_y": self.background.y,
            "background_z": self.background.z,
            "thermal_basis": self.background.thermal_basis,
            "em_basis": self.accelerator.basis,
            "current_offset": self.current_offset,
            "current_matrix": self.current_matrix,
        }
        for key, value in network.state_dict().items():
            arrays["network__" + key.replace(".", "__DOT__")] = value.detach().cpu().numpy()
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path, *, device="cpu"):
        import torch

        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            if meta.get("architecture") != ARCHITECTURE or int(meta.get("format_version", -1)) != FORMAT_VERSION:
                raise ValueError("model is not the current unified residual-corrected architecture")
            saved_thermal_basis = np.asarray(data["thermal_basis"], float)
            if saved_thermal_basis.ndim != 2 or saved_thermal_basis.shape[1] < 1 or np.any(~np.isfinite(saved_thermal_basis)):
                raise ValueError("saved thermal basis is invalid")
            background = FixedMultiscaleBackground(
                data["background_x"], data["background_y"], data["background_z"],
                frequency_hz=meta["frequency_hz"], materials=meta["materials"],
                coil_materials=meta["coil_materials"], package_materials=meta["package_materials"],
                seawater_material=meta["seawater_material"], thermal_basis=saved_thermal_basis,
                ambient_temperature=meta["ambient_temperature"],
            )
            if background.thermal_rank != int(meta["thermal_rank"]):
                raise ValueError("saved thermal basis rank does not match model metadata")

            config = ResidualMLPConfig(**meta["network_config"])
            normalizer = FeatureNormalizer(data["network__input_mean"], data["network__input_scale"])
            network = build_residual_mlp(config, normalizer)
            dtype = torch.float64 if meta.get("network_dtype") == "float64" else torch.float32
            network = network.to(device=device, dtype=dtype)
            state = {}
            for key in data.files:
                if key.startswith("network__"):
                    state[key[len("network__"):].replace("__DOT__", ".")] = torch.as_tensor(
                        data[key], dtype=dtype, device=device,
                    )
            network.load_state_dict(state, strict=True)
            network.eval()
            accelerator = NeuralMaxwellAccelerator(
                network,
                data["em_basis"],
                residual_tolerance=meta["residual_tolerance"],
                max_iterations=meta["maxwell_max_iterations"],
            )
            return cls(
                background,
                accelerator,
                default_geometry=meta["default_geometry"],
                current_offset=data["current_offset"],
                current_matrix=data["current_matrix"],
            )


__all__ = [
    "ARCHITECTURE",
    "UnifiedNeuralElectroThermalModel",
    "UnifiedPrediction",
    "UnifiedSteadyState",
]
