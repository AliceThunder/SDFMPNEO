"""Unified online electrothermal model driven by geometry-only EM tensors.

Online prediction never solves Maxwell. The neural model is evaluated once per
geometry to obtain ``Z_field``, ``D_vol`` and thermal modal Joule matrices;
current magnitude/phase, wire resistance and thermal dynamics then remain
explicit physics.
"""
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
from .unified_tensor_surrogate import UnifiedTensorSurrogate

ARCHITECTURE = "unified-geometry-tensor-electrothermal-rom"
FORMAT_VERSION = 4


@dataclass(frozen=True)
class UnifiedPrediction:
    time: float
    state: np.ndarray
    derivative: np.ndarray
    heat_source: np.ndarray
    maximum_temperature: float
    impedance: np.ndarray
    currents: np.ndarray
    volume_power: float
    wire_power: float
    outward_power: float
    tensor_projection_correction: float
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
    currents: np.ndarray
    volume_power: float
    wire_power: float
    outward_power: float
    tensor_projection_correction: float


class _ThermalFamily:
    geometry_dimension = 0

    def __init__(self, context):
        if not context.has_thermal_operators:
            raise ValueError("thermal integration requires reduced thermal operators")
        self._operator = ReducedThermalOperator(
            context.thermal_mass_reduced,
            context.thermal_stiffness_reduced,
        )

    def operator(self, _):
        return self._operator


class _Field:
    def __init__(self, model, context, tensors, operating):
        self.model = model
        self.context = context
        self.tensors = tensors
        self.operating = operating
        self.thermal_operators = _ThermalFamily(context)

    def heat_source(self, state, _geometry, _operating):
        return self.model._heat_source_with_tensors(
            state, self.context, self.tensors, self.operating
        )[0]

    def vector_field(self, state, _geometry, _operating):
        op = self.thermal_operators.operator(None)
        a = np.asarray(state, float)
        return np.linalg.solve(op.mass, -op.stiffness @ a + self.heat_source(a, None, None))


class UnifiedNeuralElectroThermalModel:
    """Shared thermal ROM plus static geometry-to-EM-tensor surrogate."""

    def __init__(
        self,
        background,
        surrogate,
        *,
        default_geometry,
        current_offset=None,
        current_matrix=None,
        context_cache_size=16,
    ):
        if background.thermal_basis is None or background.thermal_rank < 1:
            raise ValueError("unified model requires a constructed thermal basis")
        if surrogate.thermal_rank != background.thermal_rank:
            raise ValueError("tensor surrogate and thermal basis ranks differ")
        n_ports = len(background.coil_materials)
        if surrogate.n_ports != n_ports:
            raise ValueError("tensor surrogate port count differs from background")
        self.background = background
        self.surrogate = surrogate
        self.default_geometry = dict(default_geometry)
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
        self._tensor_cache = OrderedDict()

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

    def tensors(self, geometry=None):
        context = self.geometry_context(geometry)
        key = context.geometry.canonical_json()
        if key in self._tensor_cache:
            self._tensor_cache.move_to_end(key)
            return self._tensor_cache[key]
        value = self.surrogate.predict(context.geometry)
        self._tensor_cache[key] = value
        if len(self._tensor_cache) > self.context_cache_size:
            self._tensor_cache.popitem(last=False)
        return value

    def temperature_field(self, state):
        a = np.asarray(state, float).reshape(-1)
        if a.shape != (self.thermal_rank,) or np.any(~np.isfinite(a)):
            raise ValueError("thermal state dimension mismatch")
        return self.background.ambient_temperature + self.background.thermal_basis @ a

    def maximum_temperature(self, state):
        return float(max(self.background.ambient_temperature, np.max(self.temperature_field(state))))

    def _current_drive(self, operating):
        u = np.asarray(operating, complex).reshape(-1)
        if u.shape != (self.current_dimension,) or np.any(~np.isfinite(u)):
            raise ValueError("invalid operating current vector")
        return self.current_offset + self.current_matrix @ u

    @staticmethod
    def _series_impedance(value, n_ports):
        if value is None:
            return np.zeros((n_ports, n_ports), complex)
        z = np.asarray(value, complex)
        if z.ndim == 0:
            return np.eye(n_ports, dtype=complex) * z
        if z.ndim == 1:
            if z.shape != (n_ports,):
                raise ValueError("series impedance vector has wrong size")
            return np.diag(z)
        if z.shape != (n_ports, n_ports):
            raise ValueError("series impedance matrix has wrong size")
        return z

    def _currents(self, state, context, tensors, operating):
        """Resolve current-driven or voltage-driven operating conditions."""
        if isinstance(operating, dict):
            if "voltage" not in operating:
                raise ValueError("voltage-driven operating mapping requires 'voltage'")
            voltage = np.asarray(operating["voltage"], complex).reshape(-1)
            if voltage.shape != (tensors.z_field.shape[0],):
                raise ValueError("drive voltage has wrong port dimension")
            resistance = self.background.wire_resistances(context, state)
            total = tensors.z_field + np.diag(resistance)
            total = total + self._series_impedance(
                operating.get("series_impedance"), len(resistance)
            )
            try:
                return np.linalg.solve(total, voltage)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(total, voltage, rcond=None)[0]
        return self._current_drive(operating)

    def impedance(self, state, context, tensors=None):
        tensors = self.tensors(context.geometry) if tensors is None else tensors
        resistance = self.background.wire_resistances(context, state)
        return tensors.z_field + np.diag(resistance)

    def _wire_modal_heat(self, state, context, currents):
        resistance = self.background.wire_resistances(context, state)
        phi = self.background.thermal_basis
        reduced = np.zeros(self.thermal_rank, float)
        power = 0.0
        for p, (r, weights) in enumerate(zip(resistance, context.line_heat_weights)):
            local_power = 0.5 * float(r) * float(abs(currents[p]) ** 2)
            power += local_power
            reduced += local_power * (phi.T @ np.asarray(weights, float))
        return reduced, np.asarray(resistance, float), float(power)

    def _heat_source_with_tensors(self, state, context, tensors, operating):
        a = np.asarray(state, float).reshape(-1)
        if a.shape != (self.thermal_rank,) or np.any(~np.isfinite(a)):
            raise ValueError("invalid thermal state")
        currents = self._currents(a, context, tensors, operating)
        volume_modal = tensors.modal_heat(currents)
        wire_modal, resistance, wire_power = self._wire_modal_heat(a, context, currents)
        q = np.asarray(volume_modal + wire_modal, float)
        return (
            q,
            currents,
            resistance,
            tensors.volume_power(currents),
            wire_power,
            tensors.implied_outward_power(currents),
        )

    def heat_source(self, state, context, operating):
        tensors = self.tensors(context.geometry)
        return self._heat_source_with_tensors(state, context, tensors, operating)

    def evaluate(self, state, geometry, operating):
        context = self.geometry_context(geometry)
        tensors = self.tensors(context.geometry)
        q, currents, resistance, volume_power, wire_power, outward_power = (
            self._heat_source_with_tensors(state, context, tensors, operating)
        )
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
            "impedance": tensors.z_field + np.diag(resistance),
            "currents": currents,
            "volume_power": float(volume_power),
            "wire_power": float(wire_power),
            "outward_power": float(outward_power),
            "tensor_projection_correction": float(tensors.projection_correction),
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
        tensors = self.tensors(context.geometry)
        field = _Field(self, context, tensors, operating)
        empty = np.empty(0)
        spectrum = (
            GeneralizedThermalSpectrum(field.thermal_operators.operator(empty))
            if t > 0 and method in {"etd2", "etd2_adaptive", "adaptive_etd2"}
            else None
        )
        if method == "etd2":
            result = integrate_etd2(
                field, t, initial_state=a0, geometry=empty, operating=empty,
                max_step=max_step, spectrum=spectrum,
            )
        elif method in {"etd2_adaptive", "adaptive_etd2"}:
            result = integrate_etd2_adaptive(
                field, t, initial_state=a0, geometry=empty, operating=empty,
                max_step=max_step, rtol=rtol, atol=atol, initial_step=initial_step,
                spectrum=spectrum,
            )
        elif method == "imex":
            result = integrate_imex_euler(
                field, t, initial_state=a0, geometry=empty, operating=empty, max_step=max_step,
            )
        elif method == "reference":
            result = integrate_reference(
                field, t, initial_state=a0, geometry=empty, operating=empty, rtol=rtol, atol=atol,
            )
        else:
            raise ValueError("method must be etd2, etd2_adaptive, imex, or reference")
        evaluated = self.evaluate(result.state, context.geometry, operating)
        return UnifiedPrediction(
            time=t,
            state=result.state,
            derivative=evaluated["derivative"],
            heat_source=evaluated["heat_source"],
            maximum_temperature=evaluated["maximum_temperature"],
            impedance=evaluated["impedance"],
            currents=evaluated["currents"],
            volume_power=evaluated["volume_power"],
            wire_power=evaluated["wire_power"],
            outward_power=evaluated["outward_power"],
            tensor_projection_correction=evaluated["tensor_projection_correction"],
            steps=result.steps,
            rejected_steps=result.rejected_steps,
        )

    def steady_state(self, *, initial_guess, geometry, operating, tolerance=1e-10, max_iterations=40):
        state = np.asarray(initial_guess, float).reshape(-1).copy()
        if state.shape != (self.thermal_rank,) or np.any(~np.isfinite(state)):
            raise ValueError("invalid steady-state initial guess")
        context = self.geometry_context(geometry)
        tensors = self.tensors(context.geometry)
        operator = ReducedThermalOperator(
            context.thermal_mass_reduced,
            context.thermal_stiffness_reduced,
        )

        def residual(a):
            q = self._heat_source_with_tensors(a, context, tensors, operating)[0]
            return np.linalg.solve(operator.mass, -operator.stiffness @ a + q)

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
            state=state,
            residual_norm=float(np.linalg.norm(final_residual)),
            iterations=last_iteration,
            converged=float(np.linalg.norm(final_residual)) <= tolerance,
            maximum_temperature=evaluated["maximum_temperature"],
            impedance=evaluated["impedance"],
            currents=evaluated["currents"],
            volume_power=evaluated["volume_power"],
            wire_power=evaluated["wire_power"],
            outward_power=evaluated["outward_power"],
            tensor_projection_correction=evaluated["tensor_projection_correction"],
        )

    def save(self, path, *, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.surrogate.checkpoint()
        meta = {
            "format_version": FORMAT_VERSION,
            "architecture": ARCHITECTURE,
            "frequency_hz": self.background.frequency_hz,
            "ambient_temperature": self.background.ambient_temperature,
            "materials": self.background.materials,
            "coil_materials": self.background.coil_materials,
            "package_materials": self.background.package_materials,
            "seawater_material": self.background.seawater_material,
            "thermal_rank": self.thermal_rank,
            "default_geometry": self.default_geometry,
            "network_config": checkpoint["network_config"],
            "network_dtype": checkpoint["dtype"],
            "n_ports": checkpoint["n_ports"],
            "metadata": dict(metadata or {}),
        }
        arrays = {
            "metadata_json": np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
            "background_x": self.background.x,
            "background_y": self.background.y,
            "background_z": self.background.z,
            "thermal_basis": self.background.thermal_basis,
            "current_offset": self.current_offset,
            "current_matrix": self.current_matrix,
            "input_mean": checkpoint["input_mean"],
            "input_scale": checkpoint["input_scale"],
            "output_mean": checkpoint["output_mean"],
            "output_scale": checkpoint["output_scale"],
            "phi_min": checkpoint["phi_min"],
            "phi_max": checkpoint["phi_max"],
        }
        for key, value in checkpoint["network_state"].items():
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
                raise ValueError("model is not the current geometry-tensor electrothermal architecture")
            thermal_basis = np.asarray(data["thermal_basis"], float)
            background = FixedMultiscaleBackground(
                data["background_x"], data["background_y"], data["background_z"],
                frequency_hz=meta["frequency_hz"], materials=meta["materials"],
                coil_materials=meta["coil_materials"], package_materials=meta["package_materials"],
                seawater_material=meta["seawater_material"], thermal_basis=thermal_basis,
                ambient_temperature=meta["ambient_temperature"],
            )
            config = ResidualMLPConfig(**dict(meta["network_config"]))
            normalizer = FeatureNormalizer(
                np.asarray(data["input_mean"], float),
                np.asarray(data["input_scale"], float),
            )
            network = build_residual_mlp(config, normalizer)
            dtype = torch.float32 if meta.get("network_dtype") == "float32" else torch.float64
            network = network.to(device=device, dtype=dtype)
            state = {}
            for name in data.files:
                if name.startswith("network__"):
                    key = name[len("network__"):].replace("__DOT__", ".")
                    state[key] = torch.as_tensor(data[name], dtype=dtype, device=device)
            network.load_state_dict(state)
            network.eval()
            surrogate = UnifiedTensorSurrogate(
                network,
                np.asarray(data["output_mean"], float),
                np.asarray(data["output_scale"], float),
                int(meta["n_ports"]),
                np.asarray(data["phi_min"], float),
                np.asarray(data["phi_max"], float),
            )
            return cls(
                background,
                surrogate,
                default_geometry=meta["default_geometry"],
                current_offset=np.asarray(data["current_offset"], complex),
                current_matrix=np.asarray(data["current_matrix"], complex),
            )


__all__ = [
    "ARCHITECTURE", "FORMAT_VERSION", "UnifiedNeuralElectroThermalModel",
    "UnifiedPrediction", "UnifiedSteadyState",
]
