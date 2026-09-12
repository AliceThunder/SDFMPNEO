"""Deployable structure-preserving neural electrothermal ROM facade."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .integrators import integrate_etd2, integrate_imex_euler, integrate_reference
from .network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .pod import TensorPOD
from .surrogate import NeuralTensorSurrogate
from .vector_field import FixedThermalOperatorFamily, NeuralElectroThermalVectorField

_MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class NeuralROMPrediction:
    time: float
    state: np.ndarray
    derivative: np.ndarray
    heat_source: np.ndarray
    steps: int
    step_sizes: tuple[float, ...]


@dataclass(frozen=True)
class NeuralROMSteadyState:
    state: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool


class StructurePreservingNeuralElectroThermalROM:
    def __init__(
        self,
        surrogate: NeuralTensorSurrogate,
        thermal_operators,
        *,
        physical_signature: str | None = None,
        training_domain: dict | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.thermal_operators = thermal_operators
        self.field = NeuralElectroThermalVectorField(surrogate, thermal_operators)
        self.physical_signature = None if physical_signature is None else str(physical_signature)
        self.training_domain = {} if training_domain is None else {
            key: np.asarray(value, dtype=float) for key, value in training_domain.items()
        }

    def _check_domain(self, state, geometry, operating, *, allow_extrapolation: bool):
        a = np.asarray(state, dtype=float).reshape(-1)
        g = np.asarray(geometry, dtype=float).reshape(-1)
        u = np.asarray(operating, dtype=float).reshape(-1)
        if allow_extrapolation or not self.training_domain:
            return a, g, u
        for name, value in (("state", a), ("geometry", g), ("operating", u)):
            lower = self.training_domain.get(name + "_lower")
            upper = self.training_domain.get(name + "_upper")
            if lower is None or upper is None:
                continue
            if value.shape != lower.shape or upper.shape != lower.shape:
                raise ValueError(f"saved {name} training domain is incompatible")
            if np.any(value < lower) or np.any(value > upper):
                raise ValueError(f"{name} is outside the trained domain")
        return a, g, u

    def predict(
        self,
        time: float,
        *,
        initial_state: np.ndarray,
        geometry: np.ndarray,
        operating: np.ndarray,
        max_step: float,
        method: str = "etd2",
        allow_extrapolation: bool = False,
    ) -> NeuralROMPrediction:
        a0, g, u = self._check_domain(
            initial_state, geometry, operating, allow_extrapolation=allow_extrapolation
        )
        if method == "etd2":
            result = integrate_etd2(
                self.field, time, initial_state=a0, geometry=g, operating=u, max_step=max_step
            )
        elif method == "imex":
            result = integrate_imex_euler(
                self.field, time, initial_state=a0, geometry=g, operating=u, max_step=max_step
            )
        elif method == "reference":
            result = integrate_reference(
                self.field, time, initial_state=a0, geometry=g, operating=u
            )
        else:
            raise ValueError("method must be 'etd2', 'imex' or 'reference'")
        derivative = self.field.vector_field(result.state, g, u)
        heat = self.field.heat_source(result.state, g, u)
        return NeuralROMPrediction(
            time=float(time),
            state=result.state,
            derivative=derivative,
            heat_source=heat,
            steps=result.steps,
            step_sizes=result.step_sizes,
        )

    def steady_state(
        self,
        *,
        initial_guess: np.ndarray,
        geometry: np.ndarray,
        operating: np.ndarray,
        tolerance: float = 1e-10,
        max_iterations: int = 40,
        allow_extrapolation: bool = False,
    ) -> NeuralROMSteadyState:
        state, g, u = self._check_domain(
            initial_guess, geometry, operating, allow_extrapolation=allow_extrapolation
        )
        state = state.copy()
        for iteration in range(int(max_iterations) + 1):
            residual = self.field.vector_field(state, g, u)
            norm = float(np.linalg.norm(residual))
            if norm <= float(tolerance):
                return NeuralROMSteadyState(state, norm, iteration, True)
            if iteration == int(max_iterations):
                break
            jacobian = self.field.state_jacobian(state, g, u)
            try:
                step = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            accepted = False
            factor = 1.0
            for _ in range(14):
                trial = state + factor * step
                trial_norm = float(np.linalg.norm(self.field.vector_field(trial, g, u)))
                if np.isfinite(trial_norm) and trial_norm < norm:
                    state = trial
                    accepted = True
                    break
                factor *= 0.5
            if not accepted:
                break
        residual = self.field.vector_field(state, g, u)
        return NeuralROMSteadyState(state, float(np.linalg.norm(residual)), int(max_iterations), False)

    def save(self, path: str | Path, *, metadata: dict | None = None) -> Path:
        """Save without pickle; geometry operator families remain application-owned."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError("install sdfmpneo[neural] to save neural models") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        network = self.surrogate.network
        state_dict = network.state_dict()
        first_parameter = next(network.parameters())
        dtype_name = "float64" if first_parameter.dtype == torch.float64 else "float32"
        payload = {
            "format_version": _MODEL_FORMAT_VERSION,
            "architecture": "structure-preserving-quadratic-current-neural-rom",
            "network_config": network.config.to_dict(),
            "state_dimension": self.surrogate.state_dimension,
            "geometry_dimension": self.surrogate.geometry_dimension,
            "current_dimension": self.surrogate.pod.current_dimension,
            "network_dtype": dtype_name,
            "physical_signature": self.physical_signature,
            "training_domain": {key: value.tolist() for key, value in self.training_domain.items()},
            "metadata": {} if metadata is None else dict(metadata),
            "operator_kind": "fixed" if isinstance(self.thermal_operators, FixedThermalOperatorFamily) else "external",
        }
        arrays = {
            "metadata_json": np.array(json.dumps(payload, sort_keys=True, allow_nan=False)),
            "pod_mean": self.surrogate.pod.mean,
            "pod_basis": self.surrogate.pod.basis,
            "pod_singular_values": self.surrogate.pod.singular_values,
            "coefficient_mean": self.surrogate.coefficient_mean,
            "coefficient_scale": self.surrogate.coefficient_scale,
        }
        for key, tensor in state_dict.items():
            arrays["network__" + key.replace(".", "__DOT__")] = tensor.detach().cpu().numpy()
        if isinstance(self.thermal_operators, FixedThermalOperatorFamily):
            operator = self.thermal_operators.operator(np.zeros(self.thermal_operators.geometry_dimension))
            arrays["fixed_mass"] = operator.mass
            arrays["fixed_stiffness"] = operator.stiffness
        with path.open("wb") as output:
            np.savez_compressed(output, **arrays)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        thermal_operators=None,
        expected_physical_signature: str | None = None,
        device: str = "cpu",
    ) -> "StructurePreservingNeuralElectroThermalROM":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("install sdfmpneo[neural] to load neural models") from exc
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            if int(meta.get("format_version", -1)) != _MODEL_FORMAT_VERSION:
                raise ValueError("unsupported neural electrothermal ROM format")
            signature = meta.get("physical_signature")
            if expected_physical_signature is not None and signature != expected_physical_signature:
                raise ValueError("physical model signature mismatch")
            config = ResidualMLPConfig(**meta["network_config"])
            input_mean = data["network__input_mean"]
            input_scale = data["network__input_scale"]
            normalizer = FeatureNormalizer(input_mean, input_scale)
            network = build_residual_mlp(config, normalizer)
            dtype = torch.float64 if meta.get("network_dtype") == "float64" else torch.float32
            network = network.to(device=device, dtype=dtype)
            torch_state = {}
            for key in data.files:
                if key.startswith("network__"):
                    name = key[len("network__"):].replace("__DOT__", ".")
                    torch_state[name] = torch.as_tensor(data[key], dtype=dtype, device=device)
            network.load_state_dict(torch_state, strict=True)
            pod = TensorPOD(
                mean=data["pod_mean"],
                basis=data["pod_basis"],
                singular_values=data["pod_singular_values"],
                thermal_rank=int(meta["state_dimension"]),
                current_dimension=int(meta["current_dimension"]),
            )
            surrogate = NeuralTensorSurrogate(
                network,
                pod,
                state_dimension=int(meta["state_dimension"]),
                geometry_dimension=int(meta["geometry_dimension"]),
                coefficient_mean=data["coefficient_mean"],
                coefficient_scale=data["coefficient_scale"],
            )
            if meta.get("operator_kind") == "fixed":
                thermal_operators = FixedThermalOperatorFamily(
                    data["fixed_mass"],
                    data["fixed_stiffness"],
                    geometry_dimension=int(meta["geometry_dimension"]),
                )
            elif thermal_operators is None:
                raise ValueError("geometry-family model load requires thermal_operators")
            return cls(
                surrogate,
                thermal_operators,
                physical_signature=signature,
                training_domain={key: np.asarray(value, dtype=float) for key, value in meta.get("training_domain", {}).items()},
            )


__all__ = [
    "NeuralROMPrediction",
    "NeuralROMSteadyState",
    "StructurePreservingNeuralElectroThermalROM",
]
