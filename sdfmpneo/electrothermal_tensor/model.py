"""Deployable structure-preserving neural electrothermal ROM facade."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .geometry_thermal import AffineGeometryThermalOperatorFamily
from .integrators import (
    GeneralizedThermalSpectrum,
    integrate_etd2,
    integrate_etd2_adaptive,
    integrate_imex_euler,
    integrate_reference,
)
from .network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .pod import TensorPOD
from .runtime_metadata import git_revision, software_environment_summary
from .surrogate import NeuralTensorSurrogate
from .vector_field import FixedThermalOperatorFamily, NeuralElectroThermalVectorField

_MODEL_FORMAT_VERSION = 3
_SUPPORTED_MODEL_FORMAT_VERSIONS = (1, 2, 3)


@dataclass(frozen=True)
class NeuralROMPrediction:
    time: float
    state: np.ndarray
    derivative: np.ndarray
    heat_source: np.ndarray
    steps: int
    step_sizes: tuple[float, ...]
    rejected_steps: int = 0


@dataclass(frozen=True)
class NeuralROMSteadyState:
    state: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool


def _merge_metadata(base: dict, update: dict) -> dict:
    """Recursively merge JSON-compatible artifact metadata without mutation."""
    result = dict(base)
    for key, value in dict(update).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_metadata(result[key], value)
        else:
            result[key] = value
    return result


class StructurePreservingNeuralElectroThermalROM:
    def __init__(
        self,
        surrogate: NeuralTensorSurrogate,
        thermal_operators,
        *,
        physical_signature: str | None = None,
        training_domain: dict | None = None,
        artifact_metadata: dict | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.thermal_operators = thermal_operators
        self.field = NeuralElectroThermalVectorField(surrogate, thermal_operators)
        self.physical_signature = None if physical_signature is None else str(physical_signature)
        self.training_domain = {} if training_domain is None else {
            key: np.asarray(value, dtype=float) for key, value in training_domain.items()
        }
        self.artifact_metadata = {} if artifact_metadata is None else dict(artifact_metadata)
        self._spectrum_cache_size = max(1, int(getattr(thermal_operators, "cache_size", 64)))
        self._spectrum_cache: OrderedDict[tuple[float, ...], GeneralizedThermalSpectrum] = OrderedDict()
        self._spectrum_cache_enabled = isinstance(
            thermal_operators,
            (FixedThermalOperatorFamily, AffineGeometryThermalOperatorFamily),
        )

    def _check_domain(self, state, geometry, operating, *, allow_extrapolation: bool):
        a = np.asarray(state, dtype=float).reshape(-1)
        g = np.asarray(geometry, dtype=float).reshape(-1)
        u = np.asarray(operating, dtype=float).reshape(-1)
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(g)) or np.any(~np.isfinite(u)):
            raise ValueError("state, geometry and operating inputs must be finite")
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

    def _state_validator(self, *, allow_extrapolation: bool):
        if allow_extrapolation or not self.training_domain:
            return None
        lower = self.training_domain.get("state_lower")
        upper = self.training_domain.get("state_upper")
        if lower is None or upper is None:
            return None
        lo = np.asarray(lower, dtype=float).reshape(-1)
        hi = np.asarray(upper, dtype=float).reshape(-1)

        def validate(state):
            value = np.asarray(state, dtype=float).reshape(-1)
            if value.shape != lo.shape or np.any(~np.isfinite(value)):
                raise ValueError("trajectory state is incompatible with the trained state domain")
            if np.any(value < lo) or np.any(value > hi):
                raise ValueError(
                    "trajectory left the trained thermal-state domain; "
                    "expand the snapshot state box or set allow_extrapolation=True explicitly"
                )

        return validate

    def _spectrum_for_geometry(self, geometry: np.ndarray) -> GeneralizedThermalSpectrum:
        """Return a bounded per-model cached generalized thermal spectrum.

        Only persisted deterministic operator families are cached. Application-
        owned callable families may legally change their operator for the same
        geometry, so they are deliberately recomputed on every request.
        """
        g = np.asarray(geometry, dtype=float).reshape(-1)
        operator = self.thermal_operators.operator(g)
        if not self._spectrum_cache_enabled:
            return GeneralizedThermalSpectrum(operator)
        key = tuple(float(v) for v in g)
        cached = self._spectrum_cache.get(key)
        if cached is not None:
            self._spectrum_cache.move_to_end(key)
            return cached
        spectrum = GeneralizedThermalSpectrum(operator)
        self._spectrum_cache[key] = spectrum
        if len(self._spectrum_cache) > self._spectrum_cache_size:
            self._spectrum_cache.popitem(last=False)
        return spectrum

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
        rtol: float = 1e-5,
        atol: float = 1e-8,
        initial_step: float | None = None,
        max_attempts: int = 100000,
    ) -> NeuralROMPrediction:
        a0, g, u = self._check_domain(
            initial_state, geometry, operating, allow_extrapolation=allow_extrapolation
        )
        state_validator = self._state_validator(allow_extrapolation=allow_extrapolation)
        t = float(time)
        spectrum = None
        if t > 0.0 and method in {"etd2", "etd2_adaptive", "adaptive_etd2"}:
            spectrum = self._spectrum_for_geometry(g)
        if method == "etd2":
            result = integrate_etd2(
                self.field,
                t,
                initial_state=a0,
                geometry=g,
                operating=u,
                max_step=max_step,
                state_validator=state_validator,
                spectrum=spectrum,
            )
        elif method in {"etd2_adaptive", "adaptive_etd2"}:
            result = integrate_etd2_adaptive(
                self.field,
                t,
                initial_state=a0,
                geometry=g,
                operating=u,
                max_step=max_step,
                rtol=rtol,
                atol=atol,
                initial_step=initial_step,
                max_attempts=max_attempts,
                state_validator=state_validator,
                spectrum=spectrum,
            )
        elif method == "imex":
            result = integrate_imex_euler(
                self.field,
                t,
                initial_state=a0,
                geometry=g,
                operating=u,
                max_step=max_step,
                state_validator=state_validator,
            )
        elif method == "reference":
            result = integrate_reference(
                self.field,
                t,
                initial_state=a0,
                geometry=g,
                operating=u,
                rtol=rtol,
                atol=atol,
                state_validator=state_validator,
            )
        else:
            raise ValueError(
                "method must be 'etd2', 'etd2_adaptive', 'imex' or 'reference'"
            )
        derivative = self.field.vector_field(result.state, g, u)
        heat = self.field.heat_source(result.state, g, u)
        return NeuralROMPrediction(
            time=t,
            state=result.state,
            derivative=derivative,
            heat_source=heat,
            steps=result.steps,
            step_sizes=result.step_sizes,
            rejected_steps=result.rejected_steps,
        )

    def predict_batch_fixed_etd2(
        self,
        time: float,
        *,
        initial_states: np.ndarray,
        geometry: np.ndarray,
        operating: np.ndarray,
        max_step: float,
        allow_extrapolation: bool = False,
    ):
        """Batch independent fixed-step ETD2 queries sharing one geometry/time."""
        from .batch import predict_batch_fixed_etd2

        return predict_batch_fixed_etd2(
            self,
            time,
            initial_states=initial_states,
            geometry=geometry,
            operating=operating,
            max_step=max_step,
            allow_extrapolation=allow_extrapolation,
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
        tolerance = float(tolerance)
        max_iterations = int(max_iterations)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("steady-state tolerance must be finite and positive")
        if max_iterations < 1:
            raise ValueError("steady-state max_iterations must be positive")
        state, g, u = self._check_domain(
            initial_guess, geometry, operating, allow_extrapolation=allow_extrapolation
        )
        validator = self._state_validator(allow_extrapolation=allow_extrapolation)
        state = state.copy()
        last_iteration = 0
        for iteration in range(max_iterations + 1):
            last_iteration = iteration
            residual = self.field.vector_field(state, g, u)
            norm = float(np.linalg.norm(residual))
            if norm <= tolerance:
                return NeuralROMSteadyState(state, norm, iteration, True)
            if iteration == max_iterations:
                break
            jacobian = self.field.state_jacobian(state, g, u)
            try:
                step = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            if np.any(~np.isfinite(step)):
                break
            accepted = False
            factor = 1.0
            for _ in range(14):
                trial = state + factor * step
                try:
                    if validator is not None:
                        validator(trial)
                    trial_norm = float(np.linalg.norm(self.field.vector_field(trial, g, u)))
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    trial_norm = float("inf")
                if np.isfinite(trial_norm) and trial_norm < norm:
                    state = trial
                    accepted = True
                    break
                factor *= 0.5
            if not accepted:
                break
        residual = self.field.vector_field(state, g, u)
        return NeuralROMSteadyState(
            state,
            float(np.linalg.norm(residual)),
            last_iteration,
            False,
        )

    def save(self, path: str | Path, *, metadata: dict | None = None) -> Path:
        """Save a pickle-free, fail-closed neural ROM artifact.

        ``metadata`` is merged into metadata loaded from any prior artifact, so
        a later frozen audit can append certification evidence without erasing
        training provenance. Runtime spectrum caches are intentionally not
        persisted; they are rebuilt lazily from the saved exact thermal operators.
        """
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
        if isinstance(self.thermal_operators, FixedThermalOperatorFamily):
            operator_kind = "fixed"
        elif isinstance(self.thermal_operators, AffineGeometryThermalOperatorFamily):
            operator_kind = "affine_geometry"
        else:
            operator_kind = "external"
        artifact_metadata = _merge_metadata(
            self.artifact_metadata,
            {} if metadata is None else dict(metadata),
        )
        payload = {
            "format_version": _MODEL_FORMAT_VERSION,
            "architecture": "structure-preserving-quadratic-current-neural-rom",
            "source_revision": git_revision(),
            "software_environment": software_environment_summary(),
            "network_config": network.config.to_dict(),
            "state_dimension": self.surrogate.state_dimension,
            "geometry_dimension": self.surrogate.geometry_dimension,
            "current_dimension": self.surrogate.pod.current_dimension,
            "network_dtype": dtype_name,
            "physical_signature": self.physical_signature,
            "training_domain": {key: value.tolist() for key, value in self.training_domain.items()},
            "metadata": artifact_metadata,
            "operator_kind": operator_kind,
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
        if operator_kind == "fixed":
            operator = self.thermal_operators.operator(
                np.zeros(self.thermal_operators.geometry_dimension)
            )
            arrays["fixed_mass"] = operator.mass
            arrays["fixed_stiffness"] = operator.stiffness
        elif operator_kind == "affine_geometry":
            arrays.update(self.thermal_operators.persistence_arrays())
        with path.open("wb") as output:
            np.savez_compressed(output, **arrays)
        self.artifact_metadata = artifact_metadata
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
            version = int(meta.get("format_version", -1))
            if version not in _SUPPORTED_MODEL_FORMAT_VERSIONS:
                raise ValueError("unsupported neural electrothermal ROM format")
            if meta.get("architecture") != "structure-preserving-quadratic-current-neural-rom":
                raise ValueError("unexpected neural ROM architecture")
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
            operator_kind = meta.get("operator_kind")
            if operator_kind == "fixed":
                thermal_operators = FixedThermalOperatorFamily(
                    data["fixed_mass"],
                    data["fixed_stiffness"],
                    geometry_dimension=int(meta["geometry_dimension"]),
                )
            elif operator_kind == "affine_geometry":
                thermal_operators = AffineGeometryThermalOperatorFamily.from_persistence(data)
            elif thermal_operators is None:
                raise ValueError(
                    "model uses an external thermal operator family; provide thermal_operators explicitly"
                )
            if int(thermal_operators.geometry_dimension) != int(meta["geometry_dimension"]):
                raise ValueError("saved neural and thermal geometry dimensions differ")
            return cls(
                surrogate,
                thermal_operators,
                physical_signature=signature,
                training_domain={
                    key: np.asarray(value, dtype=float)
                    for key, value in meta.get("training_domain", {}).items()
                },
                artifact_metadata=dict(meta.get("metadata") or {}),
            )


__all__ = [
    "NeuralROMPrediction",
    "NeuralROMSteadyState",
    "StructurePreservingNeuralElectroThermalROM",
]
