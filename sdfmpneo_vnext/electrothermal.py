from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .em import DenseMQSTeacher, MQSConfig
from .loss import ConductorLossField
from .scene import CoilObject, Scene
from .thermal import StableThermalModel


@dataclass(frozen=True)
class CoilThermalProperties:
    heat_capacity: float
    ambient_conductance: float

    def __post_init__(self):
        if (
            not np.isfinite(self.heat_capacity)
            or self.heat_capacity <= 0.0
        ):
            raise ValueError(
                "heat_capacity must be positive and finite"
            )
        if (
            not np.isfinite(self.ambient_conductance)
            or self.ambient_conductance < 0.0
        ):
            raise ValueError(
                "ambient_conductance must be finite and nonnegative"
            )


@dataclass(frozen=True)
class ElectroThermalStep:
    state: np.ndarray
    temperatures: np.ndarray
    impedance: np.ndarray
    coil_power: np.ndarray
    iterations: int
    coupling_residual: float
    converged: bool


def build_lumped_coil_thermal_model(
    properties,
    *,
    mutual_conductance=None,
    ambient_temperature: float = 293.15,
) -> StableThermalModel:
    props = tuple(properties)
    if not props:
        raise ValueError(
            "at least one coil thermal property is required"
        )
    n = len(props)
    C = np.diag(
        [
            p.heat_capacity
            for p in props
        ]
    ).astype(float)
    G = np.diag(
        [
            p.ambient_conductance
            for p in props
        ]
    ).astype(float)
    if mutual_conductance is not None:
        M = np.asarray(
            mutual_conductance,
            dtype=float,
        )
        if M.shape != (n, n):
            raise ValueError(
                "mutual_conductance must have shape (n_coils,n_coils)"
            )
        if (
            not np.all(np.isfinite(M))
            or not np.allclose(M, M.T, atol=1e-12)
            or np.any(M < -1e-15)
        ):
            raise ValueError(
                "mutual_conductance must be finite, symmetric, and nonnegative"
            )
        M = M.copy()
        np.fill_diagonal(M, 0.0)
        G += np.diag(np.sum(M, axis=1)) - M
    return StableThermalModel(
        C,
        G,
        np.eye(n),
        np.eye(n),
        reference_temperature=float(ambient_temperature),
    )


class CurrentControlledEnvelope:
    """Narrowband current-controlled electrothermal envelope.

    Carrier electromagnetic fields are solved quasi-statically at each slow
    thermal coupling iterate. The carrier itself is never time-stepped.
    """

    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model: StableThermalModel,
        *,
        em_config: MQSConfig | None = None,
        coupling_tolerance: float = 1e-7,
        max_coupling_iterations: int = 12,
    ):
        if thermal_model.n_states != len(scene.coils):
            raise ValueError(
                "first MVP requires one thermal state per coil"
            )
        if coupling_tolerance <= 0.0:
            raise ValueError(
                "coupling_tolerance must be positive"
            )
        if max_coupling_iterations < 1:
            raise ValueError(
                "max_coupling_iterations must be >= 1"
            )
        self.scene = scene
        self.frequency_hz = float(frequency_hz)
        self.thermal_model = thermal_model
        self.em_config = em_config or MQSConfig()
        self.coupling_tolerance = float(
            coupling_tolerance
        )
        self.max_coupling_iterations = int(
            max_coupling_iterations
        )

    def _scene_at_temperatures(
        self,
        temperatures,
    ) -> Scene:
        temperatures = np.asarray(
            temperatures,
            dtype=float,
        )
        if temperatures.shape != (
            len(self.scene.coils),
        ):
            raise ValueError(
                "temperatures have wrong shape"
            )
        coils = []
        for coil, temperature in zip(
            self.scene.coils,
            temperatures,
        ):
            coils.append(
                CoilObject(
                    coil.geometry,
                    coil.material.at_temperature(
                        float(temperature)
                    ),
                    coil.name,
                )
            )
        return Scene(
            tuple(coils),
            self.scene.medium,
        )

    def _electromagnetic_power(
        self,
        temperatures,
        currents,
    ):
        warm_scene = self._scene_at_temperatures(
            temperatures
        )
        teacher = DenseMQSTeacher(
            warm_scene,
            self.frequency_hz,
            self.em_config,
        )
        result = teacher.solve()
        loss = ConductorLossField(
            teacher,
            result,
            currents,
        )
        return (
            result,
            loss.coil_power(),
        )

    def step(
        self,
        state,
        currents,
        dt: float,
    ) -> ElectroThermalStep:
        z0 = np.asarray(
            state,
            dtype=float,
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if z0.shape != (
            self.thermal_model.n_states,
        ):
            raise ValueError(
                "state has wrong shape"
            )
        if currents.shape != (
            len(self.scene.coils),
        ):
            raise ValueError(
                "currents have wrong shape"
            )
        if dt < 0.0:
            raise ValueError(
                "dt must be nonnegative"
            )

        guess = z0.copy()
        residual = np.inf
        result = None
        power = None
        converged = False
        for iteration in range(
            1,
            self.max_coupling_iterations + 1,
        ):
            temperatures = (
                self.thermal_model.temperature(
                    guess
                )
            )
            result, power = (
                self._electromagnetic_power(
                    temperatures,
                    currents,
                )
            )
            candidate = (
                self.thermal_model.advance_constant_power(
                    z0,
                    power,
                    dt,
                )
            )
            scale = max(
                float(
                    np.linalg.norm(candidate)
                ),
                1.0,
            )
            residual = float(
                np.linalg.norm(
                    candidate - guess
                )
                / scale
            )
            guess = candidate
            if residual <= self.coupling_tolerance:
                converged = True
                break

        final_temperatures = (
            self.thermal_model.temperature(
                guess
            )
        )
        # Re-evaluate electrical outputs at the accepted thermal state so the
        # returned Z and Joule powers are mutually consistent with temperature.
        result, power = (
            self._electromagnetic_power(
                final_temperatures,
                currents,
            )
        )
        return ElectroThermalStep(
            guess,
            final_temperatures,
            result.impedance,
            power,
            iteration,
            residual,
            converged,
        )

    def trajectory(
        self,
        initial_state,
        currents,
        times,
    ) -> tuple[ElectroThermalStep, ...]:
        times = np.asarray(
            times,
            dtype=float,
        )
        if (
            times.ndim != 1
            or len(times) == 0
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) < 0.0)
        ):
            raise ValueError(
                "times must be a finite nondecreasing 1D array"
            )
        state = np.asarray(
            initial_state,
            dtype=float,
        )
        out = []
        previous = 0.0
        for time in times:
            if time < previous:
                raise ValueError(
                    "times must be nondecreasing"
                )
            step = self.step(
                state,
                currents,
                float(time - previous),
            )
            out.append(step)
            state = step.state
            previous = float(time)
        return tuple(out)
