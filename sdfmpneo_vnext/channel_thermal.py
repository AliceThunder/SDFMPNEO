from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .scene import CoilObject, Scene
from .thermal import StableThermalModel


@dataclass(frozen=True)
class ThermalNodeProperties:
    heat_capacity: float
    ambient_conductance: float

    def __post_init__(self):
        if (
            not np.isfinite(
                self.heat_capacity
            )
            or self.heat_capacity
            <= 0.0
        ):
            raise ValueError(
                "heat_capacity must be positive and finite"
            )
        if (
            not np.isfinite(
                self.ambient_conductance
            )
            or self.ambient_conductance
            < 0.0
        ):
            raise ValueError(
                "ambient_conductance must be finite and nonnegative"
            )


def build_lumped_channel_thermal_model(
    properties,
    *,
    source_map=None,
    mutual_conductance=None,
    observation_map=None,
    ambient_temperature: float = 293.15,
) -> StableThermalModel:
    props = tuple(
        properties
    )
    if not props:
        raise ValueError(
            "at least one thermal node is required"
        )
    n = len(
        props
    )
    capacity = np.diag(
        [
            item.heat_capacity
            for item
            in props
        ]
    ).astype(
        float
    )
    conductance = np.diag(
        [
            item.ambient_conductance
            for item
            in props
        ]
    ).astype(
        float
    )

    if mutual_conductance is not None:
        mutual = np.asarray(
            mutual_conductance,
            dtype=float,
        )
        if mutual.shape != (
            n,
            n,
        ):
            raise ValueError(
                "mutual_conductance must have shape (n_states,n_states)"
            )
        if (
            not np.all(
                np.isfinite(
                    mutual
                )
            )
            or not np.allclose(
                mutual,
                mutual.T,
                atol=1e-12,
            )
            or np.any(
                mutual
                < -1e-15
            )
        ):
            raise ValueError(
                "mutual_conductance must be finite, symmetric, and nonnegative"
            )
        mutual = (
            mutual.copy()
        )
        np.fill_diagonal(
            mutual,
            0.0,
        )
        conductance += (
            np.diag(
                np.sum(
                    mutual,
                    axis=1,
                )
            )
            - mutual
        )

    if source_map is None:
        source = np.eye(
            n,
            dtype=float,
        )
    else:
        source = np.asarray(
            source_map,
            dtype=float,
        )
        if (
            source.ndim != 2
            or source.shape[0]
            != n
            or not np.all(
                np.isfinite(
                    source
                )
            )
        ):
            raise ValueError(
                "source_map must have shape (n_states,n_channels)"
            )

    if observation_map is None:
        decoder = np.eye(
            n,
            dtype=float,
        )
    else:
        decoder = np.asarray(
            observation_map,
            dtype=float,
        )
        if (
            decoder.ndim != 2
            or decoder.shape[1]
            != n
            or not np.all(
                np.isfinite(
                    decoder
                )
            )
        ):
            raise ValueError(
                "observation_map must have shape (n_observations,n_states)"
            )

    return StableThermalModel(
        capacity,
        conductance,
        source,
        decoder,
        reference_temperature=float(
            ambient_temperature
        ),
    )


@dataclass(frozen=True)
class ChannelElectroThermalStep:
    state: np.ndarray
    temperatures: np.ndarray
    impedance: np.ndarray
    currents: np.ndarray
    channel_power: np.ndarray
    iterations: int
    coupling_residual: float
    converged: bool

    @property
    def total_power(
        self,
    ) -> float:
        return float(
            np.sum(
                self.channel_power
            )
        )


class _ChannelResolvedEnvelopeBase:
    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model: StableThermalModel,
        artifact,
        *,
        coil_temperature_indices=None,
        coupling_tolerance: float = 1e-7,
        max_coupling_iterations: int = 12,
        power_closure_tolerance: float = 1e-5,
    ):
        if not hasattr(
            artifact,
            "predict_structured",
        ):
            raise TypeError(
                "artifact must expose predict_structured"
            )
        if (
            not np.isfinite(
                frequency_hz
            )
            or frequency_hz
            < 0.0
        ):
            raise ValueError(
                "frequency_hz must be finite and nonnegative"
            )
        if coupling_tolerance <= 0.0:
            raise ValueError(
                "coupling_tolerance must be positive"
            )
        if max_coupling_iterations < 1:
            raise ValueError(
                "max_coupling_iterations must be >= 1"
            )
        if power_closure_tolerance < 0.0:
            raise ValueError(
                "power_closure_tolerance must be nonnegative"
            )

        n_coils = len(
            scene.coils
        )
        if coil_temperature_indices is None:
            indices = np.arange(
                n_coils,
                dtype=int,
            )
        else:
            indices = np.asarray(
                coil_temperature_indices,
                dtype=int,
            )
        if indices.shape != (
            n_coils,
        ):
            raise ValueError(
                "coil_temperature_indices must contain one observation "
                "index per coil"
            )
        n_observations = (
            thermal_model.decoder.shape[
                0
            ]
        )
        if (
            np.any(
                indices < 0
            )
            or np.any(
                indices
                >= n_observations
            )
        ):
            raise ValueError(
                "coil_temperature_indices are outside the thermal observation range"
            )

        self.scene = scene
        self.frequency_hz = float(
            frequency_hz
        )
        self.thermal_model = (
            thermal_model
        )
        self.artifact = artifact
        self.coil_temperature_indices = (
            indices
        )
        self.coupling_tolerance = float(
            coupling_tolerance
        )
        self.max_coupling_iterations = int(
            max_coupling_iterations
        )
        self.power_closure_tolerance = float(
            power_closure_tolerance
        )

    def _scene_at_state(
        self,
        state,
    ):
        temperatures = (
            self.thermal_model.temperature(
                state
            )
        )
        coil_temperatures = temperatures[
            self.coil_temperature_indices
        ]
        coils = []
        for coil, temperature in zip(
            self.scene.coils,
            coil_temperatures,
        ):
            coils.append(
                CoilObject(
                    coil.geometry,
                    coil.material.at_temperature(
                        float(
                            temperature
                        )
                    ),
                    coil.name,
                )
            )
        warm_scene = Scene(
            tuple(
                coils
            ),
            self.scene.medium,
            self.scene.packages,
        )
        return (
            warm_scene,
            temperatures,
        )

    def _prediction_at_state(
        self,
        state,
    ):
        (
            warm_scene,
            temperatures,
        ) = self._scene_at_state(
            state
        )
        prediction = (
            self.artifact.predict_structured(
                warm_scene,
                self.frequency_hz,
            )
        )
        impedance = np.asarray(
            prediction.impedance,
            dtype=complex,
        )
        channels = np.asarray(
            prediction.dissipation_channels,
            dtype=complex,
        )
        n_ports = len(
            self.scene.coils
        )
        if impedance.shape != (
            n_ports,
            n_ports,
        ):
            raise ValueError(
                "artifact returned wrong impedance shape"
            )
        if (
            channels.ndim != 3
            or channels.shape[1:]
            != (
                n_ports,
                n_ports,
            )
        ):
            raise ValueError(
                "artifact returned wrong dissipation-channel shape"
            )
        if (
            self.thermal_model.source.shape[
                1
            ]
            != channels.shape[
                0
            ]
        ):
            raise ValueError(
                "thermal source-map channel count does not match "
                "electromagnetic dissipation channels"
            )
        closure = float(
            prediction.power_closure_error()
        )
        if (
            closure
            > self.power_closure_tolerance
        ):
            raise RuntimeError(
                "artifact violates dissipation-channel power closure"
            )
        return (
            prediction,
            temperatures,
        )

    def trajectory(
        self,
        initial_state,
        drive,
        times,
    ):
        times = np.asarray(
            times,
            dtype=float,
        )
        if (
            times.ndim != 1
            or len(
                times
            )
            == 0
            or np.any(
                ~np.isfinite(
                    times
                )
            )
            or np.any(
                np.diff(
                    times
                )
                < 0.0
            )
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
            step = self.step(
                state,
                drive,
                float(
                    time
                    - previous
                ),
            )
            out.append(
                step
            )
            state = (
                step.state
            )
            previous = float(
                time
            )
        return tuple(
            out
        )


class ChannelResolvedCurrentEnvelope(
    _ChannelResolvedEnvelopeBase
):
    def step(
        self,
        state,
        currents,
        dt: float,
    ) -> ChannelElectroThermalStep:
        state0 = np.asarray(
            state,
            dtype=float,
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if state0.shape != (
            self.thermal_model.n_states,
        ):
            raise ValueError(
                "state has wrong shape"
            )
        if currents.shape != (
            len(
                self.scene.coils
            ),
        ):
            raise ValueError(
                "currents have wrong shape"
            )
        if dt < 0.0:
            raise ValueError(
                "dt must be nonnegative"
            )

        guess = (
            state0.copy()
        )
        residual = np.inf
        converged = False

        for iteration in range(
            1,
            self.max_coupling_iterations
            + 1,
        ):
            (
                prediction,
                _,
            ) = self._prediction_at_state(
                guess
            )
            power = (
                prediction.channel_power(
                    currents
                )
            )
            candidate = (
                self.thermal_model.advance_constant_power(
                    state0,
                    power,
                    dt,
                )
            )
            scale = max(
                float(
                    np.linalg.norm(
                        candidate
                    )
                ),
                1.0,
            )
            residual = float(
                np.linalg.norm(
                    candidate
                    - guess
                )
                / scale
            )
            guess = candidate
            if (
                residual
                <= self.coupling_tolerance
            ):
                converged = True
                break

        (
            prediction,
            temperatures,
        ) = self._prediction_at_state(
            guess
        )
        power = (
            prediction.channel_power(
                currents
            )
        )
        return ChannelElectroThermalStep(
            guess,
            temperatures,
            prediction.impedance,
            currents.copy(),
            power,
            iteration,
            residual,
            converged,
        )


class ChannelResolvedVoltageEnvelope(
    _ChannelResolvedEnvelopeBase
):
    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model: StableThermalModel,
        artifact,
        *,
        external_impedance=None,
        coil_temperature_indices=None,
        coupling_tolerance: float = 1e-7,
        max_coupling_iterations: int = 12,
        power_closure_tolerance: float = 1e-5,
    ):
        super().__init__(
            scene,
            frequency_hz,
            thermal_model,
            artifact,
            coil_temperature_indices=(
                coil_temperature_indices
            ),
            coupling_tolerance=(
                coupling_tolerance
            ),
            max_coupling_iterations=(
                max_coupling_iterations
            ),
            power_closure_tolerance=(
                power_closure_tolerance
            ),
        )
        n = len(
            scene.coils
        )
        if external_impedance is None:
            external = np.zeros(
                (
                    n,
                    n,
                ),
                dtype=complex,
            )
        else:
            external = np.asarray(
                external_impedance,
                dtype=complex,
            )
        if external.shape != (
            n,
            n,
        ):
            raise ValueError(
                "external_impedance must have shape (n_ports,n_ports)"
            )
        if not np.all(
            np.isfinite(
                external
            )
        ):
            raise ValueError(
                "external_impedance must be finite"
            )
        dissipative = 0.5 * (
            external
            + external.conj().T
        )
        if (
            np.min(
                np.linalg.eigvalsh(
                    dissipative
                )
            )
            < -1e-12
        ):
            raise ValueError(
                "external_impedance must be passive"
            )
        self.external_impedance = (
            external
        )

    def step(
        self,
        state,
        source_voltage,
        dt: float,
    ) -> ChannelElectroThermalStep:
        state0 = np.asarray(
            state,
            dtype=float,
        )
        source_voltage = np.asarray(
            source_voltage,
            dtype=complex,
        )
        n_ports = len(
            self.scene.coils
        )
        if state0.shape != (
            self.thermal_model.n_states,
        ):
            raise ValueError(
                "state has wrong shape"
            )
        if source_voltage.shape != (
            n_ports,
        ):
            raise ValueError(
                "source_voltage has wrong shape"
            )
        if dt < 0.0:
            raise ValueError(
                "dt must be nonnegative"
            )

        guess = (
            state0.copy()
        )
        residual = np.inf
        converged = False

        for iteration in range(
            1,
            self.max_coupling_iterations
            + 1,
        ):
            (
                prediction,
                _,
            ) = self._prediction_at_state(
                guess
            )
            currents = np.linalg.solve(
                prediction.impedance
                + self.external_impedance,
                source_voltage,
            )
            power = (
                prediction.channel_power(
                    currents
                )
            )
            candidate = (
                self.thermal_model.advance_constant_power(
                    state0,
                    power,
                    dt,
                )
            )
            scale = max(
                float(
                    np.linalg.norm(
                        candidate
                    )
                ),
                1.0,
            )
            residual = float(
                np.linalg.norm(
                    candidate
                    - guess
                )
                / scale
            )
            guess = candidate
            if (
                residual
                <= self.coupling_tolerance
            ):
                converged = True
                break

        (
            prediction,
            temperatures,
        ) = self._prediction_at_state(
            guess
        )
        currents = np.linalg.solve(
            prediction.impedance
            + self.external_impedance,
            source_voltage,
        )
        power = (
            prediction.channel_power(
                currents
            )
        )
        return ChannelElectroThermalStep(
            guess,
            temperatures,
            prediction.impedance,
            currents,
            power,
            iteration,
            residual,
            converged,
        )
