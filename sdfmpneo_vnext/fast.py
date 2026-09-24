from __future__ import annotations

import numpy as np

from .electrothermal import ElectroThermalStep
from .scene import CoilObject, Scene
from .thermal import StableThermalModel


def _coil_power_from_channels(
    channels,
    currents,
) -> np.ndarray:
    channels = np.asarray(
        channels,
        dtype=complex,
    )
    currents = np.asarray(
        currents,
        dtype=complex,
    )
    if (
        channels.ndim != 3
        or channels.shape[1:]
        != (
            currents.size,
            currents.size,
        )
    ):
        raise ValueError(
            "dissipation channels have incompatible shape"
        )
    return np.asarray(
        [
            0.5
            * np.real(
                np.vdot(
                    currents,
                    channel @ currents,
                )
            )
            for channel in channels
        ],
        dtype=float,
    )


class _FastEnvelopeBase:
    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model: StableThermalModel,
        artifact,
        *,
        coupling_tolerance: float = 1e-7,
        max_coupling_iterations: int = 12,
        power_closure_tolerance: float = 1e-5,
    ):
        if (
            thermal_model.n_states
            != len(scene.coils)
        ):
            raise ValueError(
                "first MVP requires one thermal state per coil"
            )
        if (
            not np.isfinite(frequency_hz)
            or frequency_hz < 0.0
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
        if not hasattr(
            artifact,
            "predict_structured",
        ):
            raise TypeError(
                "FAST electrothermal artifact must expose predict_structured"
            )
        self.scene = scene
        self.frequency_hz = float(
            frequency_hz
        )
        self.thermal_model = (
            thermal_model
        )
        self.artifact = artifact
        self.coupling_tolerance = float(
            coupling_tolerance
        )
        self.max_coupling_iterations = int(
            max_coupling_iterations
        )
        self.power_closure_tolerance = float(
            power_closure_tolerance
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
                        float(
                            temperature
                        )
                    ),
                    coil.name,
                )
            )
        return Scene(
            tuple(coils),
            self.scene.medium,
        )

    def _predict(
        self,
        temperatures,
    ):
        warm_scene = (
            self._scene_at_temperatures(
                temperatures
            )
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
        n = len(
            self.scene.coils
        )
        if impedance.shape != (
            n,
            n,
        ):
            raise ValueError(
                "FAST artifact returned wrong impedance shape"
            )
        if channels.shape != (
            n,
            n,
            n,
        ):
            raise ValueError(
                "FAST artifact returned wrong loss-channel shape"
            )
        dissipation = 0.5 * (
            impedance
            + impedance.conj().T
        )
        closure = float(
            np.linalg.norm(
                np.sum(
                    channels,
                    axis=0,
                )
                - dissipation
            )
            / max(
                np.linalg.norm(
                    dissipation
                ),
                1e-30,
            )
        )
        if (
            closure
            > self.power_closure_tolerance
        ):
            raise RuntimeError(
                "FAST artifact violates dissipation-channel power closure"
            )
        return (
            impedance,
            channels,
        )

    def trajectory(
        self,
        initial_state,
        drive,
        times,
    ) -> tuple[
        ElectroThermalStep,
        ...,
    ]:
        times = np.asarray(
            times,
            dtype=float,
        )
        if (
            times.ndim != 1
            or len(times) == 0
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
            state = step.state
            previous = float(
                time
            )
        return tuple(
            out
        )


class FastCurrentControlledEnvelope(
    _FastEnvelopeBase
):
    """FAST current-controlled electrothermal evolution without Maxwell solves."""

    def step(
        self,
        state,
        currents,
        dt: float,
    ) -> ElectroThermalStep:
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

        guess = state0.copy()
        residual = np.inf
        converged = False
        impedance = None
        channels = None
        power = None

        for iteration in range(
            1,
            self.max_coupling_iterations
            + 1,
        ):
            temperatures = (
                self.thermal_model.temperature(
                    guess
                )
            )
            impedance, channels = (
                self._predict(
                    temperatures
                )
            )
            power = (
                _coil_power_from_channels(
                    channels,
                    currents,
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

        final_temperatures = (
            self.thermal_model.temperature(
                guess
            )
        )
        impedance, channels = (
            self._predict(
                final_temperatures
            )
        )
        power = (
            _coil_power_from_channels(
                channels,
                currents,
            )
        )
        return ElectroThermalStep(
            guess,
            final_temperatures,
            impedance,
            currents.copy(),
            power,
            iteration,
            residual,
            converged,
        )


class FastVoltageControlledEnvelope(
    _FastEnvelopeBase
):
    """FAST voltage/circuit-controlled electrothermal evolution."""

    def __init__(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model: StableThermalModel,
        artifact,
        *,
        external_impedance=None,
        coupling_tolerance: float = 1e-7,
        max_coupling_iterations: int = 12,
        power_closure_tolerance: float = 1e-5,
    ):
        super().__init__(
            scene,
            frequency_hz,
            thermal_model,
            artifact,
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
        if (
            external_impedance
            is None
        ):
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
    ) -> ElectroThermalStep:
        state0 = np.asarray(
            state,
            dtype=float,
        )
        source_voltage = np.asarray(
            source_voltage,
            dtype=complex,
        )
        n = len(
            self.scene.coils
        )
        if state0.shape != (
            self.thermal_model.n_states,
        ):
            raise ValueError(
                "state has wrong shape"
            )
        if source_voltage.shape != (
            n,
        ):
            raise ValueError(
                "source_voltage has wrong shape"
            )
        if dt < 0.0:
            raise ValueError(
                "dt must be nonnegative"
            )

        guess = state0.copy()
        residual = np.inf
        converged = False
        impedance = None
        channels = None
        currents = None
        power = None

        for iteration in range(
            1,
            self.max_coupling_iterations
            + 1,
        ):
            temperatures = (
                self.thermal_model.temperature(
                    guess
                )
            )
            impedance, channels = (
                self._predict(
                    temperatures
                )
            )
            currents = np.linalg.solve(
                impedance
                + self.external_impedance,
                source_voltage,
            )
            power = (
                _coil_power_from_channels(
                    channels,
                    currents,
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

        final_temperatures = (
            self.thermal_model.temperature(
                guess
            )
        )
        impedance, channels = (
            self._predict(
                final_temperatures
            )
        )
        currents = np.linalg.solve(
            impedance
            + self.external_impedance,
            source_voltage,
        )
        power = (
            _coil_power_from_channels(
                channels,
                currents,
            )
        )
        return ElectroThermalStep(
            guess,
            final_temperatures,
            impedance,
            currents,
            power,
            iteration,
            residual,
            converged,
        )
