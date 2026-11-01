"""BFZI quasi-static two-coil electrothermal envelopes.

Reuse the existing vNext StableThermalModel and ElectroThermalStep. Seawater
MQS remains at its reference conductivity; round-wire internal impedance and
conductor Joule heating are updated at every temperature iterate. An external
voltage-drive circuit may be supplied without modifying the seawater solver.
"""
from __future__ import annotations

import numpy as np

from sdfmpneo_vnext.electrothermal import ElectroThermalStep
from sdfmpneo_vnext.thermal import StableThermalModel
from .sea_reference import SeawaterResponse


class BFZICurrentControlledEnvelope:
    """Implicitly update two conductor temperatures for prescribed currents."""

    def __init__(
        self, reference: SeawaterResponse, thermal_model: StableThermalModel,
        *, coupling_tolerance: float = 1e-7,
        max_coupling_iterations: int = 12,
    ):
        if thermal_model.n_states != 2 or thermal_model.temperature(
            np.zeros(2)
        ).shape != (2,):
            raise ValueError("BFZI lumped thermal model must have two coil states")
        if not np.isfinite(coupling_tolerance) or coupling_tolerance <= 0:
            raise ValueError("coupling_tolerance must be finite and positive")
        if not isinstance(max_coupling_iterations, int) or max_coupling_iterations < 1:
            raise ValueError("max_coupling_iterations must be a positive integer")
        self.reference = reference
        self.thermal_model = thermal_model
        self.coupling_tolerance = float(coupling_tolerance)
        self.max_coupling_iterations = max_coupling_iterations

    def _drive_currents(self, impedance, drive):
        return drive.copy()

    def step(self, state, currents, dt: float) -> ElectroThermalStep:
        """Advance from the old thermal state using the supplied peak phasors.

        For voltage control, the second argument instead denotes the source
        voltage phasors. Only conductor power heats these two thermal states;
        seawater dissipation remains a separate spatial source channel.
        """
        z0 = np.asarray(state, dtype=float)
        drive = np.asarray(currents, dtype=complex)
        if z0.shape != (2,) or not np.isfinite(z0).all():
            raise ValueError("state must be a finite two-vector")
        if drive.shape != (2,) or not np.isfinite(drive).all():
            raise ValueError("drive must be a finite complex two-vector")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        guess = z0.copy()
        residual = np.inf
        converged = False
        for iteration in range(1, self.max_coupling_iterations + 1):
            temperature = self.thermal_model.temperature(guess)
            impedance = self.reference.port_impedance_at(temperature)
            i = self._drive_currents(impedance, drive)
            power = self.reference.port_conductor_power(i, temperature)
            candidate = self.thermal_model.advance_constant_power(z0, power, dt)
            residual = float(
                np.linalg.norm(candidate - guess) /
                max(1.0, float(np.linalg.norm(candidate)))
            )
            guess = candidate
            if residual <= self.coupling_tolerance:
                converged = True
                break
        temperature = self.thermal_model.temperature(guess)
        impedance = self.reference.port_impedance_at(temperature)
        i = self._drive_currents(impedance, drive)
        power = self.reference.port_conductor_power(i, temperature)
        return ElectroThermalStep(
            guess, temperature, impedance, i, power, iteration,
            residual, converged,
        )

    def trajectory(self, initial_state, currents, times):
        t = np.asarray(times, dtype=float)
        if (t.ndim != 1 or t.size == 0 or not np.isfinite(t).all()
                or np.any(np.diff(t) < 0) or t[0] < 0):
            raise ValueError("times must be a finite nondecreasing nonnegative array")
        state = np.asarray(initial_state, dtype=float)
        previous = 0.0
        steps = []
        for time in t:
            step = self.step(state, currents, float(time - previous))
            steps.append(step)
            state = step.state
            previous = float(time)
        return tuple(steps)


class BFZIVoltageControlledEnvelope(BFZICurrentControlledEnvelope):
    """Voltage/circuit control with temperature-dependent BFZI port currents.

    External circuit impedance is a fixed passive 2x2 matrix, permitting
    series resistance, capacitive tuning and mutual passive circuit elements.
    Its dissipation is not assigned to either coil's internal heat channel.
    """

    def __init__(
        self, reference: SeawaterResponse, thermal_model: StableThermalModel,
        *, external_impedance=None,
        coupling_tolerance: float = 1e-7,
        max_coupling_iterations: int = 12,
    ):
        super().__init__(
            reference, thermal_model,
            coupling_tolerance=coupling_tolerance,
            max_coupling_iterations=max_coupling_iterations,
        )
        external = (
            np.zeros((2, 2), dtype=complex)
            if external_impedance is None
            else np.asarray(external_impedance, dtype=complex)
        )
        if external.shape != (2, 2) or not np.isfinite(external).all():
            raise ValueError("external_impedance must be a finite 2x2 matrix")
        dissipative = 0.5 * (external + external.conj().T)
        if np.min(np.linalg.eigvalsh(dissipative)) < -1e-12:
            raise ValueError("external_impedance must be passive")
        self.external_impedance = external.copy()

    def _drive_currents(self, impedance, drive):
        return np.linalg.solve(impedance + self.external_impedance, drive)
