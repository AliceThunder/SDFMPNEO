"""BFZI quasi-static electrothermal envelopes with three physical heat channels.

Reuse vNext StableThermalModel/ElectroThermalStep. With two thermal input
channels, evolve the existing two conductor powers as before. If the thermal
source operator has a third column, inject the total seawater Joule power as
a distinct heat channel into arbitrary reduced thermal states. First two
decoded temperatures must represent the transmitter and receiver coils;
additional decoded temperatures may represent seawater or enclosure nodes.

Seawater MQS remains at reference conductivity, while conductor internal
impedances respond to coil temperature. Optional external passive impedance
enables temperature-dependent voltage/circuit drive.
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
        decoded = thermal_model.temperature(
            np.zeros(thermal_model.n_states)
        )
        if (decoded.ndim != 1 or len(decoded) < 2
                or thermal_model.source.shape[1] not in (2, 3)):
            raise ValueError(
                "BFZI thermal model needs two coil temperatures and "
                "two or three heat input channels"
            )
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
        """Advance one slow step using current or voltage peak phasors.

        The thermal source has either two (coil) or three (coil + seawater)
        heat inputs. Extra thermal states are allowed as long as the first
        two decoded temperatures correspond to the two conductors. The
        fixed reference seawater MQS has no temperature feedback yet.
        """
        z0 = np.asarray(state, dtype=float)
        drive = np.asarray(currents, dtype=complex)
        if (z0.shape != (self.thermal_model.n_states,)
                or not np.isfinite(z0).all()):
            raise ValueError("state must be a finite thermal state vector")
        if drive.shape != (2,) or not np.isfinite(drive).all():
            raise ValueError("drive must be a finite complex two-vector")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        guess = z0.copy()
        residual = np.inf
        converged = False
        for iteration in range(1, self.max_coupling_iterations + 1):
            temperature = self.thermal_model.temperature(guess)
            coil_t = temperature[:2]
            impedance = self.reference.port_impedance_at(coil_t)
            i = self._drive_currents(impedance, drive)
            power = self.reference.port_conductor_power(i, coil_t)
            thermal_power = (
                self.reference.thermal_channel_power(i, coil_t)
                if self.thermal_model.source.shape[1] == 3 else power
            )
            candidate = self.thermal_model.advance_constant_power(
                z0, thermal_power, dt,
            )
            residual = float(
                np.linalg.norm(candidate - guess) /
                max(1.0, float(np.linalg.norm(candidate)))
            )
            guess = candidate
            if residual <= self.coupling_tolerance:
                converged = True
                break
        temperature = self.thermal_model.temperature(guess)
        coil_t = temperature[:2]
        impedance = self.reference.port_impedance_at(coil_t)
        i = self._drive_currents(impedance, drive)
        power = self.reference.port_conductor_power(i, coil_t)
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
