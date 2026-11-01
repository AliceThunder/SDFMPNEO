"""BFZI current-driven electrothermal envelope on the existing vNext thermal ROM.

The finite-domain seawater MQS response is frozen over slow thermal steps;
finite circular-conductor internal impedance and Joule heat are recomputed
from the vNext material temperature laws at each implicit coupling iterate.
This is a two-coil lumped thermal MVP, not a resolved seawater heat solver.
"""
from __future__ import annotations

import numpy as np

from sdfmpneo_vnext.electrothermal import ElectroThermalStep
from sdfmpneo_vnext.thermal import StableThermalModel
from .sea_reference import SeawaterResponse


class BFZICurrentControlledEnvelope:
    """Advance two coil temperatures with a structure-preserving EM response."""

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
        if max_coupling_iterations < 1:
            raise ValueError("max_coupling_iterations must be positive")
        self.reference = reference
        self.thermal_model = thermal_model
        self.coupling_tolerance = float(coupling_tolerance)
        self.max_coupling_iterations = int(max_coupling_iterations)

    def step(self, state, currents, dt: float) -> ElectroThermalStep:
        z0 = np.asarray(state, dtype=float)
        i = np.asarray(currents, dtype=complex)
        if z0.shape != (2,) or not np.isfinite(z0).all():
            raise ValueError("state must be a finite two-vector")
        if i.shape != (2,) or not np.isfinite(i).all():
            raise ValueError("currents must be a finite complex two-vector")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        guess = z0.copy()
        residual = np.inf
        converged = False
        for iteration in range(1, self.max_coupling_iterations + 1):
            temperature = self.thermal_model.temperature(guess)
            power = self.reference.port_conductor_power(i, temperature)
            candidate = self.thermal_model.advance_constant_power(z0, power, dt)
            scale = max(1.0, float(np.linalg.norm(candidate)))
            residual = float(np.linalg.norm(candidate - guess) / scale)
            guess = candidate
            if residual <= self.coupling_tolerance:
                converged = True
                break
        temperature = self.thermal_model.temperature(guess)
        return ElectroThermalStep(
            guess, temperature, self.reference.port_impedance_at(temperature),
            i.copy(), self.reference.port_conductor_power(i, temperature),
            iteration, residual, converged,
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
