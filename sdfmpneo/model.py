from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .training import ElectroThermalResidual


@dataclass(frozen=True)
class OnlinePrediction:
    time: float
    thermal_coordinates: np.ndarray
    thermal_derivative: np.ndarray
    temperature_field: np.ndarray
    reduced_heat_source: np.ndarray
    physical_residual: np.ndarray
    physical_residual_norm: float
    impedance: object | None
    region_losses: dict[str, float] | None
    drive_rhs_residual_dual_norm: float


class ExecutableSDFMPNEOModel:
    """Current executable online map for a fixed geometry/operating model.

    The analytic evolution graph contains the already-trained time evolution for
    the declared operating condition. The electromagnetic reduced model remains
    deterministic physics. If a port set and drive currents are supplied, the
    actual electromagnetic excitation is B @ I and the coupled heat source is
    evaluated for that same excitation.

    This class deliberately does not pretend that changing `drive_currents`
    after training leaves the analytic thermal evolution valid. A future
    parameter-conditioned analytic graph will make operating condition U an
    explicit network input. For the current executable core, the drive vector is
    part of the model's fixed operating condition.
    """

    def __init__(
        self,
        *,
        evolution,
        thermal_model,
        electromagnetic_model,
        ports=None,
        drive_currents: np.ndarray | None = None,
        region_loss_projector=None,
    ) -> None:
        self.evolution = evolution
        self.thermal_model = thermal_model
        self.electromagnetic_model = electromagnetic_model
        self.ports = ports
        self.region_loss_projector = region_loss_projector

        if len(self.thermal_model.lambdas) != self.electromagnetic_model.problem.n_thermal:
            raise ValueError("thermal/electromagnetic reduced dimensions do not match")

        if drive_currents is not None:
            if ports is None:
                raise ValueError("drive_currents require a port set")
            self.drive_currents = np.asarray(drive_currents, dtype=complex)
            self.drive_rhs = ports.rhs_for_currents(self.drive_currents)
        else:
            self.drive_currents = None
            self.drive_rhs = np.asarray(self.electromagnetic_model.problem.b, dtype=complex)

        self.residual_evaluator = ElectroThermalResidual(
            self.thermal_model.lambdas,
            self.electromagnetic_model,
            rhs=self.drive_rhs,
        )

    def evaluate(self, t: float) -> OnlinePrediction:
        if t < 0:
            raise ValueError("query time must be non-negative")

        a, da = self.evolution.evaluate(float(t))
        a = np.asarray(a, dtype=float)
        da = np.asarray(da, dtype=float)

        residual = self.residual_evaluator.evaluate(a, da)
        temperature = self.thermal_model.reconstruct(a)
        heat_source = self.electromagnetic_model.heat_source_for_rhs(a, self.drive_rhs)
        drive_residual = self.electromagnetic_model.residual_dual_norm_for_rhs(a, self.drive_rhs)

        port_result = None
        if self.ports is not None:
            port_result = self.ports.evaluate(
                self.electromagnetic_model.problem,
                a,
                reduced_basis=self.electromagnetic_model.V,
            )

        region_losses = None
        if self.region_loss_projector is not None:
            x = self.electromagnetic_model.state_for_rhs(a, self.drive_rhs)
            region_losses = self.region_loss_projector.evaluate_state(x, a)

        return OnlinePrediction(
            time=float(t),
            thermal_coordinates=a,
            thermal_derivative=da,
            temperature_field=np.asarray(temperature, dtype=float),
            reduced_heat_source=np.asarray(heat_source, dtype=float),
            physical_residual=np.asarray(residual.residual, dtype=float),
            physical_residual_norm=float(residual.norm),
            impedance=port_result,
            region_losses=region_losses,
            drive_rhs_residual_dual_norm=float(drive_residual),
        )
