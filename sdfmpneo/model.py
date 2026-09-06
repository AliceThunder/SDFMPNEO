from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .certification import certify_multiport_impedance
from .training import (
    AffineOperatingRHSMap,
    ElectroThermalResidual,
    ParametricElectroThermalResidual,
)


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


@dataclass(frozen=True)
class ParametricOnlinePrediction:
    time: float
    initial_coordinates: np.ndarray
    operating: np.ndarray
    thermal_coordinates: np.ndarray
    thermal_derivative: np.ndarray
    temperature_field: np.ndarray
    state_operating_jacobian: np.ndarray
    derivative_operating_jacobian: np.ndarray
    reduced_heat_source: np.ndarray
    heat_source_operating_jacobian: np.ndarray
    physical_residual: np.ndarray
    physical_residual_norm: float
    residual_operating_jacobian: np.ndarray
    drive_rhs: np.ndarray
    drive_rhs_residual_dual_norm: float
    impedance: object | None
    impedance_certificate: object | None
    region_losses: dict[str, float] | None


class ExecutableSDFMPNEOModel:
    """Executable arbitrary-time map for one fixed geometry and operating point."""

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


class ParametricExecutableSDFMPNEOModel:
    """Arbitrary-time analytic map for arbitrary a0 and declared static U.

    This is the first executable `(a0,U,t)->state` model. `U` enters the neural
    graph as explicit zero-dynamics analytic nodes and enters the electromagnetic
    physics through an exact affine RHS map `b(U)=b0+B_U U`.

    The thermal spectrum and spatial electromagnetic operator family are fixed in
    this class. Hence it is suitable for operating parameters that change the
    excitation but not the underlying spatial operators (for example declared
    current-source amplitudes). Frequency, material, load/circuit constraints,
    or geometry parameters that change the electromagnetic operator require the
    corresponding certified physical parameter interface and are not silently
    treated as RHS-only inputs.
    """

    def __init__(
        self,
        *,
        evolution,
        thermal_model,
        electromagnetic_model,
        rhs_map: AffineOperatingRHSMap,
        ports=None,
        region_loss_projector=None,
    ) -> None:
        self.evolution = evolution
        self.thermal_model = thermal_model
        self.electromagnetic_model = electromagnetic_model
        self.rhs_map = rhs_map
        self.ports = ports
        self.region_loss_projector = region_loss_projector

        if evolution.n_modes != len(thermal_model.lambdas):
            raise ValueError("analytic/thermal dimensions do not match")
        if evolution.n_modes != electromagnetic_model.problem.n_thermal:
            raise ValueError("analytic/electromagnetic thermal dimensions do not match")
        if rhs_map.n_operating != len(evolution.operating_names):
            raise ValueError("operating parameter dimensions do not match")
        if rhs_map.n_em != electromagnetic_model.problem.n_em:
            raise ValueError("RHS/electromagnetic coordinate dimensions do not match")
        if not np.array_equal(np.asarray(evolution.lambdas), np.asarray(thermal_model.lambdas)):
            raise ValueError("analytic and thermal spectra must be identical")

        self.residual_evaluator = ParametricElectroThermalResidual(
            evolution,
            electromagnetic_model,
            rhs_map,
        )

    def evaluate(
        self,
        t: float,
        *,
        a0: np.ndarray,
        operating: np.ndarray,
    ) -> ParametricOnlinePrediction:
        if t < 0:
            raise ValueError("query time must be non-negative")
        initial = np.asarray(a0, dtype=float)
        u = np.asarray(operating, dtype=float)
        sample = self.residual_evaluator.evaluate(float(t), a0=initial, operating=u)

        temperature = self.thermal_model.reconstruct(sample.a)
        drive_residual = self.electromagnetic_model.residual_dual_norm_for_rhs(
            sample.a,
            sample.rhs,
        )

        port_result = None
        port_certificate = None
        if self.ports is not None:
            port_result = self.ports.evaluate(
                self.electromagnetic_model.problem,
                sample.a,
                reduced_basis=self.electromagnetic_model.V,
            )
            port_certificate = certify_multiport_impedance(
                self.electromagnetic_model.problem,
                self.ports,
                sample.a,
                self.electromagnetic_model.V,
            )

        region_losses = None
        if self.region_loss_projector is not None:
            x = self.electromagnetic_model.state_for_rhs(sample.a, sample.rhs)
            region_losses = self.region_loss_projector.evaluate_state(x, sample.a)

        return ParametricOnlinePrediction(
            time=float(t),
            initial_coordinates=initial.copy(),
            operating=u.copy(),
            thermal_coordinates=sample.a,
            thermal_derivative=sample.da,
            temperature_field=np.asarray(temperature, dtype=float),
            state_operating_jacobian=sample.state_operating_jacobian,
            derivative_operating_jacobian=sample.derivative_operating_jacobian,
            reduced_heat_source=sample.g_em,
            heat_source_operating_jacobian=sample.heat_source_operating_jacobian,
            physical_residual=sample.residual,
            physical_residual_norm=sample.norm,
            residual_operating_jacobian=sample.residual_operating_jacobian,
            drive_rhs=sample.rhs,
            drive_rhs_residual_dual_norm=float(drive_residual),
            impedance=port_result,
            impedance_certificate=port_certificate,
            region_losses=region_losses,
        )
