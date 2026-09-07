from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WPTMultiPhysicsOutput:
    impedance: object | None
    copper_loss: float | None
    seawater_loss: float | None
    maximum_temperature: float | None
    efficiency: float | None

    @staticmethod
    def from_losses(impedance=None, copper_loss=None, seawater_loss=None,
                    maximum_temperature=None, input_power=None, output_power=None):
        eta = None
        if input_power is not None and input_power != 0 and output_power is not None:
            eta = float(output_power / input_power)
        return WPTMultiPhysicsOutput(
            impedance, copper_loss, seawater_loss,
            maximum_temperature, eta
        )
