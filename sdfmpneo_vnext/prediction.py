from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StructuredPortPrediction:
    impedance: np.ndarray
    dissipation_channels: np.ndarray

    def __post_init__(self):
        impedance = np.asarray(
            self.impedance,
            dtype=complex,
        )
        channels = np.asarray(
            self.dissipation_channels,
            dtype=complex,
        )
        if (
            impedance.ndim != 2
            or impedance.shape[0]
            != impedance.shape[1]
        ):
            raise ValueError(
                "impedance must be square"
            )
        n = impedance.shape[0]
        if channels.shape != (
            n,
            n,
            n,
        ):
            raise ValueError(
                "dissipation_channels must have shape (n_ports,n_ports,n_ports)"
            )
        object.__setattr__(
            self,
            "impedance",
            impedance,
        )
        object.__setattr__(
            self,
            "dissipation_channels",
            channels,
        )

    def coil_power(
        self,
        currents,
    ) -> np.ndarray:
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if currents.shape != (
            self.impedance.shape[0],
        ):
            raise ValueError(
                "currents have wrong shape"
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
                for channel
                in self.dissipation_channels
            ],
            dtype=float,
        )

    def power_closure_error(
        self,
    ) -> float:
        summed = np.sum(
            self.dissipation_channels,
            axis=0,
        )
        target = 0.5 * (
            self.impedance
            + self.impedance.conj().T
        )
        return float(
            np.linalg.norm(
                summed
                - target
            )
            / max(
                np.linalg.norm(
                    target
                ),
                1e-30,
            )
        )

    def reciprocity_defect(
        self,
    ) -> float:
        return float(
            np.linalg.norm(
                self.impedance
                - self.impedance.T
            )
            / max(
                np.linalg.norm(
                    self.impedance
                ),
                1e-30,
            )
        )
