from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .em import DenseMQSTeacher, MQSResult


@dataclass(frozen=True)
class ConductorLossField:
    teacher: DenseMQSTeacher
    result: MQSResult
    currents: np.ndarray

    def __post_init__(self):
        i = np.asarray(self.currents, dtype=complex)
        if i.shape != (self.result.n_ports,):
            raise ValueError("currents has wrong shape")
        object.__setattr__(self, "currents", i)

    @property
    def coefficients(self) -> np.ndarray:
        return self.result.mode_current(self.currents)

    def coil_power(self) -> np.ndarray:
        seg = self.result.segment_power(self.currents)
        out = np.zeros(
            len(self.teacher.scene.coils),
            dtype=float,
        )
        for value, coil in zip(
            seg,
            self.result.segment_coils,
        ):
            out[int(coil)] += value
        return out

    def segment_power(self) -> np.ndarray:
        return self.result.segment_power(
            self.currents
        )

    def local_current_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> complex:
        transfer = (
            self.teacher.local_current_transfer(
                self.result,
                coil_index,
                arc_fraction,
                xy,
            )
        )
        return complex(
            transfer
            @ self.currents
        )

    def local_dissipation_matrix(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return (
            self.teacher.local_dissipation_matrix(
                self.result,
                coil_index,
                arc_fraction,
                xy,
            )
        )

    def local_joule_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> float:
        matrix = (
            self.local_dissipation_matrix(
                coil_index,
                arc_fraction,
                xy,
            )
        )
        return float(
            0.5
            * np.real(
                np.vdot(
                    self.currents,
                    matrix
                    @ self.currents,
                )
            )
        )
