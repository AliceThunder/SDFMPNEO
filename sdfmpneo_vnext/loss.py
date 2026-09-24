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
        if not (
            0
            <= coil_index
            < len(self.teacher.scene.coils)
        ):
            raise IndexError(
                "coil_index out of range"
            )
        if not (
            0.0
            <= arc_fraction
            <= 1.0
        ):
            raise ValueError(
                "arc_fraction must lie in [0,1]"
            )
        segs = [
            s
            for s in self.teacher._segments
            if s.coil == coil_index
        ]
        idx = min(
            int(
                np.floor(
                    arc_fraction
                    * len(segs)
                )
            ),
            len(segs) - 1,
        )
        seg = segs[idx]
        local = np.asarray(
            xy,
            dtype=float,
        )
        if local.shape != (2,):
            raise ValueError(
                "xy must have shape (2,)"
            )
        geometry = (
            self.teacher.scene.coils[
                coil_index
            ].geometry
        )
        w = (
            0.5
            * geometry.conductor_width
        )
        h = (
            0.5
            * geometry.conductor_thickness
        )
        m = (
            geometry.cross_section_exponent
        )
        inside = (
            (abs(local[0]) / w) ** m
            + (abs(local[1]) / h) ** m
            <= 1.0 + 1e-12
        )
        if not inside:
            return 0.0 + 0.0j
        values = seg.basis.evaluate_xy(
            local
        )
        coeff = self.coefficients[
            seg.mode_slice
        ]
        return complex(
            values @ coeff
        )

    def local_joule_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> float:
        J = self.local_current_density(
            coil_index,
            arc_fraction,
            xy,
        )
        sigma = (
            self.teacher.scene.coils[
                coil_index
            ].material.conductivity
        )
        return float(
            0.5
            * (abs(J) ** 2)
            / sigma
        )
