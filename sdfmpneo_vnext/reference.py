from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .em import MQSConfig
from .mixed import DenseMixedConductorTeacher
from .prediction import StructuredPortPrediction
from .scene import Scene


@dataclass(frozen=True)
class PreparedReferenceLossField:
    scene: Scene
    frequency_hz: float
    teacher: DenseMixedConductorTeacher
    result: object
    port_prediction: StructuredPortPrediction

    @property
    def normalization_closure_error(
        self,
    ) -> float:
        return float(
            self.port_prediction.power_closure_error()
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
        xy,
        currents,
    ) -> float:
        matrix = (
            self.local_dissipation_matrix(
                coil_index,
                arc_fraction,
                xy,
            )
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        return float(
            0.5
            * np.real(
                np.vdot(
                    currents,
                    matrix
                    @ currents,
                )
            )
        )


class MixedReferenceArtifact:
    """Canonical mesh-free current-potential-charge REFERENCE artifact."""

    def __init__(
        self,
        *,
        config: MQSConfig | None = None,
    ):
        self.config = (
            config
            or MQSConfig()
        )

    def solve(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        teacher = (
            DenseMixedConductorTeacher(
                scene,
                frequency_hz,
                self.config,
            )
        )
        return (
            teacher,
            teacher.solve(),
        )

    def predict_structured(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> StructuredPortPrediction:
        _, result = self.solve(
            scene,
            frequency_hz,
        )
        return StructuredPortPrediction(
            result.impedance,
            result.coil_dissipation_matrices(),
        )

    def predict(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> np.ndarray:
        return (
            self.predict_structured(
                scene,
                frequency_hz,
            ).impedance
        )

    def prepare_spatial(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> PreparedReferenceLossField:
        teacher, result = self.solve(
            scene,
            frequency_hz,
        )
        prediction = (
            StructuredPortPrediction(
                result.impedance,
                result.coil_dissipation_matrices(),
            )
        )
        return PreparedReferenceLossField(
            scene,
            float(
                frequency_hz
            ),
            teacher,
            result,
            prediction,
        )
