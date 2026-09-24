from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .scene import Scene


def _inside_superellipse(
    geometry,
    xy,
) -> bool:
    xy = np.asarray(
        xy,
        dtype=float,
    )
    if xy.shape != (2,):
        raise ValueError(
            "xy must have shape (2,)"
        )
    a = 0.5 * geometry.conductor_width
    b = 0.5 * geometry.conductor_thickness
    m = geometry.cross_section_exponent
    return bool(
        (abs(xy[0]) / a) ** m
        + (abs(xy[1]) / b) ** m
        <= 1.0 + 1e-12
    )


@dataclass(frozen=True)
class PreparedUniformLossField:
    scene: Scene
    frequency_hz: float
    prediction: object
    volumes: np.ndarray

    @property
    def port_prediction(self):
        return self.prediction

    @property
    def normalization_closure_error(self) -> float:
        if hasattr(
            self.prediction,
            "power_closure_error",
        ):
            return float(
                self.prediction.power_closure_error()
            )
        return 0.0

    def local_dissipation_matrices(
        self,
        coil_index,
        arc_fraction,
        xy,
    ) -> np.ndarray:
        coil_index = np.asarray(
            coil_index,
            dtype=int,
        )
        arc_fraction = np.asarray(
            arc_fraction,
            dtype=float,
        )
        xy = np.asarray(
            xy,
            dtype=float,
        )
        if coil_index.ndim != 1:
            raise ValueError(
                "coil_index must be one-dimensional"
            )
        n_query = len(
            coil_index
        )
        if (
            arc_fraction.shape != (n_query,)
            or xy.shape != (n_query, 2)
        ):
            raise ValueError(
                "spatial query arrays have incompatible shapes"
            )
        if np.any(
            (arc_fraction < 0.0)
            | (arc_fraction > 1.0)
        ):
            raise ValueError(
                "arc_fraction must lie in [0,1]"
            )
        if np.any(
            (coil_index < 0)
            | (
                coil_index
                >= len(
                    self.scene.coils
                )
            )
        ):
            raise IndexError(
                "coil_index out of range"
            )
        n_ports = len(
            self.scene.coils
        )
        out = np.zeros(
            (
                n_query,
                n_ports,
                n_ports,
            ),
            dtype=complex,
        )
        for index in range(
            n_query
        ):
            coil = int(
                coil_index[index]
            )
            geometry = (
                self.scene.coils[
                    coil
                ].geometry
            )
            if _inside_superellipse(
                geometry,
                xy[index],
            ):
                out[index] = (
                    np.asarray(
                        self.prediction.dissipation_channels[
                            coil
                        ],
                        dtype=complex,
                    )
                    / self.volumes[
                        coil
                    ]
                )
        return out

    def local_dissipation_matrix(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return self.local_dissipation_matrices(
            np.asarray(
                [coil_index],
                dtype=int,
            ),
            np.asarray(
                [arc_fraction],
                dtype=float,
            ),
            np.asarray(
                [xy],
                dtype=float,
            ),
        )[0]

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


class UniformLossFieldDecoder:
    """Continuous PSD spatial baseline with exact integrated coil power.

    The decoder distributes each coil-level dissipation channel uniformly over
    that conductor's physical volume. It is intentionally simple and serves as
    the structure-preserving baseline for later learned spatial-shape
    corrections.
    """

    def __init__(
        self,
        port_artifact,
        *,
        length_segments: int = 96,
    ):
        if not hasattr(
            port_artifact,
            "predict_structured",
        ):
            raise TypeError(
                "port_artifact must expose predict_structured"
            )
        if length_segments < 8:
            raise ValueError(
                "length_segments must be >= 8"
            )
        self.port_artifact = (
            port_artifact
        )
        self.length_segments = int(
            length_segments
        )

    def _coil_volume(
        self,
        scene: Scene,
        coil_index: int,
    ) -> float:
        geometry = (
            scene.coils[
                coil_index
            ].geometry
        )
        length = (
            geometry.polyline(
                self.length_segments
            ).total_length
        )
        return float(
            length
            * geometry.cross_section_area
        )

    def prepare(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> PreparedUniformLossField:
        prediction = (
            self.port_artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )
        volumes = np.asarray(
            [
                self._coil_volume(
                    scene,
                    coil
                )
                for coil in range(
                    len(
                        scene.coils
                    )
                )
            ],
            dtype=float,
        )
        return PreparedUniformLossField(
            scene,
            float(
                frequency_hz
            ),
            prediction,
            volumes,
        )

    def local_dissipation_matrix(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return self.prepare(
            scene,
            frequency_hz,
        ).local_dissipation_matrix(
            coil_index,
            arc_fraction,
            xy,
        )

    def local_joule_density(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        return self.prepare(
            scene,
            frequency_hz,
        ).local_joule_density(
            coil_index,
            arc_fraction,
            xy,
            currents,
        )

    def coil_integrated_matrix(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
    ) -> np.ndarray:
        """Return the exact analytic integral of the uniform field baseline."""
        prediction = (
            self.port_artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )
        return np.asarray(
            prediction.dissipation_channels[
                coil_index
            ],
            dtype=complex,
        )
