from __future__ import annotations

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

    def local_dissipation_matrix(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        if not (
            0
            <= coil_index
            < len(scene.coils)
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
        geometry = (
            scene.coils[
                coil_index
            ].geometry
        )
        if not _inside_superellipse(
            geometry,
            xy,
        ):
            n = len(
                scene.coils
            )
            return np.zeros(
                (n, n),
                dtype=complex,
            )
        prediction = (
            self.port_artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )
        volume = self._coil_volume(
            scene,
            coil_index,
        )
        return (
            prediction.dissipation_channels[
                coil_index
            ]
            / volume
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
        matrix = (
            self.local_dissipation_matrix(
                scene,
                frequency_hz,
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
