from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StructuredPortPrediction:
    impedance: np.ndarray
    dissipation_channels: np.ndarray
    channel_labels: tuple[str, ...] | None = None

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
        if (
            channels.ndim != 3
            or channels.shape[1:]
            != (
                n,
                n,
            )
            or channels.shape[0] < 1
        ):
            raise ValueError(
                "dissipation_channels must have shape "
                "(n_channels,n_ports,n_ports)"
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
        labels = self.channel_labels
        if labels is not None:
            labels = tuple(
                str(
                    label
                )
                for label in labels
            )
            if (
                len(
                    labels
                )
                != channels.shape[
                    0
                ]
                or any(
                    not label
                    for label in labels
                )
                or len(
                    set(
                        labels
                    )
                )
                != len(
                    labels
                )
            ):
                raise ValueError(
                    "channel_labels must be unique nonempty labels with one "
                    "entry per dissipation channel"
                )
            object.__setattr__(
                self,
                "channel_labels",
                labels,
            )

    def channel_index(
        self,
        label: str,
    ) -> int:
        if self.channel_labels is None:
            raise ValueError(
                "prediction does not provide channel labels"
            )
        try:
            return int(
                self.channel_labels.index(
                    str(
                        label
                    )
                )
            )
        except ValueError as exc:
            raise KeyError(
                f"unknown dissipation channel label: {label}"
            ) from exc

    def coil_channel_indices(
        self,
        n_coils: int,
    ) -> tuple[int, ...]:
        if (
            not isinstance(
                n_coils,
                (int, np.integer),
            )
            or n_coils < 1
        ):
            raise ValueError(
                "n_coils must be a positive integer"
            )
        if self.channel_labels is None:
            if self.n_channels != n_coils:
                raise ValueError(
                    "unlabeled prediction cannot identify coil channels when "
                    "extra dissipation channels are present"
                )
            return tuple(
                range(
                    n_coils
                )
            )
        indices = []
        for coil in range(
            n_coils
        ):
            candidates = (
                f"coil:{coil}",
                f"conductor:{coil}",
            )
            found = [
                self.channel_labels.index(
                    label
                )
                for label in candidates
                if label in self.channel_labels
            ]
            if len(
                found
            ) != 1:
                raise ValueError(
                    f"prediction does not uniquely identify coil channel {coil}"
                )
            indices.append(
                int(
                    found[
                        0
                    ]
                )
            )
        return tuple(
            indices
        )

    @property
    def n_channels(
        self,
    ) -> int:
        return int(
            self.dissipation_channels.shape[0]
        )

    def channel_power(
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

    def coil_power(
        self,
        currents,
    ) -> np.ndarray:
        if (
            self.n_channels
            != self.impedance.shape[0]
        ):
            raise ValueError(
                "coil_power is defined only when there is exactly one "
                "dissipation channel per port/coil; use channel_power for "
                "general structured predictions"
            )
        return self.channel_power(
            currents
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
