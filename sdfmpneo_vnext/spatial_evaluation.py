from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpatialSurrogateAudit:
    samples: int
    mean_weighted_relative_error: float
    maximum_weighted_relative_error: float
    maximum_channel_closure_error: float
    minimum_local_eigenvalue: float
    passed: bool

    def to_dict(self):
        return {
            "samples": self.samples,
            "mean_weighted_relative_error": self.mean_weighted_relative_error,
            "maximum_weighted_relative_error": self.maximum_weighted_relative_error,
            "maximum_channel_closure_error": self.maximum_channel_closure_error,
            "minimum_local_eigenvalue": self.minimum_local_eigenvalue,
            "passed": self.passed,
        }


def audit_spatial_surrogate(
    spatial_artifact,
    samples,
    *,
    mean_relative_error_limit: float = 0.10,
    maximum_relative_error_limit: float = 0.20,
    channel_closure_tolerance: float = 1e-5,
    passivity_tolerance: float = 1e-10,
) -> SpatialSurrogateAudit:
    samples = tuple(
        samples
    )
    if not samples:
        raise ValueError(
            "spatial audit requires at least one sample"
        )
    if (
        mean_relative_error_limit
        <= 0.0
        or maximum_relative_error_limit
        <= 0.0
        or channel_closure_tolerance
        < 0.0
        or passivity_tolerance
        < 0.0
    ):
        raise ValueError(
            "invalid spatial audit tolerances"
        )

    errors = []
    closure_errors = []
    minimum_eigenvalue = np.inf

    for sample in samples:
        spatial = sample.spatial_loss
        if (
            spatial is None
            or sample.target_dissipation_channels
            is None
        ):
            raise ValueError(
                "spatial audit sample is missing physical field truth"
            )

        prepared = spatial_artifact.prepare(
            sample.scene,
            sample.frequency_hz,
        )
        predicted = []
        for (
            coil,
            arc,
            xy,
        ) in zip(
            spatial.coil_index,
            spatial.arc_fraction,
            spatial.xy,
        ):
            matrix = (
                prepared.local_dissipation_matrix(
                    int(coil),
                    float(arc),
                    xy,
                )
            )
            predicted.append(
                matrix
            )
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                float(
                    np.min(
                        np.linalg.eigvalsh(
                            0.5
                            * (
                                matrix
                                + matrix.conj().T
                            )
                        )
                    )
                ),
            )
        predicted = np.asarray(
            predicted,
            dtype=complex,
        )
        target = np.asarray(
            spatial.dissipation_matrix,
            dtype=complex,
        )
        weights = np.asarray(
            spatial.weights,
            dtype=float,
        )
        numerator = float(
            np.sum(
                weights[
                    :,
                    None,
                    None,
                ]
                * np.abs(
                    predicted
                    - target
                ) ** 2
            )
        )
        denominator = max(
            float(
                np.sum(
                    weights[
                        :,
                        None,
                        None,
                    ]
                    * np.abs(
                        target
                    ) ** 2
                )
            ),
            1e-30,
        )
        errors.append(
            float(
                np.sqrt(
                    numerator
                    / denominator
                )
            )
        )

        closure_errors.append(
            float(
                prepared.normalization_closure_error
            )
        )

    errors = np.asarray(
        errors,
        dtype=float,
    )
    maximum_closure = float(
        np.max(
            closure_errors
        )
    )
    passed = bool(
        float(
            np.mean(
                errors
            )
        )
        <= mean_relative_error_limit
        and float(
            np.max(
                errors
            )
        )
        <= maximum_relative_error_limit
        and maximum_closure
        <= channel_closure_tolerance
        and minimum_eigenvalue
        >= -passivity_tolerance
    )
    return SpatialSurrogateAudit(
        samples=len(
            samples
        ),
        mean_weighted_relative_error=float(
            np.mean(
                errors
            )
        ),
        maximum_weighted_relative_error=float(
            np.max(
                errors
            )
        ),
        maximum_channel_closure_error=(
            maximum_closure
        ),
        minimum_local_eigenvalue=float(
            minimum_eigenvalue
        ),
        passed=passed,
    )
