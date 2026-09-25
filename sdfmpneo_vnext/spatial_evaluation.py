from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpatialSurrogateAudit:
    samples: int
    mean_weighted_relative_error: float
    maximum_weighted_relative_error: float
    maximum_channel_closure_error: float
    maximum_probe_joule_error: float
    minimum_local_eigenvalue: float
    passed: bool

    def to_dict(self):
        return {
            "samples": self.samples,
            "mean_weighted_relative_error": self.mean_weighted_relative_error,
            "maximum_weighted_relative_error": self.maximum_weighted_relative_error,
            "maximum_channel_closure_error": self.maximum_channel_closure_error,
            "maximum_probe_joule_error": self.maximum_probe_joule_error,
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
    maximum_probe_joule_error_limit: float = 0.20,
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
        or maximum_probe_joule_error_limit
        <= 0.0
        or passivity_tolerance
        < 0.0
    ):
        raise ValueError(
            "invalid spatial audit tolerances"
        )

    errors = []
    closure_errors = []
    maximum_probe_error = 0.0
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
        predicted = np.asarray(
            prepared.local_dissipation_matrices(
                spatial.coil_index,
                spatial.arc_fraction,
                spatial.xy,
            ),
            dtype=complex,
        )
        hermitian = 0.5 * (
            predicted
            + predicted.conj().transpose(
                0,
                2,
                1,
            )
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue,
            float(
                np.min(
                    np.linalg.eigvalsh(
                        hermitian
                    )
                )
            ),
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

        n_ports = int(
            target.shape[-1]
        )
        probes = [
            np.eye(
                n_ports,
                dtype=complex,
            )[:, port]
            for port in range(
                n_ports
            )
        ]
        probes.extend(
            [
                np.ones(
                    n_ports,
                    dtype=complex,
                ),
                (
                    np.arange(
                        1,
                        n_ports + 1,
                        dtype=float,
                    )
                    + 0.3j
                ),
            ]
        )
        for current in probes:
            predicted_joule = 0.5 * np.real(
                np.einsum(
                    "i,qij,j->q",
                    current.conj(),
                    predicted,
                    current,
                )
            )
            target_joule = 0.5 * np.real(
                np.einsum(
                    "i,qij,j->q",
                    current.conj(),
                    target,
                    current,
                )
            )
            joule_numerator = float(
                np.sum(
                    weights
                    * (
                        predicted_joule
                        - target_joule
                    ) ** 2
                )
            )
            joule_denominator = max(
                float(
                    np.sum(
                        weights
                        * target_joule**2
                    )
                ),
                1e-30,
            )
            maximum_probe_error = max(
                maximum_probe_error,
                float(
                    np.sqrt(
                        joule_numerator
                        / joule_denominator
                    )
                ),
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
        and maximum_probe_error
        <= maximum_probe_joule_error_limit
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
        maximum_probe_joule_error=float(
            maximum_probe_error
        ),
        minimum_local_eigenvalue=float(
            minimum_eigenvalue
        ),
        passed=passed,
    )
