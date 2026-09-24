from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpatialFieldAudit:
    samples: int
    mean_relative_error: float
    p95_relative_error: float
    maximum_relative_error: float
    maximum_probe_joule_error: float
    maximum_channel_closure_error: float
    minimum_local_eigenvalue: float
    passed: bool

    def to_dict(self):
        return {
            "samples": self.samples,
            "mean_relative_error": self.mean_relative_error,
            "p95_relative_error": self.p95_relative_error,
            "maximum_relative_error": self.maximum_relative_error,
            "maximum_probe_joule_error": self.maximum_probe_joule_error,
            "maximum_channel_closure_error": self.maximum_channel_closure_error,
            "minimum_local_eigenvalue": self.minimum_local_eigenvalue,
            "passed": self.passed,
        }


def _weighted_matrix_error(
    predicted,
    target,
    weights,
):
    difference = np.abs(
        predicted - target
    ) ** 2
    target_energy = np.abs(
        target
    ) ** 2
    numerator = float(
        np.sum(
            weights[
                :,
                None,
                None,
            ]
            * difference
        )
    )
    denominator = float(
        np.sum(
            weights[
                :,
                None,
                None,
            ]
            * target_energy
        )
    )
    return float(
        np.sqrt(
            numerator
            / max(
                denominator,
                1e-30,
            )
        )
    )


def _probe_currents(
    n_ports: int,
):
    probes = [
        np.eye(
            n_ports,
            dtype=complex,
        )[:, index]
        for index in range(
            n_ports
        )
    ]
    probes.append(
        np.ones(
            n_ports,
            dtype=complex,
        )
    )
    probes.append(
        np.arange(
            1,
            n_ports + 1,
            dtype=float,
        )
        + 0.3j
    )
    return probes


def _weighted_joule_error(
    predicted,
    target,
    weights,
    current,
):
    q_pred = 0.5 * np.real(
        np.einsum(
            "i,qij,j->q",
            current.conj(),
            predicted,
            current,
        )
    )
    q_true = 0.5 * np.real(
        np.einsum(
            "i,qij,j->q",
            current.conj(),
            target,
            current,
        )
    )
    numerator = float(
        np.sum(
            weights
            * (
                q_pred
                - q_true
            ) ** 2
        )
    )
    denominator = float(
        np.sum(
            weights
            * q_true**2
        )
    )
    return float(
        np.sqrt(
            numerator
            / max(
                denominator,
                1e-30,
            )
        )
    )


def audit_spatial_loss(
    artifact,
    samples,
    *,
    mean_relative_error_limit: float = 0.05,
    maximum_relative_error_limit: float = 0.10,
    maximum_probe_joule_error_limit: float = 0.10,
    channel_closure_tolerance: float = 1e-5,
    passivity_tolerance: float = 1e-9,
) -> SpatialFieldAudit:
    samples = tuple(
        samples
    )
    if not samples:
        raise ValueError(
            "spatial audit requires at least one sample"
        )
    if (
        mean_relative_error_limit <= 0.0
        or maximum_relative_error_limit <= 0.0
        or maximum_probe_joule_error_limit <= 0.0
        or channel_closure_tolerance < 0.0
        or passivity_tolerance < 0.0
    ):
        raise ValueError(
            "invalid spatial audit tolerances"
        )

    sample_errors = []
    max_probe_error = 0.0
    max_closure = 0.0
    minimum_eigenvalue = np.inf

    for sample in samples:
        spatial = sample.spatial_loss
        if (
            spatial is None
            or sample.target_dissipation_channels
            is None
        ):
            raise ValueError(
                "spatial audit sample is missing continuous loss truth"
            )

        predicted_all = np.zeros_like(
            spatial.dissipation_matrix,
            dtype=complex,
        )
        for coil in range(
            len(
                sample.scene.coils
            )
        ):
            mask = (
                spatial.coil_index
                == coil
            )
            if not np.any(
                mask
            ):
                continue
            predicted = (
                artifact.local_dissipation_matrices(
                    sample.scene,
                    sample.frequency_hz,
                    coil,
                    spatial.arc_fraction[
                        mask
                    ],
                    spatial.xy[
                        mask
                    ],
                )
            )
            predicted_all[
                mask
            ] = predicted

            integrated = (
                artifact.integrated_matrix(
                    sample.scene,
                    sample.frequency_hz,
                    coil,
                )
            )
            channel = (
                artifact.predict_structured(
                    sample.scene,
                    sample.frequency_hz,
                ).dissipation_channels[
                    coil
                ]
            )
            closure = float(
                np.linalg.norm(
                    integrated
                    - channel
                )
                / max(
                    np.linalg.norm(
                        channel
                    ),
                    1e-30,
                )
            )
            max_closure = max(
                max_closure,
                closure,
            )

        sample_errors.append(
            _weighted_matrix_error(
                predicted_all,
                spatial.dissipation_matrix,
                spatial.weights,
            )
        )

        eigenvalues = np.linalg.eigvalsh(
            0.5
            * (
                predicted_all
                + predicted_all.conj().transpose(
                    0,
                    2,
                    1,
                )
            )
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue,
            float(
                np.min(
                    eigenvalues
                )
            ),
        )

        for current in _probe_currents(
            sample.target_impedance.shape[
                0
            ]
        ):
            max_probe_error = max(
                max_probe_error,
                _weighted_joule_error(
                    predicted_all,
                    spatial.dissipation_matrix,
                    spatial.weights,
                    current,
                ),
            )

    errors = np.asarray(
        sample_errors,
        dtype=float,
    )
    mean_error = float(
        np.mean(
            errors
        )
    )
    maximum_error = float(
        np.max(
            errors
        )
    )
    passed = bool(
        mean_error
        <= mean_relative_error_limit
        and maximum_error
        <= maximum_relative_error_limit
        and max_probe_error
        <= maximum_probe_joule_error_limit
        and max_closure
        <= channel_closure_tolerance
        and minimum_eigenvalue
        >= -passivity_tolerance
    )
    return SpatialFieldAudit(
        samples=len(
            samples
        ),
        mean_relative_error=(
            mean_error
        ),
        p95_relative_error=float(
            np.quantile(
                errors,
                0.95,
            )
        ),
        maximum_relative_error=(
            maximum_error
        ),
        maximum_probe_joule_error=float(
            max_probe_error
        ),
        maximum_channel_closure_error=float(
            max_closure
        ),
        minimum_local_eigenvalue=float(
            minimum_eigenvalue
        ),
        passed=passed,
    )
