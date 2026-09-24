from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from .analytic_baseline import analytic_port_baseline


@dataclass(frozen=True)
class FastErrorCalibrator:
    quantile: float
    scale: float
    indicator_floor: float
    validation_samples: int
    baseline_weight: float
    ensemble_weight: float

    def __post_init__(self):
        if not (
            0.0
            < self.quantile
            < 1.0
        ):
            raise ValueError(
                "quantile must lie in (0,1)"
            )
        if (
            self.scale < 0.0
            or self.indicator_floor
            <= 0.0
            or self.validation_samples
            < 1
        ):
            raise ValueError(
                "invalid calibrator parameters"
            )

    def error_bound(
        self,
        indicator: float,
    ) -> float:
        return float(
            self.scale
            * (
                max(
                    float(
                        indicator
                    ),
                    0.0,
                )
                + self.indicator_floor
            )
        )

    def save(
        self,
        path,
    ):
        Path(
            path
        ).write_text(
            json.dumps(
                asdict(
                    self
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def load(
        path,
    ) -> "FastErrorCalibrator":
        return FastErrorCalibrator(
            **json.loads(
                Path(
                    path
                ).read_text(
                    encoding="utf-8"
                )
            )
        )


@dataclass(frozen=True)
class CalibratedFastPrediction:
    impedance: np.ndarray
    relative_error_bound: float
    indicator: float
    baseline_disagreement: float
    ensemble_uncertainty: float


def _relative_error(
    predicted,
    target,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=complex,
    )
    target = np.asarray(
        target,
        dtype=complex,
    )
    return float(
        np.linalg.norm(
            predicted
            - target
        )
        / max(
            np.linalg.norm(
                target
            ),
            1e-30,
        )
    )


def fast_error_indicator(
    artifacts,
    scene,
    frequency_hz: float,
    *,
    baseline_segments: int = 64,
    baseline_weight: float = 0.5,
    ensemble_weight: float = 1.0,
):
    artifacts = tuple(
        artifacts
    )
    if not artifacts:
        raise ValueError(
            "at least one FAST artifact is required"
        )
    if (
        baseline_weight < 0.0
        or ensemble_weight < 0.0
    ):
        raise ValueError(
            "indicator weights must be nonnegative"
        )
    predictions = np.asarray(
        [
            artifact.predict(
                scene,
                frequency_hz,
            )
            for artifact
            in artifacts
        ],
        dtype=complex,
    )
    mean = np.mean(
        predictions,
        axis=0,
    )
    if len(
        predictions
    ) > 1:
        spread = float(
            np.sqrt(
                np.mean(
                    np.abs(
                        predictions
                        - mean[
                            None,
                            :,
                            :,
                        ]
                    ) ** 2
                )
            )
            / max(
                np.sqrt(
                    np.mean(
                        np.abs(
                            mean
                        ) ** 2
                    )
                ),
                1e-30,
            )
        )
    else:
        spread = 0.0
    baseline = (
        analytic_port_baseline(
            scene,
            frequency_hz,
            segments_per_coil=(
                baseline_segments
            ),
        ).impedance
    )
    disagreement = (
        _relative_error(
            mean,
            baseline,
        )
    )
    indicator = float(
        np.sqrt(
            (
                baseline_weight
                * disagreement
            ) ** 2
            + (
                ensemble_weight
                * spread
            ) ** 2
        )
    )
    return (
        mean,
        indicator,
        disagreement,
        spread,
    )


def fit_fast_error_calibrator(
    artifacts,
    validation_samples,
    *,
    quantile: float = 0.95,
    baseline_segments: int = 64,
    baseline_weight: float = 0.5,
    ensemble_weight: float = 1.0,
) -> FastErrorCalibrator:
    validation_samples = tuple(
        validation_samples
    )
    if not validation_samples:
        raise ValueError(
            "calibration requires validation samples"
        )
    if not (
        0.0
        < quantile
        < 1.0
    ):
        raise ValueError(
            "quantile must lie in (0,1)"
        )
    indicators = []
    errors = []
    for sample in validation_samples:
        prediction, indicator, _, _ = (
            fast_error_indicator(
                artifacts,
                sample.scene,
                sample.frequency_hz,
                baseline_segments=(
                    baseline_segments
                ),
                baseline_weight=(
                    baseline_weight
                ),
                ensemble_weight=(
                    ensemble_weight
                ),
            )
        )
        indicators.append(
            indicator
        )
        errors.append(
            _relative_error(
                prediction,
                sample.target_impedance,
            )
        )
    indicators = np.asarray(
        indicators,
        dtype=float,
    )
    errors = np.asarray(
        errors,
        dtype=float,
    )
    positive = indicators[
        indicators > 0.0
    ]
    if len(
        positive
    ):
        indicator_floor = max(
            float(
                np.quantile(
                    positive,
                    0.10,
                )
            )
            * 0.05,
            1e-6,
        )
    else:
        indicator_floor = 1e-6
    ratios = (
        errors
        / (
            indicators
            + indicator_floor
        )
    )
    try:
        scale = float(
            np.quantile(
                ratios,
                quantile,
                method="higher",
            )
        )
    except TypeError:  # NumPy < 1.22 compatibility
        scale = float(
            np.quantile(
                ratios,
                quantile,
                interpolation="higher",
            )
        )
    return FastErrorCalibrator(
        quantile=float(
            quantile
        ),
        scale=max(
            scale,
            0.0,
        ),
        indicator_floor=(
            indicator_floor
        ),
        validation_samples=len(
            validation_samples
        ),
        baseline_weight=float(
            baseline_weight
        ),
        ensemble_weight=float(
            ensemble_weight
        ),
    )


def calibrated_fast_predict(
    artifacts,
    calibrator: FastErrorCalibrator,
    scene,
    frequency_hz: float,
    *,
    baseline_segments: int = 64,
) -> CalibratedFastPrediction:
    (
        impedance,
        indicator,
        disagreement,
        spread,
    ) = fast_error_indicator(
        artifacts,
        scene,
        frequency_hz,
        baseline_segments=(
            baseline_segments
        ),
        baseline_weight=(
            calibrator.baseline_weight
        ),
        ensemble_weight=(
            calibrator.ensemble_weight
        ),
    )
    return CalibratedFastPrediction(
        impedance=np.asarray(
            impedance,
            dtype=complex,
        ),
        relative_error_bound=(
            calibrator.error_bound(
                indicator
            )
        ),
        indicator=float(
            indicator
        ),
        baseline_disagreement=float(
            disagreement
        ),
        ensemble_uncertainty=float(
            spread
        ),
    )
