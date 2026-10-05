from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .analytic_baseline import analytic_port_baseline
from .features import encode_scene_invariant
from .sampling import MVPSceneSamplerConfig, sample_two_coil_mvp_scene
from .serialization import content_hash, scene_to_dict


@dataclass(frozen=True)
class ActiveLearningCandidate:
    scene: object
    frequency_hz: float
    acquisition_score: float
    baseline_disagreement: float
    ensemble_uncertainty: float
    coverage_distance: float
    feature_vector: np.ndarray


@dataclass(frozen=True)
class ActiveLearningRound:
    candidates_evaluated: int
    selected: tuple[ActiveLearningCandidate, ...]
    records: tuple[object, ...]


def _relative_matrix_distance(
    left,
    right,
) -> float:
    left = np.asarray(
        left,
        dtype=complex,
    )
    right = np.asarray(
        right,
        dtype=complex,
    )
    return float(
        np.linalg.norm(
            left - right
        )
        / max(
            np.linalg.norm(
                right
            ),
            1e-30,
        )
    )


def scene_regime_vector(
    scene,
    frequency_hz: float,
) -> np.ndarray:
    """Permutation-insensitive dimensionless summary for coverage scoring."""
    encoded = encode_scene_invariant(
        scene,
        frequency_hz,
    )
    node = np.asarray(
        encoded.node_features,
        dtype=float,
    )
    pair = np.asarray(
        encoded.pair_features,
        dtype=float,
    )
    n = node.shape[0]
    node_stats = np.concatenate(
        (
            np.mean(
                node,
                axis=0,
            ),
            np.std(
                node,
                axis=0,
            ),
            np.min(
                node,
                axis=0,
            ),
            np.max(
                node,
                axis=0,
            ),
        )
    )
    if n > 1:
        mask = ~np.eye(
            n,
            dtype=bool,
        )
        edge = pair[
            mask
        ]
        pair_stats = np.concatenate(
            (
                np.mean(
                    edge,
                    axis=0,
                ),
                np.std(
                    edge,
                    axis=0,
                ),
                np.min(
                    edge,
                    axis=0,
                ),
                np.max(
                    edge,
                    axis=0,
                ),
            )
        )
    else:
        pair_stats = np.zeros(
            4
            * pair.shape[-1],
            dtype=float,
        )
    return np.concatenate(
        (
            np.asarray(
                [
                    float(n),
                    np.log1p(
                        float(
                            frequency_hz
                        )
                    ),
                ]
            ),
            node_stats,
            pair_stats,
        )
    )


def _coverage_model(
    training_samples,
):
    vectors = np.asarray(
        [
            scene_regime_vector(
                sample.scene,
                sample.frequency_hz,
            )
            for sample
            in training_samples
        ],
        dtype=float,
    )
    if len(
        vectors
    ) == 0:
        return (
            None,
            None,
            None,
        )
    mean = np.mean(
        vectors,
        axis=0,
    )
    scale = np.maximum(
        np.std(
            vectors,
            axis=0,
        ),
        1e-6,
    )
    standardized = (
        vectors - mean
    ) / scale
    return (
        mean,
        scale,
        standardized,
    )


def _coverage_distance(
    vector,
    model,
) -> float:
    mean, scale, training = model
    if training is None:
        return 1.0
    point = (
        vector - mean
    ) / scale
    distances = np.linalg.norm(
        training
        - point[
            None,
            :,
        ],
        axis=1,
    )
    return float(
        np.min(
            distances
        )
        / np.sqrt(
            point.size
        )
    )


def _ensemble_prediction(
    artifacts,
    scene,
    frequency_hz,
):
    matrices = np.asarray(
        [
            artifact.predict(
                scene,
                frequency_hz,
            )
            for artifact in artifacts
        ],
        dtype=complex,
    )
    mean = np.mean(
        matrices,
        axis=0,
    )
    if len(
        matrices
    ) < 2:
        uncertainty = 0.0
    else:
        spread = np.sqrt(
            np.mean(
                np.abs(
                    matrices
                    - mean[
                        None,
                        :,
                        :,
                    ]
                ) ** 2
            )
        )
        uncertainty = float(
            spread
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
    return (
        mean,
        uncertainty,
    )


def score_active_learning_candidates(
    artifacts,
    training_samples,
    candidate_scenes,
    *,
    baseline_segments: int = 64,
    uncertainty_weight: float = 1.0,
    baseline_weight: float = 0.5,
    coverage_weight: float = 0.5,
):
    artifacts = tuple(
        artifacts
    )
    if not artifacts:
        raise ValueError(
            "at least one FAST artifact is required"
        )
    if (
        uncertainty_weight < 0.0
        or baseline_weight < 0.0
        or coverage_weight < 0.0
    ):
        raise ValueError(
            "acquisition weights must be nonnegative"
        )
    training_samples = tuple(
        training_samples
    )
    coverage_model = (
        _coverage_model(
            training_samples
        )
    )
    scored = []
    seen = set()
    for scene, frequency_hz in candidate_scenes:
        identity = content_hash(
            {
                "scene": scene_to_dict(
                    scene
                ),
                "frequency_hz": float(
                    frequency_hz
                ),
            }
        )
        if identity in seen:
            continue
        seen.add(
            identity
        )
        mean_prediction, uncertainty = (
            _ensemble_prediction(
                artifacts,
                scene,
                frequency_hz,
            )
        )
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
            _relative_matrix_distance(
                mean_prediction,
                baseline,
            )
        )
        vector = scene_regime_vector(
            scene,
            frequency_hz,
        )
        coverage = _coverage_distance(
            vector,
            coverage_model,
        )
        score = (
            uncertainty_weight
            * uncertainty
            + baseline_weight
            * disagreement
            + coverage_weight
            * coverage
        )
        scored.append(
            ActiveLearningCandidate(
                scene,
                float(
                    frequency_hz
                ),
                float(
                    score
                ),
                float(
                    disagreement
                ),
                float(
                    uncertainty
                ),
                float(
                    coverage
                ),
                vector,
            )
        )
    return tuple(
        sorted(
            scored,
            key=lambda candidate: (
                -candidate.acquisition_score
            ),
        )
    )


def select_diverse_candidates(
    candidates,
    count: int,
    *,
    diversity_weight: float = 0.25,
):
    candidates = tuple(
        candidates
    )
    if count < 1:
        raise ValueError(
            "count must be >= 1"
        )
    if diversity_weight < 0.0:
        raise ValueError(
            "diversity_weight must be nonnegative"
        )
    if not candidates:
        return ()
    count = min(
        count,
        len(
            candidates
        ),
    )
    vectors = np.asarray(
        [
            candidate.feature_vector
            for candidate in candidates
        ],
        dtype=float,
    )
    mean = np.mean(
        vectors,
        axis=0,
    )
    scale = np.maximum(
        np.std(
            vectors,
            axis=0,
        ),
        1e-6,
    )
    normalized = (
        vectors - mean
    ) / scale
    remaining = set(
        range(
            len(
                candidates
            )
        )
    )
    selected = []
    while (
        len(
            selected
        )
        < count
    ):
        best = None
        best_score = -np.inf
        for index in remaining:
            base = (
                candidates[
                    index
                ].acquisition_score
            )
            if selected:
                distance = min(
                    float(
                        np.linalg.norm(
                            normalized[
                                index
                            ]
                            - normalized[
                                chosen
                            ]
                        )
                        / np.sqrt(
                            normalized.shape[
                                1
                            ]
                        )
                    )
                    for chosen
                    in selected
                )
            else:
                distance = 0.0
            adjusted = (
                base
                * (
                    1.0
                    + diversity_weight
                    * distance
                )
            )
            if adjusted > best_score:
                best_score = (
                    adjusted
                )
                best = index
        selected.append(
            int(
                best
            )
        )
        remaining.remove(
            best
        )
    return tuple(
        candidates[
            index
        ]
        for index in selected
    )


def run_active_learning_round(
    dataset,
    artifacts,
    *,
    candidate_count: int = 128,
    select_count: int = 8,
    seed: int = 101,
    sampler_config: MVPSceneSamplerConfig | None = None,
    teacher_config=None,
    baseline_segments: int = 64,
    uncertainty_weight: float = 1.0,
    baseline_weight: float = 0.5,
    coverage_weight: float = 0.5,
    diversity_weight: float = 0.25,
) -> ActiveLearningRound:
    if (
        candidate_count < 1
        or select_count < 1
    ):
        raise ValueError(
            "candidate_count and select_count must be >= 1"
        )
    if hasattr(
        dataset,
        "reference_backends",
    ):
        existing_backends = (
            dataset.reference_backends(
                "train"
            )
        )
        if (
            existing_backends
            and existing_backends
            != (
                "mixed",
            )
        ):
            raise ValueError(
                "active learning requires a pure mixed-reference training split"
            )
    rng = np.random.default_rng(
        seed
    )
    candidate_scenes = [
        sample_two_coil_mvp_scene(
            rng,
            sampler_config,
        )
        for _ in range(
            candidate_count
        )
    ]
    training_samples = tuple(
        dataset.iter_samples(
            "train"
        )
    )
    scored = (
        score_active_learning_candidates(
            artifacts,
            training_samples,
            candidate_scenes,
            baseline_segments=(
                baseline_segments
            ),
            uncertainty_weight=(
                uncertainty_weight
            ),
            baseline_weight=(
                baseline_weight
            ),
            coverage_weight=(
                coverage_weight
            ),
        )
    )
    selected = (
        select_diverse_candidates(
            scored,
            select_count,
            diversity_weight=(
                diversity_weight
            ),
        )
    )
    records = []
    for candidate in selected:
        records.append(
            dataset.generate_and_add(
                candidate.scene,
                candidate.frequency_hz,
                teacher_config=(
                    teacher_config
                ),
                baseline_segments=(
                    baseline_segments
                ),
                reference_backend="mixed",
                source="active",
                split="train",
            )
        )
    return ActiveLearningRound(
        candidates_evaluated=len(
            scored
        ),
        selected=selected,
        records=tuple(
            records
        ),
    )
