from __future__ import annotations

import argparse
from pathlib import Path

from sdfmpneo_vnext import (
    ImmutableTeacherDataset,
    MQSConfig,
)
from sdfmpneo_vnext.active_learning import (
    run_active_learning_round,
)
from sdfmpneo_vnext.neural import NeuralResidualArtifact


def main():
    parser = argparse.ArgumentParser(
        description="Run one vNext physics-guided active-learning round."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--select",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    args = parser.parse_args()

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    artifacts = tuple(
        NeuralResidualArtifact.load(
            path,
            device=args.device,
        )
        for path in args.artifacts
    )
    teacher = MQSConfig(
        segments_per_turn=12,
        min_segments=16,
        section_degree=1,
        radial_order=4,
        angular_order=24,
        line_order=2,
    )
    result = run_active_learning_round(
        dataset,
        artifacts,
        candidate_count=args.candidates,
        select_count=args.select,
        seed=args.seed,
        teacher_config=teacher,
        baseline_segments=64,
    )
    print(
        "active-learning:",
        {
            "candidates_evaluated": result.candidates_evaluated,
            "selected": len(
                result.selected
            ),
            "dataset_counts": dataset.counts(),
        },
    )
    for candidate, record in zip(
        result.selected,
        result.records,
    ):
        print(
            record.sample_id,
            {
                "score": candidate.acquisition_score,
                "uncertainty": candidate.ensemble_uncertainty,
                "baseline_disagreement": candidate.baseline_disagreement,
                "coverage": candidate.coverage_distance,
                "split": record.split,
                "backend": record.reference_backend,
            },
        )


if __name__ == "__main__":
    main()
