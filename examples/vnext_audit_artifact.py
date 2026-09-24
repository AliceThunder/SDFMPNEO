from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdfmpneo_vnext import (
    ImmutableTeacherDataset,
    audit_surrogate,
)
from sdfmpneo_vnext.neural import NeuralResidualArtifact


def main():
    parser = argparse.ArgumentParser(
        description="Audit a frozen vNext neural artifact on a locked dataset split."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--split",
        choices=("test", "release"),
        default="release",
    )
    parser.add_argument(
        "--mean-limit",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-limit",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    args = parser.parse_args()

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    samples = tuple(
        dataset.iter_samples(args.split)
    )
    if not samples:
        raise SystemExit(
            f"dataset contains no {args.split} samples"
        )
    artifact = NeuralResidualArtifact.load(
        args.artifact,
        device=args.device,
    )
    audit = audit_surrogate(
        artifact,
        samples,
        mean_relative_error_limit=args.mean_limit,
        maximum_relative_error_limit=args.max_limit,
    )
    print(
        json.dumps(
            audit.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(
        0 if audit.passed else 2
    )


if __name__ == "__main__":
    main()
