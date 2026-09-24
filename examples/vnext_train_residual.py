from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from sdfmpneo_vnext import ImmutableTeacherDataset
from sdfmpneo_vnext.neural import (
    train_residual_surrogate,
)


def relative_error(predicted, target):
    return float(
        np.linalg.norm(predicted - target)
        / max(
            np.linalg.norm(target),
            1e-30,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the vNext physics-factored neural residual."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--factor-rank", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    train_samples = tuple(
        dataset.iter_samples("train")
    )
    validation_samples = tuple(
        dataset.iter_samples("validation")
    )
    if not train_samples:
        raise SystemExit(
            "dataset contains no training samples"
        )

    artifact, report = train_residual_surrogate(
        train_samples,
        hidden_dim=args.hidden,
        factor_rank=args.factor_rank,
        epochs=args.epochs,
        device=args.device,
    )
    artifact.save(args.artifact)

    train_error = [
        relative_error(
            artifact.predict(
                sample.scene,
                sample.frequency_hz,
            ),
            sample.target_impedance,
        )
        for sample in train_samples
    ]
    validation_error = [
        relative_error(
            artifact.predict(
                sample.scene,
                sample.frequency_hz,
            ),
            sample.target_impedance,
        )
        for sample in validation_samples
    ]
    print(
        "training:",
        {
            "epochs": report.epochs,
            "samples": report.samples,
            "final_loss": report.final_loss,
            "mean_relative_error": float(
                np.mean(train_error)
            ),
            "max_relative_error": float(
                np.max(train_error)
            ),
        },
    )
    if validation_error:
        print(
            "validation:",
            {
                "samples": len(validation_error),
                "mean_relative_error": float(
                    np.mean(validation_error)
                ),
                "max_relative_error": float(
                    np.max(validation_error)
                ),
            },
        )
    else:
        print(
            "validation: no samples in validation split"
        )
    print("artifact:", args.artifact)


if __name__ == "__main__":
    main()
