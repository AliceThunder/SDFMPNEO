from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from sdfmpneo_vnext import ImmutableHybridTeacherDataset
from sdfmpneo_vnext.hybrid_neural import (
    train_hybrid_residual_surrogate,
)


def relative_error(
    predicted,
    target,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(
                predicted
            )
            - np.asarray(
                target
            )
        )
        / max(
            np.linalg.norm(
                np.asarray(
                    target
                )
            ),
            1e-30,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the package-aware vNext hybrid port surrogate, including "
            "declared lossy-background media when present in the dataset."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
    )
    parser.add_argument(
        "artifact",
        type=Path,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--factor-rank",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    args = parser.parse_args()

    dataset = ImmutableHybridTeacherDataset(
        args.dataset
    )
    train = tuple(
        dataset.iter_samples(
            "train"
        )
    )
    validation = tuple(
        dataset.iter_samples(
            "validation"
        )
    )
    if not train:
        raise SystemExit(
            "dataset contains no training samples"
        )
    if not validation:
        raise SystemExit(
            "dataset contains no validation samples"
        )

    lossy_present = any(
        sample.scene.medium.conductivity
        > 0.0
        for sample
        in train
        + validation
    )
    background_domain = (
        dataset.background_conductivity_domain
    )
    if (
        lossy_present
        and background_domain
        is None
    ):
        raise SystemExit(
            "lossy-background samples require dataset design-domain metadata; "
            "regenerate the hybrid dataset with the current generator"
        )

    artifact, report = (
        train_hybrid_residual_surrogate(
            train,
            validation_samples=(
                validation
            ),
            hidden_dim=args.hidden,
            factor_rank=(
                args.factor_rank
            ),
            epochs=args.epochs,
            patience=args.patience,
            background_conductivity_range=(
                background_domain
            ),
            device=args.device,
        )
    )
    artifact.save(
        args.artifact
    )

    validation_z = []
    validation_channels = []
    for sample in validation:
        predicted = (
            artifact.predict_structured(
                sample.scene,
                sample.frequency_hz,
            )
        )
        validation_z.append(
            relative_error(
                predicted.impedance,
                sample.target_impedance,
            )
        )
        validation_channels.append(
            relative_error(
                predicted.dissipation_channels,
                sample.target_dissipation_channels,
            )
        )

    print(
        {
            "epochs_run": report.epochs,
            "samples": report.samples,
            "best_epoch": report.best_epoch,
            "final_loss": report.final_loss,
            "best_validation_score": (
                report.best_validation_score
            ),
            "best_validation_z_error": (
                report.best_validation_z_error
            ),
            "best_validation_channel_error": (
                report.best_validation_channel_error
            ),
            "stopped_early": (
                report.stopped_early
            ),
            "validation_mean_z_error": float(
                np.mean(
                    validation_z
                )
            ),
            "validation_mean_channel_error": float(
                np.mean(
                    validation_channels
                )
            ),
            "background_conductivity_domain": (
                background_domain
            ),
            "artifact": str(
                args.artifact
            ),
        }
    )


if __name__ == "__main__":
    main()
