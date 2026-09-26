from __future__ import annotations

import argparse
from pathlib import Path

from sdfmpneo_vnext import ImmutableHybridTeacherDataset
from sdfmpneo_vnext.hybrid_neural import (
    HybridNeuralResidualArtifact,
)
from sdfmpneo_vnext.hybrid_spatial_neural import (
    train_hybrid_spatial_loss_surrogate,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the package-aware continuous PSD loss surrogate for "
            "conductors, dielectric packages, and optional unbounded lossy "
            "background media."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
    )
    parser.add_argument(
        "port_artifact",
        type=Path,
    )
    parser.add_argument(
        "spatial_artifact",
        type=Path,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--field-hidden",
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
        default=20,
    )
    parser.add_argument(
        "--background-segments-per-turn",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--background-radial-order",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--background-angular-order",
        type=int,
        default=48,
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
    if any(
        not sample.has_spatial_truth
        for sample
        in train
        + validation
    ):
        raise SystemExit(
            "hybrid dataset does not contain complete spatial loss truth"
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
            "lossy-background samples require dataset design-domain metadata"
        )

    port = HybridNeuralResidualArtifact.load(
        args.port_artifact,
        device=args.device,
    )
    if lossy_present:
        if not port.supports_lossy_background:
            raise SystemExit(
                "port artifact was not trained for lossy background media"
            )
        if (
            port.background_conductivity_range
            != background_domain
        ):
            raise SystemExit(
                "port artifact background domain does not match the dataset"
            )

    artifact, report = (
        train_hybrid_spatial_loss_surrogate(
            port,
            train,
            validation_samples=(
                validation
            ),
            field_hidden_dim=(
                args.field_hidden
            ),
            factor_rank=(
                args.factor_rank
            ),
            epochs=args.epochs,
            patience=args.patience,
            background_segments_per_turn=(
                args.background_segments_per_turn
            ),
            background_radial_order=(
                args.background_radial_order
            ),
            background_angular_order=(
                args.background_angular_order
            ),
            background_conductivity_range=(
                background_domain
            ),
            device=args.device,
        )
    )
    artifact.save(
        args.spatial_artifact
    )
    print(
        {
            "epochs_run": report.epochs,
            "best_epoch": report.best_epoch,
            "final_loss": report.final_loss,
            "best_validation_error": (
                report.best_validation_error
            ),
            "best_validation_shape_error": (
                report.best_validation_shape_error
            ),
            "stopped_early": (
                report.stopped_early
            ),
            "background_conductivity_domain": (
                background_domain
            ),
            "artifact": str(
                args.spatial_artifact
            ),
        }
    )


if __name__ == "__main__":
    main()
