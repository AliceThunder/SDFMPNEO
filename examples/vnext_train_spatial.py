from __future__ import annotations

import argparse
from pathlib import Path

from sdfmpneo_vnext import ImmutableTeacherDataset
from sdfmpneo_vnext.neural import NeuralResidualArtifact
from sdfmpneo_vnext.spatial_neural import train_spatial_loss_surrogate


def main():
    parser = argparse.ArgumentParser(
        description="Train the vNext continuous spatial Joule field head."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("port_artifact", type=Path)
    parser.add_argument("spatial_artifact", type=Path)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--factor-rank", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
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
    if not validation_samples:
        raise SystemExit(
            "dataset contains no validation samples"
        )

    port_artifact = NeuralResidualArtifact.load(
        args.port_artifact,
        device=args.device,
    )
    artifact, report = train_spatial_loss_surrogate(
        port_artifact,
        train_samples,
        validation_samples=validation_samples,
        field_hidden_dim=args.hidden,
        factor_rank=args.factor_rank,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )
    artifact.save(
        args.spatial_artifact
    )
    print(
        "spatial training:",
        {
            "epochs_run": report.epochs,
            "final_loss": report.final_loss,
            "best_epoch": report.best_epoch,
            "best_validation_error": report.best_validation_error,
            "stopped_early": report.stopped_early,
        },
    )
    print(
        "artifact:",
        args.spatial_artifact,
    )


if __name__ == "__main__":
    main()
