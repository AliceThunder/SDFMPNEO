from __future__ import annotations

import argparse
from pathlib import Path

from sdfmpneo_vnext import ImmutableTeacherDataset
from sdfmpneo_vnext.neural import NeuralResidualArtifact
from sdfmpneo_vnext.uncertainty import fit_fast_error_calibrator


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate vNext FAST error bounds on the fixed validation split."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    args = parser.parse_args()

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    validation = tuple(
        dataset.iter_samples(
            "validation"
        )
    )
    if not validation:
        raise SystemExit(
            "dataset contains no validation samples"
        )
    artifacts = tuple(
        NeuralResidualArtifact.load(
            path,
            device=args.device,
        )
        for path in args.artifacts
    )
    calibrator = fit_fast_error_calibrator(
        artifacts,
        validation,
        quantile=args.quantile,
        baseline_segments=64,
    )
    calibrator.save(
        args.output
    )
    print(
        "calibrator:",
        {
            "quantile": calibrator.quantile,
            "scale": calibrator.scale,
            "indicator_floor": calibrator.indicator_floor,
            "validation_samples": calibrator.validation_samples,
            "output": str(args.output),
        },
    )


if __name__ == "__main__":
    main()
