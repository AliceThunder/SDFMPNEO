from __future__ import annotations

import argparse
from pathlib import Path

from sdfmpneo_vnext.bundle import publish_bundle
from sdfmpneo_vnext.neural import NeuralResidualArtifact
from sdfmpneo_vnext.spatial_neural import NeuralSpatialLossArtifact
from sdfmpneo_vnext.uncertainty import FastErrorCalibrator


def main():
    parser = argparse.ArgumentParser(
        description="Publish a checksum-verified vNext inference bundle."
    )
    parser.add_argument("port_artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--spatial-artifact",
        type=Path,
    )
    parser.add_argument(
        "--calibrator",
        type=Path,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    port = NeuralResidualArtifact.load(
        args.port_artifact,
        device=args.device,
    )
    spatial = None
    if args.spatial_artifact is not None:
        spatial = NeuralSpatialLossArtifact.load(
            args.spatial_artifact,
            port,
            device=args.device,
        )
    calibrator = None
    if args.calibrator is not None:
        calibrator = FastErrorCalibrator.load(
            args.calibrator
        )

    manifest = publish_bundle(
        args.output,
        port,
        spatial_artifact=spatial,
        calibrator=calibrator,
        metadata={
            "source_port_artifact": str(
                args.port_artifact
            ),
            "source_spatial_artifact": (
                None
                if args.spatial_artifact is None
                else str(
                    args.spatial_artifact
                )
            ),
        },
        overwrite=args.overwrite,
    )
    print(
        "published:",
        args.output,
    )
    print(
        "files:",
        sorted(
            manifest[
                "files"
            ]
        ),
    )


if __name__ == "__main__":
    main()
