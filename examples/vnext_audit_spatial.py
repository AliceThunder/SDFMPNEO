from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdfmpneo_vnext import ImmutableTeacherDataset
from sdfmpneo_vnext.neural import NeuralResidualArtifact
from sdfmpneo_vnext.spatial_evaluation import (
    audit_spatial_surrogate,
)
from sdfmpneo_vnext.spatial_neural import (
    SpatialLossArtifact,
)


def main():
    parser = argparse.ArgumentParser(
        description="Audit a frozen vNext continuous spatial Joule artifact."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("port_artifact", type=Path)
    parser.add_argument("spatial_artifact", type=Path)
    parser.add_argument(
        "--split",
        choices=("test", "release"),
        default="release",
    )
    parser.add_argument(
        "--mean-limit",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-limit",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--joule-limit",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--offgrid-mean-limit",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--offgrid-max-limit",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--offgrid-joule-limit",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--cross-grid-closure-limit",
        type=float,
        default=0.02,
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = ImmutableTeacherDataset(
        args.dataset
    )
    samples = tuple(
        dataset.iter_samples(
            args.split
        )
    )
    if not samples:
        raise SystemExit(
            f"dataset contains no {args.split} samples"
        )
    port = NeuralResidualArtifact.load(
        args.port_artifact,
        device=args.device,
    )
    spatial = SpatialLossArtifact.load(
        args.spatial_artifact,
        port,
        device=args.device,
    )
    report = audit_spatial_surrogate(
        spatial,
        samples,
        mean_relative_error_limit=args.mean_limit,
        maximum_relative_error_limit=args.max_limit,
        maximum_probe_joule_error_limit=args.joule_limit,
        mean_offgrid_relative_error_limit=args.offgrid_mean_limit,
        maximum_offgrid_relative_error_limit=args.offgrid_max_limit,
        maximum_offgrid_probe_joule_error_limit=args.offgrid_joule_limit,
        cross_grid_closure_tolerance=args.cross_grid_closure_limit,
    )
    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(
        0 if report.passed else 2
    )


if __name__ == "__main__":
    main()
