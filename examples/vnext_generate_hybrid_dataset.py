from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from sdfmpneo_vnext import (
    ImmutableHybridTeacherDataset,
    MQSConfig,
    sample_hybrid_package_scene,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate immutable vNext hybrid dielectric teacher data, "
            "including conductor and package spatial loss truth."
        )
    )
    parser.add_argument(
        "output",
        type=Path,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=37,
    )
    parser.add_argument(
        "--baseline-segments",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--surface-vertical-order",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--surface-azimuthal-order",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--package-volume-axial-order",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--package-volume-radial-order",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--package-volume-azimuthal-order",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit(
            "--count must be >= 1"
        )

    if (
        args.output
        / "manifest.json"
    ).exists():
        dataset = (
            ImmutableHybridTeacherDataset(
                args.output
            )
        )
    else:
        dataset = (
            ImmutableHybridTeacherDataset.create(
                args.output,
                split_seed=args.seed,
            )
        )

    rng = np.random.default_rng(
        args.seed
    )
    teacher = MQSConfig(
        segments_per_turn=12,
        min_segments=16,
        section_degree=1,
        radial_order=4,
        angular_order=24,
        line_order=2,
    )
    for index in range(
        args.count
    ):
        scene, frequency = (
            sample_hybrid_package_scene(
                rng
            )
        )
        record = (
            dataset.generate_and_add(
                scene,
                frequency,
                teacher_config=(
                    teacher
                ),
                baseline_segments=(
                    args.baseline_segments
                ),
                surface_vertical_order=(
                    args.surface_vertical_order
                ),
                surface_azimuthal_order=(
                    args.surface_azimuthal_order
                ),
                include_spatial_truth=True,
                package_volume_axial_order=(
                    args.package_volume_axial_order
                ),
                package_volume_radial_order=(
                    args.package_volume_radial_order
                ),
                package_volume_azimuthal_order=(
                    args.package_volume_azimuthal_order
                ),
                source="initial",
            )
        )
        print(
            f"[{index + 1:04d}/{args.count:04d}] "
            f"{record.sample_id[:12]} "
            f"{record.split} "
            f"{frequency / 1e3:.2f} kHz"
        )
    print(
        "counts:",
        dataset.counts(),
    )


if __name__ == "__main__":
    main()
