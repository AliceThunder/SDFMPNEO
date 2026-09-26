from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from sdfmpneo_vnext import (
    HybridSceneSamplerConfig,
    ImmutableHybridTeacherDataset,
    MQSConfig,
    sample_hybrid_package_scene,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate immutable vNext hybrid dielectric teacher data, "
            "including conductor, package, and optional unbounded-background "
            "spatial loss truth."
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
    parser.add_argument(
        "--lossy-background-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--background-epsilon-min",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--background-epsilon-max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--background-conductivity-min",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--background-conductivity-max",
        type=float,
        default=5e-3,
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
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit(
            "--count must be >= 1"
        )

    sampler = HybridSceneSamplerConfig(
        background_relative_permittivity_range=(
            args.background_epsilon_min,
            args.background_epsilon_max,
        ),
        background_conductivity_range=(
            args.background_conductivity_min,
            args.background_conductivity_max,
        ),
        lossy_background_probability=(
            args.lossy_background_probability
        ),
    )
    domain_metadata = {
        "background": {
            "relative_permittivity_range": [
                float(
                    args.background_epsilon_min
                ),
                float(
                    args.background_epsilon_max
                ),
            ],
            "conductivity_range": [
                float(
                    args.background_conductivity_min
                ),
                float(
                    args.background_conductivity_max
                ),
            ],
            "lossy_probability": float(
                args.lossy_background_probability
            ),
        }
    }

    if (
        args.output
        / "manifest.json"
    ).exists():
        dataset = (
            ImmutableHybridTeacherDataset(
                args.output
            )
        )
        if (
            dataset.domain_metadata
            != domain_metadata
        ):
            raise SystemExit(
                "existing dataset background design domain does not match "
                "the requested generator arguments"
            )
    else:
        dataset = (
            ImmutableHybridTeacherDataset.create(
                args.output,
                split_seed=args.seed,
                domain_metadata=(
                    domain_metadata
                ),
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
                rng,
                sampler,
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
                background_radial_order=(
                    args.background_radial_order
                ),
                background_angular_order=(
                    args.background_angular_order
                ),
                source="initial",
            )
        )
        print(
            f"[{index + 1:04d}/{args.count:04d}] "
            f"{record.sample_id[:12]} "
            f"{record.split} "
            f"{frequency / 1e3:.2f} kHz "
            f"sigma_bg={scene.medium.conductivity:.3e} S/m"
        )
    print(
        "counts:",
        dataset.counts(),
    )


if __name__ == "__main__":
    main()
