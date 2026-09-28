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
        "--package-count-min",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--package-count-max",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--nested-package-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--nested-package-scale-min",
        type=float,
        default=1.15,
    )
    parser.add_argument(
        "--nested-package-scale-max",
        type=float,
        default=1.45,
    )
    parser.add_argument(
        "--package-offset-fraction-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--package-offset-fraction-max",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--package-mu-min",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--package-mu-max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--debye-package-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--multi-debye-package-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--package-debye-epsilon-infinite-min",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--package-debye-epsilon-infinite-max",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--package-debye-delta-epsilon-min",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--package-debye-delta-epsilon-max",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--package-debye-relaxation-time-min",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--package-debye-relaxation-time-max",
        type=float,
        default=1e-4,
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
        "--debye-background-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--multi-debye-background-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--multi-debye-min-poles",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--multi-debye-max-poles",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--background-debye-epsilon-infinite-min",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--background-debye-epsilon-infinite-max",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--background-debye-delta-epsilon-min",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--background-debye-delta-epsilon-max",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--background-debye-relaxation-time-min",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--background-debye-relaxation-time-max",
        type=float,
        default=1e-4,
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
        package_count_range=(
            args.package_count_min,
            args.package_count_max,
        ),
        nested_package_probability=(
            args.nested_package_probability
        ),
        nested_package_scale_range=(
            args.nested_package_scale_min,
            args.nested_package_scale_max,
        ),
        package_center_offset_fraction_range=(
            args.package_offset_fraction_min,
            args.package_offset_fraction_max,
        ),
        package_relative_permeability_range=(
            args.package_mu_min,
            args.package_mu_max,
        ),
        debye_package_probability=(
            args.debye_package_probability
        ),
        multi_debye_package_probability=(
            args.multi_debye_package_probability
        ),
        package_debye_epsilon_infinite_range=(
            args.package_debye_epsilon_infinite_min,
            args.package_debye_epsilon_infinite_max,
        ),
        package_debye_delta_epsilon_range=(
            args.package_debye_delta_epsilon_min,
            args.package_debye_delta_epsilon_max,
        ),
        package_debye_relaxation_time_range=(
            args.package_debye_relaxation_time_min,
            args.package_debye_relaxation_time_max,
        ),
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
        debye_background_probability=(
            args.debye_background_probability
        ),
        multi_debye_background_probability=(
            args.multi_debye_background_probability
        ),
        multi_debye_poles_range=(
            args.multi_debye_min_poles,
            args.multi_debye_max_poles,
        ),
        background_debye_epsilon_infinite_range=(
            args.background_debye_epsilon_infinite_min,
            args.background_debye_epsilon_infinite_max,
        ),
        background_debye_delta_epsilon_range=(
            args.background_debye_delta_epsilon_min,
            args.background_debye_delta_epsilon_max,
        ),
        background_debye_relaxation_time_range=(
            args.background_debye_relaxation_time_min,
            args.background_debye_relaxation_time_max,
        ),
    )
    domain_metadata = {
        "geometry": (
            sampler.geometry_domain_metadata()
        ),
        "background": (
            sampler.background_domain_metadata()
        ),
        "package": (
            sampler.package_domain_metadata()
        ),
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
                "existing dataset material design domain does not match "
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
            f"sigma_eff_bg={scene.medium.loss_conductivity(frequency):.3e} S/m "
            f"medium={type(scene.medium).__name__} "
            f"n_packages={len(scene.packages)} "
            f"packages={[type(package.material).__name__ for package in scene.packages]}"
        )
    print(
        "counts:",
        dataset.counts(),
    )


if __name__ == "__main__":
    main()
